"""
Fault catalogue and propagation engine for IT Fault Environment.

This module defines all 15 fault types and the propagation queue system
that handles fault spread through the microservice topology.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import heapq
import numpy as np
import networkx as nx

from .graph import ServiceNode


@dataclass
class FaultType:
    """
    Defines a fault type and its propagation characteristics.

    Attributes:
        name: Internal fault identifier
        display: Human-readable fault name
        targets: List of tiers this fault can spawn on
        health_decay: Health lost per step on the host node
        latency_multiplier: Multiplier applied to node latency
        error_rate_delta: Increase in error rate
        propagation_delay: (min_steps, max_steps) before spreading downstream
        downstream_severity: Fraction of decay passed to downstream nodes
        cascade_to: Fault names that can emerge if untreated
        cascade_threshold: Host health below which cascade fires
        log_template_key: Key into templates/*.json for log generation
        alert_codes: List of alert codes this fault generates
    """

    name: str
    display: str
    targets: List[str]  # which tiers this fault spawns on
    health_decay: float  # health lost per step on host node
    latency_multiplier: float
    error_rate_delta: float
    propagation_delay: Tuple[int, int]  # (min_steps, max_steps)
    downstream_severity: float  # fraction of decay passed downstream
    cascade_to: List[str]  # fault names that can emerge if untreated
    cascade_threshold: float  # host health below this -> cascade fires
    log_template_key: str  # key into templates/*.json
    alert_codes: List[str]


# =============================================================================
# FAULT CATALOGUE - 15 fault types across 4 categories
# =============================================================================

FAULT_CATALOGUE: Dict[str, FaultType] = {
    # -------------------------------------------------------------------------
    # DATABASE FAULTS (targets=["db"])
    # -------------------------------------------------------------------------

    "connection_pool_exhausted": FaultType(
        name="connection_pool_exhausted",
        display="Connection Pool Exhausted",
        targets=["db"],
        health_decay=0.06,
        latency_multiplier=3.5,
        error_rate_delta=0.08,
        propagation_delay=(2, 5),
        downstream_severity=0.7,
        cascade_to=["upstream_timeout"],
        cascade_threshold=0.5,
        log_template_key="connection_pool_exhausted",
        alert_codes=["DB_CONN_POOL_HIGH", "DB_LATENCY_CRIT"],
    ),
    "disk_io_saturation": FaultType(
        name="disk_io_saturation",
        display="Disk I/O Saturation",
        targets=["db"],
        health_decay=0.04,
        latency_multiplier=4.0,
        error_rate_delta=0.03,
        propagation_delay=(3, 7),
        downstream_severity=0.5,
        cascade_to=["memory_leak"],
        cascade_threshold=0.4,
        log_template_key="disk_io_saturation",
        alert_codes=["DISK_IO_HIGH", "DISK_WAIT_CRIT"],
    ),
    "replication_lag": FaultType(
        name="replication_lag",
        display="Replication Lag",
        targets=["db"],
        health_decay=0.03,
        latency_multiplier=2.0,
        error_rate_delta=0.05,
        propagation_delay=(4, 8),
        downstream_severity=0.4,
        cascade_to=[],
        cascade_threshold=0.3,
        log_template_key="replication_lag",
        alert_codes=["DB_REPL_LAG_HIGH"],
    ),
    "deadlock_storm": FaultType(
        name="deadlock_storm",
        display="Deadlock Storm",
        targets=["db"],
        health_decay=0.08,
        latency_multiplier=5.0,
        error_rate_delta=0.12,
        propagation_delay=(1, 3),
        downstream_severity=0.8,
        cascade_to=["connection_pool_exhausted"],
        cascade_threshold=0.6,
        log_template_key="deadlock_storm",
        alert_codes=["DB_DEADLOCK", "DB_CONN_POOL_HIGH"],
    ),
    # -------------------------------------------------------------------------
    # NETWORK FAULTS (targets=["frontend", "api", "backend"])
    # -------------------------------------------------------------------------

    "packet_loss": FaultType(
        name="packet_loss",
        display="Packet Loss",
        targets=["frontend", "api", "backend"],
        health_decay=0.05,
        latency_multiplier=2.5,
        error_rate_delta=0.06,
        propagation_delay=(1, 4),
        downstream_severity=0.6,
        cascade_to=["dependency_timeout"],
        cascade_threshold=0.5,
        log_template_key="packet_loss",
        alert_codes=["NET_PACKET_LOSS", "NET_LATENCY_HIGH"],
    ),
    "dns_resolution_failure": FaultType(
        name="dns_resolution_failure",
        display="DNS Resolution Failure",
        targets=["frontend", "api", "backend"],
        health_decay=0.07,
        latency_multiplier=6.0,
        error_rate_delta=0.15,
        propagation_delay=(1, 2),
        downstream_severity=0.9,
        cascade_to=["circuit_breaker_open"],
        cascade_threshold=0.6,
        log_template_key="dns_resolution_failure",
        alert_codes=["DNS_FAIL", "NET_UNREACHABLE"],
    ),
    "network_partition": FaultType(
        name="network_partition",
        display="Network Partition",
        targets=["frontend", "api", "backend"],
        health_decay=0.09,
        latency_multiplier=8.0,
        error_rate_delta=0.20,
        propagation_delay=(1, 2),
        downstream_severity=1.0,
        cascade_to=["circuit_breaker_open", "dependency_timeout"],
        cascade_threshold=0.7,
        log_template_key="network_partition",
        alert_codes=["NET_PARTITION", "NET_UNREACHABLE", "SVC_DOWN"],
    ),
    # -------------------------------------------------------------------------
    # COMPUTE FAULTS (targets=["api", "backend"])
    # -------------------------------------------------------------------------

    "memory_leak": FaultType(
        name="memory_leak",
        display="Memory Leak",
        targets=["api", "backend"],
        health_decay=0.03,
        latency_multiplier=1.5,
        error_rate_delta=0.01,
        propagation_delay=(5, 10),
        downstream_severity=0.3,
        cascade_to=["oom_kill_loop"],
        cascade_threshold=0.3,
        log_template_key="memory_leak",
        alert_codes=["MEM_HIGH", "MEM_LEAK_DETECTED"],
    ),
    "cpu_saturation": FaultType(
        name="cpu_saturation",
        display="CPU Saturation",
        targets=["api", "backend"],
        health_decay=0.05,
        latency_multiplier=2.0,
        error_rate_delta=0.04,
        propagation_delay=(2, 6),
        downstream_severity=0.5,
        cascade_to=["thread_pool_exhaustion"],
        cascade_threshold=0.4,
        log_template_key="cpu_saturation",
        alert_codes=["CPU_HIGH", "CPU_THROTTLE"],
    ),
    "oom_kill_loop": FaultType(
        name="oom_kill_loop",
        display="OOM Kill Loop",
        targets=["api", "backend"],
        health_decay=0.10,
        latency_multiplier=3.0,
        error_rate_delta=0.18,
        propagation_delay=(1, 3),
        downstream_severity=0.7,
        cascade_to=[],
        cascade_threshold=0.8,
        log_template_key="oom_kill_loop",
        alert_codes=["OOM_KILL", "SVC_RESTART_LOOP"],
    ),
    "thread_pool_exhaustion": FaultType(
        name="thread_pool_exhaustion",
        display="Thread Pool Exhaustion",
        targets=["api", "backend"],
        health_decay=0.06,
        latency_multiplier=2.8,
        error_rate_delta=0.09,
        propagation_delay=(2, 4),
        downstream_severity=0.6,
        cascade_to=["dependency_timeout"],
        cascade_threshold=0.5,
        log_template_key="thread_pool_exhaustion",
        alert_codes=["THREAD_POOL_FULL", "REQ_QUEUE_HIGH"],
    ),
    # -------------------------------------------------------------------------
    # APPLICATION FAULTS (targets=["api", "backend", "frontend"])
    # -------------------------------------------------------------------------

    "bad_deploy": FaultType(
        name="bad_deploy",
        display="Bad Deploy",
        targets=["api", "backend", "frontend"],
        health_decay=0.07,
        latency_multiplier=2.0,
        error_rate_delta=0.10,
        propagation_delay=(1, 3),
        downstream_severity=0.6,
        cascade_to=["circuit_breaker_open"],
        cascade_threshold=0.5,
        log_template_key="bad_deploy",
        alert_codes=["DEPLOY_ERROR", "SVC_DEGRADED"],
    ),
    "config_misconfiguration": FaultType(
        name="config_misconfiguration",
        display="Config Misconfiguration",
        targets=["api", "backend", "frontend"],
        health_decay=0.05,
        latency_multiplier=1.8,
        error_rate_delta=0.08,
        propagation_delay=(2, 5),
        downstream_severity=0.5,
        cascade_to=[],
        cascade_threshold=0.4,
        log_template_key="config_misconfiguration",
        alert_codes=["CONFIG_ERROR", "SVC_DEGRADED"],
    ),
    "dependency_timeout": FaultType(
        name="dependency_timeout",
        display="Dependency Timeout",
        targets=["api", "backend", "frontend"],
        health_decay=0.04,
        latency_multiplier=3.0,
        error_rate_delta=0.07,
        propagation_delay=(2, 4),
        downstream_severity=0.7,
        cascade_to=["circuit_breaker_open"],
        cascade_threshold=0.5,
        log_template_key="dependency_timeout",
        alert_codes=["DEP_TIMEOUT", "UPSTREAM_SLOW"],
    ),
    "circuit_breaker_open": FaultType(
        name="circuit_breaker_open",
        display="Circuit Breaker Open",
        targets=["api", "backend", "frontend"],
        health_decay=0.02,
        latency_multiplier=1.2,
        error_rate_delta=0.25,
        propagation_delay=(1, 2),
        downstream_severity=0.4,
        cascade_to=[],
        cascade_threshold=0.2,
        log_template_key="circuit_breaker_open",
        alert_codes=["CIRCUIT_OPEN", "SVC_UNAVAILABLE"],
    ),
}


class PropagationQueue:
    """
    Min-heap based priority queue for fault propagation events.

    Events are scheduled to fire at specific steps, propagating faults
    from upstream nodes to downstream nodes with appropriate delays.

    Internal structure: min-heap via heapq
    Entries: (fire_at_step: int, node_id: str, fault_name: str, severity: float)
    """

    def __init__(self) -> None:
        """Initialize an empty propagation queue."""
        self._queue: List[Tuple[int, str, str, float]] = []

    def schedule(
        self,
        source_node_id: str,
        fault: FaultType,
        graph: nx.DiGraph,
        current_step: int,
    ) -> None:
        """
        Schedule fault propagation to all downstream neighbors.

        For each downstream neighbor, calculates delay and severity,
        then pushes the event onto the priority queue.

        Args:
            source_node_id: ID of the node where the fault originated
            fault: The FaultType being propagated
            graph: The topology graph
            current_step: Current simulation step
        """
        for neighbour in graph.successors(source_node_id):
            delay = int(
                np.random.randint(
                    fault.propagation_delay[0], fault.propagation_delay[1] + 1
                )
            )
            edge_data = graph[source_node_id][neighbour]
            weight = edge_data.get("weight", 1.0)
            severity = fault.downstream_severity * weight
            fire_at_step = current_step + delay
            heapq.heappush(
                self._queue, (fire_at_step, neighbour, fault.name, severity)
            )

    def tick(self, current_step: int) -> List[Tuple[str, str, float]]:
        """
        Pop all events that should fire at or before the current step.

        Args:
            current_step: Current simulation step

        Returns:
            List of (node_id, fault_name, severity) tuples for events to process
        """
        ready_events = []
        while self._queue and self._queue[0][0] <= current_step:
            event = heapq.heappop(self._queue)
            fire_at_step, node_id, fault_name, severity = event
            ready_events.append((node_id, fault_name, severity))
        return ready_events

    def check_cascades(
        self, node: ServiceNode, fault_catalogue: Dict[str, FaultType]
    ) -> List[FaultType]:
        """
        Check if any active faults on this node should cascade.

        For each active fault on the node, if node health has fallen
        below the cascade threshold, returns the cascade fault types.

        Args:
            node: The ServiceNode to check
            fault_catalogue: Global fault catalogue

        Returns:
            List of FaultType objects that should cascade
        """
        cascades = []
        for fault_name in node.active_faults:
            if fault_name not in fault_catalogue:
                continue
            fault = fault_catalogue[fault_name]
            if node.health < fault.cascade_threshold:
                for cascade_name in fault.cascade_to:
                    # Only cascade if not already active
                    if cascade_name not in node.active_faults:
                        if cascade_name in fault_catalogue:
                            cascades.append(fault_catalogue[cascade_name])
        return cascades

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self._queue) == 0

    def __len__(self) -> int:
        """Return the number of pending events."""
        return len(self._queue)


# Import numpy at module level for schedule method
import numpy as np
