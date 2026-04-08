"""
Microservice topology graph generation for IT Fault Environment.

This module provides dataclasses and functions for generating realistic
microservice topology graphs as directed acyclic graphs (DAGs).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import networkx as nx
import numpy as np


# Tier names pool - sampled without replacement within each episode
TIER_NAMES: Dict[str, List[str]] = {
    "frontend": [
        "web-gateway",
        "nginx-ingress",
        "cdn-edge",
        "api-gateway",
        "load-balancer",
    ],
    "api": [
        "payment-api",
        "order-svc",
        "cart-service",
        "auth-svc",
        "user-api",
        "checkout-svc",
        "pricing-svc",
        "shipping-api",
    ],
    "backend": [
        "billing-worker",
        "notification-svc",
        "inventory-svc",
        "email-worker",
        "analytics-svc",
        "fraud-detector",
        "recommendation-svc",
    ],
    "db": [
        "postgres-primary",
        "redis-cache",
        "mongo-orders",
        "elasticsearch",
        "postgres-replica",
        "memcached",
        "cassandra-cluster",
    ],
}

# Tier baseline metrics
TIER_BASELINES: Dict[str, Dict[str, float]] = {
    "frontend": {"cpu": 0.15, "memory": 0.30, "latency_ms": 20.0},
    "api": {"cpu": 0.25, "memory": 0.45, "latency_ms": 45.0},
    "backend": {"cpu": 0.35, "memory": 0.55, "latency_ms": 80.0},
    "db": {"cpu": 0.20, "memory": 0.70, "latency_ms": 15.0},
}

# Status thresholds
HEALTH_THRESHOLDS: Dict[str, float] = {
    "healthy": 0.80,
    "degraded": 0.50,
    "failing": 0.20,
}


@dataclass
class ServiceNode:
    """
    Represents a single microservice node in the topology.

    Attributes:
        id: Unique identifier for the service
        tier: Service tier (frontend|api|backend|db)
        health: Health score from 0.0 (dead) to 1.0 (healthy)
        cpu: CPU utilization fraction
        memory: Memory utilization fraction
        latency_ms: Service latency in milliseconds
        error_rate: Error rate as a fraction
        request_queue: Number of pending requests
        status: Current status (healthy|degraded|failing|down)
        active_faults: List of active fault names affecting this node
    """

    id: str
    tier: str  # frontend | api | backend | db
    health: float = 1.0  # 0.0 (dead) -> 1.0 (healthy)
    cpu: float = 0.20
    memory: float = 0.40
    latency_ms: float = 45.0
    error_rate: float = 0.001
    request_queue: int = 0
    status: str = "healthy"  # healthy | degraded | failing | down
    active_faults: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Set tier-appropriate baseline values."""
        if self.tier in TIER_BASELINES:
            baseline = TIER_BASELINES[self.tier]
            self.cpu = baseline["cpu"]
            self.memory = baseline["memory"]
            self.latency_ms = baseline["latency_ms"]
        self.status = health_to_status(self.health)


def health_to_status(health: float) -> str:
    """
    Convert health score to status string.

    Args:
        health: Health score from 0.0 to 1.0

    Returns:
        Status string: 'healthy', 'degraded', 'failing', or 'down'
    """
    if health >= HEALTH_THRESHOLDS["healthy"]:
        return "healthy"
    elif health >= HEALTH_THRESHOLDS["degraded"]:
        return "degraded"
    elif health >= HEALTH_THRESHOLDS["failing"]:
        return "failing"
    else:
        return "down"


def sample_topology(
    n_services: int = 8, seed: Optional[int] = None, rng: Optional[np.random.Generator] = None
) -> nx.DiGraph:
    """
    Sample a microservice topology as a 4-tier DAG.

    Creates a directed acyclic graph with edges pointing downstream:
    frontend -> api -> backend -> db

    Every non-leaf node has >= 1 downstream edge.
    Every non-root node has >= 1 upstream edge.

    Args:
        n_services: Total number of services in the topology
        seed: Random seed for reproducibility
        rng: Optional numpy random generator (if provided, seed is ignored)

    Returns:
        networkx.DiGraph with nodes containing ServiceNode objects
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    # Determine tier distribution based on n_services
    # Ensure at least 1 frontend, 2 api, 2 backend, 1 db for minimal topology
    if n_services < 6:
        n_services = 6  # Minimum viable topology

    # Distribute services across tiers
    # Typical distribution: 1-2 frontend, 2-3 api, 2-3 backend, 1-2 db
    tier_counts = _distribute_tier_counts(n_services, rng)

    # Sample service names without replacement
    tier_to_services: Dict[str, List[str]] = {}
    for tier, count in tier_counts.items():
        available_names = TIER_NAMES[tier].copy()
        assert rng is not None
        rng.shuffle(available_names)
        tier_to_services[tier] = available_names[:count]

    # Create the graph
    graph = nx.DiGraph()

    # Track nodes by tier for edge creation
    tier_nodes: Dict[str, List[str]] = {
        "frontend": [],
        "api": [],
        "backend": [],
        "db": [],
    }

    # Create nodes
    for tier, service_names in tier_to_services.items():
        for name in service_names:
            node = ServiceNode(id=name, tier=tier)
            graph.add_node(name, service=node)
            tier_nodes[tier].append(name)

    # Create edges: connect tiers in order
    # frontend -> api -> backend -> db
    _connect_tiers(graph, tier_nodes["frontend"], tier_nodes["api"], rng)
    _connect_tiers(graph, tier_nodes["api"], tier_nodes["backend"], rng)
    _connect_tiers(graph, tier_nodes["backend"], tier_nodes["db"], rng)

    return graph


def _distribute_tier_counts(
    n_services: int, rng: np.random.Generator
) -> Dict[str, int]:
    """
    Distribute services across tiers based on total count.

    Args:
        n_services: Total number of services
        rng: Numpy random generator

    Returns:
        Dict mapping tier name to service count
    """
    # Minimum viable: 1 frontend, 2 api, 2 backend, 1 db = 6
    remaining = n_services - 6

    # Start with minimum distribution
    counts = {"frontend": 1, "api": 2, "backend": 2, "db": 1}

    # Distribute remaining services
    # Weight towards api and backend tiers
    tier_weights = [0.15, 0.35, 0.35, 0.15]  # frontend, api, backend, db
    tiers = ["frontend", "api", "backend", "db"]

    for _ in range(remaining):
        chosen_tier = rng.choice(tiers, p=tier_weights)
        counts[chosen_tier] += 1

    return counts


def _connect_tiers(
    graph: nx.DiGraph,
    upstream_tier: List[str],
    downstream_tier: List[str],
    rng: np.random.Generator,
) -> None:
    """
    Connect two adjacent tiers with edges.

    Ensures every upstream node has at least one downstream connection
    and every downstream node has at least one upstream connection.

    Args:
        graph: The graph to add edges to
        upstream_tier: List of node IDs in the upstream tier
        downstream_tier: List of node IDs in the downstream tier
        rng: Numpy random generator
    """
    if not upstream_tier or not downstream_tier:
        return

    # First pass: ensure every upstream node has at least one downstream edge
    for upstream_node in upstream_tier:
        # Each upstream node connects to 1-2 downstream nodes
        n_connections = rng.integers(1, min(3, len(downstream_tier) + 1))
        targets = rng.choice(downstream_tier, size=n_connections, replace=False)
        for target in targets:
            weight = rng.uniform(0.4, 1.0)
            graph.add_edge(upstream_node, target, weight=weight)

    # Second pass: ensure every downstream node has at least one upstream edge
    for downstream_node in downstream_tier:
        predecessors = list(graph.predecessors(downstream_node))
        if not predecessors:
            # Add a random upstream connection
            source = rng.choice(upstream_tier)
            weight = rng.uniform(0.4, 1.0)
            graph.add_edge(source, downstream_node, weight=weight)
