"""
Tests for the fault system and propagation engine.

Verifies fault catalogue, propagation queue, and cascade behavior.
"""

import pytest
import sys
import os
import heapq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.graph import sample_topology, ServiceNode
from env.faults import FAULT_CATALOGUE, FaultType, PropagationQueue


def test_all_15_faults_in_catalogue() -> None:
    """Test that all 15 fault types are defined."""
    expected_faults = {
        # Database faults
        "connection_pool_exhausted",
        "disk_io_saturation",
        "replication_lag",
        "deadlock_storm",
        # Network faults
        "packet_loss",
        "dns_resolution_failure",
        "network_partition",
        # Compute faults
        "memory_leak",
        "cpu_saturation",
        "oom_kill_loop",
        "thread_pool_exhaustion",
        # Application faults
        "bad_deploy",
        "config_misconfiguration",
        "dependency_timeout",
        "circuit_breaker_open",
    }

    assert len(FAULT_CATALOGUE) == 15, f"Expected 15 faults, got {len(FAULT_CATALOGUE)}"
    assert set(FAULT_CATALOGUE.keys()) == expected_faults


def test_fault_type_attributes() -> None:
    """Test that all fault types have required attributes."""
    required_attrs = [
        "name", "display", "targets", "health_decay", "latency_multiplier",
        "error_rate_delta", "propagation_delay", "downstream_severity",
        "cascade_to", "cascade_threshold", "log_template_key", "alert_codes"
    ]

    for fault_name, fault in FAULT_CATALOGUE.items():
        assert isinstance(fault, FaultType), f"{fault_name} is not a FaultType"
        for attr in required_attrs:
            assert hasattr(fault, attr), f"Fault {fault_name} missing attribute {attr}"


def test_propagation_queue_schedules_correctly() -> None:
    """Test that propagation queue schedules events correctly."""
    graph = sample_topology(n_services=6, seed=42)
    queue = PropagationQueue()

    # Get a fault and root node
    fault = FAULT_CATALOGUE["connection_pool_exhausted"]

    # Find a db node
    db_nodes = [n for n in graph.nodes if graph.nodes[n]["service"].tier == "db"]
    assert len(db_nodes) > 0, "No db nodes in graph"

    root_node = db_nodes[0]

    # Schedule propagation
    queue.schedule(root_node, fault, graph, current_step=0)

    # Should have events for downstream nodes (but db is leaf, so no successors)
    # Let's test with a frontend node instead
    frontend_nodes = [n for n in graph.nodes if graph.nodes[n]["service"].tier == "frontend"]
    if frontend_nodes:
        queue2 = PropagationQueue()
        frontend_fault = FAULT_CATALOGUE["packet_loss"]
        root = frontend_nodes[0]
        queue2.schedule(root, frontend_fault, graph, current_step=0)

        # Should have scheduled events
        assert len(queue2) > 0, "Queue should have scheduled events"

        # Check event structure
        events = queue2.tick(100)  # Tick far in the future
        for event in events:
            node_id, fault_name, severity = event
            assert isinstance(node_id, str)
            assert isinstance(fault_name, str)
            assert isinstance(severity, float)
            assert 0.0 <= severity <= 1.0


def test_cascade_fires_below_threshold() -> None:
    """Test that cascades fire when health drops below threshold."""
    node = ServiceNode(id="test-node", tier="api")
    node.health = 1.0
    node.active_faults = ["cpu_saturation"]

    fault = FAULT_CATALOGUE["cpu_saturation"]
    assert fault.cascade_threshold < 1.0, "Test fault should have cascade threshold < 1.0"

    # Node health above threshold - no cascade
    queue = PropagationQueue()
    cascades = queue.check_cascades(node, FAULT_CATALOGUE)
    # cpu_saturation cascades to thread_pool_exhaustion when health < 0.4
    assert len(cascades) == 0, "Should not cascade when health above threshold"

    # Node health below threshold - should cascade
    node.health = 0.3  # Below 0.4 threshold
    cascades = queue.check_cascades(node, FAULT_CATALOGUE)
    assert len(cascades) > 0, "Should cascade when health below threshold"
    assert any(c.name == "thread_pool_exhaustion" for c in cascades)


def test_downstream_severity_decays_with_edge_weight() -> None:
    """Test that downstream severity is modulated by edge weight."""
    graph = sample_topology(n_services=6, seed=42)
    queue = PropagationQueue()

    # Find frontend and api nodes
    frontend_nodes = [n for n in graph.nodes if graph.nodes[n]["service"].tier == "frontend"]
    api_nodes = [n for n in graph.nodes if graph.nodes[n]["service"].tier == "api"]

    if frontend_nodes and api_nodes:
        fault = FAULT_CATALOGUE["packet_loss"]
        root = frontend_nodes[0]

        # Get edge weight
        for api_node in api_nodes:
            if graph.has_edge(root, api_node):
                edge_weight = graph[root][api_node]["weight"]
                expected_severity = fault.downstream_severity * edge_weight

                queue.schedule(root, fault, graph, current_step=0)
                events = queue.tick(100)

                for node_id, _, severity in events:
                    if node_id == api_node:
                        # Allow small floating point tolerance
                        assert abs(severity - expected_severity) < 0.01, \
                            f"Severity {severity} != expected {expected_severity}"
                        break


def test_fault_targets_correct_tiers() -> None:
    """Test that faults only target their specified tiers."""
    db_faults = [f for f in FAULT_CATALOGUE.values() if "db" in f.targets]
    network_faults = [f for f in FAULT_CATALOGUE.values() if "frontend" in f.targets or "api" in f.targets or "backend" in f.targets]

    # All db faults should have "db" in targets
    for fault in db_faults:
        assert "db" in fault.targets

    # Verify specific fault tier assignments
    assert FAULT_CATALOGUE["connection_pool_exhausted"].targets == ["db"]
    assert FAULT_CATALOGUE["packet_loss"].targets == ["frontend", "api", "backend"]
    assert FAULT_CATALOGUE["memory_leak"].targets == ["api", "backend"]
    assert FAULT_CATALOGUE["bad_deploy"].targets == ["api", "backend", "frontend"]


def test_propagation_delay_range() -> None:
    """Test that propagation delays are reasonable."""
    for fault in FAULT_CATALOGUE.values():
        min_delay, max_delay = fault.propagation_delay
        assert min_delay >= 1, f"Fault {fault.name} min_delay should be >= 1"
        assert max_delay >= min_delay, f"Fault {fault.name} max_delay should be >= min_delay"
        assert max_delay <= 10, f"Fault {fault.name} max_delay seems too high"
