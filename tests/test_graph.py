"""
Tests for the graph topology module.

Verifies that sampled topologies are valid DAGs with correct tier structure.
"""

import pytest
import networkx as nx
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.graph import sample_topology, ServiceNode, TIER_NAMES


def test_topology_is_connected_dag() -> None:
    """Test that sampled topology is a connected DAG."""
    graph = sample_topology(n_services=8, seed=42)

    # Should be a DAG
    assert nx.is_directed_acyclic_graph(graph), "Graph must be a DAG"

    # Should be weakly connected
    assert nx.is_weakly_connected(graph), "Graph must be weakly connected"

    # All nodes should have service attribute
    for node_id, node_data in graph.nodes(data=True):
        assert "service" in node_data, f"Node {node_id} missing service attribute"
        assert isinstance(node_data["service"], ServiceNode)


def test_all_nodes_have_tier() -> None:
    """Test that all nodes have valid tier assignments."""
    valid_tiers = {"frontend", "api", "backend", "db"}
    graph = sample_topology(n_services=10, seed=123)

    for node_id, node_data in graph.nodes(data=True):
        node = node_data["service"]
        assert node.tier in valid_tiers, f"Node {node_id} has invalid tier: {node.tier}"


def test_names_unique_within_episode() -> None:
    """Test that service names are unique within an episode."""
    graph = sample_topology(n_services=12, seed=456)

    node_names = list(graph.nodes)
    assert len(node_names) == len(set(node_names)), "Node names must be unique"


def test_sample_10_topologies_no_orphans() -> None:
    """Test that multiple sampled topologies have no orphan nodes."""
    for seed in range(10):
        graph = sample_topology(n_services=8, seed=seed)

        # Every non-leaf node must have at least one downstream edge
        for node_id in graph.nodes:
            successors = list(graph.successors(node_id))
            predecessors = list(graph.predecessors(node_id))

            # Non-leaf nodes (not db tier) must have downstream edges
            node = graph.nodes[node_id]["service"]
            if node.tier != "db":
                assert len(successors) > 0, f"Non-leaf node {node_id} has no downstream edges"

            # Non-root nodes (not frontend tier) must have upstream edges
            if node.tier != "frontend":
                assert len(predecessors) > 0, f"Non-root node {node_id} has no upstream edges"


def test_tier_distribution() -> None:
    """Test that tier distribution is reasonable."""
    graph = sample_topology(n_services=10, seed=789)

    tier_counts = {"frontend": 0, "api": 0, "backend": 0, "db": 0}
    for node_data in graph.nodes.values():
        tier_counts[node_data["service"].tier] += 1

    # Should have at least one of each tier
    for tier, count in tier_counts.items():
        assert count >= 1, f"Tier {tier} should have at least 1 node"

    # Total should match
    assert sum(tier_counts.values()) == 10


def test_edge_weights_in_range() -> None:
    """Test that edge weights are in the expected range."""
    graph = sample_topology(n_services=8, seed=111)

    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 0)
        assert 0.4 <= weight <= 1.0, f"Edge ({u}, {v}) weight {weight} out of range [0.4, 1.0]"


def test_node_status_initialized() -> None:
    """Test that node status is correctly initialized from health."""
    graph = sample_topology(n_services=6, seed=222)

    for node_data in graph.nodes.values():
        node = node_data["service"]
        # Fresh nodes should be healthy
        assert node.health == 1.0, f"Node {node.id} should start with health=1.0"
        assert node.status == "healthy", f"Node {node.id} should start with status='healthy'"
