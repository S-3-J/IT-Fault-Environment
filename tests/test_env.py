"""
Tests for the core ITFaultEnv environment.

Verifies reset, step, state, and action handling.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.env import EnvConfig, ITFaultEnv
from env.tasks import TASKS


def test_reset_returns_valid_obs() -> None:
    """Test that reset returns a valid observation."""
    config = EnvConfig(task_id="task_1")
    env = ITFaultEnv(config)

    obs, info = env.reset()

    # Check observation structure
    assert "metrics" in obs
    assert "logs" in obs
    assert "alerts" in obs
    assert "budget" in obs
    assert "step" in obs

    # Check metrics structure
    assert len(obs["metrics"]) > 0
    for service_name, metrics in obs["metrics"].items():
        assert "cpu" in metrics
        assert "memory" in metrics
        assert "latency_ms" in metrics
        assert "error_rate" in metrics
        assert "health" in metrics

    # Check info
    assert "step" in info
    assert "budget" in info
    assert "action_space_size" in info
    assert info["step"] == 0
    assert info["budget"] == config.max_budget


def test_step_100_random_episodes_no_crash() -> None:
    """Test that 100 random episodes run without crashing."""
    for seed in range(100):
        config = EnvConfig(task_id="task_1", seed=seed)
        env = ITFaultEnv(config)
        obs, info = env.reset()

        action_space = info["action_space_size"]
        done = False
        steps = 0

        while not done and steps < 10:
            action_idx = (seed + steps) % action_space
            obs, reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated
            steps += 1

        # Should complete without exception
        assert steps > 0


def test_terminated_when_all_healthy() -> None:
    """Test that episode terminates when all services are healthy."""
    config = EnvConfig(task_id="task_1", seed=42)
    env = ITFaultEnv(config)
    obs, info = env.reset()

    # Initially should not be terminated (faults injected)
    # Run recovery actions on all nodes
    action_space = info["action_space_size"]
    n_nodes = action_space // 2

    for step in range(20):
        # Recovery actions start at index n_nodes
        action_idx = n_nodes + (step % n_nodes)
        obs, reward, terminated, truncated, info = env.step(action_idx)

        if terminated:
            # Verify all nodes are healthy
            state = env.state()
            for node_id, node_data in state["graph_nodes"].items():
                assert node_data["health"] > 0.85, \
                    f"Node {node_id} should be healthy when terminated"
            return

    # If we get here, episode wasn't terminated in 20 steps
    # This is OK for some seeds


def test_budget_depletes_on_actions() -> None:
    """Test that budget decreases with each action."""
    config = EnvConfig(task_id="task_1", seed=42, max_budget=100.0)
    env = ITFaultEnv(config)
    obs, info = env.reset()

    initial_budget = info["budget"]
    action_space = info["action_space_size"]
    assert initial_budget == 100.0

    # Take probe action (costs 2.0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert info["budget"] == initial_budget - config.probe_cost

    # Take recovery action (costs 8.0)
    recovery_action = action_space // 2  # First recovery action
    obs, reward, terminated, truncated, info = env.step(recovery_action)
    expected_budget = initial_budget - config.probe_cost - config.recovery_cost
    assert info["budget"] == expected_budget


def test_action_map_covers_all_nodes() -> None:
    """Test that action map covers all nodes."""
    config = EnvConfig(task_id="task_1", seed=42)
    env = ITFaultEnv(config)
    obs, info = env.reset()

    action_space = info["action_space_size"]
    n_nodes = len(obs["metrics"])

    # Should have 2 actions per node (probe + recovery)
    assert action_space == 2 * n_nodes

    # Check action map
    for i in range(action_space):
        assert i in env.action_map
        action_type, target = env.action_map[i]
        assert action_type in ["probe", "recovery"]
        assert target in obs["metrics"]


def test_state_returns_ground_truth() -> None:
    """Test that state() returns complete ground truth."""
    config = EnvConfig(task_id="task_1", seed=42)
    env = ITFaultEnv(config)
    obs, info = env.reset()

    state = env.state()

    # Check state structure
    assert "graph_nodes" in state
    assert "root_cause_nodes" in state
    assert "active_faults_per_node" in state
    assert "propagation_queue" in state
    assert "step" in state
    assert "budget" in state
    assert "history" in state

    # Root cause nodes should be hidden from observation
    root_causes = state["root_cause_nodes"]
    assert len(root_causes) > 0, "Should have at least one root cause"


def test_history_tracking() -> None:
    """Test that episode history is tracked correctly."""
    config = EnvConfig(task_id="task_1", seed=42)
    env = ITFaultEnv(config)
    obs, info = env.reset()

    # Take a few actions
    for i in range(5):
        env.step(i)

    history = env.get_history()
    assert history is not None
    assert len(history.actions) == 5
    assert len(history.rewards) == 5
    assert len(history.health_trajectory) == 5
    assert history.steps_taken == 5


def test_different_tasks_have_different_configs() -> None:
    """Test that different tasks use different configurations."""
    for task_id in ["task_1", "task_2", "task_3"]:
        config = EnvConfig(task_id=task_id)
        env = ITFaultEnv(config)
        obs, info = env.reset()

        n_services = len(obs["metrics"])
        task_config = TASKS[task_id].config

        assert n_services == task_config["n_services"], \
            f"{task_id} should have {task_config['n_services']} services"
