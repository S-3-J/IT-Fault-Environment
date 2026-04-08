"""
Tests for task graders.

Verifies that grader functions return correct scores for various scenarios.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.tasks import (
    EpisodeHistory,
    TASKS,
    grade_task_1,
    grade_task_2,
    grade_task_3,
)


def test_task1_grader_perfect_score() -> None:
    """Test that perfect play gets score 1.0."""
    history = EpisodeHistory(
        task_id="task_1",
        root_cause_nodes=["db-primary"],
        actions=[
            {"step": 0, "action_type": "probe", "target": "web-gateway"},
            {"step": 1, "action_type": "probe", "target": "api-gateway"},
            {"step": 2, "action_type": "probe", "target": "db-primary"},
            {"step": 3, "action_type": "recovery", "target": "db-primary"},
        ],
        observations=[],
        rewards=[0.0, 0.0, 0.0, 1.5],
        health_trajectory=[0.7, 0.7, 0.7, 1.0],
        final_health=1.0,
        terminated=True,
        truncated=False,
        steps_taken=4,
        cascades_triggered=[],
        db_nodes_affected=[],
    )

    score = grade_task_1(history)

    # Root cause targeted: 0.4
    # Final health 1.0: 0.3
    # Efficiency (30-4)/30: 0.173
    # No healthy hits: 0.1
    # Total: ~0.97 (may not be exactly 1.0 due to efficiency calc)
    assert score >= 0.9, f"Perfect play should score >= 0.9, got {score}"


def test_task1_grader_zero_score() -> None:
    """Test that terrible play gets score near 0."""
    history = EpisodeHistory(
        task_id="task_1",
        root_cause_nodes=["db-primary"],
        actions=[
            {"step": 0, "action_type": "recovery", "target": "web-gateway"},
            {"step": 1, "action_type": "recovery", "target": "api-gateway"},
        ],
        observations=[],
        rewards=[-0.15, -0.15],
        health_trajectory=[0.5, 0.4],
        final_health=0.4,
        terminated=False,
        truncated=True,
        steps_taken=30,  # Used all steps
        cascades_triggered=[],
        db_nodes_affected=[],
    )

    score = grade_task_1(history)

    # Root cause NOT targeted: 0.0
    # Final health 0.4: 0.12
    # Efficiency 0: 0.0
    # All actions on healthy nodes: 0.0
    # Total: ~0.12
    assert score < 0.3, f"Terrible play should score < 0.3, got {score}"


def test_task2_grader_partial_credit() -> None:
    """Test task 2 grader with partial success."""
    history = EpisodeHistory(
        task_id="task_2",
        root_cause_nodes=["api-1", "backend-2"],
        actions=[
            {"step": 0, "action_type": "probe", "target": "api-1"},
            {"step": 1, "action_type": "recovery", "target": "api-1"},
            # Missed backend-2
        ],
        observations=[],
        rewards=[0.0, 0.5],
        health_trajectory=[0.6, 0.8],
        final_health=0.8,
        terminated=False,
        truncated=True,
        steps_taken=45,
        cascades_triggered=["dependency_timeout"],
        db_nodes_affected=[],  # No DB affected - good!
    )

    score = grade_task_2(history)

    # 1/2 root causes targeted: 0.175
    # Final health 0.8: 0.28
    # No DB affected: 0.2
    # Efficiency 0: 0.0
    # Total: ~0.655
    assert 0.5 <= score <= 0.8, f"Partial success should score 0.5-0.8, got {score}"


def test_task3_grader_ordering_bonus() -> None:
    """Test task 3 grader with probe-before-recovery ordering."""
    # Good ordering: probe before recovery
    history_good = EpisodeHistory(
        task_id="task_3",
        root_cause_nodes=["api-1", "backend-2", "db-1"],
        actions=[
            {"step": 0, "action_type": "probe", "target": "api-1"},
            {"step": 1, "action_type": "probe", "target": "backend-2"},
            {"step": 2, "action_type": "probe", "target": "db-1"},
            {"step": 3, "action_type": "recovery", "target": "api-1"},
            {"step": 4, "action_type": "recovery", "target": "backend-2"},
            {"step": 5, "action_type": "recovery", "target": "db-1"},
        ],
        observations=[],
        rewards=[0.0, 0.0, 0.0, 0.5, 0.5, 1.0],
        health_trajectory=[0.5, 0.5, 0.5, 0.7, 0.85, 1.0],
        final_health=1.0,
        terminated=True,
        truncated=False,
        steps_taken=6,
        cascades_triggered=[],
        db_nodes_affected=[],
    )

    score_good = grade_task_3(history_good)

    # All root causes targeted: 0.3
    # Final health 1.0: 0.3
    # All probed before recovery: 0.2
    # No unnecessary recoveries: 0.1
    # Resolved before step 45: 0.1
    # Total: 1.0
    assert score_good >= 0.9, f"Good ordering should score >= 0.9, got {score_good}"

    # Bad ordering: recovery without probing
    history_bad = EpisodeHistory(
        task_id="task_3",
        root_cause_nodes=["api-1", "backend-2", "db-1"],
        actions=[
            {"step": 0, "action_type": "recovery", "target": "api-1"},
            {"step": 1, "action_type": "recovery", "target": "backend-2"},
            {"step": 2, "action_type": "recovery", "target": "db-1"},
        ],
        observations=[],
        rewards=[0.0, 0.0, 0.5],
        health_trajectory=[0.5, 0.6, 0.9],
        final_health=0.9,
        terminated=False,
        truncated=True,
        steps_taken=60,
        cascades_triggered=[],
        db_nodes_affected=[],
    )

    score_bad = grade_task_3(history_bad)

    # All root causes targeted: 0.3
    # Final health 0.9: 0.27
    # No probing before recovery: 0.0
    # No unnecessary recoveries: 0.1
    # Not resolved before 45: 0.0
    # Total: ~0.67
    assert score_bad < score_good, "Bad ordering should score lower than good ordering"


def test_grader_pure_functions() -> None:
    """Test that graders are pure (same input = same output)."""
    history = EpisodeHistory(
        task_id="task_1",
        root_cause_nodes=["db-1"],
        actions=[{"step": 0, "action_type": "recovery", "target": "db-1"}],
        observations=[],
        rewards=[0.5],
        health_trajectory=[0.8],
        final_health=0.9,
        terminated=True,
        truncated=False,
        steps_taken=1,
        cascades_triggered=[],
        db_nodes_affected=[],
    )

    # Run multiple times - should get same score
    scores = [grade_task_1(history) for _ in range(10)]
    assert all(s == scores[0] for s in scores), "Grader should be pure"


def test_task_registry_has_all_tasks() -> None:
    """Test that TASKS registry has all 3 tasks."""
    assert "task_1" in TASKS
    assert "task_2" in TASKS
    assert "task_3" in TASKS

    # Check task properties
    assert TASKS["task_1"].difficulty == "easy"
    assert TASKS["task_2"].difficulty == "medium"
    assert TASKS["task_3"].difficulty == "hard"

    # Check graders are callable
    assert callable(TASKS["task_1"].grader)
    assert callable(TASKS["task_2"].grader)
    assert callable(TASKS["task_3"].grader)

    # Check passing scores
    assert TASKS["task_1"].passing_score == 0.70
    assert TASKS["task_2"].passing_score == 0.55
    assert TASKS["task_3"].passing_score == 0.40


def test_cascade_penalty_in_task2() -> None:
    """Test that DB cascade is penalized in task 2."""
    history_no_cascade = EpisodeHistory(
        task_id="task_2",
        root_cause_nodes=["api-1", "backend-2"],
        actions=[],
        observations=[],
        rewards=[],
        health_trajectory=[0.8],
        final_health=0.8,
        terminated=False,
        truncated=True,
        steps_taken=45,
        cascades_triggered=[],
        db_nodes_affected=[],
    )

    history_with_cascade = EpisodeHistory(
        task_id="task_2",
        root_cause_nodes=["api-1", "backend-2"],
        actions=[],
        observations=[],
        rewards=[],
        health_trajectory=[0.8],
        final_health=0.8,
        terminated=False,
        truncated=True,
        steps_taken=45,
        cascades_triggered=["dependency_timeout"],
        db_nodes_affected=["postgres-primary"],  # DB affected!
    )

    score_no_cascade = grade_task_2(history_no_cascade)
    score_with_cascade = grade_task_2(history_with_cascade)

    # Should lose 0.20 points for DB cascade
    assert score_no_cascade - score_with_cascade == pytest.approx(0.20, abs=0.01)
