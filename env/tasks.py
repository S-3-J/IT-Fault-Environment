"""
Task definitions and graders for IT Fault Environment.

Defines 3 tasks with increasing difficulty:
1. Single Fault Isolation (easy)
2. Cascade Containment (medium)
3. Full Incident Response (hard)

Each task includes a programmatic grader that scores agent performance 0.0-1.0.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import numpy as np


@dataclass
class EpisodeHistory:
    """
    Records the complete history of an episode for grading.

    Attributes:
        task_id: ID of the task being evaluated
        root_cause_nodes: List of nodes where faults were originally injected
        actions: List of action dicts with step, action_type, target, reasoning
        observations: List of observation dicts (one per step)
        rewards: List of rewards received at each step
        health_trajectory: Mean system health at each step
        final_health: Mean system health at episode end
        terminated: Whether the episode terminated naturally (recovered)
        truncated: Whether the episode was truncated (max steps/budget)
        steps_taken: Total number of steps in the episode
        cascades_triggered: List of cascade fault names that were triggered
        db_nodes_affected: List of DB tier nodes affected by cascades
    """

    task_id: str
    root_cause_nodes: List[str]
    actions: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    health_trajectory: List[float] = field(default_factory=list)
    final_health: float = 0.0
    terminated: bool = False
    truncated: bool = False
    steps_taken: int = 0
    cascades_triggered: List[str] = field(default_factory=list)
    db_nodes_affected: List[str] = field(default_factory=list)


@dataclass
class Task:
    """
    Defines a task configuration and its grader.

    Attributes:
        id: Unique task identifier
        name: Human-readable task name
        description: Task description shown to the agent
        difficulty: Difficulty level (easy|medium|hard)
        config: Environment config overrides for this task
        grader: Function that takes EpisodeHistory and returns score 0.0-1.0
        max_steps: Maximum steps allowed
        passing_score: Score threshold for considering the task solved
    """

    id: str
    name: str
    description: str
    difficulty: str  # easy | medium | hard
    config: Dict[str, Any]
    grader: Callable[[EpisodeHistory], float]
    max_steps: int
    passing_score: float


# =============================================================================
# HELPER FUNCTIONS FOR GRADERS
# =============================================================================


def _count_actions_on_healthy_nodes(
    episode_history: EpisodeHistory, healthy_threshold: float = 0.85
) -> int:
    """
    Count the number of actions taken on nodes that were healthy at the time.

    Args:
        episode_history: The episode history to analyze
        healthy_threshold: Health threshold for considering a node healthy

    Returns:
        Number of unnecessary actions on healthy nodes
    """
    healthy_hits = 0
    for action in episode_history.actions:
        target = action.get("target", "")
        step = action.get("step", 0)

        # Check if node was healthy at that step
        if step < len(episode_history.health_trajectory):
            # We don't have per-node health, so use trajectory as proxy
            # This is a simplification - in reality we'd track per-node health
            if episode_history.health_trajectory[step] > healthy_threshold:
                healthy_hits += 1

    return healthy_hits


def _check_probe_precedes_recovery(episode_history: EpisodeHistory) -> float:
    """
    Check if the agent probed before recovering for each target.

    Returns a score from 0.0 to 1.0 based on diagnostic discipline.

    Args:
        episode_history: The episode history to analyze

    Returns:
        Score from 0.0 (no probing) to 1.0 (always probed before recovery)
    """
    # Track which nodes were probed
    probed_nodes: set = set()
    recovery_targets: List[str] = []

    for action in episode_history.actions:
        action_type = action.get("action_type", "")
        target = action.get("target", "")

        if action_type == "probe":
            probed_nodes.add(target)
        elif action_type == "recovery":
            recovery_targets.append(target)

    if not recovery_targets:
        return 0.0

    # Count how many recovery targets were probed first
    probed_before_recovery = sum(1 for t in recovery_targets if t in probed_nodes)
    return probed_before_recovery / len(recovery_targets)


def _count_unnecessary_recoveries(
    episode_history: EpisodeHistory, healthy_threshold: float = 0.85
) -> float:
    """
    Calculate the rate of unnecessary recoveries on healthy nodes.

    Args:
        episode_history: The episode history to analyze
        healthy_threshold: Health threshold for considering a node healthy

    Returns:
        Rate of unnecessary recoveries (0.0 = none, 1.0 = all unnecessary)
    """
    recovery_actions = [
        a for a in episode_history.actions if a.get("action_type") == "recovery"
    ]

    if not recovery_actions:
        return 0.0

    # Count recoveries when system was already healthy
    unnecessary = 0
    for action in recovery_actions:
        step = action.get("step", 0)
        if step < len(episode_history.health_trajectory):
            if episode_history.health_trajectory[step] > healthy_threshold:
                unnecessary += 1

    return unnecessary / len(recovery_actions)


# =============================================================================
# TASK 1: SINGLE FAULT ISOLATION (EASY)
# =============================================================================


def grade_task_1(episode_history: EpisodeHistory) -> float:
    """
    Grade Task 1: Single Fault Isolation.

    Scoring breakdown:
    - 0.4 pts: Was the root cause node targeted with a recovery action?
    - 0.3 pts: System health at episode end (normalized)
    - 0.2 pts: Efficiency (fewer steps is better)
    - 0.1 pts: No unnecessary actions on healthy nodes

    Args:
        episode_history: The episode history to grade

    Returns:
        Score from 0.0 to 1.0
    """
    score = 0.0
    max_steps = 30

    # 0.4 pts: Root cause targeted
    recovery_targets = [
        a.get("target", "")
        for a in episode_history.actions
        if a.get("action_type") == "recovery"
    ]
    root_cause_targeted = any(
        rc in recovery_targets for rc in episode_history.root_cause_nodes
    )
    if root_cause_targeted:
        score += 0.40

    # 0.3 pts: Final system health
    score += 0.30 * episode_history.final_health

    # 0.2 pts: Efficiency
    steps_taken = episode_history.steps_taken
    efficiency_score = max(0, (max_steps - steps_taken) / max_steps)
    score += 0.20 * efficiency_score

    # 0.1 pts: No unnecessary actions on healthy nodes
    healthy_hits = _count_actions_on_healthy_nodes(episode_history)
    total_actions = len(episode_history.actions)
    if total_actions > 0:
        unnecessary_rate = healthy_hits / total_actions
        score += 0.10 * max(0, 1 - unnecessary_rate)
    else:
        score += 0.10

    return min(1.0, score)


# =============================================================================
# TASK 2: CASCADE CONTAINMENT (MEDIUM)
# =============================================================================


def grade_task_2(episode_history: EpisodeHistory) -> float:
    """
    Grade Task 2: Cascade Containment.

    Scoring breakdown:
    - 0.35 pts: Both root causes targeted
    - 0.35 pts: Final system health
    - 0.20 pts: No cascade reached DB tier (cascade containment)
    - 0.10 pts: Efficiency

    Args:
        episode_history: The episode history to grade

    Returns:
        Score from 0.0 to 1.0
    """
    score = 0.0
    max_steps = 45

    # 0.35 pts: Root causes targeted
    recovery_targets = [
        a.get("target", "")
        for a in episode_history.actions
        if a.get("action_type") == "recovery"
    ]
    causes_found = len(
        [rc for rc in episode_history.root_cause_nodes if rc in recovery_targets]
    )
    total_causes = len(episode_history.root_cause_nodes)
    if total_causes > 0:
        score += 0.35 * (causes_found / total_causes)

    # 0.35 pts: Final system health
    score += 0.35 * episode_history.final_health

    # 0.20 pts: Cascade containment (no DB nodes affected)
    if not episode_history.db_nodes_affected:
        score += 0.20

    # 0.10 pts: Efficiency
    steps_taken = episode_history.steps_taken
    efficiency_score = max(0, (max_steps - steps_taken) / max_steps)
    score += 0.10 * efficiency_score

    return min(1.0, score)


# =============================================================================
# TASK 3: FULL INCIDENT RESPONSE (HARD)
# =============================================================================


def grade_task_3(episode_history: EpisodeHistory) -> float:
    """
    Grade Task 3: Full Incident Response.

    Scoring breakdown:
    - 0.30 pts: All root causes identified and targeted
    - 0.30 pts: Final system health
    - 0.20 pts: Correct ordering (probed before recovering)
    - 0.10 pts: No unnecessary recoveries on healthy nodes
    - 0.10 pts: Resolved before step 45 (time bonus)

    Args:
        episode_history: The episode history to grade

    Returns:
        Score from 0.0 to 1.0
    """
    score = 0.0

    # 0.30 pts: All root causes targeted
    recovery_targets = [
        a.get("target", "")
        for a in episode_history.actions
        if a.get("action_type") == "recovery"
    ]
    causes_found = len(
        [rc for rc in episode_history.root_cause_nodes if rc in recovery_targets]
    )
    total_causes = len(episode_history.root_cause_nodes)
    if total_causes > 0:
        score += 0.30 * (causes_found / total_causes)

    # 0.30 pts: Final system health
    score += 0.30 * episode_history.final_health

    # 0.20 pts: Diagnostic discipline (probe before recovery)
    probe_score = _check_probe_precedes_recovery(episode_history)
    score += 0.20 * probe_score

    # 0.10 pts: No unnecessary recoveries
    unnecessary_rate = _count_unnecessary_recoveries(episode_history)
    score += 0.10 * (1 - unnecessary_rate)

    # 0.10 pts: Time bonus (resolved before step 45)
    if episode_history.terminated and episode_history.steps_taken < 45:
        score += 0.10

    return min(1.0, score)


# =============================================================================
# TASK REGISTRY
# =============================================================================

TASKS: Dict[str, Task] = {
    "task_1": Task(
        id="task_1",
        name="Single Fault Isolation",
        description=(
            "A single service in a 6-node microservice topology has a fault. "
            "Identify the root cause service and recover the system."
        ),
        difficulty="easy",
        config={
            "n_services": 6,
            "extra_noise": 0.0,
            "masked_sensors": 0,
            "spurious_alert_rate": 0.15,
        },
        grader=grade_task_1,
        max_steps=30,
        passing_score=0.70,
    ),
    "task_2": Task(
        id="task_2",
        name="Cascade Containment",
        description=(
            "Two concurrent faults are propagating through a 10-node topology. "
            "Contain the cascade before it reaches critical services."
        ),
        difficulty="medium",
        config={
            "n_services": 10,
            "extra_noise": 0.20,
            "masked_sensors": 0,
            "spurious_alert_rate": 0.15,
        },
        grader=grade_task_2,
        max_steps=45,
        passing_score=0.55,
    ),
    "task_3": Task(
        id="task_3",
        name="Full Incident Response",
        description=(
            "A complex incident with 3 concurrent faults, masked sensors, "
            "and spurious alerts is degrading a 15-node production topology. "
            "Diagnose, contain, and recover all affected services."
        ),
        difficulty="hard",
        config={
            "n_services": 15,
            "extra_noise": 0.30,
            "masked_sensors": 3,
            "spurious_alert_rate": 0.30,
        },
        grader=grade_task_3,
        max_steps=60,
        passing_score=0.40,
    ),
}
