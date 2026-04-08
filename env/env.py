"""
Core environment logic for IT Fault Diagnosis and Recovery.

Implements the ITFaultEnv class with reset(), step(), and state() methods
following the OpenEnv specification.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import networkx as nx

from .graph import ServiceNode, sample_topology, health_to_status
from .faults import FAULT_CATALOGUE, FaultType, PropagationQueue
from .renderer import ObservationRenderer
from .tasks import TASKS, EpisodeHistory, Task


@dataclass
class EnvConfig:
    """
    Configuration for the IT Fault Environment.

    Attributes:
        n_services: Number of services in the topology
        max_steps: Maximum steps per episode
        max_budget: Maximum budget for actions
        probe_cost: Cost of a probe action
        recovery_cost: Cost of a recovery action
        difficulty: Difficulty level (easy|medium|hard)
        seed: Random seed for reproducibility
        task_id: Task identifier
        extra_noise: Additional sensor noise for medium/hard
        masked_sensors: Number of nodes with masked metrics
        spurious_alert_rate: Rate of spurious alerts in logs
    """

    n_services: int = 8
    max_steps: int = 60
    max_budget: float = 100.0
    probe_cost: float = 2.0
    recovery_cost: float = 8.0
    difficulty: str = "easy"
    seed: Optional[int] = None
    task_id: str = "task_1"
    extra_noise: float = 0.0
    masked_sensors: int = 0
    spurious_alert_rate: float = 0.15


class ITFaultEnv:
    """
    IT Fault Diagnosis and Recovery Environment.

    The agent receives noisy observations (metrics, logs, alerts) from a
    running service graph, decides each step whether to probe a service
    for more information or take a recovery action, and is rewarded for
    efficient correct diagnosis.

    Attributes:
        config: Environment configuration
        renderer: Observation renderer
        graph: Current topology graph
        action_map: Mapping from action index to (action_type, target)
        action_space_size: Total number of available actions
        root_cause_nodes: Nodes where faults were injected (hidden from agent)
        prop_queue: Propagation queue for fault spread
        history: Episode history for grading
    """

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        """
        Initialize the environment.

        Args:
            config: Environment configuration (uses defaults if None)
        """
        self.config = config or EnvConfig()
        self.renderer = ObservationRenderer(
            "templates/", seed=self.config.seed
        )

        # State variables (initialized in reset)
        self.graph: Optional[nx.DiGraph] = None
        self.action_map: Dict[int, Tuple[str, str]] = {}
        self.action_space_size: int = 0
        self.root_cause_nodes: List[str] = []
        self.prop_queue = PropagationQueue()
        self.history: Optional[EpisodeHistory] = None

        # Episode state
        self.step_count: int = 0
        self.budget: float = 0.0
        self.episode_logs: List[str] = []
        self.prev_mean_health: float = 1.0

        # Get task configuration
        self.task: Optional[Task] = None
        if self.config.task_id in TASKS:
            self.task = TASKS[self.config.task_id]

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Optional random seed
            options: Optional additional options

        Returns:
            Tuple of (observation, info dict)
        """
        # Update seed if provided
        if seed is not None:
            self.config.seed = seed

        rng = np.random.default_rng(self.config.seed)

        # Sample topology
        n_services = self.config.n_services
        if self.task and "n_services" in self.task.config:
            n_services = self.task.config["n_services"]

        self.graph = sample_topology(n_services=n_services, seed=self.config.seed, rng=rng)

        # Determine number of faults from difficulty
        difficulty_faults = {"easy": 1, "medium": 2, "hard": 3}
        n_faults = difficulty_faults.get(self.config.difficulty, 1)
        if self.task and "n_faults" in self.task.config:
            n_faults = self.task.config["n_faults"]

        # Select faults and inject them
        self.root_cause_nodes = []
        self.prop_queue = PropagationQueue()

        # Get available faults based on topology tiers
        available_faults = self._get_available_faults()

        # Sample faults without replacement
        fault_names = list(available_faults.keys())
        rng.shuffle(fault_names)
        selected_faults = fault_names[:n_faults]

        # Track which nodes have faults to avoid duplicate injection
        nodes_with_faults: set = set()

        for fault_name in selected_faults:
            fault = available_faults[fault_name]

            # Find valid root nodes (must match fault targets tier)
            valid_nodes = self._get_valid_root_nodes(fault, nodes_with_faults)

            if valid_nodes:
                # Sample a root node
                root_node_id = str(rng.choice(valid_nodes))
                self.root_cause_nodes.append(root_node_id)
                nodes_with_faults.add(root_node_id)

                # Apply initial health decay
                node = self.graph.nodes[root_node_id]["service"]
                node.health = max(0.0, node.health - fault.health_decay * 2)
                node.active_faults.append(fault_name)
                node.status = health_to_status(node.health)

                # Schedule propagation
                self.prop_queue.schedule(root_node_id, fault, self.graph, 0)

        # Build action map
        nodes = list(self.graph.nodes)
        self.action_map = {}
        for i, node in enumerate(nodes):
            self.action_map[i] = ("probe", node)
            self.action_map[i + len(nodes)] = ("recovery", node)
        self.action_space_size = 2 * len(nodes)

        # Initialize episode state
        self.step_count = 0
        self.budget = self.config.max_budget
        self.episode_logs = []
        self.prev_mean_health = 1.0

        # Initialize history
        db_nodes = [
            n for n in self.graph.nodes
            if self.graph.nodes[n]["service"].tier == "db"
        ]
        self.history = EpisodeHistory(
            task_id=self.config.task_id,
            root_cause_nodes=self.root_cause_nodes.copy(),
            final_health=1.0,
        )

        # Generate initial observation
        masked_nodes = self._get_masked_nodes(rng)
        obs = self.renderer.render(
            self.graph,
            self.episode_logs,
            self.budget,
            self.config.max_budget,
            self.step_count,
            extra_noise=self.config.extra_noise,
            masked_nodes=masked_nodes,
        )

        info = {
            "step": 0,
            "budget": self.budget,
            "action_space_size": self.action_space_size,
        }

        return obs, info

    def _get_available_faults(self) -> Dict[str, FaultType]:
        """
        Get faults that can be injected based on topology tiers.

        Returns:
            Dict of fault name to FaultType
        """
        # Get tiers present in topology
        tiers_present = set()
        for node_data in self.graph.nodes.values():
            tiers_present.add(node_data["service"].tier)

        # Filter faults that can target at least one tier
        available = {}
        for name, fault in FAULT_CATALOGUE.items():
            if any(tier in fault.targets for tier in tiers_present):
                available[name] = fault

        return available

    def _get_valid_root_nodes(
        self, fault: FaultType, excluded: set
    ) -> List[str]:
        """
        Get valid root nodes for a fault.

        Args:
            fault: The fault to find root nodes for
            excluded: Set of node IDs to exclude

        Returns:
            List of valid node IDs
        """
        valid = []
        for node_id, node_data in self.graph.nodes.items():
            node = node_data["service"]
            if node.tier in fault.targets and node_id not in excluded:
                valid.append(node_id)
        return valid

    def _get_masked_nodes(
        self, rng: np.random.Generator
    ) -> List[str]:
        """
        Get list of nodes with masked metrics.

        Args:
            rng: Random number generator

        Returns:
            List of masked node IDs
        """
        n_masked = self.config.masked_sensors
        if self.task and "masked_sensors" in self.task.config:
            n_masked = self.task.config["masked_sensors"]

        if n_masked <= 0:
            return []

        nodes = list(self.graph.nodes)
        if len(nodes) <= n_masked:
            return nodes

        return list(rng.choice(nodes, size=n_masked, replace=False))

    def step(
        self, action_idx: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action_idx: Index of the action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Decode action
        if action_idx not in self.action_map:
            raise ValueError(f"Invalid action index: {action_idx}")

        action_type, target_node = self.action_map[action_idx]

        # Apply action effect
        if action_type == "probe":
            probe_result = self.renderer.render_probe(target_node, self.graph)
            self.episode_logs.append(probe_result)
            self.budget -= self.config.probe_cost

        elif action_type == "recovery":
            node = self.graph.nodes[target_node]["service"]
            # Heal the node
            node.health = min(1.0, node.health + 0.40)
            # Clear faults if health recovered enough
            if node.health >= 0.60:
                node.active_faults = []
            # Update status
            node.status = health_to_status(node.health)
            self.budget -= self.config.recovery_cost

        # Propagation tick - process scheduled fault propagation
        effects = self.prop_queue.tick(self.step_count)
        for node_id, fault_name, severity in effects:
            node = self.graph.nodes[node_id]["service"]
            fault = FAULT_CATALOGUE[fault_name]

            # Apply fault effects
            node.health = max(0.0, node.health - fault.health_decay * severity)
            node.latency_ms *= 1 + (fault.latency_multiplier - 1) * severity
            node.error_rate = min(1.0, node.error_rate + fault.error_rate_delta * severity)
            node.cpu = min(1.0, node.cpu + 0.1 * severity)
            node.memory = min(1.0, node.memory + 0.08 * severity)

            if fault_name not in node.active_faults:
                node.active_faults.append(fault_name)

            node.status = health_to_status(node.health)

            # Check cascades
            cascades = self.prop_queue.check_cascades(node, FAULT_CATALOGUE)
            for cascade_fault in cascades:
                self.prop_queue.schedule(node_id, cascade_fault, self.graph, self.step_count)
                if cascade_fault.name not in self.history.cascades_triggered:
                    self.history.cascades_triggered.append(cascade_fault.name)

                # Track if DB nodes affected
                if node.tier == "db" and node_id not in self.history.db_nodes_affected:
                    self.history.db_nodes_affected.append(node_id)

        # Compute dense reward
        curr_health = np.mean(
            [n["service"].health for n in self.graph.nodes.values()]
        )
        delta = curr_health - self.prev_mean_health
        self.prev_mean_health = curr_health

        reward = 0.0
        reward += 1.0 * delta  # Health delta (main signal)
        reward -= 0.01  # Step penalty

        # Penalize actions on healthy nodes
        target_health = self.graph.nodes[target_node]["service"].health
        if target_health > 0.85:
            reward -= 0.15

        # Bonus: if budget conserved and system recovered
        if all(
            n["service"].health > 0.85 for n in self.graph.nodes.values()
        ):
            budget_ratio = self.budget / self.config.max_budget
            reward += 1.0 + 0.5 * budget_ratio

        # Check termination
        terminated = all(
            n["service"].health > 0.85 for n in self.graph.nodes.values()
        )
        truncated = (
            self.step_count >= self.config.max_steps or self.budget <= 0
        )

        # Update history
        self.history.actions.append({
            "step": self.step_count,
            "action_type": action_type,
            "target": target_node,
        })
        self.history.rewards.append(reward)
        self.history.health_trajectory.append(curr_health)
        self.history.steps_taken = self.step_count + 1
        self.history.final_health = curr_health
        self.history.terminated = terminated
        self.history.truncated = truncated

        self.step_count += 1

        # Generate observation
        masked_seed = (
            None if self.config.seed is None else self.config.seed + self.step_count
        )
        masked_nodes = self._get_masked_nodes(np.random.default_rng(masked_seed))
        obs = self.renderer.render(
            self.graph,
            self.episode_logs,
            self.budget,
            self.config.max_budget,
            self.step_count,
            extra_noise=self.config.extra_noise,
            masked_nodes=masked_nodes,
        )

        info = {
            "step": self.step_count,
            "budget": self.budget,
            "mean_health": curr_health,
        }

        return obs, reward, terminated, truncated, info

    def state(self) -> Dict[str, Any]:
        """
        Get full ground truth state (not shown to agent during training).

        Returns:
            Dict with complete environment state for evaluation
        """
        graph_nodes = {}
        active_faults_per_node = {}

        for node_id, node_data in self.graph.nodes.items():
            node = node_data["service"]
            graph_nodes[node_id] = asdict(node)
            active_faults_per_node[node_id] = node.active_faults

        return {
            "graph_nodes": graph_nodes,
            "root_cause_nodes": self.root_cause_nodes,
            "active_faults_per_node": active_faults_per_node,
            "propagation_queue": list(self.prop_queue._queue),
            "step": self.step_count,
            "budget": self.budget,
            "history": asdict(self.history) if self.history else None,
        }

    def get_history(self) -> Optional[EpisodeHistory]:
        """
        Get the episode history for grading.

        Returns:
            EpisodeHistory object or None if no episode has been run
        """
        return self.history
