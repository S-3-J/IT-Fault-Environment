"""
Observation renderer for IT Fault Environment.

Converts raw environment state into observations that the agent sees,
including noisy metrics, log entries, and active alerts.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


class ObservationRenderer:
    """
    Renders raw environment state into observations for the agent.

    The renderer produces:
    - Noisy metrics for each service (CPU, memory, latency, error rate, health)
    - Recent log entries generated from fault templates
    - Active alert codes from degraded nodes
    - Budget and step information

    Attributes:
        templates: Dict mapping fault names to log template lists
        rng: Random number generator for noise and sampling
    """

    def __init__(self, template_dir: str, seed: Optional[int] = None) -> None:
        """
        Initialize the renderer with templates from the specified directory.

        Args:
            template_dir: Path to directory containing JSON template files
            seed: Optional random seed for reproducibility
        """
        self.templates: Dict[str, List[str]] = {}
        self._load_templates(template_dir)
        self.rng = np.random.default_rng(seed)

    def _load_templates(self, template_dir: str) -> None:
        """
        Load all JSON template files from the template directory.

        Args:
            template_dir: Path to directory containing template JSON files
        """
        template_path = Path(template_dir)
        if not template_path.exists():
            # Try relative to this file
            template_path = Path(__file__).parent.parent / "templates"

        for json_file in template_path.glob("*.json"):
            with open(json_file, "r") as f:
                data = json.load(f)
                for fault_name, templates_list in data.items():
                    self.templates[fault_name] = templates_list

    def render(
        self,
        graph: Any,
        episode_logs: List[str],
        budget: float,
        max_budget: float,
        step: int,
        extra_noise: float = 0.0,
        masked_nodes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Render the current state as an observation for the agent.

        Args:
            graph: NetworkX graph with ServiceNode objects
            episode_logs: List of log entries from this episode
            budget: Current remaining budget
            max_budget: Maximum budget
            step: Current step number
            extra_noise: Additional noise factor for medium/hard difficulty
            masked_nodes: List of node IDs with masked metrics

        Returns:
            Dict with keys: metrics, logs, alerts, budget, step
        """
        metrics: Dict[str, Dict[str, float]] = {}
        alerts: List[str] = []
        masked = set(masked_nodes or [])

        for node_id, node_data in graph.nodes(data=True):
            node = node_data["service"]

            if node_id in masked:
                # Return masked/placeholder metrics
                metrics[node_id] = {
                    "cpu": 0.0,
                    "memory": 0.0,
                    "latency_ms": 0.0,
                    "error_rate": 0.0,
                    "health": 0.0,
                    "masked": True,
                }
            else:
                # Add Gaussian noise to metrics
                cpu_noise = self.rng.normal(0, 0.02 + extra_noise)
                memory_noise = self.rng.normal(0, 0.01 + extra_noise)
                latency_noise = self.rng.normal(0, node.latency_ms * 0.05)
                error_noise = self.rng.normal(0, 0.001 + extra_noise)

                metrics[node_id] = {
                    "cpu": max(0.0, min(1.0, node.cpu + cpu_noise)),
                    "memory": max(0.0, min(1.0, node.memory + memory_noise)),
                    "latency_ms": max(0.0, node.latency_ms + latency_noise),
                    "error_rate": max(0.0, min(1.0, node.error_rate + error_noise)),
                    "health": max(0.0, min(1.0, node.health)),
                    "masked": False,
                }

            # Collect alerts from degraded nodes
            if node.status != "healthy":
                for fault_name in node.active_faults:
                    if fault_name in self.templates:
                        # Get alert codes from fault catalogue
                        from .faults import FAULT_CATALOGUE

                        if fault_name in FAULT_CATALOGUE:
                            alerts.extend(FAULT_CATALOGUE[fault_name].alert_codes)

        # Generate fault logs
        fault_logs = self._generate_fault_logs(graph)

        # Combine logs: recent episode logs + new fault logs
        all_logs = episode_logs[-20:] + fault_logs
        if len(all_logs) > 20:
            all_logs = all_logs[-20:]

        return {
            "metrics": metrics,
            "logs": all_logs,
            "alerts": list(set(alerts)),  # Deduplicate alerts
            "budget": budget / max_budget,
            "step": step,
        }

    def render_probe(self, node_id: str, graph: Any) -> str:
        """
        Render a detailed diagnostic probe result for a specific node.

        Returns ground truth values with slight noise on connections
        and p99 latency (1.4x * base latency_ms).

        Args:
            node_id: ID of the node to probe
            graph: NetworkX graph with ServiceNode objects

        Returns:
            Formatted probe result string
        """
        node_data = graph.nodes[node_id]
        node = node_data["service"]

        # Probe returns ground truth with p99 latency and slightly noisy connections
        p99_latency = node.latency_ms * 1.4
        noisy_connections = int(
            node.request_queue + self.rng.integers(-2, 3)
        )  # +/- 2 noise
        noisy_connections = max(0, noisy_connections)

        # Generate random uptime
        uptime_hours = self.rng.integers(0, 72)
        uptime_minutes = self.rng.integers(0, 60)

        # Format active faults
        if node.active_faults:
            faults_str = ", ".join(node.active_faults)
        else:
            faults_str = "none detected"

        return (
            f"PROBE {node_id}: cpu={node.cpu:.1%} mem={node.memory:.1%} "
            f"latency_p99={p99_latency:.0f}ms error_rate={node.error_rate:.4f} "
            f"active_connections={noisy_connections} "
            f"uptime={uptime_hours}h{uptime_minutes}m "
            f"active_faults={faults_str}"
        )

    def _generate_fault_logs(self, graph: Any) -> List[str]:
        """
        Generate log entries for nodes with active faults.

        For each node with health < 0.8, generates log entries based on
        active fault templates. Also adds spurious false-positive alerts
        with 15% probability.

        Args:
            graph: NetworkX graph with ServiceNode objects

        Returns:
            List of generated log lines
        """
        logs: List[str] = []

        for node_id, node_data in graph.nodes(data=True):
            node = node_data["service"]

            if node.health >= 0.8:
                continue

            for fault_name in node.active_faults:
                template_str = self._pick_template(fault_name, node)
                if template_str:
                    try:
                        # Get a random downstream node for upstream references
                        successors = list(graph.successors(node_id))
                        upstream = successors[0] if successors else "unknown"
                        db_host = successors[0] if successors and graph.nodes[successors[0]]["service"].tier == "db" else "postgres-primary"

                        formatted = template_str.format(
                            service=node_id,
                            latency=node.latency_ms,
                            errors=node.error_rate,
                            cpu=node.cpu,
                            memory=node.memory,
                            queue=node.request_queue,
                            db_host=db_host,
                            upstream=upstream,
                        )
                        logs.append(formatted)
                    except (KeyError, ValueError):
                        # Fallback if template formatting fails
                        logs.append(f"ERROR {node_id}: {fault_name} detected (health={node.health:.2f})")

        # Add spurious false-positive alert with 15% probability
        if self.rng.random() < 0.15:
            spurious_alerts = [
                "WARN  system: transient network glitch detected on eth0",
                "INFO  system: scheduled health check timeout (false positive)",
                "WARN  system: temporary memory pressure spike (GC pause)",
                "ERROR system: upstream timeout (transient, self-healing)",
            ]
            logs.append(self.rng.choice(spurious_alerts))

        return logs

    def _pick_template(self, fault_name: str, node: Any) -> Optional[str]:
        """
        Pick a random template for the given fault.

        Args:
            fault_name: Name of the fault
            node: ServiceNode to generate logs for

        Returns:
            Random template string or None if no templates available
        """
        if fault_name not in self.templates:
            return None
        templates_list = self.templates[fault_name]
        if not templates_list:
            return None
        return str(self.rng.choice(templates_list))

