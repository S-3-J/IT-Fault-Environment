"""
Pydantic models for IT Fault Environment wire types.

Defines the Action, Observation, and Reward types used for the OpenEnv API.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """
    Types of actions available in the IT Fault Environment.

    Members:
        probe: Gather diagnostic information about a service
        restart: Restart a service (recovery action)
        isolate: Isolate a service from the topology (recovery action)
        patch: Apply a patch to a service (recovery action)
        rollback: Rollback a service to previous version (recovery action)
    """

    probe = "probe"
    restart = "restart"
    isolate = "isolate"
    patch = "patch"
    rollback = "rollback"


class FaultAction(BaseModel):
    """
    Action taken by the agent in the IT Fault Environment.

    Attributes:
        action_idx: Flat index into the action map
        action_type: Type of action (for logging/eval only)
        target: Target service name
        reasoning: LLM's stated reasoning (logged for evaluation)
    """

    action_idx: int = Field(..., description="Flat index into action_map")
    action_type: ActionType = Field(..., description="Type of action")
    target: str = Field(..., description="Target service name")
    reasoning: str = Field(
        ..., description="LLM's stated reasoning for this action"
    )


class ServiceMetrics(BaseModel):
    """
    Metrics for a single service.

    Attributes:
        cpu: CPU utilization fraction (0.0-1.0)
        memory: Memory utilization fraction (0.0-1.0)
        latency_ms: Service latency in milliseconds
        error_rate: Error rate as a fraction (0.0-1.0)
        health: Health score (0.0-1.0)
    """

    cpu: float = Field(..., description="CPU utilization fraction")
    memory: float = Field(..., description="Memory utilization fraction")
    latency_ms: float = Field(..., description="Service latency in milliseconds")
    error_rate: float = Field(..., description="Error rate fraction")
    health: float = Field(..., description="Health score (0.0-1.0)")


class FaultObservation(BaseModel):
    """
    Observation from the IT Fault Environment.

    Attributes:
        metrics: Dict mapping service names to their metrics
        logs: List of recent log entries
        alerts: List of active alert codes
        budget: Remaining budget fraction (0.0-1.0)
        step: Current step number
    """

    metrics: Dict[str, ServiceMetrics] = Field(
        ..., description="Per-service metrics"
    )
    logs: List[str] = Field(..., description="Recent log entries")
    alerts: List[str] = Field(..., description="Active alert codes")
    budget: float = Field(..., description="Remaining budget fraction")
    step: int = Field(..., description="Current step number")

    def to_prompt(self) -> str:
        """
        Render the observation as a text prompt for LLM consumption.

        Formats the observation as a real SRE dashboard readout with:
        - Services sorted by health (most critical first)
        - Status emojis for each service
        - Active alerts
        - Recent log entries

        Returns:
            Formatted string suitable for LLM input
        """
        # Status emojis
        status_emojis = {
            "down": "🔴",
            "failing": "🟠",
            "degraded": "🟡",
            "healthy": "🟢",
        }

        def get_status(health: float) -> str:
            if health < 0.20:
                return "down"
            elif health < 0.50:
                return "failing"
            elif health < 0.80:
                return "degraded"
            else:
                return "healthy"

        # Build services section sorted by health ascending
        services_lines = []

        # Sort services by health (ascending - most critical first)
        sorted_services = sorted(
            self.metrics.items(), key=lambda x: x[1].health
        )

        for service_name, metrics in sorted_services:
            status = get_status(metrics.health)
            emoji = status_emojis.get(status, "⚪")

            services_lines.append(
                f"{emoji} {service_name} (health={metrics.health:.2f})"
            )
            services_lines.append(
                f"    CPU: {metrics.cpu:.1%}  "
                f"MEM: {metrics.memory:.1%}  "
                f"LAT: {metrics.latency_ms:.0f}ms  "
                f"ERR: {metrics.error_rate:.3f}"
            )

        # Build alerts section
        alerts_section = ""
        if self.alerts:
            alerts_section = "\n".join(f"  [{alert}]" for alert in self.alerts)
        else:
            alerts_section = "  No active alerts"

        # Build logs section (last 10)
        recent_logs = self.logs[-10:] if len(self.logs) > 10 else self.logs
        logs_section = "\n".join(f"  {log}" for log in recent_logs) if recent_logs else "  No recent logs"

        # Assemble the full prompt
        prompt = (
            f"=== SYSTEM STATUS (step {self.step}, budget {self.budget:.0%}) ===\n\n"
            f"SERVICES:\n"
            f"{chr(10).join(services_lines)}\n\n"
            f"ACTIVE ALERTS:\n"
            f"{alerts_section}\n\n"
            f"RECENT LOGS:\n"
            f"{logs_section}"
        )

        return prompt


class TaskReward(BaseModel):
    """
    Reward information for a task step or episode.

    Attributes:
        step_reward: Reward for the current step
        cumulative_reward: Total cumulative reward
        grader_score: Final score from the task grader (0.0-1.0)
        components: Breakdown of reward components
    """

    step_reward: float = Field(..., description="Reward for current step")
    cumulative_reward: float = Field(..., description="Total cumulative reward")
    grader_score: float = Field(
        ..., description="Final grader score (0.0-1.0)"
    )
    components: Dict[str, float] = Field(
        ..., description="Breakdown of reward components"
    )
