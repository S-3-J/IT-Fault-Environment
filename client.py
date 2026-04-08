"""IT Fault Environment Client.

Client for connecting to the IT Fault Environment server.
"""

from typing import Dict, Any, Optional

from openenv.core import EnvClient
from openenv.core.env_server.types import State

from .models import FaultAction, FaultObservation, ServiceMetrics


class ItFaultEnvClient(
    EnvClient[FaultAction, FaultObservation, State]
):
    """
    Client for the IT Fault Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with ItFaultEnvClient(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.budget)
        ...
        ...     result = client.step(FaultAction(
        ...         action_idx=0,
        ...         action_type="probe",
        ...         target="web-gateway",
        ...         reasoning="Checking frontend health"
        ...     ))

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = ItFaultEnvClient.from_docker_image("it-fault-env:latest")
        >>> try:
        ...     result = client.reset()
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: FaultAction) -> Dict[str, Any]:
        """
        Convert FaultAction to JSON payload for step message.

        Args:
            action: FaultAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_idx": action.action_idx,
            "action_type": action.action_type.value,
            "target": action.target,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> Any:
        """
        Parse server response into StepResult.

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with FaultObservation
        """
        from openenv.core.client_types import StepResult

        obs_data = payload.get("observation", {})

        # Parse metrics
        metrics = {}
        for service_name, metric_data in obs_data.get("metrics", {}).items():
            metrics[service_name] = ServiceMetrics(
                cpu=metric_data.get("cpu", 0.0),
                memory=metric_data.get("memory", 0.0),
                latency_ms=metric_data.get("latency_ms", 0.0),
                error_rate=metric_data.get("error_rate", 0.0),
                health=metric_data.get("health", 0.0),
            )

        observation = FaultObservation(
            metrics=metrics,
            logs=obs_data.get("logs", []),
            alerts=obs_data.get("alerts", []),
            budget=obs_data.get("budget", 1.0),
            step=obs_data.get("step", 0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("terminated", False) or payload.get("truncated", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id", "unknown"),
            step_count=payload.get("step", 0),
        )
