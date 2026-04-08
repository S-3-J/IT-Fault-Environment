"""
FastAPI server for IT Fault Environment.

Implements the OpenEnv server with endpoints for:
- POST /reset - Reset environment
- POST /step - Execute one step
- GET /state - Get ground truth state
- GET /health - Liveness check
- GET /tasks - List available tasks
- POST /reset/{task_id} - Reset with specific task
"""

import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.env import EnvConfig, ITFaultEnv
from env.tasks import TASKS
from models import FaultAction, FaultObservation, ServiceMetrics, ActionType

app = FastAPI(title="IT Fault Environment", version="1.0.0")

# Session storage for concurrent environments
_sessions: Dict[str, ITFaultEnv] = {}


class ResetResponse(BaseModel):
    """Response from reset endpoint."""

    observation: FaultObservation
    info: Dict[str, Any]


class StepResponse(BaseModel):
    """Response from step endpoint."""

    observation: FaultObservation
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


def _get_or_create_env(session_id: str, task_id: Optional[str] = None) -> ITFaultEnv:
    """
    Get existing environment or create a new one for the session.

    Args:
        session_id: Unique session identifier
        task_id: Optional task ID to override default

    Returns:
        ITFaultEnv instance for this session
    """
    if session_id not in _sessions:
        # Determine task_id
        if task_id is None:
            task_id = "task_1"

        # Create environment with task-specific config
        config = EnvConfig(task_id=task_id)
        if task_id in TASKS:
            task = TASKS[task_id]
            # Apply task config overrides
            if "n_services" in task.config:
                config.n_services = task.config["n_services"]
            if "extra_noise" in task.config:
                config.extra_noise = task.config["extra_noise"]
            if "masked_sensors" in task.config:
                config.masked_sensors = task.config["masked_sensors"]
            if "spurious_alert_rate" in task.config:
                config.spurious_alert_rate = task.config["spurious_alert_rate"]
            config.difficulty = task.difficulty

        env = ITFaultEnv(config)
        _sessions[session_id] = env

    return _sessions[session_id]


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Liveness check endpoint."""
    return {"status": "ok"}


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    """List available tasks."""
    return {
        "tasks": [
            {
                "id": task.id,
                "name": task.name,
                "description": task.description,
                "difficulty": task.difficulty,
                "max_steps": task.max_steps,
                "passing_score": task.passing_score,
            }
            for task in TASKS.values()
        ]
    }


@app.post("/reset", response_model=ResetResponse)
async def reset(
    session_id: Optional[str] = Header(None),
    task_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> ResetResponse:
    """
    Reset the environment.

    Args:
        session_id: Optional session ID for persistent sessions
        task_id: Optional task ID to reset with

    Returns:
        Initial observation and info dict
    """
    # Generate session ID if not provided
    if session_id is None:
        session_id = str(uuid4())

    env = _get_or_create_env(session_id, task_id)
    obs_dict, info = env.reset(seed=seed)

    # Convert observation to FaultObservation model
    fault_obs = _dict_to_fault_observation(obs_dict)

    return ResetResponse(observation=fault_obs, info=info)


@app.post("/reset/{task_id}", response_model=ResetResponse)
async def reset_with_task(
    task_id: str,
    session_id: Optional[str] = Header(None),
    seed: Optional[int] = None,
) -> ResetResponse:
    """
    Reset the environment with a specific task.

    Args:
        task_id: Task ID (task_1, task_2, task_3)
        session_id: Optional session ID

    Returns:
        Initial observation and info dict
    """
    if task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id: {task_id}. Available: {list(TASKS.keys())}",
        )

    # Clear existing session to create fresh env with new task
    if session_id and session_id in _sessions:
        del _sessions[session_id]

    return await reset(session_id=session_id, task_id=task_id, seed=seed)


@app.post("/step", response_model=StepResponse)
async def step(
    action: FaultAction,
    x_session_id: Optional[str] = Header(None),
) -> StepResponse:
    """
    Execute one step in the environment.

    Args:
        action: Action to execute
        x_session_id: Session ID from X-Session-ID header

    Returns:
        Observation, reward, terminated, truncated, and info
    """
    session_id = x_session_id
    if session_id is None:
        session_id = str(uuid4())

    env = _get_or_create_env(session_id)

    # Validate action index
    if action.action_idx not in env.action_map:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action_idx: {action.action_idx}. Valid range: 0-{env.action_space_size-1}"
        )

    obs_dict, reward, terminated, truncated, info = env.step(action.action_idx)

    # Convert observation to FaultObservation model
    fault_obs = _dict_to_fault_observation(obs_dict)

    return StepResponse(
        observation=fault_obs,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )


@app.get("/state")
async def get_state(session_id: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Get full ground truth state.

    Args:
        session_id: Optional session ID

    Returns:
        Complete environment state for evaluation
    """
    if session_id is None:
        raise HTTPException(status_code=400, detail="session_id required")

    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return env.state()


@app.get("/history")
async def get_history(session_id: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Get episode history for grading.

    Args:
        session_id: Optional session ID

    Returns:
        Episode history
    """
    if session_id is None:
        raise HTTPException(status_code=400, detail="session_id required")

    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found")

    history = env.get_history()
    if history is None:
        return {"history": None}

    return {"history": asdict(history)}


def _dict_to_fault_observation(obs_dict: Dict[str, Any]) -> FaultObservation:
    """
    Convert raw observation dict to FaultObservation model.

    Args:
        obs_dict: Raw observation dictionary from environment

    Returns:
        FaultObservation pydantic model
    """
    metrics = {}
    for service_name, service_metrics in obs_dict.get("metrics", {}).items():
        metrics[service_name] = ServiceMetrics(
            cpu=service_metrics.get("cpu", 0.0),
            memory=service_metrics.get("memory", 0.0),
            latency_ms=service_metrics.get("latency_ms", 0.0),
            error_rate=service_metrics.get("error_rate", 0.0),
            health=service_metrics.get("health", 0.0),
        )

    return FaultObservation(
        metrics=metrics,
        logs=obs_dict.get("logs", []),
        alerts=obs_dict.get("alerts", []),
        budget=obs_dict.get("budget", 1.0),
        step=obs_dict.get("step", 0),
    )


# WebSocket support for persistent sessions
@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """
    WebSocket endpoint for persistent session interaction.

    Supports JSON frames with:
    - {"type": "reset", "task_id": "task_1"}
    - {"type": "step", "action_idx": 0, "action_type": "probe", "target": "service-1", "reasoning": "..."}
    - {"type": "state"}
    """
    await websocket.accept()

    session_id = str(uuid4())
    env = _get_or_create_env(session_id)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "reset":
                task_id = data.get("task_id", "task_1")
                env = _get_or_create_env(session_id, task_id)
                obs_dict, info = env.reset()
                fault_obs = _dict_to_fault_observation(obs_dict)
                await websocket.send_json({
                    "type": "observation",
                    "observation": fault_obs.dict(),
                    "info": info,
                })

            elif msg_type == "step":
                action_idx = data.get("action_idx")
                if action_idx is None:
                    await websocket.send_json({
                        "type": "error",
                        "error": "action_idx required",
                    })
                    continue

                obs_dict, reward, terminated, truncated, info = env.step(action_idx)
                fault_obs = _dict_to_fault_observation(obs_dict)
                await websocket.send_json({
                    "type": "step_result",
                    "observation": fault_obs.dict(),
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info,
                })

            elif msg_type == "state":
                state = env.state()
                await websocket.send_json({
                    "type": "state",
                    "state": state,
                })

            elif msg_type == "history":
                history = env.get_history()
                if history:
                    await websocket.send_json({
                        "type": "history",
                        "history": asdict(history),
                    })
                else:
                    await websocket.send_json({
                        "type": "history",
                        "history": None,
                    })

            else:
                await websocket.send_json({
                    "type": "error",
                    "error": f"Unknown message type: {msg_type}",
                })

    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "error": str(e),
        })
    finally:
        # Clean up session on disconnect
        if session_id in _sessions:
            del _sessions[session_id]


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uvicorn.

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IT Fault Environment Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()
    if args.host == "0.0.0.0" and args.port == 8000:
        main()
    else:
        main(host=args.host, port=args.port)
