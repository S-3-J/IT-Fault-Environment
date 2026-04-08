#!/usr/bin/env python3
"""
Inference entrypoint for IT Fault Environment.

This script conforms to the benchmark stdout contract:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-Turbo")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
BENCHMARK_NAME = os.getenv("BENCHMARK_NAME", "it-fault-env")


SYSTEM_PROMPT = """You are an expert SRE diagnosing a production incident.

You will receive a dashboard with service metrics, logs, alerts, and budget.
You must respond with exactly one JSON object and nothing else.

Rules:
- Do not use markdown fences.
- Do not return an empty object.
- action_idx must be a valid integer action index.
- action_type must be exactly "probe" or "recovery".
- target must exactly match one service name from the observation.
- reasoning must be a short plain-text explanation.

Respond with JSON only in this form:
{
  "action_idx": <integer>,
  "action_type": "probe" or "recovery",
  "target": "<service_name>",
  "reasoning": "<brief reasoning>"
}
"""


@dataclass
class EpisodeResult:
    """Final episode result."""

    success: bool
    steps: int
    score: float
    rewards: List[float]


def log_debug(message: str) -> None:
    """Write debugging information to stderr without affecting stdout contract."""
    print(message, file=sys.stderr)


class HTTPFaultEnv:
    """Small HTTP wrapper that gives the script a reset/step/close shape."""

    def __init__(self, base_url: str, task_id: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.task_id = task_id
        self.session_id = str(uuid.uuid4())

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        response = requests.post(
            f"{self.base_url}/reset/{self.task_id}",
            headers={"session-id": self.session_id},
            params={"seed": seed} if seed is not None else None,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["observation"], payload["info"]

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/step",
            headers={"x-session-id": self.session_id, "Content-Type": "application/json"},
            json=action,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def history(self) -> Optional[Dict[str, Any]]:
        response = requests.get(
            f"{self.base_url}/history",
            headers={"session-id": self.session_id},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("history")

    def close(self) -> None:
        """No-op hook to satisfy the benchmark lifecycle."""
        return None


def format_observation(obs: Dict[str, Any]) -> str:
    """Render the observation as a concise text prompt."""
    lines: List[str] = []
    lines.append(
        f"SYSTEM STATUS step={obs.get('step', 0)} budget={obs.get('budget', 1.0):.0%}"
    )

    metrics = obs.get("metrics", {})
    sorted_services = sorted(metrics.items(), key=lambda item: item[1].get("health", 0.0))
    lines.append("SERVICES:")
    for service_name, metric in sorted_services:
        lines.append(
            f"- {service_name}: health={metric.get('health', 0.0):.2f} "
            f"cpu={metric.get('cpu', 0.0):.2%} mem={metric.get('memory', 0.0):.2%} "
            f"lat={metric.get('latency_ms', 0.0):.0f}ms err={metric.get('error_rate', 0.0):.4f}"
        )

    service_order = list(metrics.keys())
    lines.append("AVAILABLE ACTIONS:")
    for index, service_name in enumerate(service_order):
        lines.append(f"- {index}: probe {service_name}")
    offset = len(service_order)
    for index, service_name in enumerate(service_order):
        lines.append(f"- {index + offset}: recovery {service_name}")

    alerts = obs.get("alerts", [])
    lines.append("ALERTS:")
    if alerts:
        lines.extend(f"- {alert}" for alert in alerts)
    else:
        lines.append("- none")

    logs = obs.get("logs", [])
    lines.append("LOGS:")
    if logs:
        lines.extend(f"- {log}" for log in logs[-10:])
    else:
        lines.append("- none")

    lines.append("Return a JSON action only.")
    return "\n".join(lines)


def parse_action(
    response_text: str,
    action_space_size: int,
    metrics: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Parse and minimally validate a model-produced action."""
    try:
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        action = json.loads(text)
        action_idx = action.get("action_idx")
        action_type = action.get("action_type")
        target = action.get("target")

        if not isinstance(action_idx, int):
            return None
        if action_idx < 0 or action_idx >= action_space_size:
            return None
        if action_type not in {"probe", "recovery"}:
            return None
        if target not in metrics:
            return None

        action["reasoning"] = str(action.get("reasoning", ""))
        return action
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def fallback_action(
    obs: Dict[str, Any],
    service_order: List[str],
    action_space_size: int,
) -> Dict[str, Any]:
    """Pick a safe heuristic action when the model response is unusable."""
    metrics = obs.get("metrics", {})
    if not metrics or not service_order:
        return {
            "action_idx": 0,
            "action_type": "probe",
            "target": "",
            "reasoning": "Fallback due to missing metrics",
        }

    lowest_service, lowest_metric = min(
        metrics.items(),
        key=lambda item: item[1].get("health", 1.0),
    )
    service_idx = service_order.index(lowest_service)
    health = float(lowest_metric.get("health", 1.0))
    if health < 0.5:
        action_idx = service_idx + (action_space_size // 2)
        action_type = "recovery"
    else:
        action_idx = service_idx
        action_type = "probe"

    return {
        "action_idx": action_idx,
        "action_type": action_type,
        "target": lowest_service,
        "reasoning": f"Fallback action for lowest-health service ({health:.2f})",
    }


def compute_grader_score(task_id: str, history: Optional[Dict[str, Any]]) -> float:
    """Compute a bounded score from episode history."""
    if not history:
        return 0.0

    from env.tasks import EpisodeHistory, TASKS

    task = TASKS.get(task_id)
    if task is None:
        return 0.0

    episode_history = EpisodeHistory(
        task_id=history.get("task_id", task_id),
        root_cause_nodes=history.get("root_cause_nodes", []),
        actions=history.get("actions", []),
        observations=history.get("observations", []),
        rewards=history.get("rewards", []),
        health_trajectory=history.get("health_trajectory", []),
        final_health=history.get("final_health", 0.0),
        terminated=history.get("terminated", False),
        truncated=history.get("truncated", False),
        steps_taken=history.get("steps_taken", 0),
        cascades_triggered=history.get("cascades_triggered", []),
        db_nodes_affected=history.get("db_nodes_affected", []),
    )
    score = float(task.grader(episode_history))
    return max(0.0, min(1.0, score))


def action_to_str(action: Dict[str, Any]) -> str:
    """Render the action in a stable, single-line format."""
    action_type = action.get("action_type", "probe")
    target = str(action.get("target", "")).replace("\n", " ").replace("\r", " ")
    if action_type == "recovery":
        return f"recover('{target}')"
    return f"probe('{target}')"


def emit_start(task_id: str, model_name: str) -> None:
    """Emit the required episode start line."""
    print(f"[START] task={task_id} env={BENCHMARK_NAME} model={model_name}")


def emit_step(step_num: int, action: Dict[str, Any], reward: float, done: bool, error: Optional[str]) -> None:
    """Emit the required step line."""
    error_text = "null" if error is None else str(error).replace("\n", " ").replace("\r", " ")
    done_text = "true" if done else "false"
    print(
        f"[STEP] step={step_num} action={action_to_str(action)} "
        f"reward={reward:.2f} done={done_text} error={error_text}"
    )


def emit_end(result: EpisodeResult) -> None:
    """Emit the required episode end line."""
    success_text = "true" if result.success else "false"
    rewards_text = ",".join(f"{reward:.2f}" for reward in result.rewards)
    print(
        f"[END] success={success_text} steps={result.steps} "
        f"score={result.score:.2f} rewards={rewards_text}"
    )


def run_episode(
    task_id: str,
    env_url: str,
    model_name: str,
    max_steps: int,
    seed: Optional[int] = None,
) -> EpisodeResult:
    """Run one episode and emit benchmark-formatted stdout."""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required")

    from openai import OpenAI

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    env = HTTPFaultEnv(base_url=env_url, task_id=task_id)

    rewards: List[float] = []
    steps = 0
    score = 0.0
    success = False

    emit_start(task_id, model_name)

    try:
        observation, info = env.reset(seed=seed)
        action_space_size = int(info.get("action_space_size", 0))
        if action_space_size <= 0:
            raise RuntimeError("Environment returned an empty action space")

        service_order = list(observation.get("metrics", {}).keys())
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step_index in range(1, max_steps + 1):
            prompt = format_observation(observation)
            messages.append({"role": "user", "content": prompt})

            action: Optional[Dict[str, Any]] = None
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=300,
                )
                content = response.choices[0].message.content or ""
                action = parse_action(
                    content,
                    action_space_size=action_space_size,
                    metrics=observation.get("metrics", {}),
                )
                if action is None:
                    log_debug(
                        f"[debug] task={task_id} step={step_index} invalid_model_response="
                        f"{content[:500]!r}"
                    )
            except Exception as exc:
                log_debug(f"[debug] task={task_id} step={step_index} model_error={exc}")
                action = None

            if action is None:
                action = fallback_action(observation, service_order, action_space_size)
                if not action.get("target"):
                    raise RuntimeError("Unable to derive a valid fallback action")

            step_payload = {
                "action_idx": action["action_idx"],
                "action_type": action["action_type"],
                "target": action["target"],
                "reasoning": action.get("reasoning", ""),
            }

            step_result = env.step(step_payload)
            observation = step_result["observation"]
            reward = float(step_result["reward"])
            done = bool(step_result["terminated"] or step_result["truncated"])

            rewards.append(reward)
            steps = step_index
            emit_step(step_index, action, reward, done, None)

            messages.append({"role": "assistant", "content": json.dumps(step_payload)})

            if done:
                break

        history = env.history()
        if history is None:
            log_debug(f"[debug] task={task_id} history_missing_for_session={env.session_id}")
        score = compute_grader_score(task_id, history)

        from env.tasks import TASKS

        passing_score = TASKS[task_id].passing_score if task_id in TASKS else 1.0
        success = score >= passing_score
        return EpisodeResult(success=success, steps=steps, score=score, rewards=rewards)
    except Exception as exc:
        log_debug(f"[debug] task={task_id} fatal_error={exc}")
        return EpisodeResult(success=False, steps=steps, score=score, rewards=rewards)
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments without affecting stdout contract."""
    parser = argparse.ArgumentParser(description="Run a benchmark-compatible inference episode.")
    parser.add_argument("--task", default="task_1", help="Task ID to run")
    parser.add_argument("--all-tasks", action="store_true", help="Run task_1, task_2, and task_3 sequentially")
    parser.add_argument("--env-url", default=ENV_URL, help="Environment server URL")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name override")
    parser.add_argument("--max-steps", type=int, default=60, help="Maximum number of environment steps")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible resets")
    return parser.parse_args()


def main() -> None:
    """Script entrypoint."""
    args = parse_args()
    task_ids = ["task_1", "task_2", "task_3"] if args.all_tasks else [args.task]

    for index, task_id in enumerate(task_ids):
        episode_seed = None if args.seed is None else args.seed + index
        result = run_episode(
            task_id=task_id,
            env_url=args.env_url,
            model_name=args.model,
            max_steps=args.max_steps,
            seed=episode_seed,
        )
        emit_end(result)


if __name__ == "__main__":
    main()
