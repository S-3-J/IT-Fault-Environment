---
title: IT Fault Environment
emoji: 🏥
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
---

# IT Fault Environment

IT Systems Fault Diagnosis and Recovery — an RL environment where an LLM agent diagnoses and recovers failing microservice topologies. Based on real-world SRE incident response workflows.

## Overview and Motivation

This environment simulates a production SRE (Site Reliability Engineering) incident response scenario. In real-world operations, SREs must:

1. **Diagnose** issues from noisy, incomplete observability data (metrics, logs, alerts)
2. **Prioritize** which services to investigate based on symptoms
3. **Contain** cascading failures before they reach critical infrastructure
4. **Recover** services while minimizing downtime and resource expenditure

The agent receives observations mirroring real monitoring dashboards and must decide whether to:
- **Probe** a service for detailed diagnostics (costs budget, provides ground truth)
- **Recover** a service (costs more budget, restores health if correctly targeted)

This is not a toy problem — it directly models the cognitive workload of on-call engineers responding to PagerDuty alerts at 3 AM.

## Environment Architecture

### Topology Graph

The environment generates a 4-tier microservice DAG (Directed Acyclic Graph):

```
frontend → api → backend → db
```

- **Frontend tier**: 1-2 services (web-gateway, nginx-ingress, cdn-edge, etc.)
- **API tier**: 2-3 services (payment-api, order-svc, auth-svc, etc.)
- **Backend tier**: 2-3 services (billing-worker, notification-svc, etc.)
- **Database tier**: 1-2 services (postgres-primary, redis-cache, etc.)

Edges have weights sampled from Uniform(0.4, 1.0) that modulate fault propagation severity.

### Fault System

15 fault types across 4 categories:

| Category | Faults |
|----------|--------|
| **Database** | connection_pool_exhausted, disk_io_saturation, replication_lag, deadlock_storm |
| **Network** | packet_loss, dns_resolution_failure, network_partition |
| **Compute** | memory_leak, cpu_saturation, oom_kill_loop, thread_pool_exhaustion |
| **Application** | bad_deploy, config_misconfiguration, dependency_timeout, circuit_breaker_open |

Each fault has:
- Health decay rate per step
- Latency multiplier and error rate delta
- Propagation delay (steps before spreading downstream)
- Cascade conditions (triggers secondary faults when health drops below threshold)

### Propagation Engine

Faults propagate downstream via a priority queue:
1. Root fault injected at step 0
2. Scheduled propagation events fire after delay (scaled by edge weight)
3. Cascades trigger when node health drops below fault-specific threshold
4. Effects compound: latency increases, error rates climb, CPU/memory pressure grows

### Observation Space

At each step, the agent sees:

```python
{
    "metrics": {
        "service-name": {
            "cpu": 0.25,      # Noisy (±0.02 Gaussian)
            "memory": 0.45,   # Noisy (±0.01 Gaussian)
            "latency_ms": 52, # Noisy (±5% Gaussian)
            "error_rate": 0.015,
            "health": 0.72
        }
    },
    "logs": ["ERROR service: connection timeout...", ...],  # Last 20 lines
    "alerts": ["DB_CONN_POOL_HIGH", "NET_LATENCY_HIGH"],
    "budget": 0.85,  # Remaining budget fraction
    "step": 5
}
```

**Probe action** returns detailed diagnostics:
```
PROBE payment-api: cpu=45% mem=62% latency_p99=180ms error_rate=0.0234
active_connections=12 uptime=24h37m active_faults=connection_pool_exhausted
```

## Action Space

Discrete actions indexed as:
- `0` to `N-1`: Probe service `i`
- `N` to `2N-1`: Recover service `i`

Where `N` is the number of services in the topology.

```python
# Example: 8-service topology
action_idx=0  → probe service[0]
action_idx=7  → probe service[7]
action_idx=8  → recover service[0]
action_idx=15 → recover service[7]
```

## Task Descriptions

### Task 1: Single Fault Isolation (Easy)
- **Topology**: 6 services
- **Faults**: 1 injected fault
- **Sensors**: Full visibility
- **Max steps**: 30
- **Passing score**: 0.70

> "A single service in a 6-node microservice topology has a fault. Identify the root cause service and recover the system."

### Task 2: Cascade Containment (Medium)
- **Topology**: 10 services
- **Faults**: 2 concurrent faults with cascades enabled
- **Sensors**: 20% extra noise
- **Max steps**: 45
- **Passing score**: 0.55

> "Two concurrent faults are propagating through a 10-node topology. Contain the cascade before it reaches critical services."

### Task 3: Full Incident Response (Hard)
- **Topology**: 15 services
- **Faults**: 3 concurrent faults with cascades
- **Sensors**: 3 masked nodes, 30% spurious alerts
- **Max steps**: 60
- **Passing score**: 0.40

> "A complex incident with 3 concurrent faults, masked sensors, and spurious alerts is degrading a 15-node production topology. Diagnose, contain, and recover all affected services."

## Reward Function

Dense reward at every step:

```python
reward = 0.0
reward += 1.0 * delta_health          # Health improvement (main signal)
reward -= 0.01                         # Step penalty
if target_health > 0.85:
    reward -= 0.15                     # Penalty for acting on healthy nodes
if all_services_healthy:
    reward += 1.0 + 0.5 * budget_ratio # Completion bonus
```

**Grader scores** (0.0-1.0) are computed at episode end based on:
- Root causes targeted with recovery actions
- Final system health
- Efficiency (fewer steps = higher score)
- Diagnostic discipline (probe before recovery)
- Cascade containment (for Task 2/3)

## Setup Instructions

### Prerequisites

- Python 3.10+
- pip or uv for dependency management

### Installation

```bash
cd it_fault_env

# Install dependencies
pip install openenv-core networkx numpy pydantic fastapi uvicorn openai requests

# Or with uv
uv sync
```

### Running the Server

```bash
# Development
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

# Docker
docker build -t it-fault-env:latest .
docker run -p 8000:8000 it-fault-env:latest
```

### Validation

```bash
# Validate against OpenEnv spec
openenv validate

# Should output: ✓ Validation passed
```

## Running the Baseline

```bash
# Set required credentials
export HF_TOKEN=your_hf_token_here
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct

# Run a single task
python inference.py --task task_1

# Run all tasks sequentially with reproducible seeds
python inference.py --all-tasks --seed 42

# Custom environment URL
python inference.py --all-tasks --seed 42 --env-url http://localhost:8000
```

## Baseline Performance Scores

Run `inference.py --all-tasks --seed 42` with a real model endpoint to record baseline scores for submission. The current script emits benchmark-style `[START]`, `[STEP]`, and `[END]` lines for each task.

## Hugging Face Spaces Deployment

```bash
# From the project root
docker build -t it-fault-env:latest .
docker run -p 8000:8000 it-fault-env:latest
```

To deploy on Hugging Face Spaces:

1. Create a new Space and choose `Docker` as the SDK.
2. Upload this repository contents to the Space.
3. Ensure the Space keeps port `8000` exposed.
4. Add any runtime secrets you need, such as `HF_TOKEN`, in the Space settings.
5. Let Hugging Face build the image from the root [`Dockerfile`](/Users/shreyasjoshi/funprojs/mph/recovery_env/it_fault_env/Dockerfile).
6. After the build completes, verify [`/health`](/Users/shreyasjoshi/funprojs/mph/recovery_env/it_fault_env/server/app.py#L81) and run `openenv validate` locally before submission.

## Extending the Environment

### Adding New Fault Types

Edit `env/faults.py`:

```python
FAULT_CATALOGUE["my_new_fault"] = FaultType(
    name="my_new_fault",
    display="My New Fault",
    targets=["api", "backend"],
    health_decay=0.05,
    latency_multiplier=2.0,
    error_rate_delta=0.05,
    propagation_delay=(2, 5),
    downstream_severity=0.6,
    cascade_to=["circuit_breaker_open"],
    cascade_threshold=0.5,
    log_template_key="my_new_fault",
    alert_codes=["MY_CUSTOM_ALERT"],
)
```

Then add log templates to `templates/app.json`.

### Adding New Tasks

Edit `env/tasks.py`:

```python
def grade_task_4(history: EpisodeHistory) -> float:
    # Custom grading logic
    return score

TASKS["task_4"] = Task(
    id="task_4",
    name="Custom Task",
    description="Task description...",
    difficulty="hard",
    config={"n_services": 20, ...},
    grader=grade_task_4,
    max_steps=90,
    passing_score=0.50,
)
```

Update `openenv.yaml` with the new task metadata.

## Project Structure

```
it_fault_env/
├── env/
│   ├── __init__.py
│   ├── graph.py       # Topology generation
│   ├── faults.py      # Fault catalogue + propagation
│   ├── env.py         # Core environment (ITFaultEnv)
│   ├── renderer.py    # Observation rendering
│   └── tasks.py       # Task definitions + graders
├── templates/
│   ├── db.json        # Database fault log templates
│   ├── network.json   # Network fault log templates
│   ├── compute.json   # Compute fault log templates
│   └── app.json       # Application fault log templates
├── server/
│   └── app.py         # FastAPI server
├── models.py          # Pydantic wire types
├── inference.py       # Baseline inference script
├── openenv.yaml       # HF Spaces manifest
├── Dockerfile
└── README.md
```

## License

BSD-style license (see LICENSE file).
