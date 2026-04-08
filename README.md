# ATC TRACON RL Environment

---
title: OpenEnv ATC
emoji: ✈️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

> An **OpenEnv-compliant Reinforcement Learning environment** for Air Traffic Control (ATC) decision support in Terminal Radar Approach Control (TRACON) scenarios.

The system acts as a **decision-support agent** — not full automation. It provides minimal, interpretable advisory actions to assist human controllers across five procedural ATC tasks.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Supported Tasks](#supported-tasks)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Reward & Score Design](#reward--score-design)
- [Quick Start](#quick-start)
- [Running with Docker](#running-with-docker)
- [Python Client](#python-client)
- [Inference Agent](#inference-agent)
- [Environment Variables](#environment-variables)
- [Data Models](#data-models)
- [OpenEnv Compliance](#openenv-compliance)
- [License](#license)

---

## Overview

The ATC TRACON RL Environment simulates five real-world ATC decision tasks at a TRACON facility. Each task exposes a REST API following the OpenEnv standard — `reset`, `step`, and `state` — allowing any RL agent or LLM to interact with it over HTTP.

Key design principles:

- **Safety-first rewards** — violations are penalised heavily; safe behaviour is rewarded incrementally
- **Normalised scores** — all episode scores are bounded to `[0.0, 1.0]` for fair comparison
- **Deterministic replay** — every task accepts a `seed` parameter for reproducible scenarios
- **LLM-compatible** — the inference script supports rule-based, Grok (xAI), and HuggingFace models with a fallback chain

---

## Project Structure

```
my-openenv/
├── models.py              # Pydantic typed models (actions, state, results)
├── client.py              # Python HTTP client with retry logic
├── inference.py           # Hybrid rule-based + LLM inference agent
├── prevalidation.py       # Streamlit UI for manual environment testing
├── test_env.py            # Automated end-to-end test suite
├── openenv.yaml           # OpenEnv specification (tasks, endpoints, schemas)
├── pyproject.toml         # Package metadata and dependencies
├── requirements.txt       # Server dependencies (pinned)
├── __init__.py
├── Dockerfile
└── server/
    ├── app.py             # FastAPI REST server (5 endpoints)
    ├── environment.py     # ATCEnvironment — episode management, state transitions
    ├── graders.py         # Per-task reward logic (5 grader classes)
    └── requirements.txt   # Server-only dependencies
```

### Key file responsibilities

| File | Responsibility |
|------|----------------|
| `server/graders.py` | Implements reward/penalty logic for all 5 tasks. Each task is a self-contained dataclass with a `step()` method that returns `(reward, log)`. |
| `server/environment.py` | Manages episode lifecycle: `reset()`, `step()`, `state()`. Translates HTTP actions into grader calls, normalises raw rewards to `[0,1]`, and tracks terminal conditions. |
| `server/app.py` | FastAPI application. Stateless wrapper around a single `ATCEnvironment` instance. |
| `models.py` | All shared Pydantic types: `ATCAction`, `EnvironmentState`, `StepResult`, `TaskType`, `ActionType`. |
| `client.py` | `ATCClient` — thin HTTP wrapper with exponential-backoff retry on 5xx errors. |
| `inference.py` | Hybrid agent: rule-based handlers for all 5 tasks, with Grok API and HuggingFace LLM as fallbacks for unsupported scenarios. |

---

## Supported Tasks

| Task | `task` value | Description | Max Steps |
|------|-------------|-------------|-----------|
| Wake Turbulence Separation | `wake_turbulence` | Maintain FAA-compliant separation between a leading and trailing aircraft pair | 40 |
| Go-Around Prevention | `go_around_prevention` | Optimise the inbound landing sequence to prevent go-arounds while managing fuel-critical aircraft | 40 |
| Emergency Vectoring | `emergency_vectoring` | Vector a MAYDAY aircraft (EMER1) to the runway conflict-free as quickly as possible | 40 |
| Conflict Resolution | `conflict_resolution` | Resolve a converging aircraft pair with minimal heading or speed corrections | 40 |
| Gate Assignment | `gate_assignment` | Assign arriving aircraft to the best available compatible gate from pool A1–C2 | 40 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    inference.py / client                 │
│   Rule-based agent → Grok API → HF LLM → fallback       │
└────────────────────┬────────────────────────────────────┘
                     │  HTTP  POST /reset, /step, GET /state
┌────────────────────▼────────────────────────────────────┐
│                    server/app.py  (FastAPI)              │
│          /health  /reset  /step  /state  /tasks          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              server/environment.py                       │
│  ATCEnvironment  ─── episode state, step dispatch,       │
│                       reward normalisation, terminal      │
└────────────────────┬────────────────────────────────────┘
                     │  calls graders per task
┌────────────────────▼────────────────────────────────────┐
│              server/graders.py                           │
│  WakeTurbulenceEnv  │  sequence_flights()                │
│  EmergencyVectorEnv │  ConflictAlertEnv                  │
│  GateAssignmentEnv  │                                    │
└─────────────────────────────────────────────────────────┘
```

Each episode follows the standard RL loop:

```
reset(task, seed) → initial EnvironmentState
    ↓
step(actions)     → StepResult { state, reward, score, done, violations, info }
    ↓  (repeat until done=True or max_steps reached)
summary
```

---

## API Reference

All endpoints are served at `http://localhost:7860` by default.

### `GET /health`

Returns server status, version, and available task list.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "tasks": ["wake_turbulence", "go_around_prevention", "emergency_vectoring",
            "conflict_resolution", "gate_assignment"]
}
```

---

### `POST /reset`

Initialise a new episode. Returns `EnvironmentState`.

**Request body** (all fields optional):

```json
{
  "task": "conflict_resolution",
  "seed": 42,
  "options": { "max_steps": 30 }
}
```

- `task` — one of the five task IDs. Omit to pick randomly.
- `seed` — integer for reproducible scenarios.
- `options.max_steps` — override the default 40-step episode limit.

---

### `POST /step`

Advance the episode by one step. Returns `StepResult`.

**Request body:**

```json
{
  "actions": [
    {
      "action_type": "heading_change",
      "target_callsign": "AC1",
      "value": -10
    }
  ]
}
```

**Action fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `action_type` | string | ✅ | See action types below |
| `target_callsign` | string | ✅ | Callsign of the target aircraft |
| `value` | number | depends | Heading degrees, speed knots, altitude feet |
| `secondary_target` | string | `sequence_swap` only | Second callsign to swap with |
| `gate_id` | string | `assign_gate` only | Gate identifier (A1–C2) |
| `rationale` | string | no | Free-text reason (for `sequence_swap` strategy hint) |

**Action types by task:**

| Task | Valid `action_type` | `value` meaning |
|------|---------------------|-----------------|
| `wake_turbulence` | `speed_change`, `heading_change` | Speed delta (kts) or heading delta (deg) |
| `go_around_prevention` | `sequence_swap` | — (use `rationale`: `rl_agent` \| `fuel_priority` \| `eta_optimized` \| `fcfs`) |
| `emergency_vectoring` | `vector`, `altitude_change` | Target heading (0–360°) or target altitude (ft) |
| `conflict_resolution` | `heading_change`, `speed_change` | Negative = left/slow, positive = right/fast |
| `gate_assignment` | `assign_gate` | — (use `gate_id`) |

**Step result:**

```json
{
  "state": { ... },
  "reward": 0.7344,
  "score":  0.8000,
  "done":   false,
  "violations": [],
  "info": {
    "raw_reward": 12.0,
    "sep_nm": 5.2,
    "required_nm": 5.0,
    "action": "slow_down_trailing",
    "terminal": false
  }
}
```

- `reward` — shaped step reward, normalised to `[0.0, 1.0]`
- `score` — clean episodic performance metric, normalised to `[0.0, 1.0]`
- `violations` — list of safety violation strings for this step
- `info.raw_reward` — un-normalised reward from the grader

---

### `GET /state`

Returns the current `EnvironmentState` without advancing the episode. Requires a prior `/reset`.

---

### `GET /tasks`

Returns `{ "tasks": ["wake_turbulence", ...] }`.

---

### `GET /tasks/{task_name}`

Returns the description, valid actions, and reward basis for a specific task.

---

## Reward & Score Design

### Raw reward structure (per task)

#### Task 1 — Wake Turbulence Separation

| Condition | Reward |
|-----------|--------|
| Perfect FAA-compliant spacing (within 1 NM of required) | +12 |
| Safe but wider than optimal | +6 |
| Loss of separation (below required) | −20 |
| Approaching minimum separation (warning) | −5 |
| Delay accumulation > 40 s | −5 |

**FAA required minimums** (NM):

| Leading → Trailing | Required |
|--------------------|----------|
| Heavy → Heavy | 4.0 NM |
| Heavy → Medium | 5.0 NM |
| Heavy → Light | 6.0 NM |
| Medium → Medium | 3.0 NM |
| Medium → Light | 4.0 NM |
| Light → Light | 3.0 NM |

#### Task 2 — Go-Around Prevention

| Condition | Reward |
|-----------|--------|
| Go-around rate < 5% (episode bonus) | +12 |
| On-time landing (hold ≤ 1 min) | +3 |
| Go-around triggered | −20 |
| Holding penalty | −8 × ⌈wait minutes⌉ |

Go-around triggers: hold time > 8 min, or fuel < 3,000 lbs with any hold > 4 min.

Sequencing strategies available via `rationale` field:

| Strategy | Behaviour |
|----------|-----------|
| `rl_agent` | Multi-factor: ETA × 0.4 + fuel × 0.3 − priority × 3 |
| `fuel_priority` | Critical fuel (< 3,000 lbs) aircraft land first, then by ETA |
| `eta_optimized` | Tightest ETA first |
| `fcfs` | First come, first served (baseline) |

#### Task 3 — Emergency Vectoring

| Condition | Reward |
|-----------|--------|
| Conflict-free insertion in ≤ 2 min | +18 |
| Conflict-free but slow (> 2 min) | partial credit |
| Flow smooth (degradation < 10%) | +4 |
| Safety violation (< 3 NM lateral AND < 1,000 ft vertical) | −22 each |
| Flow breakdown (degradation ≥ 10%) | −12 |
| Runway alignment bonus (heading near 270°) | up to +5 |
| Low-fuel urgency bonus (clean insertion only) | up to +6 |

#### Task 4 — Conflict Resolution

| Condition | Reward |
|-----------|--------|
| Separation > 8 NM AND both aircraft within 20° of target heading | +10 |
| Separation > 8 NM but off-course | +5 |
| Safe-time streak bonus | +2 × consecutive safe steps |
| Separation decreased vs previous step | −5 |
| Critical proximity (< 5 NM) | −15 |

Aircraft positions advance 1 minute along their current headings each step. Scoring is on post-tick positions so the effect of the action is what gets evaluated.

#### Task 5 — Gate Assignment

| Condition | Reward |
|-----------|--------|
| Quick arrival (ETA ≤ 12 min) AND short taxi (≤ 7 min) | +15 |
| Short taxi ≤ 5 min | +5 |
| Long taxi > 10 min | −8 |
| Long ETA > 15 min | −8 |
| Blocked taxi path | −20 |
| Incompatible gate size | −20 |
| Occupied gate | −30 |

Available gates: `A1` (4 min taxi), `A2` (3.5 min), `B1` (12 min), `B2` (6 min, size-restricted), `C1` (5.5 min), `C2` (7.5 min).

### Normalisation

All raw rewards are normalised per step to `[0.0, 1.0]` using:

```
norm = clamp((raw − low) / (high − low), 0.0, 1.0)
```

Normalisation bounds per task:

| Task | `low` | `high` |
|------|-------|--------|
| Wake turbulence | −55 | 25 |
| Go-around prevention | −28 × n\_flights | 12 + 3 × n\_flights |
| Emergency vectoring | −22 × n\_traffic − 12 | 33 |
| Conflict resolution | −25 | 80 |
| Gate assignment | −60 | 20 |

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### 1. Install dependencies

```bash
# Clone the repo
git clone https://github.com/your-org/my-openenv.git
cd my-openenv

# Install server dependencies
pip install -r server/requirements.txt

# Install client/inference dependencies
pip install -r requirements.txt
```

### 2. Start the server

```bash
python server/app.py
# Server running at http://localhost:7860
```

### 3. Run the test suite

```bash
python test_env.py
# All 5 tasks × 3 steps each — should print [PASS] for every check
```

### 4. Run a task manually

```bash
# Default task (conflict_resolution), random seed
python inference.py

# Specify task and seed
python inference.py wake_turbulence 42

# Specify task, seed, and model
python inference.py emergency_vectoring 7 mistralai/Mistral-7B-Instruct-v0.2
```

---

## Running with Docker

```bash
# Build the image
docker build -t atc-rl .

# Run with default settings (no LLM required)
docker run -p 7860:7860 atc-rl

# Run with HuggingFace LLM
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_your_token_here \
  -e MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct \
  atc-rl

# Run with Grok API (xAI)
docker run -p 7860:7860 \
  -e GROK_API_KEY=xai_your_key_here \
  -e GROK_MODEL=grok-3-mini \
  atc-rl
```

---

## Python Client

The `ATCClient` class provides a clean Python interface with automatic retry on transient server errors.

```python
from client import ATCClient

# Connect
client = ATCClient("http://localhost:7860")

# Check health
info = client.health()
print(info["tasks"])

# Reset a specific task
state = client.reset(task="wake_turbulence", seed=42)
print(f"Aircraft: {len(state['aircraft'])}")

# Run an episode
while not state["done"]:
    actions = [
        {
            "action_type":     "speed_change",
            "target_callsign": "TRAIL",
            "value":           -10
        }
    ]
    result = client.step(actions)
    state  = result["state"]
    print(f"Reward: {result['reward']:.3f}  Score: {result['score']:.3f}  Done: {result['done']}")

client.close()
```

Using it as a context manager:

```python
with ATCClient("http://localhost:7860") as client:
    state = client.reset(task="gate_assignment", seed=1)
    result = client.step([{
        "action_type":     "assign_gate",
        "target_callsign": "UAL101",
        "gate_id":         "A1"
    }])
    print(result["reward"])
```

### Client configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_url` | `http://localhost:7860` | Server URL |
| `timeout` | 30.0 s | Per-request timeout |
| `max_retries` | 3 | Retries on 5xx errors |
| `retry_delay` | 1.0 s | Base delay (doubles each retry) |

---

## Inference Agent

`inference.py` implements a hybrid decision agent with a three-tier fallback chain:

```
Rule-based (primary) → Grok API → HuggingFace LLM → hard fallback
```

### Rule-based agent (Tier 1)

Deterministic logic for all 5 tasks — no API calls required:

| Task | Strategy |
|------|----------|
| `wake_turbulence` | Speed/heading adjustments based on Euclidean separation vs required minimum |
| `go_around_prevention` | Emergency/fuel-priority sort, spacing check on approach train |
| `emergency_vectoring` | Direct vector toward runway heading 180° |
| `conflict_resolution` | Proactive divergence at 6 NM, emergency separation at 3 NM |
| `gate_assignment` | Best available gate by taxi distance and occupancy status |

### LLM fallbacks (Tiers 2 & 3)

Set environment variables to enable:

```bash
export GROK_API_KEY=xai_...         # enables Grok API fallback
export HF_TOKEN=hf_...              # enables HuggingFace fallback
export MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
export GROK_MODEL=grok-3-mini       # optional, default: grok-3-mini
```

Both LLM backends receive the same structured prompt:
- Full current aircraft state (position, altitude, heading, speed, fuel)
- Active conflicts list
- Available gates
- Step number and task name

### Running inference

```bash
# All positional arguments are optional
python inference.py [task] [seed]

python inference.py wake_turbulence
python inference.py go_around_prevention 42
python inference.py emergency_vectoring 7
python inference.py conflict_resolution
python inference.py gate_assignment 99
```

### Output format

```
[START] Initialising environment ...
[STEP] step=1 | actions=[speed_change(TRAIL,Δ-20)]
[STEP] step=1 | reward=0.6364 | score=0.7000 | conflicts=0 | done=False
[STEP] step=2 | actions=[no_action(TRAIL)]
[STEP] step=2 | reward=0.7273 | score=0.8000 | conflicts=0 | done=False
...
[STEP] ── Episode Summary ──────────────────────────
[STEP] Task:              wake_turbulence
[STEP] Completed steps:   3
[STEP] Avg reward:        0.6980
[STEP] Avg score:         0.7500
[STEP] Total violations:  0
[END]
```

### Prevalidation UI

A Streamlit UI is available for manual testing:

```bash
pip install streamlit
streamlit run prevalidation.py
```

This connects to `https://dcode7-openenv.hf.space` (the hosted environment). Change `BASE_URL` in the file to point to a local server.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | no | `http://localhost:7860` | URL of the running environment server |
| `MODEL_NAME` | no | `mistralai/Mistral-7B-Instruct-v0.2` | LLM model identifier for HuggingFace inference |
| `HF_TOKEN` | for HF LLM | — | HuggingFace API token |
| `GROK_API_KEY` | for Grok | — | xAI Grok API key |
| `GROK_MODEL` | no | `grok-3-mini` | Grok model identifier |

---

## Data Models

All models are defined in `models.py` using Pydantic v2.

### `ATCAction`

```python
class ATCAction(BaseModel):
    action_type:      ActionType          # required
    target_callsign:  str                 # required
    value:            Optional[float]     # speed delta, heading delta, altitude
    secondary_target: Optional[str]       # for sequence_swap
    gate_id:          Optional[str]       # for assign_gate
    rationale:        Optional[str]       # sequencing strategy hint or free text
    time:             Optional[float]     # reserved
```

### `AircraftState`

```python
class AircraftState(BaseModel):
    callsign:      str                    # e.g. "UAL101"
    category:      AircraftCategory       # Heavy | Large | Small
    status:        AircraftStatus         # approach | enroute | emergency | ...
    x:             float                  # NM east
    y:             float                  # NM north (or ETA minutes for sequencing)
    altitude:      float                  # feet MSL
    heading:       float                  # degrees
    speed:         float                  # knots
    is_emergency:  bool
    fuel_state:    float                  # 0.0 (empty) → 1.0 (full)
    sequence_pos:  Optional[int]
    assigned_gate: Optional[str]
```

### `EnvironmentState`

```python
class EnvironmentState(BaseModel):
    task:             TaskType
    step:             int
    max_steps:        int
    aircraft:         List[AircraftState]
    gates:            List[GateState]
    runways:          List[RunwayState]
    active_conflicts: List[Dict]          # [{ac1, ac2, lateral_nm, vertical_ft}]
    done:             bool
    info:             Dict[str, Any]      # task-specific diagnostics
```

### `StepResult`

```python
class StepResult(BaseModel):
    state:      EnvironmentState
    reward:     float             # normalised [0.0, 1.0]
    done:       bool
    info:       Dict[str, Any]
    violations: List[str]
    score:      float             # normalised [0.0, 1.0]
```

---

## OpenEnv Compliance

This environment conforms to the OpenEnv specification defined in `openenv.yaml`.

| Requirement | Status |
|-------------|--------|
| `POST /reset` with `task`, `seed`, `options` | ✅ |
| `POST /step` with `actions` array | ✅ |
| `GET /state` | ✅ |
| `GET /health` with `status`, `version`, `tasks` | ✅ |
| Typed Pydantic models | ✅ |
| Scores normalised to `[0.0, 1.0]` | ✅ |
| 5 tasks implemented with graders | ✅ |
| `openenv.yaml` specification file | ✅ |
| Dockerfile | ✅ |
| Inference script with `[START]` / `[STEP]` / `[END]` logging | ✅ |
| Deterministic seed support | ✅ |

---

## License

MIT
