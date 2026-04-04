# 🌧️ Monsoon Flood Gate Control
### OpenEnv RL Environment · Mumbai Stormwater Management

> **Real-world task** · **Multi-agent** · **Partial observability** · **3 difficulty tiers**

---

## Overview

Mumbai receives ~2,400 mm of annual rainfall, nearly 80% concentrated in the June–September monsoon window. Low-lying areas like **Dharavi** face recurring floods when stormwater drains cannot discharge fast enough into the overloaded Mithi River.

This environment simulates a **3-basin stormwater network** where RL agents dynamically control **gates** (gravity-fed, free) and **pumps** (powered, expensive) in response to real-time rain gauge readings and river level sensors. The key challenge is **multi-agent coordination under partial observability**: each agent sees its own basin clearly but only gets noisy, coarse signals about its neighbours.

**Why this is not a toy:** The hydrology follows standard orifice-flow equations (`Q = Cd·A·√(2g·Δh)`). Storm profiles are parameterised from documented Mumbai monsoon statistics. Energy costs are calibrated to typical municipal pump specifications.

---

## Environment Architecture

```
Rain ──→ [Basin 0] ──gates/pumps──→ Mithi River
                ↕ (partial obs)
Rain ──→ [Basin 1] ──gates/pumps──→ Mithi River
                ↕ (partial obs)
Rain ──→ [Basin 2] ──gates/pumps──→ Mithi River
```

Each agent receives:
- Full sensor data for its own basin
- Noisy, coarse signals from 2 neighbours (water fraction ± 10%, flood status)
- Rainfall forecast for next 3 steps (± 30% noise)

---

## Action & Observation Spaces

### Action Space

```json
{
  "gate_positions": [0.0, 0.0, 0.0],  // 3 gates, each 0.0 (closed) → 1.0 (fully open)
  "pump_speeds":    [0.0, 0.0]         // 2 pumps, each 0.0 (off) → 1.0 (full speed)
}
```

Gates use **gravity-driven orifice flow** — free to operate but only work when basin > river level. Pumps always discharge but cost energy.

### Observation Space

| Field | Type | Range | Description |
|---|---|---|---|
| `basin_id` | int | 0–2 | Which sub-basin |
| `water_level_m` | float | 0–2.2 | Current water depth (metres) |
| `water_level_fraction` | float | 0.0–1.1 | water / capacity |
| `rainfall_mm_per_hr` | float | 0–250 | Live rain gauge |
| `river_level_m` | float | 0.3–3.5 | Downstream river level |
| `gate_positions` | float[3] | 0.0–1.0 | Current gate states |
| `pump_speeds` | float[2] | 0.0–1.0 | Current pump states |
| `is_flooding` | bool | — | Water ≥ capacity |
| `neighbors` | list | — | Noisy partial view of other basins |
| `timestep` | int | 0–72 | Steps elapsed |
| `storm_phase` | str | enum | `pre_monsoon / peak / recession / clear` |
| `rainfall_forecast` | float[3] | 0–250 | Noisy 3-step-ahead forecast |

---

## Tasks

### Task 1 — `task_easy`: Single-Peak Monsoon Response
**Difficulty:** Easy | **Storm seed:** 42

A single smooth monsoon burst follows a Gaussian profile peaking around step 36. The forecast is noisy but reliable enough to pre-drain basins. Gate-only strategy can succeed.

**Success criteria:**
- Flood volume < 500 m³ (70% weight)
- Energy usage < 8 kWh (20% weight)
- No basin floods > 10 minutes (10% weight)

**Expected baseline scores:** Heuristic ~0.82, Reactive ~0.65, Random ~0.25

---

### Task 2 — `task_medium`: Double-Burst Storm Coordination
**Difficulty:** Medium | **Storm seed:** 137

Two storm bursts (steps 12–24 and 42–54) separated by a partial lull. Agents must use the lull to drain water and create buffer capacity for the second burst. Forecast noise is higher (±30%). Coordination matters — a basin that wastes capacity on the first burst will flood on the second.

**Success criteria:**
- Flood volume < 1,500 m³ (60% weight)
- Inter-basin flood-time standard deviation minimised (25% weight)
- Energy usage < 15 kWh (15% weight)

**Expected baseline scores:** Heuristic ~0.71, Reactive ~0.52, Random ~0.18

---

### Task 3 — `task_hard`: Erratic Extreme Event (Dharavi Worst-Case)
**Difficulty:** Hard | **Storm seed:** 999

Simulates documented worst-case Mumbai rainfall: random spikes up to 200 mm/hr, sudden drops, and rising river level that reduces gravity-drain effectiveness. Forecast is nearly useless. Partial observability of neighbours is most limiting.

**Success criteria:**
- Flood volume < 5,000 m³ (50% weight)
- Worst-performing basin < 60 flood-minutes (30% weight)
- Moderate energy use (not too little = under-responding, not too much = wasteful) (20% weight)

**Expected baseline scores:** Heuristic ~0.56, Reactive ~0.41, Random ~0.12

---

## Reward Function

Each step returns a shaped reward with 5 components:

```python
reward = (
    flood_penalty        # –2.0 when flooding; –0.5 when water > 85% capacity
  + drainage_bonus       # +0.0–0.5 when outflow > inflow and basin >50% full
  + energy_cost          # –proportional to pump kWh used this step
  + gate_chatter_penalty # –0.05 per unit of gate oscillation
  + forecast_bonus       # +0.15 for proactive pre-draining before forecast spike
)
```

**Partial progress:** Reward is non-zero throughout the trajectory, not just at episode end. An agent can improve by reducing water fraction even without completely preventing flooding.

---

## Setup & Usage

### Prerequisites
- Python 3.10+
- Docker Desktop
- (Optional) OpenAI API key for LLM agent baseline

### Local Development

```bash
# Clone / create project
cd monsoon-floodgate-env

# Install dependencies
pip install -r requirements.txt

# Run server directly (no Docker)
cd server
python main.py
# Server starts at http://localhost:8000
```

### Docker

```bash
# Build image (from project root — Dockerfile must be here)
docker build -t monsoon-floodgate-control:latest .

# Run container
docker run -p 8000:8000 monsoon-floodgate-control:latest

# With web UI enabled
docker run -p 8000:8000 -e ENABLE_WEB_INTERFACE=true monsoon-floodgate-control:latest
# Visit http://localhost:8000/web
```

### Test Endpoints

```bash
# Health check
curl http://localhost:8000/health

# List tasks
curl http://localhost:8000/tasks

# Reset environment
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy", "basin_idx": 0}'

# Send a step action
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task_easy",
    "basin_idx": 0,
    "action": {
      "gate_positions": [0.8, 0.8, 0.5],
      "pump_speeds": [0.3, 0.2]
    }
  }'

# Get grader score (after episode)
curl "http://localhost:8000/grade?task_id=task_easy&basin_idx=0"
```

### Run Baseline Inference

```bash
# All strategies, all tasks
python baseline_inference.py --base-url http://localhost:8000 --strategy all

# LLM agent only (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python baseline_inference.py --strategy llm

# Heuristic only (no API key needed)
python baseline_inference.py --strategy heuristic
```

---

## OpenEnv Validation

```bash
pip install openenv-core[cli]
openenv validate .
```

---

## Deploy to Hugging Face Spaces

```bash
# Login
huggingface-cli login

# Create Space
huggingface-cli repo create monsoon-floodgate-control --type space --space-sdk docker

# Push
git init && git add .
git commit -m "Initial OpenEnv submission"
git remote add origin https://huggingface.co/spaces/<YOUR_USERNAME>/monsoon-floodgate-control
git push -u origin main
```

Tag your Space with `openenv` for hackathon evaluation.

---

## Baseline Scores (Reproducible)

| Strategy | task_easy | task_medium | task_hard | Average |
|---|---|---|---|---|
| Rule-Based Heuristic | ~0.82 | ~0.71 | ~0.56 | ~0.70 |
| Reactive Threshold | ~0.65 | ~0.52 | ~0.41 | ~0.53 |
| LLM Agent (GPT-4o-mini) | ~0.79 | ~0.68 | ~0.53 | ~0.67 |
| Random | ~0.25 | ~0.18 | ~0.12 | ~0.18 |

Scores are averaged over all 3 basin indices with fixed storm seeds per task.

---

## File Structure

```
monsoon-floodgate-env/
├── Dockerfile                  # Must be at project root
├── openenv.yaml                # OpenEnv metadata spec
├── requirements.txt
├── baseline_inference.py       # Baseline runner (3 strategies × 3 tasks)
├── README.md
└── server/
    ├── main.py                 # FastAPI HTTP server
    ├── environment.py          # MonsoonFloodGateEnv (OpenEnv interface)
    ├── simulation.py           # Physics simulation engine
    ├── models.py               # Typed Pydantic models (Action/Observation/Reward/State)
    └── graders.py              # Task definitions + programmatic graders
```

---

## Innovation Highlights

**Multi-agent with partial observability:** Neighbours share only coarse, noisy signals — agents cannot perfectly coordinate, mirroring real SCADA system limitations.

**Physics-based hydrology:** Orifice-flow equations, runoff coefficients, catchment areas and river backpressure are all derived from documented hydraulic engineering principles.

**Erratic storm generation:** The hard task uses a random-walk rainfall generator calibrated to Mumbai's documented extreme event statistics (Maharashtra State Disaster Management Authority reports).

**Shaped reward with forecast bonus:** Encourages proactive behaviour (pre-draining before rain arrives) rather than purely reactive control — a key real-world desideratum for flood management systems.
