# ATC TRACON RL Environment

An OpenEnv-compliant Reinforcement Learning environment for **Air Traffic Control (ATC) decision support** in Terminal Radar Approach Control (TRACON) scenarios.

The system acts as a **decision-support agent** (not full automation), providing minimal, interpretable advisory actions to assist human controllers.

---

## 📁 Project Structure

```
my_env/
├── models.py              # Pydantic typed models
├── client.py              # Python HTTP client
├── openenv.yaml           # OpenEnv specification
├── __init__.py
├── README.md
├── outputs/               # Episode outputs / logs
└── server/
    ├── your_environment.py  # Core RL environment simulation
    ├── app.py               # FastAPI REST server
    ├── requirements.txt
    └── Dockerfile
inference.py               # LLM-powered inference script
pyproject.toml
```

---

## 🛫 Supported Tasks

| Task                       | ID                     | Description                                      |
| -------------------------- | ---------------------- | ------------------------------------------------ |
| Wake Turbulence Separation | `wake_turbulence`      | Maintain safe spacing between sequenced aircraft |
| Go-Around Prevention       | `go_around_prevention` | Optimise landing order to prevent go-arounds     |
| Emergency Vectoring        | `emergency_vectoring`  | Priority routing for MAYDAY aircraft             |
| Conflict Resolution        | `conflict_resolution`  | Resolve near-conflicts with minimal actions      |
| Gate Assignment            | `gate_assignment`      | Efficient airport stand allocation               |

---

## ⚙️ API Endpoints

| Method | Path          | Description                                   |
| ------ | ------------- | --------------------------------------------- |
| `GET`  | `/health`     | Health check                                  |
| `POST` | `/reset`      | Reset environment, returns `EnvironmentState` |
| `POST` | `/step`       | Apply actions, returns `StepResult`           |
| `GET`  | `/state`      | Get current state                             |
| `GET`  | `/tasks`      | List available tasks                          |
| `GET`  | `/tasks/{id}` | Task details                                  |

---

## 🚀 Quick Start

### Run the server

```bash
cd my_env/server
pip install -r requirements.txt
python app.py
# → http://localhost:7860
```

### Run with Docker

```bash
docker build -f my_env/server/Dockerfile -t atc-rl .
docker run -p 7860:7860 \
  -e API_BASE_URL=http://localhost:7860 \
  -e MODEL_NAME=claude-sonnet-4-20250514 \
  -e HF_TOKEN=your_token \
  atc-rl
```

### Python client

```python
from my_env import ATCClient, ATCAction, ActionType, TaskType

client = ATCClient("http://localhost:7860")
state  = client.reset(task=TaskType.CONFLICT_RESOLUTION, seed=42)
print(f"Aircraft: {len(state.aircraft)}, Conflicts: {len(state.active_conflicts)}")

actions = [
    ATCAction(action_type=ActionType.HEADING_CHANGE,
              target_callsign="AAL201",
              value=20,
              rationale="Turn right to avoid conflict")
]
result = client.step(actions)
print(f"Reward: {result.reward:.3f} | Score: {result.score:.3f}")
```

### Run inference

```bash
export API_BASE_URL=http://localhost:7860
export MODEL_NAME=claude-sonnet-4-20250514
export HF_TOKEN=your_anthropic_key

python inference.py conflict_resolution
```

---

## 🎯 Reward / Score Design

All rewards and scores are **normalised to [0.0, 1.0]**.

| Task                 | Score formula                                               |
| -------------------- | ----------------------------------------------------------- |
| Wake Turbulence      | `mean(actual_sep / required_sep)` clipped to [0,1] per pair |
| Go-Around Prevention | `1 - (inversions / pairs)` × fuel bonus                     |
| Emergency Vectoring  | `0.4·proximity + 0.4·alignment + 0.2·fuel_urgency`          |
| Conflict Resolution  | `1 - (conflicts / aircraft_count)`                          |
| Gate Assignment      | `0.5·assign_rate + 0.5·compat_rate`                         |

---

## ✈️ Aircraft State

```json
{
  "callsign": "UAL100",
  "category": "heavy",
  "status": "approach",
  "x": -12.4,
  "y": 18.7,
  "altitude": 3200,
  "heading": 175.0,
  "speed": 158.0,
  "is_emergency": false,
  "fuel_state": 0.72,
  "sequence_pos": 1,
  "assigned_gate": null
}
```

---

## 🔐 Environment Variables

| Variable       | Description                       |
| -------------- | --------------------------------- |
| `API_BASE_URL` | ATC environment server URL        |
| `MODEL_NAME`   | LLM model identifier              |
| `HF_TOKEN`     | API key (Anthropic / HuggingFace) |

---

## 📋 OpenEnv Compliance

- ✅ `reset()`, `step()`, `state()` APIs
- ✅ Typed Pydantic models
- ✅ 5 tasks implemented with graders
- ✅ Scores normalised [0.0, 1.0]
- ✅ `openenv.yaml` specification
- ✅ Dockerfile builds successfully
- ✅ Inference script with `[START]` / `[STEP]` / `[END]` format

---

## 📄 License

MIT
