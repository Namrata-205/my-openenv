"""
Monsoon Flood Gate Control — OpenEnv HTTP Server
=================================================
FastAPI server exposing the standard OpenEnv endpoint interface:
  POST /reset          → Observation
  POST /step           → {observation, reward, done, info}
  GET  /state          → GlobalState
  GET  /health         → {"status": "ok"}
  GET  /tasks          → list of available tasks
  GET  /info           → environment metadata
  GET  /web            → web UI (if ENABLE_WEB_INTERFACE=true)
"""

import os
import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from models import Action, Observation, Reward, GlobalState
from environment import MonsoonFloodGateEnv
from graders import TASKS, grade


# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Monsoon Flood Gate Control",
    description="OpenEnv environment simulating stormwater gate management in Mumbai",
    version="1.0.0",
)

# Session-level environment instances (one per basin_idx + task_id combo)
_envs: dict[str, MonsoonFloodGateEnv] = {}

ENABLE_WEB = os.environ.get("ENABLE_WEB_INTERFACE", "false").lower() == "true"


def _get_env(task_id: str = "task_easy", basin_idx: int = 0, storm_seed: int | None = None) -> MonsoonFloodGateEnv:
    key = f"{task_id}_basin{basin_idx}_seed{storm_seed}"
    if key not in _envs:
        _envs[key] = MonsoonFloodGateEnv(task_id=task_id, basin_idx=basin_idx, storm_seed=storm_seed)
    return _envs[key]


# ─── Request / Response models ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_easy"
    basin_idx: int = 0
    storm_seed: Optional[int] = None


class StepRequest(BaseModel):
    task_id: str = "task_easy"
    basin_idx: int = 0
    storm_seed: Optional[int] = None
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "environment": "monsoon-floodgate-control"}


@app.get("/tasks")
def list_tasks():
    return {k: {
        "name": v["name"],
        "description": v["description"],
        "difficulty": v["difficulty"],
    } for k, v in TASKS.items()}


@app.get("/info")
def env_info():
    env = MonsoonFloodGateEnv()
    return {
        "name": "Monsoon Flood Gate Control",
        "version": "1.0.0",
        "description": "Multi-agent stormwater gate management for Mumbai flood prevention",
        "tasks": list(TASKS.keys()),
        "num_basins": 3,
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "step_duration_minutes": 5,
        "episode_length_steps": 72,
        "episode_duration_hours": 6,
    }


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest):
    env = _get_env(req.task_id, req.basin_idx, req.storm_seed)
    try:
        obs = env.reset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return obs


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = _get_env(req.task_id, req.basin_idx, req.storm_seed)
    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=GlobalState)
def get_state(task_id: str = "task_easy", basin_idx: int = 0, storm_seed: Optional[int] = None):
    env = _get_env(task_id, basin_idx, storm_seed)
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/grade")
def grade_episode(task_id: str = "task_easy", basin_idx: int = 0, storm_seed: Optional[int] = None):
    env = _get_env(task_id, basin_idx, storm_seed)
    try:
        score = env.grade_episode()
        return {"task_id": task_id, "score": score}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─── Web Interface ────────────────────────────────────────────────────────────

if ENABLE_WEB:
    @app.get("/web", response_class=HTMLResponse)
    def web_ui():
        return HTMLResponse(content=WEB_UI_HTML, status_code=200)


WEB_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Monsoon Flood Gate Control</title>
<style>
  body { font-family: monospace; background: #0a0f1e; color: #e0e8ff; padding: 20px; }
  h1 { color: #4fc3f7; }
  .basin { border: 1px solid #1e88e5; padding: 12px; margin: 8px 0; border-radius: 6px; }
  .flood { background: rgba(244,67,54,0.2); border-color: #f44336; }
  button { background: #1e88e5; color: white; border: none; padding: 8px 16px; cursor: pointer; border-radius: 4px; }
  pre { background: #111827; padding: 10px; border-radius: 4px; overflow: auto; }
  input[type=range] { width: 200px; }
  label { display: inline-block; width: 120px; }
</style>
</head>
<body>
<h1>🌧️ Monsoon Flood Gate Control</h1>
<div>
  <label>Task:</label>
  <select id="taskId">
    <option value="task_easy">Easy — Single Peak</option>
    <option value="task_medium">Medium — Double Burst</option>
    <option value="task_hard">Hard — Erratic Extreme</option>
  </select>
  <label style="margin-left:20px">Basin:</label>
  <select id="basinIdx"><option value="0">Basin 0</option><option value="1">Basin 1</option><option value="2">Basin 2</option></select>
  <button onclick="doReset()">🔄 Reset</button>
</div>
<h3>Controls</h3>
<div>
  <label>Gate 1:</label><input type="range" id="g0" min="0" max="1" step="0.05" value="0"><br>
  <label>Gate 2:</label><input type="range" id="g1" min="0" max="1" step="0.05" value="0"><br>
  <label>Gate 3:</label><input type="range" id="g2" min="0" max="1" step="0.05" value="0"><br>
  <label>Pump 1:</label><input type="range" id="p0" min="0" max="1" step="0.05" value="0"><br>
  <label>Pump 2:</label><input type="range" id="p1" min="0" max="1" step="0.05" value="0"><br>
  <button onclick="doStep()">⏩ Step</button>
</div>
<h3>Observation</h3>
<div id="basins"></div>
<pre id="obs">Run reset to start.</pre>
<script>
async function doReset() {
  const r = await fetch('/reset', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({task_id: taskId(), basin_idx: basinIdx()})});
  const obs = await r.json();
  renderObs(obs);
}
async function doStep() {
  const body = {
    task_id: taskId(), basin_idx: basinIdx(),
    action: {
      gate_positions: [+g('g0'), +g('g1'), +g('g2')],
      pump_speeds: [+g('p0'), +g('p1')]
    }
  };
  const r = await fetch('/step', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const data = await r.json();
  renderObs(data.observation);
  document.getElementById('obs').textContent = JSON.stringify(data, null, 2);
}
function g(id) { return parseFloat(document.getElementById(id).value); }
function taskId() { return document.getElementById('taskId').value; }
function basinIdx() { return parseInt(document.getElementById('basinIdx').value); }
function renderObs(obs) {
  const cls = obs.is_flooding ? 'basin flood' : 'basin';
  document.getElementById('basins').innerHTML =
    `<div class="${cls}">
      Basin ${obs.basin_id} | Water: ${(obs.water_level_fraction*100).toFixed(1)}%
      ${obs.is_flooding ? '🚨 FLOODING' : '✅ OK'}
      | Rain: ${obs.rainfall_mm_per_hr} mm/hr | Phase: ${obs.storm_phase}
      | Step: ${obs.timestep}/72
    </div>`;
  document.getElementById('obs').textContent = JSON.stringify(obs, null, 2);
}
</script>
</body>
</html>
"""


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
