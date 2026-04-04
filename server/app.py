"""
FastAPI application exposing the ATC TRACON RL Environment via OpenEnv-compliant REST API.
"""

from __future__ import annotations

import sys
import os

# ensure local imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    ActionBatch, EnvironmentState, HealthResponse,
    ResetRequest, StepRequest, StepResult, TaskType,
)
from environment import ATCEnvironment

app = FastAPI(
    title="ATC TRACON RL Environment",
    description=(
        "OpenEnv-compliant RL environment for Air Traffic Control decision support. "
        "Supports Wake Turbulence Separation, Go-Around Prevention, Emergency Vectoring, "
        "Conflict Resolution, and Gate Assignment tasks."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared environment instance (single-session server)
_env = ATCEnvironment()
_initialized = False


# ──────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        version=ATCEnvironment.VERSION,
        tasks=[t.value for t in TaskType],
    )


@app.get("/", response_model=HealthResponse)
def root():
    return health()


# ──────────────────────────────────────────────
# OpenEnv core endpoints
# ──────────────────────────────────────────────

@app.post("/reset", response_model=EnvironmentState)
def reset(request: ResetRequest = None):
    global _initialized
    if request is None:
        request = ResetRequest()
    state = _env.reset(
        task=request.task,
        seed=request.seed,
        options=request.options,
    )
    _initialized = True
    return state


@app.post("/step", response_model=StepResult)
def step(request: StepRequest):
    global _initialized
    if not _initialized:
        raise HTTPException(status_code=400, detail="Call /reset before /step")
    try:
        result = _env.step(request.actions)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/state", response_model=EnvironmentState)
def state():
    if not _initialized:
        raise HTTPException(status_code=400, detail="Call /reset before /state")
    return _env.state()


# ──────────────────────────────────────────────
# Info endpoints
# ──────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    return {"tasks": [t.value for t in TaskType]}


@app.get("/tasks/{task_name}")
def task_info(task_name: str):
    descriptions = {
        "wake_turbulence": {
            "description": "Maintain safe wake-turbulence separation between sequenced aircraft.",
            "actions": ["heading_change", "speed_change", "sequence_swap"],
            "reward_basis": "Ratio of actual vs required separation distance, averaged over all consecutive pairs.",
        },
        "go_around_prevention": {
            "description": "Optimise landing sequence to prevent go-arounds and fuel emergencies.",
            "actions": ["sequence_swap", "speed_change"],
            "reward_basis": "Fraction of correctly ordered (faster-ahead) consecutive pairs, with fuel bonus.",
        },
        "emergency_vectoring": {
            "description": "Vector a MAYDAY aircraft to the runway with minimum delay.",
            "actions": ["vector", "heading_change", "sequence_swap"],
            "reward_basis": "Weighted score of proximity to runway, heading alignment, and fuel urgency.",
        },
        "conflict_resolution": {
            "description": "Resolve lateral/vertical conflicts with the fewest corrective actions.",
            "actions": ["heading_change", "altitude_change", "speed_change"],
            "reward_basis": "1 - (active_conflicts / number_of_aircraft).",
        },
        "gate_assignment": {
            "description": "Assign arriving aircraft to compatible and available gates efficiently.",
            "actions": ["assign_gate"],
            "reward_basis": "Weighted combination of assignment rate and compatibility score.",
        },
    }
    info = descriptions.get(task_name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_name}")
    return {"task": task_name, **info}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)