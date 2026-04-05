"""
app.py — FastAPI server for the ATC TRACON RL Environment.
"""
from __future__ import annotations

import sys
import os
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import (
    ATCAction,
    EnvironmentState,
    HealthResponse,
    ResetRequest,
    StepResult,
    TaskType,
)
from environment import ATCEnvironment


class StepRequest(BaseModel):
    actions: List[ATCAction]


app = FastAPI(
    title="ATC TRACON RL Environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env = ATCEnvironment()
_initialized = False


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


@app.post("/reset", response_model=EnvironmentState)
def reset(request: ResetRequest = None):
    global _initialized
    if request is None:
        request = ResetRequest()
    state = _env.reset(task=request.task, seed=request.seed, options=request.options)
    _initialized = True
    return state


@app.post("/step", response_model=StepResult)
def step(request: StepRequest):
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


@app.get("/tasks")
def list_tasks():
    return {"tasks": [t.value for t in TaskType]}


@app.get("/tasks/{task_name}")
def task_info(task_name: str):
    descriptions = {
        "wake_turbulence": {
            "description": "Maintain FAA-compliant wake-turbulence separation.",
            "actions": ["speed_change", "heading_change", "no_action"],
            "reward_basis": "+15 perfect spacing, -25 loss of separation.",
        },
        "go_around_prevention": {
            "description": "Optimise landing sequence to prevent go-arounds.",
            "actions": ["sequence_swap (rationale: rl_agent|fuel_priority|eta_optimized|fcfs)"],
            "reward_basis": "+12 low GA rate, -20 per go-around, -8/min holding.",
        },
        "emergency_vectoring": {
            "description": "Vector MAYDAY aircraft to runway conflict-free.",
            "actions": ["vector (value=heading)", "altitude_change (value=altitude)"],
            "reward_basis": "+18 fast clean insertion, -22 per conflict.",
        },
        "conflict_resolution": {
            "description": "Resolve converging aircraft with minimal corrections.",
            "actions": ["heading_change (negative=left, positive=right)", "speed_change"],
            "reward_basis": "+10 safely apart on-course, -15 critical proximity.",
        },
        "gate_assignment": {
            "description": "Assign arriving aircraft to best available gate.",
            "actions": ["assign_gate (gate_id: A1|A2|B1|B2|C1|C2)"],
            "reward_basis": "+15 quick arrival, -20 occupied/incompatible/blocked.",
        },
    }
    info = descriptions.get(task_name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_name}")
    return {"task": task_name, **info}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)