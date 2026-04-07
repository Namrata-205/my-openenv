"""
models.py — Pydantic typed models for the ATC TRACON RL Environment.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enumerations ───────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    WAKE_TURBULENCE      = "wake_turbulence"
    GO_AROUND_PREVENTION = "go_around_prevention"
    EMERGENCY_VECTORING  = "emergency_vectoring"
    CONFLICT_RESOLUTION  = "conflict_resolution"
    GATE_ASSIGNMENT      = "gate_assignment"


class ActionType(str, Enum):
    SPEED_CHANGE    = "speed_change"
    HEADING_CHANGE  = "heading_change"
    ALTITUDE_CHANGE = "altitude_change"
    SEQUENCE_SWAP   = "sequence_swap"
    ASSIGN_GATE     = "assign_gate"
    VECTOR          = "vector"
    NO_ACTION       = "no_action"


class AircraftCategory(str, Enum):
    HEAVY  = "Heavy"
    LARGE  = "Large"
    SMALL  = "Small"


class AircraftStatus(str, Enum):
    APPROACH   = "approach"
    ENROUTE    = "enroute"
    EMERGENCY  = "emergency"
    GO_AROUND  = "go_around"
    LANDED     = "landed"
    DEPARTING  = "departing"


# ── Action ─────────────────────────────────────────────────────────────────────

class ATCAction(BaseModel):
    """A single controller action directed at one aircraft."""
    action_type:      ActionType
    target_callsign:  str
    value:            Optional[float] = None
    secondary_target: Optional[str]   = None
    gate_id:          Optional[str]   = None
    rationale:        Optional[str]   = None
    time:             Optional[float] = None


# ── State objects ──────────────────────────────────────────────────────────────

class AircraftState(BaseModel):
    callsign:       str
    category:       AircraftCategory = AircraftCategory.LARGE
    status:         AircraftStatus   = AircraftStatus.ENROUTE
    x:              float = 0.0
    y:              float = 0.0
    altitude:       float = 3000.0
    heading:        float = 180.0
    speed:          float = 160.0
    is_emergency:   bool  = False
    fuel_state:     float = 1.0          # 0.0 (empty) → 1.0 (full)
    sequence_pos:   Optional[int]   = None
    assigned_gate:  Optional[str]   = None


class GateState(BaseModel):
    gate_id:    str
    occupied:   bool                    = False
    aircraft:   Optional[str]           = None   # callsign of occupying aircraft
    gate_type:  Optional[AircraftCategory] = None


class RunwayState(BaseModel):
    runway_id: str
    active:    bool = True


class EnvironmentState(BaseModel):
    task:             TaskType
    step:             int
    max_steps:        int
    aircraft:         List[AircraftState]
    gates:            List[GateState]      = Field(default_factory=list)
    runways:          List[RunwayState]    = Field(default_factory=list)
    active_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    done:             bool                 = False
    info:             Dict[str, Any]       = Field(default_factory=dict)


# ── Step result ────────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    state:      EnvironmentState
    reward:     float
    done:       bool
    info:       Dict[str, Any]       = Field(default_factory=dict)
    violations: List[str]            = Field(default_factory=list)
    score:      float                = 0.0


# ── Ancillary request / response models ───────────────────────────────────────

class ResetRequest(BaseModel):
    task:    Optional[TaskType]        = None
    seed:    Optional[int]             = None
    options: Optional[Dict[str, Any]]  = None


class HealthResponse(BaseModel):
    status:  str
    version: str
    tasks:   List[str]