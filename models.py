"""
models.py — Typed Pydantic models for the ATC TRACON RL Environment.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    WAKE_TURBULENCE       = "wake_turbulence"
    GO_AROUND_PREVENTION  = "go_around_prevention"
    EMERGENCY_VECTORING   = "emergency_vectoring"
    CONFLICT_RESOLUTION   = "conflict_resolution"
    GATE_ASSIGNMENT       = "gate_assignment"


class AircraftCategory(str, Enum):
    HEAVY  = "heavy"
    LARGE  = "large"
    SMALL  = "small"
    SUPER  = "super"


class AircraftStatus(str, Enum):
    ENROUTE    = "enroute"
    APPROACH   = "approach"
    LANDING    = "landing"
    LANDED     = "landed"
    GO_AROUND  = "go_around"
    EMERGENCY  = "emergency"
    TAXIING    = "taxiing"
    PARKED     = "parked"


class ActionType(str, Enum):
    HEADING_CHANGE  = "heading_change"
    SPEED_CHANGE    = "speed_change"
    ALTITUDE_CHANGE = "altitude_change"
    SEQUENCE_SWAP   = "sequence_swap"
    ASSIGN_GATE     = "assign_gate"
    VECTOR          = "vector"
    NO_ACTION       = "no_action"


class AircraftState(BaseModel):
    callsign:      str
    category:      AircraftCategory
    status:        AircraftStatus
    x:             float
    y:             float
    altitude:      float
    heading:       float = Field(..., ge=0, le=360)
    speed:         float
    is_emergency:  bool            = False
    assigned_gate: Optional[str]   = None
    sequence_pos:  Optional[int]   = None
    fuel_state:    float           = Field(default=1.0, ge=0.0, le=1.0)


class GateState(BaseModel):
    gate_id:   str
    occupied:  bool                       = False
    aircraft:  Optional[str]              = None
    gate_type: AircraftCategory


class RunwayState(BaseModel):
    runway_id:          str
    active:             bool                       = True
    occupied:           bool                       = False
    last_departure_cat: Optional[AircraftCategory] = None


class ATCAction(BaseModel):
    action_type:      ActionType
    target_callsign:  str
    value:            Optional[float] = None
    secondary_target: Optional[str]   = None
    gate_id:          Optional[str]   = None
    rationale:        Optional[str]   = None

    class Config:
        use_enum_values = True


class EnvironmentState(BaseModel):
    task:              TaskType
    step:              int
    max_steps:         int
    aircraft:          List[AircraftState]
    gates:             List[GateState]
    runways:           List[RunwayState]
    active_conflicts:  List[Dict[str, Any]] = Field(default_factory=list)
    separation_matrix: Dict[str, float]     = Field(default_factory=dict)
    done:              bool                 = False
    info:              Dict[str, Any]       = Field(default_factory=dict)


class StepResult(BaseModel):
    state:      EnvironmentState
    reward:     float = Field(..., ge=0.0, le=1.0)
    done:       bool
    info:       Dict[str, Any] = Field(default_factory=dict)
    violations: List[str]      = Field(default_factory=list)
    score:      float          = Field(..., ge=0.0, le=1.0)


class ResetRequest(BaseModel):
    task:    Optional[TaskType] = None
    seed:    Optional[int]      = None
    options: Dict[str, Any]     = Field(default_factory=dict)


class StepRequest(BaseModel):
    actions: List[ATCAction]


class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str = "1.0.0"
    tasks:   List[str]