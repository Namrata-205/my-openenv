"""
Pydantic models for the ATC TRACON RL Environment.
Defines typed data structures for aircraft state, actions, observations, and API responses.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# Aircraft & Airport Models
# ──────────────────────────────────────────────

class AircraftState(BaseModel):
    """Complete state representation of a single aircraft."""
    callsign:       str             = Field(..., description="Unique flight identifier, e.g. UAL123")
    category:       AircraftCategory
    status:         AircraftStatus
    x:              float           = Field(..., description="X position in nautical miles from airport center")
    y:              float           = Field(..., description="Y position in nautical miles from airport center")
    altitude:       float           = Field(..., description="Altitude in feet")
    heading:        float           = Field(..., ge=0, le=360, description="True heading in degrees")
    speed:          float           = Field(..., description="Ground speed in knots")
    is_emergency:   bool            = False
    assigned_gate:  Optional[str]   = None
    sequence_pos:   Optional[int]   = None     # position in landing queue (1 = next)
    fuel_state:     float           = Field(default=1.0, ge=0.0, le=1.0, description="Fuel fraction remaining")

    def distance_to(self, other: "AircraftState") -> float:
        """Euclidean distance in nautical miles."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class GateState(BaseModel):
    gate_id:   str
    occupied:  bool         = False
    aircraft:  Optional[str] = None   # callsign of occupying aircraft
    gate_type: AircraftCategory       # max category the gate can handle


class RunwayState(BaseModel):
    runway_id:          str
    active:             bool  = True
    occupied:           bool  = False
    last_departure_cat: Optional[AircraftCategory] = None


# ──────────────────────────────────────────────
# Action Models
# ──────────────────────────────────────────────

class ATCAction(BaseModel):
    """A single advisory action issued by the RL agent."""
    action_type:     ActionType
    target_callsign: str                   = Field(..., description="Aircraft this action targets")
    value:           Optional[float]       = None   # heading/speed/alt delta or new value
    secondary_target: Optional[str]        = None   # for sequence_swap or vector
    gate_id:         Optional[str]         = None   # for gate assignment
    rationale:       Optional[str]         = None   # human-readable explanation

    class Config:
        use_enum_values = True


class ActionBatch(BaseModel):
    """Batch of actions for a single step (agent may act on multiple aircraft)."""
    actions:   List[ATCAction]
    step:      int


# ──────────────────────────────────────────────
# Environment State & API Response Models
# ──────────────────────────────────────────────

class EnvironmentState(BaseModel):
    """Full observable state returned by state() and reset()."""
    task:               TaskType
    step:               int
    max_steps:          int
    aircraft:           List[AircraftState]
    gates:              List[GateState]
    runways:            List[RunwayState]
    active_conflicts:   List[Dict[str, Any]]    = Field(default_factory=list)
    separation_matrix:  Dict[str, float]        = Field(default_factory=dict)
    done:               bool                    = False
    info:               Dict[str, Any]          = Field(default_factory=dict)


class StepResult(BaseModel):
    """Result returned after each step()."""
    state:      EnvironmentState
    reward:     float           = Field(..., ge=0.0, le=1.0, description="Normalised reward [0,1]")
    done:       bool
    info:       Dict[str, Any]  = Field(default_factory=dict)
    violations: List[str]       = Field(default_factory=list)
    score:      float           = Field(..., ge=0.0, le=1.0, description="Task-specific grader score [0,1]")


class ResetRequest(BaseModel):
    task:     Optional[TaskType]  = None
    seed:     Optional[int]       = None
    options:  Dict[str, Any]      = Field(default_factory=dict)


class StepRequest(BaseModel):
    actions: List[ATCAction]


class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str = "1.0.0"
    tasks:   List[str]