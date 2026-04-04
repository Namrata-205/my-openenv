"""
Monsoon Flood Gate Control - OpenEnv Models
============================================
Simulates real-time stormwater management for Mumbai's low-lying areas
(inspired by Dharavi flood-risk zones).

Multi-agent system with partial observability:
- Each agent controls a local sub-basin (gates + pumps)
- Agents observe their own water levels + partial neighbor data
- Rewards balance flood reduction vs. energy cost
"""

from __future__ import annotations

import math
import random
from typing import Optional
from pydantic import BaseModel, Field


# ─── Action ──────────────────────────────────────────────────────────────────

class Action(BaseModel):
    """
    Control action for a single agent managing one sub-basin.

    gate_positions: list of 3 gate openings [0.0=closed … 1.0=fully open]
    pump_speeds:    list of 2 pump speeds   [0.0=off … 1.0=full speed]
    """
    gate_positions: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Fraction open for each stormwater gate (0.0–1.0)",
        min_length=3,
        max_length=3,
    )
    pump_speeds: list[float] = Field(
        default_factory=lambda: [0.0, 0.0],
        description="Pump speed fraction for each pump (0.0–1.0)",
        min_length=2,
        max_length=2,
    )

    def clamp(self) -> "Action":
        return Action(
            gate_positions=[max(0.0, min(1.0, g)) for g in self.gate_positions],
            pump_speeds=[max(0.0, min(1.0, p)) for p in self.pump_speeds],
        )


# ─── Observation ─────────────────────────────────────────────────────────────

class NeighborObservation(BaseModel):
    """Partial view of an adjacent sub-basin (limited information sharing)."""
    basin_id: int
    water_level_fraction: float   # 0.0–1.0 (coarse, noisy)
    is_flooding: bool


class Observation(BaseModel):
    """
    Full observation for a single agent's sub-basin.
    Includes own sensors + partial neighbor info.
    """
    # Own basin
    basin_id: int
    water_level_m: float          # current water depth in metres
    water_level_fraction: float   # water_level_m / capacity_m
    rainfall_mm_per_hr: float     # live rain gauge reading
    river_level_m: float          # downstream river level
    gate_positions: list[float]   # current gate states
    pump_speeds: list[float]      # current pump states
    is_flooding: bool             # water_level_m >= capacity_m

    # Neighbor partial observability
    neighbors: list[NeighborObservation]

    # Global context
    timestep: int
    elapsed_minutes: float
    storm_phase: str              # "pre_monsoon" | "peak" | "recession" | "clear"

    # Rainfall forecast (noisy 3-step ahead)
    rainfall_forecast: list[float]


# ─── Reward ──────────────────────────────────────────────────────────────────

class Reward(BaseModel):
    """Shaped reward with partial-progress signals."""
    total: float

    # Components
    flood_penalty: float          # negative: penalizes flooded volume
    drainage_bonus: float         # positive: reward for successfully draining
    energy_cost: float            # negative: penalizes excessive pump/gate use
    gate_chatter_penalty: float   # negative: penalizes rapid oscillation
    forecast_bonus: float         # positive: proactive action before rain peaks


# ─── State (internal, full observability for graders) ────────────────────────

class BasinState(BaseModel):
    basin_id: int
    water_level_m: float
    capacity_m: float
    inflow_m3_per_min: float
    outflow_m3_per_min: float
    gate_positions: list[float]
    pump_speeds: list[float]
    area_m2: float                # basin catchment area
    flood_minutes: float          # cumulative minutes in flood state
    total_energy_kwh: float


class GlobalState(BaseModel):
    basins: list[BasinState]
    timestep: int
    elapsed_minutes: float
    storm_event_id: int
    storm_phase: str
    global_rainfall_mm_per_hr: float
    river_level_m: float
    river_capacity_m: float
    done: bool
    episode_flood_volume_m3: float
    episode_energy_kwh: float
    task_id: str
    task_score: float             # running grader score [0.0–1.0]
