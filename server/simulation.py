"""
Monsoon Flood Gate Simulation Engine
=====================================
Physics-based simulation of Mumbai-style stormwater basins.

Hydrology model:
  dV/dt = Q_in(rain, runoff) - Q_gates(gate_pos, head) - Q_pumps(speed)
  Water level h = V / area

Storm generator produces realistic Mumbai monsoon profiles:
  - Clear day (low rain)
  - Pre-monsoon shower
  - Peak monsoon burst (erratic, high intensity)
  - Recession
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from models import Action, BasinState, GlobalState, Observation, NeighborObservation, Reward


# ─── Constants ───────────────────────────────────────────────────────────────

DT_MINUTES = 5          # simulation step = 5 minutes
GRAVITY = 9.81          # m/s²
GATE_COEFF = 0.6        # orifice discharge coefficient
PUMP_MAX_FLOW = 8.0     # m³/min per pump at full speed
GATE_AREA = 2.0         # m² effective gate area (fully open)
ENERGY_PER_PUMP_KWH = 0.05  # kWh per pump at full speed per step

# Basin configs (3 sub-basins representing Dharavi-adjacent zones)
BASIN_CONFIGS = [
    {"area_m2": 50_000, "capacity_m": 1.5, "runoff_coeff": 0.75},
    {"area_m2": 35_000, "capacity_m": 1.2, "runoff_coeff": 0.80},
    {"area_m2": 65_000, "capacity_m": 2.0, "runoff_coeff": 0.70},
]


# ─── Storm Profiles ──────────────────────────────────────────────────────────

def _generate_storm_profile(task_id: str, seed: int) -> list[float]:
    """
    Returns rainfall mm/hr for each timestep.
    task_id controls storm intensity / predictability.
    """
    rng = random.Random(seed)
    steps = 72  # 6 hours of simulation

    if task_id == "task_easy":
        # Gentle, predictable linear ramp
        base = [rng.uniform(5, 15) for _ in range(steps)]
        peak_pos = steps // 2
        profile = [b + 30 * math.exp(-0.05 * (i - peak_pos) ** 2) for i, b in enumerate(base)]

    elif task_id == "task_medium":
        # Two-burst storm with moderate noise
        peak1, peak2 = steps // 3, 2 * steps // 3
        profile = []
        for i in range(steps):
            burst1 = 60 * math.exp(-0.03 * (i - peak1) ** 2)
            burst2 = 45 * math.exp(-0.04 * (i - peak2) ** 2)
            noise = rng.uniform(-5, 5)
            profile.append(max(0, 10 + burst1 + burst2 + noise))

    else:  # task_hard
        # Erratic Mumbai-style: sudden spikes, unpredictable bursts
        profile = []
        rain = rng.uniform(15, 30)
        for i in range(steps):
            if rng.random() < 0.15:  # sudden spike
                rain = rng.uniform(80, 200)
            elif rng.random() < 0.2:  # drop
                rain *= rng.uniform(0.3, 0.6)
            else:
                rain += rng.uniform(-10, 15)
            rain = max(0, min(250, rain))
            profile.append(rain)

    return profile


def _storm_phase(step: int, total_steps: int, rainfall: float) -> str:
    frac = step / total_steps
    if frac < 0.1:
        return "pre_monsoon"
    if rainfall > 50:
        return "peak"
    if frac > 0.75:
        return "recession"
    return "clear"


# ─── Hydraulics ──────────────────────────────────────────────────────────────

def _gate_outflow(gate_positions: list[float], water_level_m: float, river_level_m: float) -> float:
    """
    Orifice equation: Q = Cd * A * sqrt(2g * deltaH)
    Only flows if water_level > river_level (gravity-driven).
    """
    delta_h = max(0.0, water_level_m - river_level_m)
    total = 0.0
    for g in gate_positions:
        if g > 0.01:
            effective_area = g * GATE_AREA
            flow_m3s = GATE_COEFF * effective_area * math.sqrt(2 * GRAVITY * delta_h + 1e-9)
            total += flow_m3s * 60  # convert to m³/min
    return total


def _pump_outflow(pump_speeds: list[float]) -> float:
    return sum(max(0.0, p) * PUMP_MAX_FLOW for p in pump_speeds)


def _rainfall_inflow(rainfall_mm_hr: float, area_m2: float, runoff_coeff: float) -> float:
    """Convert rainfall to volumetric inflow m³/min."""
    mm_per_min = rainfall_mm_hr / 60.0
    m_per_min = mm_per_min / 1000.0
    return m_per_min * area_m2 * runoff_coeff


# ─── Simulation Engine ───────────────────────────────────────────────────────

class FloodSimulation:
    def __init__(self, task_id: str = "task_easy", storm_seed: int = 42):
        self.task_id = task_id
        self.storm_seed = storm_seed
        self._storm_profile: list[float] = []
        self._basins: list[BasinState] = []
        self._timestep: int = 0
        self._river_level_m: float = 1.0
        self._river_capacity_m: float = 3.5
        self._episode_flood_volume_m3: float = 0.0
        self._episode_energy_kwh: float = 0.0
        self._prev_gates: list[list[float]] = []  # for chatter detection
        self._done: bool = False

    def reset(self) -> GlobalState:
        rng = random.Random(self.storm_seed)
        self._storm_profile = _generate_storm_profile(self.task_id, self.storm_seed)
        self._timestep = 0
        self._episode_flood_volume_m3 = 0.0
        self._episode_energy_kwh = 0.0
        self._done = False
        self._river_level_m = rng.uniform(0.5, 1.2)

        self._basins = []
        for i, cfg in enumerate(BASIN_CONFIGS):
            init_water = rng.uniform(0.1, 0.3) * cfg["capacity_m"]
            self._basins.append(BasinState(
                basin_id=i,
                water_level_m=init_water,
                capacity_m=cfg["capacity_m"],
                inflow_m3_per_min=0.0,
                outflow_m3_per_min=0.0,
                gate_positions=[0.0, 0.0, 0.0],
                pump_speeds=[0.0, 0.0],
                area_m2=cfg["area_m2"],
                flood_minutes=0.0,
                total_energy_kwh=0.0,
            ))
        self._prev_gates = [[0.0, 0.0, 0.0] for _ in self._basins]
        return self._build_state()

    def step(self, actions: list[Action]) -> tuple[GlobalState, list[Reward], bool, dict]:
        """
        Advance simulation by DT_MINUTES.
        actions: one Action per basin (len == 3).
        Returns: (global_state, rewards, done, info)
        """
        assert len(actions) == len(self._basins), "Need one action per basin"
        actions = [a.clamp() for a in actions]

        rainfall = self._storm_profile[min(self._timestep, len(self._storm_profile) - 1)]

        # River level rises with heavy rain (simplified)
        rain_pressure = min(1.0, rainfall / 150.0)
        self._river_level_m = min(
            self._river_capacity_m,
            self._river_level_m + 0.02 * rain_pressure - 0.005
        )
        self._river_level_m = max(0.3, self._river_level_m)

        rewards: list[Reward] = []
        step_flood_vol = 0.0
        step_energy = 0.0

        for i, (basin, action) in enumerate(zip(self._basins, actions)):
            cfg = BASIN_CONFIGS[i]
            prev_gates = self._prev_gates[i]

            # Update basin state
            basin.gate_positions = action.gate_positions
            basin.pump_speeds = action.pump_speeds

            inflow = _rainfall_inflow(rainfall, basin.area_m2, cfg["runoff_coeff"])
            gate_out = _gate_outflow(action.gate_positions, basin.water_level_m, self._river_level_m)
            pump_out = _pump_outflow(action.pump_speeds)
            total_out = gate_out + pump_out

            basin.inflow_m3_per_min = inflow
            basin.outflow_m3_per_min = total_out

            # dV = (inflow - outflow) * dt
            delta_vol = (inflow - total_out) * DT_MINUTES
            new_vol = max(0, basin.water_level_m * basin.area_m2 + delta_vol)
            basin.water_level_m = new_vol / basin.area_m2

            # Flood tracking
            overflow_depth = max(0, basin.water_level_m - basin.capacity_m)
            basin.water_level_m = min(basin.water_level_m, basin.capacity_m * 1.1)  # allow slight overflow
            if overflow_depth > 0:
                basin.flood_minutes += DT_MINUTES
                flood_vol = overflow_depth * basin.area_m2
                step_flood_vol += flood_vol
                self._episode_flood_volume_m3 += flood_vol

            # Energy tracking
            energy = sum(p for p in action.pump_speeds) * ENERGY_PER_PUMP_KWH
            basin.total_energy_kwh += energy
            step_energy += energy
            self._episode_energy_kwh += energy

            # Gate chatter detection
            chatter = sum(abs(g - pg) for g, pg in zip(action.gate_positions, prev_gates))
            self._prev_gates[i] = list(action.gate_positions)

            # ─── Shaped Reward ───────────────────────────────────────────
            is_flooding = basin.water_level_m >= basin.capacity_m
            water_frac = basin.water_level_m / basin.capacity_m

            # Flood penalty (non-linear: much worse when nearly full)
            if is_flooding:
                flood_pen = -2.0
            elif water_frac > 0.85:
                flood_pen = -0.5 * (water_frac - 0.85) / 0.15
            else:
                flood_pen = 0.0

            # Drainage bonus: reward for reducing water level
            drainage = max(0, total_out - inflow) / max(1, inflow + 1)
            drain_bonus = min(0.5, drainage * 0.2) if water_frac > 0.5 else 0.0

            # Energy cost
            e_cost = -energy * 2.0

            # Gate chatter penalty
            chatter_pen = -0.05 * chatter

            # Forecast bonus: open gates preemptively before peak
            if self._timestep + 3 < len(self._storm_profile):
                future_rain = self._storm_profile[self._timestep + 3]
                if future_rain > rainfall * 1.3 and water_frac < 0.6:
                    # proactively draining before incoming spike
                    total_gate = sum(action.gate_positions) / 3
                    f_bonus = 0.15 * total_gate
                else:
                    f_bonus = 0.0
            else:
                f_bonus = 0.0

            total_reward = flood_pen + drain_bonus + e_cost + chatter_pen + f_bonus

            rewards.append(Reward(
                total=round(total_reward, 4),
                flood_penalty=round(flood_pen, 4),
                drainage_bonus=round(drain_bonus, 4),
                energy_cost=round(e_cost, 4),
                gate_chatter_penalty=round(chatter_pen, 4),
                forecast_bonus=round(f_bonus, 4),
            ))

        self._timestep += 1
        self._done = self._timestep >= len(self._storm_profile)

        state = self._build_state(step_flood_vol=step_flood_vol)
        info = {
            "step_flood_volume_m3": step_flood_vol,
            "step_energy_kwh": step_energy,
            "rainfall_mm_hr": rainfall,
            "river_level_m": self._river_level_m,
        }
        return state, rewards, self._done, info

    def state(self) -> GlobalState:
        return self._build_state()

    def _build_state(self, step_flood_vol: float = 0.0) -> GlobalState:
        rainfall = self._storm_profile[min(self._timestep, len(self._storm_profile) - 1)] \
            if self._storm_profile else 0.0
        phase = _storm_phase(self._timestep, len(self._storm_profile) or 1, rainfall)

        # Running grader score (see graders.py)
        max_possible_flood = sum(
            cfg["capacity_m"] * cfg["area_m2"] * len(self._storm_profile) * DT_MINUTES
            for cfg in BASIN_CONFIGS
        )
        flood_ratio = min(1.0, self._episode_flood_volume_m3 / max(1, max_possible_flood * 0.01))
        task_score = round(max(0.0, 1.0 - flood_ratio), 4)

        return GlobalState(
            basins=[b.model_copy() for b in self._basins],
            timestep=self._timestep,
            elapsed_minutes=self._timestep * DT_MINUTES,
            storm_event_id=self.storm_seed,
            storm_phase=phase,
            global_rainfall_mm_per_hr=rainfall,
            river_level_m=self._river_level_m,
            river_capacity_m=self._river_capacity_m,
            done=self._done,
            episode_flood_volume_m3=self._episode_flood_volume_m3,
            episode_energy_kwh=self._episode_energy_kwh,
            task_id=self.task_id,
            task_score=task_score,
        )

    def get_observation(self, basin_idx: int) -> Observation:
        """Return partial observation for a single agent."""
        basin = self._basins[basin_idx]
        rainfall = self._storm_profile[min(self._timestep, len(self._storm_profile) - 1)] \
            if self._storm_profile else 0.0

        # Noisy rainfall forecast (agents get imperfect info)
        rng = random.Random(self._timestep * 100 + basin_idx)
        forecast = []
        for offset in range(1, 4):
            future_idx = min(self._timestep + offset, len(self._storm_profile) - 1)
            future_rain = self._storm_profile[future_idx] if self._storm_profile else 0.0
            noisy = future_rain * rng.uniform(0.7, 1.3)  # ±30% noise
            forecast.append(round(noisy, 1))

        # Partial neighbor info (limited data sharing)
        neighbors = []
        for j, other in enumerate(self._basins):
            if j == basin_idx:
                continue
            wf = other.water_level_m / other.capacity_m
            # Noise on neighbor observation
            noisy_wf = min(1.0, max(0.0, wf + rng.uniform(-0.1, 0.1)))
            neighbors.append(NeighborObservation(
                basin_id=j,
                water_level_fraction=round(noisy_wf, 2),
                is_flooding=other.water_level_m >= other.capacity_m,
            ))

        phase = _storm_phase(self._timestep, len(self._storm_profile) or 1, rainfall)

        return Observation(
            basin_id=basin_idx,
            water_level_m=round(basin.water_level_m, 3),
            water_level_fraction=round(basin.water_level_m / basin.capacity_m, 3),
            rainfall_mm_per_hr=round(rainfall, 1),
            river_level_m=round(self._river_level_m, 3),
            gate_positions=basin.gate_positions,
            pump_speeds=basin.pump_speeds,
            is_flooding=basin.water_level_m >= basin.capacity_m,
            neighbors=neighbors,
            timestep=self._timestep,
            elapsed_minutes=self._timestep * DT_MINUTES,
            storm_phase=phase,
            rainfall_forecast=forecast,
        )
