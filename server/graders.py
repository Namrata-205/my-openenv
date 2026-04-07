"""
ATC Reinforcement Learning Simulator — All 5 Tasks (Optimized v1.2)
Improved wake_turbulence, emergency_vectoring, and conflict_resolution.
Better multi-step signals and test compatibility.
"""

import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — WAKE TURBULENCE (Optimized)
# ═══════════════════════════════════════════════════════════════════════════════
class AircraftCategory(Enum):
    HEAVY  = "Heavy"
    MEDIUM = "Medium"
    LIGHT  = "Light"

REQUIRED_SEPARATION = {
    (AircraftCategory.HEAVY,  AircraftCategory.HEAVY):  4.0,
    (AircraftCategory.HEAVY,  AircraftCategory.MEDIUM): 5.0,
    (AircraftCategory.HEAVY,  AircraftCategory.LIGHT):  6.0,
    (AircraftCategory.MEDIUM, AircraftCategory.MEDIUM): 3.0,
    (AircraftCategory.MEDIUM, AircraftCategory.LIGHT):  4.0,
    (AircraftCategory.LIGHT,  AircraftCategory.LIGHT):  3.0,
}

@dataclass
class WakeTurbulenceEnv:
    leading_cat:   AircraftCategory
    trailing_cat:  AircraftCategory
    current_sep:   float
    elapsed_safe:  float = 0.0
    delay_sec:     float = 0.0
    violated:      bool  = False

    @property
    def required_sep(self) -> float:
        return REQUIRED_SEPARATION[(self.leading_cat, self.trailing_cat)]

    def step(self, action: str) -> Tuple[float, str]:
        reward = 0.0
        log = []
        delay_added = 0.0
        sep_before = self.current_sep
        # ❌ Penalize no meaningful change
        if abs(self.current_sep - sep_before) < 0.05:
            reward -= 2
        req = self.required_sep

        if action == "slow_down_trailing":
            delta = random.uniform(0.45, 0.95)
            self.current_sep += delta
            delay_added = random.uniform(16, 38)
            self.delay_sec += delay_added
            log.append(f"slow_down_trailing → +{delta:.2f} NM")
        elif action == "speed_up_trailing":
            delta = random.uniform(0.2, 0.6)
            self.current_sep -= delta
            log.append(f"speed_up_trailing → -{delta:.2f} NM")
        elif action == "increase_heading_gap":
            delta = random.uniform(0.65, 1.35)
            self.current_sep += delta
            delay_added = random.uniform(7, 20)
            self.delay_sec += delay_added
            log.append(f"increase_heading_gap → +{delta:.2f} NM")
        elif action == "hold":
            drift = random.uniform(-0.12, 0.12)
            self.current_sep += drift
            self.elapsed_safe += 1
            log.append(f"hold → drift {drift:+.2f} NM")

        self.current_sep = max(0.8, self.current_sep)

        # Shaping: closing gap
        gap_before = max(0.0, req - sep_before)
        gap_after  = max(0.0, req - self.current_sep)
        if gap_before > 0:
            closed = gap_before - gap_after
            if closed > 0:
                reward += 9.0 * (closed / max(0.1, gap_before))

        if self.current_sep >= req:
            diff = abs(self.current_sep - req)
            if diff <= 0.5:
                reward += 12
            else:
                reward += max(0, 10 - 5 * diff)

        if self.elapsed_safe > 0:
            reward += min(6.0, self.elapsed_safe * 1.5)

        if self.current_sep < 4.0:
            reward -= 25
            self.violated = True

        if delay_added > 35:
            reward -= 8

        if self.current_sep > req + 2.0:
            reward -= 5
            
        description = " | ".join(log)
        return round(reward, 2), description


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — GO-AROUND PREVENTION (Unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class InboundFlight:
    callsign:    str
    eta_min:     float
    fuel_lbs:    float
    priority:    int
    holding_min: float = 0.0
    landed:      bool  = False
    went_around: bool  = False


def sequence_flights(flights: List[InboundFlight], strategy: str) -> Tuple[float, List[str], dict]:
    if strategy == "fcfs":
        ordered = sorted(flights, key=lambda f: f.eta_min)
    elif strategy == "fuel_priority":
        ordered = sorted(flights, key=lambda f: f.fuel_lbs / 1000.0 - f.priority * 3.0 + f.eta_min * 0.05)
    elif strategy == "eta_optimized":
        ordered = sorted(flights, key=lambda f: f.eta_min + f.holding_min)
    elif strategy == "rl_agent":
        ordered = sorted(flights, key=lambda f: f.eta_min * 0.25 + (f.fuel_lbs / 8000.0) * 0.35 - f.priority * 5.0)
    else:
        ordered = flights

    RUNWAY_OCCUPANCY = 2.0
    runway_free_at = 0.0
    total_reward = 0.0
    go_arounds = 0
    total_holding = 0.0
    events = []

    for flight in ordered:
        earliest = max(flight.eta_min, runway_free_at)
        wait = max(0.0, earliest - flight.eta_min)
        flight.holding_min = wait
        total_holding += wait

        goes_around = (wait > 8.0) or (flight.fuel_lbs < 3000 and wait > 3.5)

        if goes_around:
            flight.went_around = True
            go_arounds += 1
            total_reward -= 20
            events.append(f"GO-AROUND {flight.callsign} (hold={wait:.1f}min)")
        else:
            flight.landed = True
            runway_free_at = earliest + RUNWAY_OCCUPANCY
            if wait <= 1.0:
                total_reward += 3

        if wait > 0:
            total_reward += -8 * math.ceil(wait)

    if go_arounds / len(flights) < 0.05:
        total_reward += 12

    stats = {"strategy": strategy, "go_arounds": go_arounds, "total_holding": round(total_holding, 2)}
    return round(total_reward, 2), events, stats


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3 — EMERGENCY VECTORING (Optimized)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Aircraft:
    callsign: str
    heading: float
    altitude: float
    speed: float
    lat: float
    lon: float


@dataclass
class EmergencyVectorEnv:
    emergency: Aircraft
    traffic: List[Aircraft]

    def _separation_nm(self, a: Aircraft, b: Aircraft) -> float:
        return math.hypot(a.lat - b.lat, a.lon - b.lon)

    def insert_emergency(self, new_heading: float, new_altitude: float, insertion_time_min: float) -> Tuple[float, List[str]]:
        reward = 0.0
        log = []

        self.emergency.heading = new_heading % 360
        self.emergency.altitude = max(1000, new_altitude)

        rad = math.radians(new_heading)
        dist = (self.emergency.speed / 60.0) * insertion_time_min
        self.emergency.lat += math.cos(rad) * dist
        self.emergency.lon += math.sin(rad) * dist

        conflicts = []
        for ac in self.traffic:
            horiz = self._separation_nm(self.emergency, ac)
            vert = abs(self.emergency.altitude - ac.altitude)
            if horiz < 3.5 and vert < 1200:
                conflicts.append(ac.callsign)

        dist_to_runway = math.hypot(self.emergency.lat, self.emergency.lon)

        if not conflicts:
            reward += 14
            if dist_to_runway < 15:
                reward += -0.5 * dist_to_runway
        else:
            reward -= 24 * len(conflicts)

        if insertion_time_min > 3.0:
            reward -= 10 * (insertion_time_min - 2.0)

        if dist_to_runway < 6.0 and not conflicts:
            reward += 9

        log.append(f"heading={new_heading:.0f}° alt={new_altitude:.0f}ft dist={dist_to_runway:.1f}NM")
        return round(reward, 2), log


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4 — CONFLICT RESOLUTION (Optimized)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class ConflictAircraftState:
    callsign: str
    x: float
    y: float
    heading_deg: float
    speed_kts: float
    target_hdg: float
    altitude: float = 5000.0


@dataclass
class ConflictAlertEnv:
    ac1: ConflictAircraftState
    ac2: ConflictAircraftState
    time_safe: float = 0.0

    def _separation(self) -> float:
        return math.hypot(self.ac1.x - self.ac2.x, self.ac1.y - self.ac2.y)

    def _move(self, ac: ConflictAircraftState, dt_min: float = 1.0):
        rad = math.radians(ac.heading_deg)
        dist = (ac.speed_kts / 60.0) * dt_min
        ac.x += math.sin(rad) * dist
        ac.y += math.cos(rad) * dist

    def step(self, action: str) -> Tuple[float, str]:
        reward = 0.0
        log = []
        sep_before = self._separation()

        if "left" in action.lower():
            self.ac1.heading_deg = (self.ac1.heading_deg - 15) % 360
            log.append("left_15")
        elif "right" in action.lower():
            self.ac1.heading_deg = (self.ac1.heading_deg + 15) % 360
            log.append("right_15")
        elif "slow" in action.lower():
            self.ac1.speed_kts = max(130, self.ac1.speed_kts - 15)
            log.append("slow_15")
        elif "speed" in action.lower():
            self.ac1.speed_kts += 15
            log.append("speed_15")

        self._move(self.ac1)
        self._move(self.ac2)

        sep_after = self._separation()

        if sep_after > sep_before + 0.3:
            reward += min(8.0, (sep_after - sep_before) * 2.5)

        if sep_after > 12.0:
            reward += 13
            self.time_safe += 1
        elif sep_after > 8.0:
            reward += 7
            self.time_safe += 1
        else:
            self.time_safe = 0

        if self.time_safe > 0:
            reward += min(6.0, self.time_safe * 1.7)

        if sep_after < 5.0:
            reward -= 20

        desc = " | ".join(log) or "no change"
        return round(reward, 2), desc


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5 — GATE ASSIGNMENT (Unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Gate:
    gate_id: str
    occupied: bool
    taxi_dist_min: float
    compatible: bool
    blocked_by: Optional[str] = None


@dataclass
class ArrivingPlane:
    callsign: str
    eta_min: float


@dataclass
class GateAssignmentEnv:
    arriving: ArrivingPlane
    gates: List[Gate]
    assigned_gate: Optional[Gate] = None

    def assign(self, gate_id: str) -> Tuple[float, List[str]]:
        gate = next((g for g in self.gates if g.gate_id == gate_id), None)
        if not gate:
            return -25.0, ["Gate not found"]

        reward = 0.0
        log = []

        if gate.occupied:
            reward -= 22
        if not gate.compatible:
            reward -= 18
        if gate.blocked_by:
            reward -= 20

        valid = not (gate.occupied or not gate.compatible or gate.blocked_by)
        if valid:
            if gate.taxi_dist_min <= 5:
                reward += 8
            if self.arriving.eta_min <= 12 and gate.taxi_dist_min <= 7:
                reward += 15

        return round(reward, 2), log

    def find_best_gate(self) -> str:
        best = max(self.gates, key=lambda g: (-10 if g.occupied else 0) + (-10 if not g.compatible else 0) + (-15 if g.blocked_by else 0) - g.taxi_dist_min)
        return best.gate_id


# Optional test runners can be added if needed