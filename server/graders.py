"""
graders.py — Task graders for the ATC TRACON RL Environment.

Performance fixes in this revision:
  - WakeTurbulenceEnv: perfect-spacing window widened (req → req+1.0 NM)
    so agents can actually reach the +12 reward band during normal operation.
  - EmergencyVectorEnv: `inserted` flag added; once a clean insertion is
    confirmed the env freezes so the terminal condition in environment.py
    fires reliably on the next step check.
  - ConflictAlertEnv: `tick()` advances both aircraft along their current
    headings each step so separation evolves naturally; heading/speed changes
    now persistently alter the track rather than being one-shot x-bumps.
  - Gate.blocked: only True when blocked_by is set AND gate is not already
    occupied — prevents double-penalising occupied+blocked gates.
  - GateAssignmentEnv.assign(): queue always advances (valid or not) so the
    queue drains deterministically and terminal fires correctly.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — WAKE TURBULENCE
# ═══════════════════════════════════════════════════════════════════════════════

class AircraftCategory(Enum):
    HEAVY  = "Heavy"
    MEDIUM = "Medium"
    LIGHT  = "Light"

REQUIRED_SEPARATION: Dict[Tuple[AircraftCategory, AircraftCategory], float] = {
    (AircraftCategory.HEAVY,  AircraftCategory.HEAVY):  4.0,
    (AircraftCategory.HEAVY,  AircraftCategory.MEDIUM): 5.0,
    (AircraftCategory.HEAVY,  AircraftCategory.LIGHT):  6.0,
    (AircraftCategory.MEDIUM, AircraftCategory.MEDIUM): 3.0,
    (AircraftCategory.MEDIUM, AircraftCategory.LIGHT):  4.0,
    (AircraftCategory.LIGHT,  AircraftCategory.LIGHT):  3.0,
}


@dataclass
class WakeTurbulenceEnv:
    leading_cat:    AircraftCategory
    trailing_cat:   AircraftCategory
    current_sep:    float = 4.5
    delay_sec:      float = 0.0
    leading_speed:  float = 160.0
    trailing_speed: float = 150.0

    @property
    def required_sep(self) -> float:
        return REQUIRED_SEPARATION[(self.leading_cat, self.trailing_cat)]

    @property
    def violated(self) -> bool:
        return self.current_sep < self.required_sep

    def step(self, action: str) -> Tuple[float, str]:
        """Return (reward, log_string)."""
        reward    = 0.0
        req       = self.required_sep
        log_parts: List[str] = [f"action={action}"]

        if action == "slow_down_trailing":
            self.current_sep += random.uniform(0.3, 0.6)
            self.delay_sec   += random.uniform(10, 25)
            log_parts.append("trailing slowed → sep increased")

        elif action == "speed_up_trailing":
            self.current_sep -= random.uniform(0.1, 0.4)
            if self.current_sep < req * 1.2:
                reward -= 5
                log_parts.append("WARNING: approaching minimum separation")

        elif action == "increase_heading_gap":
            self.current_sep += random.uniform(0.5, 1.0)
            self.delay_sec   += random.uniform(5, 15)
            log_parts.append("heading gap widened → sep increased")

        elif action == "hold":
            self.current_sep += random.uniform(-0.05, 0.05)
            log_parts.append("holding — sep drifting")

        # FIX: perfect-spacing window widened from (req, req+0.5] to (req, req+1.0]
        # The old 0.5 NM window was too narrow for hold drift (±0.05/step) to
        # land in reliably, so the +12 reward band was nearly unreachable.
        if self.current_sep >= req and self.current_sep <= req + 1.0:
            reward += 12
            log_parts.append("PERFECT spacing")
        elif self.current_sep >= req:
            reward += 6
            log_parts.append("safe separation maintained")
        else:
            reward -= 20
            log_parts.append("LOSS OF SEPARATION")

        if self.delay_sec > 40:
            reward -= 5
            log_parts.append("delay penalty applied")

        return round(reward, 2), " | ".join(log_parts)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — GO-AROUND PREVENTION
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


def sequence_flights(
    flights: List[InboundFlight],
    strategy: str,
) -> Tuple[float, List[str], Dict[str, Any]]:
    """
    Sequence the given flight list using *strategy*.

    Returns (total_reward, events, stats).
    stats keys: "landed", "go_arounds"
    """
    runway_free_at = 0.0
    total_reward   = 0.0
    events: List[str] = []

    if strategy == "fcfs":
        flights.sort(key=lambda f: f.eta_min)
    elif strategy == "fuel_priority":
        # Lowest fuel first — they have least buffer and must land soonest
        flights.sort(key=lambda f: f.fuel_lbs)
    elif strategy == "eta_optimized":
        # Sort by ETA but bump low-fuel planes forward if they'd go-around otherwise
        flights.sort(key=lambda f: (f.eta_min if f.fuel_lbs >= 3000 else f.eta_min - 20))
    # "rl_agent" → keep existing order

    for f in flights:
        landing_time  = max(f.eta_min, runway_free_at)
        wait          = landing_time - f.eta_min
        f.holding_min = wait

        if wait > 8 or (f.fuel_lbs < 3000 and wait > 4):
            f.went_around  = True
            total_reward  -= 20
            events.append(f"{f.callsign}: GO-AROUND after {wait:.1f} min wait")
        else:
            f.landed       = True
            runway_free_at = landing_time + 2
            if wait < 1:
                total_reward += 5
                events.append(f"{f.callsign}: landed immediately (+5)")
            else:
                penalty       = wait * 2
                total_reward -= penalty
                events.append(
                    f"{f.callsign}: landed after {wait:.1f} min hold (-{penalty:.1f})")

    landed_count    = sum(1 for f in flights if f.landed)
    go_around_count = sum(1 for f in flights if f.went_around)
    stats = {"landed": landed_count, "go_arounds": go_around_count}
    return round(total_reward, 2), events, stats


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3 — EMERGENCY VECTORING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Aircraft:
    callsign:   str
    heading:    float
    altitude:   float
    speed:      float
    lat:        float
    lon:        float
    fuel_state: float = 1.0


@dataclass
class EmergencyVectorEnv:
    emergency: Aircraft
    traffic:   List[Aircraft]
    # FIX: once a clean, fast insertion is achieved, freeze the env so the
    # terminal flag fires on the very next environment.py done-check rather
    # than looping forever with done=False.
    inserted:  bool = False

    def insert_emergency(
        self,
        heading: float,
        altitude: float,
        insert_time: float = 1.0,
    ) -> Tuple[float, List[str]]:
        """
        Attempt to insert the emergency aircraft at *heading* / *altitude*.
        Returns (reward, log_lines).
        """
        if self.inserted:
            # Already inserted — return stable success so environment terminates
            return 18.0, ["Already inserted — episode should terminate"]

        reward: float  = 0.0
        log: List[str] = []

        for ac in self.traffic:
            lateral  = math.hypot(self.emergency.lat - ac.lat,
                                  self.emergency.lon - ac.lon)
            vertical = abs(altitude - ac.altitude)

            if lateral < 3.0 and vertical < 1000:
                reward -= 22
                log.append(
                    f"SAFETY VIOLATION: {self.emergency.callsign} vs {ac.callsign} "
                    f"lat={lateral:.2f}NM vert={vertical:.0f}ft")
            elif lateral < 5.0:
                reward -= 8
                log.append(
                    f"CAUTION: close proximity to {ac.callsign} ({lateral:.2f} NM)")
            else:
                reward += 5
                log.append(f"CLEAR of {ac.callsign} ({lateral:.2f} NM)")

        runway_heading  = 180.0
        diff            = abs(heading - runway_heading) % 360
        diff            = min(diff, 360 - diff)
        alignment_bonus = max(0.0, 10.0 - diff / 10.0)
        reward         += alignment_bonus
        log.append(f"heading={heading:.1f}° alignment_bonus={alignment_bonus:.1f}")

        insert_time = max(0.5, min(5.0, insert_time))
        time_bonus  = max(0.0, (2.0 - insert_time) * 2.0)
        reward     += time_bonus
        log.append(f"insert_time={insert_time:.1f}min time_bonus={time_bonus:.1f}")

        urgency = 1.0 + (1.0 - self.emergency.fuel_state)
        reward *= urgency
        log.append(f"fuel_state={self.emergency.fuel_state:.2f} urgency×{urgency:.2f}")

        self.emergency.heading  = heading
        self.emergency.altitude = altitude

        conflict_free = not any("SAFETY VIOLATION" in l for l in log)
        if conflict_free:
            self.inserted = True
            log.append("INSERTION CONFIRMED — terminal")

        return round(reward, 2), log


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4 — CONFLICT RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConflictAircraftState:
    callsign:    str
    x:           float
    y:           float
    altitude:    float
    speed_kts:   float
    heading_deg: float = 90.0
    target_hdg:  float = 90.0

    def _hdg_rad(self) -> float:
        return math.radians(self.heading_deg)

    def tick(self, dt_min: float = 1.0):
        """Advance position by dt_min minutes at current heading and speed."""
        nm_per_min  = self.speed_kts / 60.0
        self.x     += nm_per_min * dt_min * math.sin(self._hdg_rad())
        self.y     += nm_per_min * dt_min * math.cos(self._hdg_rad())


@dataclass
class ConflictAlertEnv:
    ac1: ConflictAircraftState
    ac2: ConflictAircraftState

    def _separation(self) -> float:
        return math.hypot(self.ac1.x - self.ac2.x, self.ac1.y - self.ac2.y)

    def step(self, action_str: str) -> Tuple[float, str, float, float]:
        """
        Apply action, score current (pre-tick) separation, then tick both aircraft.

        Returns (reward, description, pre_tick_horiz_sep, pre_tick_vert_sep).
        Scoring on pre-tick positions means step 1 always evaluates spawn positions
        (guaranteed safe) rather than post-movement positions.
        """
        reward      = 0.0
        description = action_str

        if action_str in ("altitude_up", "altitude_change"):
            self.ac1.altitude += 1000
            description = "AC1 climb 1000 ft"

        elif action_str == "altitude_down":
            self.ac1.altitude -= 1000
            description = "AC1 descend 1000 ft"

        elif action_str == "left_10":
            self.ac1.heading_deg = (self.ac1.heading_deg - 10) % 360
            description          = "AC1 turn left 10°"

        elif action_str in ("right_10", "heading_change"):
            self.ac1.heading_deg = (self.ac1.heading_deg + 10) % 360
            description          = "AC1 turn right 10°"

        elif action_str == "slow_10":
            self.ac1.speed_kts = max(150, self.ac1.speed_kts - 10)
            description        = "AC1 slow 10 kts"

        elif action_str == "speed_10":
            self.ac1.speed_kts = min(350, self.ac1.speed_kts + 10)
            description        = "AC1 accelerate 10 kts"

        elif action_str == "ac2_left_10":
            self.ac2.heading_deg = (self.ac2.heading_deg - 10) % 360
            description          = "AC2 turn left 10°"

        elif action_str == "ac2_right_10":
            self.ac2.heading_deg = (self.ac2.heading_deg + 10) % 360
            description          = "AC2 turn right 10°"

        elif action_str == "ac2_slow_10":
            self.ac2.speed_kts = max(150, self.ac2.speed_kts - 10)
            description        = "AC2 slow 10 kts"

        elif action_str == "ac2_speed_10":
            self.ac2.speed_kts = min(350, self.ac2.speed_kts + 10)
            description        = "AC2 accelerate 10 kts"

        # Score on PRE-tick positions (current separation before movement)
        horiz_sep = self._separation()
        vert_sep  = abs(self.ac1.altitude - self.ac2.altitude)

        if horiz_sep > 5.0:
            reward += 10
        if vert_sep >= 1000:
            reward += 10
        if horiz_sep > 5.0 and vert_sep >= 1000:
            reward += 15   # hybrid bonus

        if horiz_sep < 3.0 and vert_sep < 1000:
            reward -= 25

        # Tick AFTER scoring — positions advance ready for the next step
        self.ac1.tick(dt_min=1.0)
        self.ac2.tick(dt_min=1.0)

        description += (
            f" → sep={horiz_sep:.2f}NM vert={vert_sep:.0f}ft reward={reward:.1f}")
        return round(reward, 2), description, horiz_sep, vert_sep


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5 — GATE ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Gate:
    gate_id:       str
    occupied:      bool
    taxi_dist_min: float
    compatible:    bool
    blocked_by:    Optional[str] = None

    @property
    def blocked(self) -> bool:
        # FIX: occupied gates are NOT also flagged as blocked — the occupied
        # penalty already covers them. Double-penalising skewed agent gate choice.
        return (not self.occupied) and (self.blocked_by is not None)


@dataclass
class ArrivingPlane:
    callsign: str
    eta_min:  float


@dataclass
class GateAssignmentEnv:
    arriving:    ArrivingPlane
    gates:       List[Gate]
    queue:       List[ArrivingPlane]  = field(default_factory=list)
    assignments: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def queue_empty(self) -> bool:
        return len(self.queue) == 0

    @property
    def assigned_gate(self) -> Optional[str]:
        for rec in reversed(self.assignments):
            if rec.get("valid"):
                return rec["gate_id"]
        return None

    def find_best_gate(self) -> str:
        """Unoccupied, unblocked, compatible gate with shortest taxi."""
        # Tier 1: free, unblocked, compatible
        candidates = [
            g for g in self.gates
            if not g.occupied and not g.blocked and g.compatible
        ]
        if candidates:
            return min(candidates, key=lambda g: g.taxi_dist_min).gate_id
        # Tier 2: free, unblocked (any compatibility)
        free = [g for g in self.gates if not g.occupied and not g.blocked]
        if free:
            return min(free, key=lambda g: g.taxi_dist_min).gate_id
        # Tier 3: free but blocked (still better than occupied)
        free_blocked = [g for g in self.gates if not g.occupied]
        if free_blocked:
            return min(free_blocked, key=lambda g: g.taxi_dist_min).gate_id
        # Tier 4: all gates occupied — pick shortest taxi to minimise penalty
        return min(self.gates, key=lambda g: g.taxi_dist_min).gate_id

    def assign(self, gate_id: str) -> Tuple[float, List[str]]:
        """
        Assign *arriving* to *gate_id*.  Returns (reward, log_lines).

        FIX: queue advances after every call (valid or invalid) so the queue
        always drains and queue_empty becomes True at the right time.
        """
        gate   = next((g for g in self.gates if g.gate_id == gate_id), None)
        log:   List[str] = []
        reward = 0.0

        if gate is None:
            log.append(f"!!! Unknown gate {gate_id}")
            self.assignments.append(
                {"callsign": self.arriving.callsign,
                 "gate_id": gate_id, "valid": False, "reward": 0.0})
        else:
            if gate.occupied:
                reward -= 30
                log.append(f"!!! Gate {gate_id} is OCCUPIED")
            if not gate.compatible:
                reward -= 20
                log.append(f"!!! Gate {gate_id} is INCOMPATIBLE with aircraft type")
            if gate.blocked:
                reward -= 20
                log.append(f"!!! Gate {gate_id} is BLOCKED by {gate.blocked_by}")

            taxi_bonus = max(0.0, 10.0 - gate.taxi_dist_min)
            reward    += taxi_bonus
            log.append(
                f"{self.arriving.callsign} → {gate_id} "
                f"taxi={gate.taxi_dist_min:.1f}min bonus={taxi_bonus:.1f}")

            valid = not gate.occupied and not gate.blocked
            if valid:
                gate.occupied   = True
                gate.blocked_by = self.arriving.callsign

            self.assignments.append(
                {"callsign": self.arriving.callsign,
                 "gate_id":  gate_id,
                 "valid":    valid,
                 "reward":   round(reward, 2)})

        # FIX: always advance queue regardless of assignment validity
        if self.queue:
            self.arriving = self.queue.pop(0)
            log.append(
                f"Next arriving: {self.arriving.callsign} ETA {self.arriving.eta_min} min")
        else:
            log.append("Queue exhausted — all planes processed")

        return round(reward, 2), log