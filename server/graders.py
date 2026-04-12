"""
graders.py — Task graders for the ATC TRACON RL Environment.

FIXED: All 5 tasks now return normalized scores strictly between 0 and 1 (0.01 to 0.99).
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
    _score_override: float = None  # For testing

    @property
    def required_sep(self) -> float:
        return REQUIRED_SEPARATION[(self.leading_cat, self.trailing_cat)]

    @property
    def violated(self) -> bool:
        return self.current_sep < self.required_sep

    def _calculate_normalized_score(self) -> float:
        """
        Calculate normalized score between 0.01 and 0.99 (exclusive bounds).
        Based on separation ratio and delay penalty.
        """
        req = self.required_sep
        sep_ratio = min(2.0, self.current_sep / req)  # Cap at 2x required
        
        # Base score from separation (0.01 to 0.85 range)
        if self.current_sep >= req + 1.0:
            # Excellent separation (20-50% above required)
            base_score = 0.75 + min(0.20, (self.current_sep - req - 1.0) / 10.0)
        elif self.current_sep >= req:
            # Good separation (0-20% above required)
            base_score = 0.55 + ((self.current_sep - req) / req) * 0.20
        elif self.current_sep >= req * 0.7:
            # Marginal separation (30-100% of required)
            base_score = 0.25 + ((self.current_sep / req) - 0.7) / 0.3 * 0.30
        else:
            # Poor separation (below 70% of required)
            base_score = 0.05 + (self.current_sep / req) / 0.7 * 0.20
        
        # Delay penalty (0 to 0.30)
        delay_penalty = min(0.30, self.delay_sec / 200.0)
        
        # Final score clamped to [0.01, 0.99]
        final_score = max(0.01, min(0.99, base_score - delay_penalty))
        return round(final_score, 4)

    def step(self, action: str) -> Tuple[float, str, float]:
        """Return (reward, log_string, normalized_score)."""
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

        # Reward calculation
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
        
        # Get normalized score (always between 0.01 and 0.99)
        normalized_score = self._calculate_normalized_score()
        log_parts.append(f"score={normalized_score:.4f}")

        return round(reward, 2), " | ".join(log_parts), normalized_score


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


def _calculate_ga_score(flights: List[InboundFlight]) -> float:
    """
    Calculate normalized score for go-around prevention.
    Returns value between 0.01 and 0.99.
    """
    if not flights:
        return 0.50
    
    go_around_count = sum(1 for f in flights if f.went_around)
    ga_rate = go_around_count / len(flights)
    
    avg_holding = sum(f.holding_min for f in flights) / len(flights)
    
    # Score components (0 to 1 range)
    # Perfect ga_rate = 0 gives 1.0, but we cap at 0.99
    ga_score = max(0.0, 1.0 - ga_rate * 4.0)  # 25% go-around = 0 score
    
    # Perfect holding = 0 gives 1.0, 15+ minutes gives 0
    holding_score = max(0.0, 1.0 - avg_holding / 15.0)
    
    # Weighted combination (70% go-around, 30% holding)
    raw_score = (ga_score * 0.7) + (holding_score * 0.3)
    
    # Scale to [0.01, 0.99] range
    final_score = 0.01 + (raw_score * 0.98)
    return round(final_score, 4)


def sequence_flights(
    flights: List[InboundFlight],
    strategy: str,
) -> Tuple[float, List[str], Dict[str, Any]]:
    """
    Sequence the given flight list using *strategy*.
    Returns (total_reward, events, stats).
    """
    # ── Sort by strategy ──────────────────────────────────────────────────────
    if strategy == "fcfs":
        flights.sort(key=lambda f: f.eta_min)
    elif strategy == "fuel_priority":
        def fuel_key(f: InboundFlight) -> Tuple[int, float]:
            bucket = 0 if f.fuel_lbs < 3000 else 1
            return (bucket, f.eta_min)
        flights.sort(key=fuel_key)
    elif strategy == "eta_optimized":
        flights.sort(key=lambda f: f.eta_min)
    elif strategy == "rl_agent":
        MAX_FUEL = 10_000.0
        flights.sort(key=lambda f: (
            f.eta_min * 0.4
            + (f.fuel_lbs / MAX_FUEL) * 0.3
            - f.priority * 3.0
        ))

    # ── Simulate runway operations ────────────────────────────────────────────
    RUNWAY_OCC_MIN = 2.0
    runway_free_at = 0.0
    total_reward   = 0.0
    events: List[str] = []

    for f in flights:
        landing_time  = max(f.eta_min, runway_free_at)
        wait          = landing_time - f.eta_min
        f.holding_min = wait

        goes_around = wait > 8.0 or (f.fuel_lbs < 3_000 and wait > 4.0)

        if goes_around:
            f.went_around  = True
            total_reward  -= 20
            events.append(
                f"{f.callsign}: GO-AROUND  wait={wait:.1f}min  fuel={f.fuel_lbs:.0f}lbs  → -20"
            )
        else:
            f.landed       = True
            runway_free_at = landing_time + RUNWAY_OCC_MIN
            if wait <= 1.0:
                total_reward += 3
                events.append(f"{f.callsign}: ON-TIME  wait={wait:.1f}min  → +3")
            else:
                hold_penalty  = -8 * math.ceil(wait)
                total_reward += hold_penalty
                events.append(
                    f"{f.callsign}: LANDED  wait={wait:.1f}min  "
                    f"hold_penalty={hold_penalty:.0f}"
                )

    go_around_count = sum(1 for f in flights if f.went_around)
    landed_count    = sum(1 for f in flights if f.landed)
    ga_rate         = go_around_count / len(flights) if flights else 0.0

    if ga_rate < 0.05:
        total_reward += 12
        events.append(f"BONUS: go-around rate {ga_rate*100:.1f}% < 5%  → +12")
    else:
        events.append(f"INFO: go-around rate {ga_rate*100:.1f}%  (need <5% for bonus)")
    
    # Calculate normalized score (always between 0.01 and 0.99)
    normalized_score = _calculate_ga_score(flights)
    events.append(f"score={normalized_score:.4f}")

    stats = {
        "strategy":      strategy,
        "landed":        landed_count,
        "go_arounds":    go_around_count,
        "ga_rate_pct":   round(ga_rate * 100, 1),
        "total_reward":  round(total_reward, 2),
        "total_holding": round(sum(f.holding_min for f in flights), 2),
        "normalized_score": normalized_score,
    }
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
    inserted:  bool = False
    
    def _calculate_normalized_score(self) -> float:
        """Calculate normalized score between 0.01 and 0.99."""
        if self.inserted:
            # Successfully inserted - high score based on fuel and time
            fuel_factor = self.emergency.fuel_state
            raw_score = 0.70 + (fuel_factor * 0.29)
        else:
            # Not yet inserted - lower score based on fuel remaining
            fuel_factor = max(0.1, self.emergency.fuel_state)
            raw_score = 0.05 + (fuel_factor * 0.25)
        
        # Ensure strict bounds
        final_score = max(0.01, min(0.99, raw_score))
        return round(final_score, 4)

    def insert_emergency(
        self,
        heading:     float,
        altitude:    float,
        insert_time: float = 1.0,
    ) -> Tuple[float, List[str], float]:
        """
        Vector the emergency aircraft.
        Returns (reward, log_lines, normalized_score).
        """
        if self.inserted:
            score = self._calculate_normalized_score()
            return 18.0, ["Already inserted — episode should terminate"], score

        reward: float = 0.0
        log: List[str] = []
        safety_violations = 0
        min_separation = float('inf')

        for ac in self.traffic:
            lateral = math.hypot(self.emergency.lat - ac.lat,
                                  self.emergency.lon - ac.lon)
            vertical = abs(altitude - ac.altitude)
            min_separation = min(min_separation, lateral)

            if lateral < 3.0 and vertical < 1000:
                reward -= 22
                safety_violations += 1
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

        runway_heading = 180.0
        diff = abs(heading - runway_heading) % 360
        diff = min(diff, 360 - diff)
        alignment_bonus = max(0.0, 10.0 - diff / 10.0)
        reward += alignment_bonus
        log.append(f"heading={heading:.1f}° alignment_bonus={alignment_bonus:.1f}")

        insert_time = max(0.5, min(5.0, insert_time))
        time_bonus = max(0.0, (2.0 - insert_time) * 2.0)
        reward += time_bonus
        log.append(f"insert_time={insert_time:.1f}min time_bonus={time_bonus:.1f}")

        urgency = 1.0 + (1.0 - self.emergency.fuel_state)
        reward *= urgency
        log.append(f"fuel_state={self.emergency.fuel_state:.2f} urgency×{urgency:.2f}")

        self.emergency.heading = heading
        self.emergency.altitude = altitude

        conflict_free = safety_violations == 0
        if conflict_free:
            self.inserted = True
            log.append("INSERTION CONFIRMED — terminal")
        
        normalized_score = self._calculate_normalized_score()
        log.append(f"score={normalized_score:.4f}")

        return round(reward, 2), log, normalized_score


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4 — CONFLICT ALERT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConflictAircraftState:
    callsign:    str
    x:           float
    y:           float
    speed_kts:   float
    heading_deg: float = 90.0
    target_hdg:  float = 90.0
    altitude:    float = 8000.0

    def tick(self, dt_min: float = 1.0):
        nm = self.speed_kts / 60.0 * dt_min
        rad = math.radians(self.heading_deg)
        self.x += nm * math.sin(rad)
        self.y += nm * math.cos(rad)


@dataclass
class ConflictAlertEnv:
    ac1: ConflictAircraftState
    ac2: ConflictAircraftState
    safe_seconds: int = 0

    def _separation(self) -> float:
        return math.hypot(self.ac1.x - self.ac2.x, self.ac1.y - self.ac2.y)

    @staticmethod
    def _heading_deviation(ac: ConflictAircraftState) -> float:
        diff = abs(ac.heading_deg - ac.target_hdg) % 360
        return min(diff, 360 - diff)
    
    def _calculate_normalized_score(self) -> float:
        """Calculate normalized score between 0.01 and 0.99."""
        sep = self._separation()
        dev1 = self._heading_deviation(self.ac1)
        dev2 = self._heading_deviation(self.ac2)
        
        # Separation score (0 to 1 range)
        if sep >= 10.0:
            sep_score = 1.0
        elif sep >= 8.0:
            sep_score = 0.8 + (sep - 8.0) / 2.0 * 0.2
        elif sep >= 5.0:
            sep_score = 0.5 + (sep - 5.0) / 3.0 * 0.3
        elif sep >= 3.0:
            sep_score = 0.2 + (sep - 3.0) / 2.0 * 0.3
        else:
            sep_score = max(0.0, sep / 3.0 * 0.2)
        
        # Heading deviation penalty (0 to 0.3)
        avg_deviation = (dev1 + dev2) / 2.0
        heading_penalty = min(0.3, avg_deviation / 120.0 * 0.3)
        
        # Safe time bonus (0 to 0.2)
        time_bonus = min(0.2, self.safe_seconds / 50.0 * 0.2)
        
        raw_score = sep_score - heading_penalty + time_bonus
        raw_score = max(0.0, min(1.0, raw_score))
        
        # Scale to [0.01, 0.99]
        final_score = 0.01 + (raw_score * 0.98)
        return round(final_score, 4)

    def step(self, action: str) -> Tuple[float, str, float, float, float]:
        """
        Apply action, score current separation, then tick both aircraft.
        Returns (reward, description, pre_tick_horiz_sep, pre_tick_vert_sep, normalized_score).
        """
        reward = 0.0
        log_parts: List[str] = []

        sep_before = self._separation()

        # ── Apply action ──────────────────────────────────────────────────────
        if action == "left_10":
            self.ac1.heading_deg = (self.ac1.heading_deg - 10) % 360
            log_parts.append("AC1 turn left 10°")
        elif action == "right_10":
            self.ac1.heading_deg = (self.ac1.heading_deg + 10) % 360
            log_parts.append("AC1 turn right 10°")
        elif action == "slow_10":
            self.ac1.speed_kts = max(100, self.ac1.speed_kts - 10)
            log_parts.append(f"AC1 slow 10kts → {self.ac1.speed_kts:.0f}kts")
        elif action == "speed_10":
            self.ac1.speed_kts = min(350, self.ac1.speed_kts + 10)
            log_parts.append(f"AC1 accelerate 10kts → {self.ac1.speed_kts:.0f}kts")
        elif action == "ac2_left_10":
            self.ac2.heading_deg = (self.ac2.heading_deg - 10) % 360
            log_parts.append("AC2 turn left 10°")
        elif action == "ac2_right_10":
            self.ac2.heading_deg = (self.ac2.heading_deg + 10) % 360
            log_parts.append("AC2 turn right 10°")
        elif action == "ac2_slow_10":
            self.ac2.speed_kts = max(100, self.ac2.speed_kts - 10)
            log_parts.append(f"AC2 slow 10kts → {self.ac2.speed_kts:.0f}kts")
        elif action == "ac2_speed_10":
            self.ac2.speed_kts = min(350, self.ac2.speed_kts + 10)
            log_parts.append(f"AC2 accelerate 10kts → {self.ac2.speed_kts:.0f}kts")

        horiz_sep = self._separation()
        vert_sep = abs(self.ac1.altitude - self.ac2.altitude)

        if horiz_sep > 5.0:
            reward += 10
        if vert_sep >= 1000:
            reward += 10
        if horiz_sep > 5.0 and vert_sep >= 1000:
            reward += 15

        if horiz_sep < 3.0 and vert_sep < 1000:
            reward -= 25

        self.ac1.tick(dt_min=1.0)
        self.ac2.tick(dt_min=1.0)

        sep_after = self._separation()
        dev1 = self._heading_deviation(self.ac1)
        dev2 = self._heading_deviation(self.ac2)
        on_course = dev1 <= 20.0 and dev2 <= 20.0

        if sep_after > 8.0 and on_course:
            reward += 10
            self.safe_seconds += 1
            log_parts.append(f"+10 safely apart ({sep_after:.2f}NM) and on-course")
        elif sep_after > 8.0:
            reward += 5
            self.safe_seconds += 1
            log_parts.append(f"+5 safe distance ({sep_after:.2f}NM) but off-course")
        else:
            self.safe_seconds = 0

        if self.safe_seconds > 0:
            safe_bonus = self.safe_seconds * 2
            reward += safe_bonus
            log_parts.append(f"+{safe_bonus} safe-time bonus ({self.safe_seconds}s streak)")

        if sep_after < sep_before:
            reward -= 5
            log_parts.append(f"-5 sep decreased ({sep_before:.2f}→{sep_after:.2f}NM)")

        if sep_after < 5.0:
            reward -= 15
            log_parts.append(f"-15 CRITICAL PROXIMITY ({sep_after:.2f}NM < 5NM)")

        # Calculate normalized score (always between 0.01 and 0.99)
        normalized_score = self._calculate_normalized_score()
        log_parts.append(f"score={normalized_score:.4f}")

        description = " | ".join(log_parts)
        return round(reward, 2), description, horiz_sep, vert_sep, normalized_score


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5 — GATE ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Gate:
    gate_id: str
    occupied: bool
    taxi_dist_min: float
    compatible: bool
    blocked_by: Optional[str] = None

    @property
    def blocked(self) -> bool:
        return (not self.occupied) and (self.blocked_by is not None)


@dataclass
class ArrivingPlane:
    callsign: str
    eta_min: float


@dataclass
class GateAssignmentEnv:
    arriving: ArrivingPlane
    gates: List[Gate]
    queue: List[ArrivingPlane] = field(default_factory=list)
    assignments: List[Dict[str, Any]] = field(default_factory=list)

    def _calculate_normalized_score(self) -> float:
        """Calculate normalized score between 0.01 and 0.99."""
        if not self.assignments:
            return 0.50

        valid_assignments = sum(1 for a in self.assignments if a.get("valid", False))
        total_assignments = len(self.assignments)

        if total_assignments == 0:
            return 0.50

        # Success rate (0 to 1)
        success_rate = valid_assignments / total_assignments

        # Average taxi time for valid assignments
        taxi_times = []
        for a in self.assignments:
            if a.get("valid", False):
                gate_id = a.get("gate_id")
                gate = next((g for g in self.gates if g.gate_id == gate_id), None)
                if gate:
                    taxi_times.append(gate.taxi_dist_min)

        if taxi_times:
            avg_taxi = sum(taxi_times) / len(taxi_times)
            # Taxi score: 1.0 for 0-3min, 0 for 15+ min
            taxi_score = max(0.0, 1.0 - (avg_taxi - 3.0) / 12.0)
        else:
            taxi_score = 0.5

        # Combine scores (60% success, 40% taxi time)
        raw_score = (success_rate * 0.6) + (taxi_score * 0.4)
        raw_score = max(0.0, min(1.0, raw_score))

        # Scale to [0.01, 0.99]
        final_score = 0.01 + (raw_score * 0.98)
        return round(final_score, 4)

    @property
    def queue_empty(self) -> bool:
        return len(self.queue) == 0

    @property
    def assigned_gate(self) -> Optional[str]:
        for rec in reversed(self.assignments):
            if rec.get("valid"):
                return rec["gate_id"]
        return None

    def _gate_score(self, gate: Gate, eta: float) -> float:
        score = 0.0
        if gate.occupied:
            score -= 30
            return score
        if not gate.compatible:
            score -= 20
        if gate.blocked:
            score -= 20

        if gate.taxi_dist_min <= 5.0:
            score += 5
        elif gate.taxi_dist_min > 10.0:
            score -= 8

        if eta <= 12.0 and gate.taxi_dist_min <= 7.0:
            score += 15
        elif eta > 15.0:
            score -= 8

        return score

    def find_best_gate(self) -> str:
        candidates = [
            g for g in self.gates
            if not g.occupied and not g.blocked and g.compatible
        ]
        if candidates:
            return min(candidates, key=lambda g: g.taxi_dist_min).gate_id
        free = [g for g in self.gates if not g.occupied and not g.blocked]
        if free:
            return min(free, key=lambda g: g.taxi_dist_min).gate_id
        free_blocked = [g for g in self.gates if not g.occupied]
        if free_blocked:
            return min(free_blocked, key=lambda g: g.taxi_dist_min).gate_id
        return min(self.gates, key=lambda g: g.taxi_dist_min).gate_id

    def assign(self, gate_id: str) -> Tuple[float, List[str], float]:
        """
        Assign *arriving* to *gate_id*.
        Returns (reward, log_lines, normalized_score).
        """
        gate = next((g for g in self.gates if g.gate_id == gate_id), None)
        log: List[str] = []
        reward = 0.0

        if gate is None:
            log.append(f"ERROR: unknown gate '{gate_id}'")
            self.assignments.append(
                {"callsign": self.arriving.callsign,
                 "gate_id": gate_id,
                 "valid": False,
                 "reward": 0.0}
            )
        else:
            valid = True

            if gate.occupied:
                reward -= 30
                valid = False
                log.append(f"  -30  Gate {gate_id} OCCUPIED → invalid assignment")

            if not gate.compatible:
                reward -= 20
                valid = False
                log.append(f"  -20  Gate {gate_id} INCOMPATIBLE with aircraft size")

            if gate.blocked:
                reward -= 20
                valid = False
                log.append(f"  -20  Gate {gate_id} taxi path BLOCKED by {gate.blocked_by}")

            if valid:
                if gate.taxi_dist_min <= 5.0:
                    reward += 5
                    log.append(f"  +5   Short taxi {gate.taxi_dist_min:.1f}min")
                elif gate.taxi_dist_min > 10.0:
                    reward -= 8
                    log.append(f"  -8   Long taxi {gate.taxi_dist_min:.1f}min (detour)")

                if self.arriving.eta_min <= 12.0 and gate.taxi_dist_min <= 7.0:
                    reward += 15
                    log.append(
                        f"  +15  Quick arrival "
                        f"(ETA={self.arriving.eta_min:.1f}min, "
                        f"taxi={gate.taxi_dist_min:.1f}min)")
                elif self.arriving.eta_min > 15.0:
                    reward -= 8
                    log.append(
                        f"  -8   Long ETA {self.arriving.eta_min:.1f}min → plane must wait")

                gate.occupied = True
                gate.blocked_by = self.arriving.callsign
                log.append(f"  Gate {gate_id} assigned to {self.arriving.callsign}")
            else:
                log.append(f"  Gate {gate_id} rejected — no soft rewards applied")

            self.assignments.append(
                {"callsign": self.arriving.callsign,
                 "gate_id": gate_id,
                 "valid": valid,
                 "reward": round(reward, 2)}
            )

        if self.queue:
            self.arriving = self.queue.pop(0)
            log.append(f"  Next: {self.arriving.callsign}  ETA {self.arriving.eta_min:.1f}min")
        else:
            log.append("  Queue exhausted — all planes processed")

        # Calculate normalized score (always between 0.01 and 0.99)
        normalized_score = self._calculate_normalized_score()
        log.append(f"score={normalized_score:.4f}")

        return round(reward, 2), log, normalized_score
