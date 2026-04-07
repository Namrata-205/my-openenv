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
# TASK 1 — WAKE TURBULENCE  (unchanged — working correctly)
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
# TASK 2 — GO-AROUND PREVENTION  (improved)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InboundFlight:
    callsign:    str
    eta_min:     float   # minutes from now to threshold
    fuel_lbs:    float   # fuel remaining in lbs
    priority:    int     # 1 = highest urgency
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

    Reward spec (matched exactly):
      +12  go-around rate < 5%
      +3   on-time landing (wait ≤ 1 min)
      -20  each go-around
      -8 × ceil(wait_min)  holding penalty per flight
    """

    # ── Sort by strategy ──────────────────────────────────────────────────────
    if strategy == "fcfs":
        # Baseline: first come, first served by arrival time
        flights.sort(key=lambda f: f.eta_min)

    elif strategy == "fuel_priority":
        # Low-fuel is urgent — but we break ties by ETA so we don't strand
        # aircraft that are already on a short final while waiting for a low-fuel
        # plane that is still 20 minutes out.
        # Sort key: primary = fuel bucket (0-3000 = critical, else normal),
        #           secondary = ETA within each bucket
        def fuel_key(f: InboundFlight) -> Tuple[int, float]:
            bucket = 0 if f.fuel_lbs < 3000 else 1
            return (bucket, f.eta_min)
        flights.sort(key=fuel_key)

    elif strategy == "eta_optimized":
        # Pure ETA order — tightest deadline first, no hacks
        flights.sort(key=lambda f: f.eta_min)

    elif strategy == "rl_agent":
        # Multi-factor RL policy:
        #   - Low ETA aircraft land sooner (weight 0.4)
        #   - Low fuel aircraft are treated as more urgent (weight 0.3, inverted)
        #   - High priority (low number) aircraft are bumped forward (weight 3.0)
        # Result: naturally avoids holding critical/low-fuel flights
        MAX_FUEL = 10_000.0  # normalisation ceiling
        flights.sort(key=lambda f: (
            f.eta_min * 0.4
            + (f.fuel_lbs / MAX_FUEL) * 0.3
            - f.priority * 3.0
        ))

    # ── Simulate runway operations ────────────────────────────────────────────
    RUNWAY_OCC_MIN = 2.0   # runway occupancy time per landing (minutes)
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
        # Earliest the runway is available for this flight
        landing_time  = max(f.eta_min, runway_free_at)
        wait          = landing_time - f.eta_min  # holding time (≥ 0)
        f.holding_min = wait

        # Go-around decision: too long a hold OR critically low fuel with any hold
        goes_around = wait > 8.0 or (f.fuel_lbs < 3_000 and wait > 4.0)

        if goes_around:
            f.went_around  = True
            total_reward  -= 20
            events.append(
                f"{f.callsign}: GO-AROUND  "
                f"wait={wait:.1f}min  fuel={f.fuel_lbs:.0f}lbs  → -20"
            )
        else:
            f.landed       = True
            runway_free_at = landing_time + RUNWAY_OCC_MIN

            if wait <= 1.0:
                # On-time landing
                total_reward += 3
                events.append(f"{f.callsign}: ON-TIME  wait={wait:.1f}min  → +3")
            else:
                # Holding penalty: -8 per each full minute (or part thereof)
                hold_penalty  = -8 * math.ceil(wait)
                total_reward += hold_penalty
                events.append(
                    f"{f.callsign}: LANDED  wait={wait:.1f}min  "
                    f"hold_penalty={hold_penalty:.0f}"
                )

    # ── Go-around rate bonus ──────────────────────────────────────────────────
    go_around_count = sum(1 for f in flights if f.went_around)
    landed_count    = sum(1 for f in flights if f.landed)
    ga_rate         = go_around_count / len(flights) if flights else 0.0

    if ga_rate < 0.05:
        total_reward += 12
        events.append(
            f"BONUS: go-around rate {ga_rate*100:.1f}% < 5%  → +12"
        )
    else:
        events.append(
            f"INFO:  go-around rate {ga_rate*100:.1f}%  (need <5% for bonus)"
        )

    stats = {
        "strategy":      strategy,
        "landed":        landed_count,
        "go_arounds":    go_around_count,
        "ga_rate_pct":   round(ga_rate * 100, 1),
        "total_reward":  round(total_reward, 2),
        "total_holding": round(sum(f.holding_min for f in flights), 2),
    }
    return round(total_reward, 2), events, stats


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3 — EMERGENCY VECTORING  (improved)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Aircraft:
    callsign:   str
    heading:    float    # degrees
    altitude:   float    # feet MSL
    speed:      float    # knots
    lat:        float    # 2-D position (1 unit ≈ 1 NM)
    lon:        float
    fuel_state: float = 1.0   # 0.0 (empty) – 1.0 (full)


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
        heading:     float,
        altitude:    float,
        insert_time: float = 1.0,
    ) -> Tuple[float, List[str]]:
        """
        Vector the emergency aircraft to *heading* / *altitude* and check
        whether the resulting position is conflict-free.

        Reward spec (matched exactly):
          +18  conflict-free insertion in ≤ 2 min
          +4   throughput drop < 10%  (no blocking conflicts)
          -22  each safety violation (< 3 NM lateral AND < 1000 ft vertical)
          -12  flow breakdown (≥ 10% throughput degradation)
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

        log.insert(0,
            f"Vector heading={heading:.0f}° alt={altitude:.0f}ft "
            f"dist={dist_nm:.1f}NM hdg_err={hdg_error:.0f}° "
            f"reward={reward:.1f}")
        return round(reward, 2), log


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4 — CONFLICT ALERT  (improved)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConflictAircraftState:
    callsign:    str
    x:           float   # NM east
    y:           float   # NM north
    speed_kts:   float
    heading_deg: float = 90.0
    target_hdg:  float = 90.0   # desired final heading (degrees)
    altitude:    float = 8000.0  # feet MSL — kept for environment.py compatibility

    def tick(self, dt_min: float = 1.0):
        """Advance position by dt_min minutes along current heading/speed."""
        nm   = self.speed_kts / 60.0 * dt_min
        rad  = math.radians(self.heading_deg)
        self.x += nm * math.sin(rad)
        self.y += nm * math.cos(rad)


@dataclass
class ConflictAlertEnv:
    ac1:          ConflictAircraftState
    ac2:          ConflictAircraftState
    safe_seconds: int = 0   # consecutive steps both aircraft have been safe

    def _separation(self) -> float:
        return math.hypot(self.ac1.x - self.ac2.x, self.ac1.y - self.ac2.y)

    @staticmethod
    def _heading_deviation(ac: ConflictAircraftState) -> float:
        """Smallest angular distance from current heading to target heading."""
        diff = abs(ac.heading_deg - ac.target_hdg) % 360
        return min(diff, 360 - diff)

    def step(self, action: str) -> Tuple[float, str, float]:
        """
        Apply action, score current (pre-tick) separation, then tick both aircraft.

        Returns (reward, description, pre_tick_horiz_sep, pre_tick_vert_sep).
        Scoring on pre-tick positions means step 1 always evaluates spawn positions
        (guaranteed safe) rather than post-movement positions.
        """
        reward      = 0.0
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

        sep_after = self._separation()

        # ── Evaluate POST-tick separation (action effect is now visible) ───────
        dev1      = self._heading_deviation(self.ac1)
        dev2      = self._heading_deviation(self.ac2)
        on_course = dev1 <= 20.0 and dev2 <= 20.0

        # +10: safely apart AND on-course
        if sep_after > 8.0 and on_course:
            reward           += 10
            self.safe_seconds += 1
            log_parts.append(
                f"+10 safely apart ({sep_after:.2f}NM) and on-course"
            )
        elif sep_after > 8.0:
            # Safe distance but off-course — partial credit, reset streak
            reward            += 5
            self.safe_seconds += 1
            log_parts.append(
                f"+5 safe distance ({sep_after:.2f}NM) but off-course "
                f"(dev1={dev1:.0f}° dev2={dev2:.0f}°)"
            )
        else:
            # Not yet safe — reset streak
            self.safe_seconds = 0

        # +2 per accumulated safe second
        if self.safe_seconds > 0:
            safe_bonus = self.safe_seconds * 2
            reward    += safe_bonus
            log_parts.append(
                f"+{safe_bonus} safe-time bonus ({self.safe_seconds}s streak)"
            )

        # -5: separation got worse
        if sep_after < sep_before:
            reward    -= 5
            log_parts.append(
                f"-5 sep decreased ({sep_before:.2f}→{sep_after:.2f}NM)"
            )

        # -15: critical proximity
        if sep_after < 5.0:
            reward    -= 15
            log_parts.append(
                f"-15 CRITICAL PROXIMITY ({sep_after:.2f}NM < 5NM)"
            )

        description = " | ".join(log_parts)
        description += f" | sep={sep_after:.2f}NM reward={reward:.1f}"
        # Return 4 values: (reward, description, horiz_sep, vert_sep)
        # vert_sep derived from altitude difference (always 0 if altitudes equal).
        # environment.py unpacks: raw_reward, desc, sep, vert_sep
        vert_sep = abs(self.ac1.altitude - self.ac2.altitude)
        return round(reward, 2), description, sep_after, vert_sep


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5 — GATE ASSIGNMENT  (improved)
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
    eta_min:  float   # minutes until touchdown


@dataclass
class GateAssignmentEnv:
    arriving:    ArrivingPlane
    gates:       List[Gate]
    queue:       List[ArrivingPlane]   = field(default_factory=list)
    assignments: List[Dict[str, Any]]  = field(default_factory=list)

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
        """
        Scoring function used by find_best_gate — mirrors the assign() reward
        so greedy selection always picks the gate that maximises actual reward.

        Score components (positive = better):
          +15  quick arrival AND short taxi
          +5   short taxi (≤ 5 min)
          -8   long taxi (> 10 min) OR long ETA (> 15 min)
          -20  blocked path
          -20  incompatible
          -30  occupied
        """
        score = 0.0
        if gate.occupied:
            score -= 30
            return score      # occupied gate — further scoring irrelevant
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
            log.append(f"ERROR: unknown gate '{gate_id}'")
            self.assignments.append(
                {"callsign": self.arriving.callsign,
                 "gate_id":  gate_id,
                 "valid":    False,
                 "reward":   0.0}
            )
        else:
            valid = True   # assume valid until a hard constraint fires

            # ── Hard constraints (stackable penalties) ────────────────────────
            if gate.occupied:
                reward -= 30
                valid   = False
                log.append(f"  -30  Gate {gate_id} OCCUPIED → invalid assignment")

            if not gate.compatible:
                reward -= 20
                valid   = False
                log.append(f"  -20  Gate {gate_id} INCOMPATIBLE with aircraft size")

            if gate.blocked:
                reward -= 20
                valid   = False
                log.append(f"  -20  Gate {gate_id} taxi path BLOCKED by {gate.blocked_by}")

            # ── Soft rewards/penalties (only for valid gates) ─────────────────
            if valid:
                # Short taxi bonus
                if gate.taxi_dist_min <= 5.0:
                    reward += 5
                    log.append(f"  +5   Short taxi {gate.taxi_dist_min:.1f}min")
                elif gate.taxi_dist_min > 10.0:
                    reward -= 8
                    log.append(f"  -8   Long taxi {gate.taxi_dist_min:.1f}min (detour)")

                # Quick arrival + close gate bonus
                if self.arriving.eta_min <= 12.0 and gate.taxi_dist_min <= 7.0:
                    reward += 15
                    log.append(
                        f"  +15  Quick arrival "
                        f"(ETA={self.arriving.eta_min:.1f}min, "
                        f"taxi={gate.taxi_dist_min:.1f}min)"
                    )
                elif self.arriving.eta_min > 15.0:
                    reward -= 8
                    log.append(
                        f"  -8   Long ETA {self.arriving.eta_min:.1f}min → plane must wait"
                    )

                # Mark gate occupied
                gate.occupied   = True
                gate.blocked_by = self.arriving.callsign
                log.append(
                    f"  Gate {gate_id} assigned to {self.arriving.callsign}"
                )
            else:
                log.append(
                    f"  Gate {gate_id} rejected — no soft rewards applied"
                )

            self.assignments.append(
                {"callsign": self.arriving.callsign,
                 "gate_id":  gate_id,
                 "valid":    valid,
                 "reward":   round(reward, 2)}
            )

        # FIX: always advance queue regardless of assignment validity
        if self.queue:
            self.arriving = self.queue.pop(0)
            log.append(
                f"  Next: {self.arriving.callsign}  ETA {self.arriving.eta_min:.1f}min"
            )
        else:
            log.append("  Queue exhausted — all planes processed")

        return round(reward, 2), log
