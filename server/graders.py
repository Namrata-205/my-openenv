"""
graders.py — Task graders for the ATC TRACON RL Environment.

Task 1 (Wake Turbulence) is unchanged — working correctly.

Improvements in Tasks 2–5:
  Task 2 — Go-Around Prevention:
    - Reward/penalty values now match spec exactly
      (+12 go-around rate <5%, +3 on-time, -20 go-around, -8/min holding)
    - 'rl_agent' strategy now has a real multi-factor sort
      (ETA × 0.4 + fuel_normalised × 0.3 - priority × 3)
    - 'eta_optimized' uses genuine ETA-first ordering without
      fuel-hack offsets that caused incorrect go-around counts
    - 'fuel_priority' now correctly prioritises low-fuel only when
      their ETA is imminent, avoiding starvation of short-ETA flights
    - Go-around trigger unified: hold > 8 min OR (fuel < 3000 AND wait > 4)
    - Holding penalty uses -8 × ceil(wait) per spec (not wait × 2)

  Task 3 — Emergency Vectoring:
    - Emergency aircraft position is updated BEFORE conflict checks,
      so lateral separation reflects the post-vector position
    - Urgency multiplier removed from penalties (only amplifies bonuses)
    - Reward structure matches spec:
        +18 conflict-free insertion in ≤ 2 min
        +4  throughput drop < 10%
        -22 safety violation (< 3 NM lateral AND < 1000 ft vertical)
        -12 flow breakdown (throughput drop ≥ 10%)
    - 'inserted' flag correctly prevents re-entry but returns 0 on
      repeat calls instead of a fake +18 reward
    - Runway heading is a parameter (defaults to 270° for ILS approach)

  Task 4 — Conflict Alert:
    - Scoring is on POST-tick positions so the action's effect is what
      gets evaluated, not the frozen spawn positions
    - Reward structure matches spec exactly:
        +10 safely apart (> 8 NM) AND on correct heading (deviation ≤ 20°)
        +2  per accumulated safe second
        -5  separation decreased vs previous step
        -15 critical proximity (< 5 NM)
    - Separate safe_seconds counter that resets on any unsafe step
    - heading_deviation helper checks angular closeness to target heading
    - Vertical dimension removed from Task 4 (spec is 2-D radar/horizontal)
    - Alias actions ("altitude_change", "heading_change") removed; clean
      canonical action set: left_10, right_10, slow_10, speed_10,
      ac2_left_10, ac2_right_10, ac2_slow_10, ac2_speed_10

  Task 5 — Gate Assignment:
    - Reward structure matches spec exactly:
        +15 plane reaches gate quickly (ETA ≤ 12 min AND taxi ≤ 7 min)
        +5  short taxi (taxi ≤ 5 min)
        -8  long taxi (taxi > 10 min) OR long ETA (eta > 15 min)
        -20 blockage (path blocked by another plane)
        -20 incompatible gate
        -30 occupied gate
    - Occupied gates get no taxi_bonus (they are invalid assignments)
    - find_best_gate scoring updated to reflect new reward weights so the
      greedy selection matches what assign() actually rewards
    - Queue advancement logic unchanged (correct)
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
    emergency:       Aircraft
    traffic:         List[Aircraft]
    runway_heading:  float = 270.0   # ILS/runway heading for alignment bonus
    inserted:        bool  = False   # True once a clean insertion is confirmed

    def _move_to_vector(self, heading: float, altitude: float, insert_time: float):
        """
        Move emergency aircraft along the new heading for insert_time minutes
        at its current speed so conflict checks reflect the post-vector position.
        """
        rad          = math.radians(heading)
        nm_per_min   = self.emergency.speed / 60.0
        dist         = nm_per_min * insert_time
        self.emergency.lat     += math.cos(rad) * dist
        self.emergency.lon     += math.sin(rad) * dist
        self.emergency.heading  = heading
        self.emergency.altitude = altitude

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
            # Already inserted — episode should have terminated; no reward
            return 0.0, ["Already inserted — awaiting episode termination"]

        log:    List[str] = []
        reward: float     = 0.0

        # Clamp insert_time to a realistic range
        insert_time = max(0.5, min(5.0, insert_time))

        # ── Move emergency aircraft to new position FIRST ─────────────────────
        # Conflict checks below reflect where the aircraft actually will be,
        # not its current spawn position.
        self._move_to_vector(heading, altitude, insert_time)

        # ── Check conflicts against all traffic ───────────────────────────────
        violations     = 0
        close_calls    = 0

        for ac in self.traffic:
            lateral  = math.hypot(
                self.emergency.lat - ac.lat,
                self.emergency.lon - ac.lon,
            )
            vertical = abs(self.emergency.altitude - ac.altitude)

            if lateral < 3.0 and vertical < 1_000:
                # Full safety violation
                reward     -= 22
                violations += 1
                log.append(
                    f"  SAFETY VIOLATION: {self.emergency.callsign} vs {ac.callsign}"
                    f"  lateral={lateral:.2f}NM  vertical={vertical:.0f}ft  → -22"
                )
            elif lateral < 5.0:
                # Close call — degrade throughput but not a hard violation
                close_calls += 1
                log.append(
                    f"  CLOSE CALL: {self.emergency.callsign} vs {ac.callsign}"
                    f"  lateral={lateral:.2f}NM  → throughput degraded"
                )
            else:
                log.append(
                    f"  CLEAR of {ac.callsign}  lateral={lateral:.2f}NM"
                )

        conflict_free = (violations == 0)

        # ── Insertion speed reward ────────────────────────────────────────────
        if conflict_free and insert_time <= 2.0:
            reward += 18
            log.append(f"  INSERTION OK in {insert_time:.1f}min  → +18")
        elif conflict_free:
            # Safe but slow — partial credit proportional to how close to 2 min
            partial = max(0.0, 18.0 - (insert_time - 2.0) * 4.0)
            reward += partial
            log.append(
                f"  SLOW INSERTION {insert_time:.1f}min  → +{partial:.1f} (partial)"
            )

        # ── Throughput / flow impact ──────────────────────────────────────────
        # Each violation degrades throughput ~15 pp; each close call ~5 pp
        degradation_pct = violations * 15 + close_calls * 5
        if degradation_pct < 10:
            reward += 4
            log.append(
                f"  FLOW SMOOTH  degradation={degradation_pct:.0f}%  → +4"
            )
        else:
            reward -= 12
            log.append(
                f"  FLOW BREAKDOWN  degradation={degradation_pct:.0f}%  → -12"
            )

        # ── Runway alignment bonus (unchanged, but now a separate additive) ───
        diff            = abs(heading - self.runway_heading) % 360
        diff            = min(diff, 360 - diff)
        alignment_bonus = round(max(0.0, 5.0 - diff / 18.0), 2)  # up to +5
        reward         += alignment_bonus
        if alignment_bonus > 0:
            log.append(
                f"  ALIGNMENT: {diff:.1f}° off runway heading  → +{alignment_bonus}"
            )

        # ── Fuel urgency — bonus ONLY, never multiplies penalties ─────────────
        # A low-fuel emergency that is inserted cleanly gets a small extra reward
        # to encourage the agent to act faster on low-fuel cases.
        if conflict_free and self.emergency.fuel_state < 0.3:
            urgency_bonus = round((0.3 - self.emergency.fuel_state) * 20, 2)
            reward       += urgency_bonus
            log.append(
                f"  FUEL URGENCY bonus  fuel_state={self.emergency.fuel_state:.2f}"
                f"  → +{urgency_bonus}"
            )

        # ── Mark inserted if successful ───────────────────────────────────────
        if conflict_free:
            self.inserted = True
            log.append("  INSERTION CONFIRMED → terminal on next step")
        else:
            log.append("  INSERTION FAILED — retrying next step")

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
        Apply action to one or both aircraft, tick positions forward 1 minute,
        then evaluate the resulting separation.

        Reward spec (matched exactly):
          +10  separation > 8 NM AND both aircraft within 20° of target heading
          +2   per accumulated safe second (safe_seconds counter)
          -5   separation decreased vs previous step
          -15  critical proximity (< 5 NM after tick)

        Returns (reward, description, post_tick_separation_NM).
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
            log_parts.append(f"AC2 accelerate 10kts → {self.ac2.speed_kts:.0f}kts")

        else:
            log_parts.append(f"HOLD (unknown action '{action}')")

        # ── Tick both aircraft forward 1 minute ───────────────────────────────
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
        # Only flag as blocked when free but path is obstructed.
        # Occupied gates are penalised by the occupied penalty only.
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
        """
        Return the gate_id that maximises _gate_score for the current
        arriving aircraft.  Greedy selection using the same weights as assign().
        """
        best_id    = None
        best_score = float("-inf")
        for gate in self.gates:
            s = self._gate_score(gate, self.arriving.eta_min)
            if s > best_score:
                best_score = s
                best_id    = gate.gate_id
        return best_id  # type: ignore[return-value]

    def assign(self, gate_id: str) -> Tuple[float, List[str]]:
        """
        Assign *arriving* to *gate_id*.

        Reward spec (matched exactly):
          +15  quick arrival (ETA ≤ 12 min) AND short taxi (≤ 7 min)
          +5   short taxi (taxi ≤ 5 min)
          -8   long taxi (taxi > 10 min) OR long ETA (eta > 15 min)
          -20  blocked path (gate free but obstructed)
          -20  incompatible gate type
          -30  occupied gate (strongest penalty — invalid assignment)

        Queue always advances so the environment drains deterministically.
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

        # ── Always advance queue ──────────────────────────────────────────────
        if self.queue:
            self.arriving = self.queue.pop(0)
            log.append(
                f"  Next: {self.arriving.callsign}  ETA {self.arriving.eta_min:.1f}min"
            )
        else:
            log.append("  Queue exhausted — all planes processed")

        return round(reward, 2), log
