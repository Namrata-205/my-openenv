"""
ATC Reinforcement Learning Simulator — All 5 Tasks
====================================================
No model training. Pure reward/penalty logic for each task.
Run: python app.py
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — WAKE TURBULENCE SEPARATION
# ═══════════════════════════════════════════════════════════════════════════════
"""
Reward Structure:
  +15  → 100% FAA-compliant spacing (perfect interval)
  +2   → Every safe second that passes
  -25  → Loss of separation (< 4 NM)
  -10  → Unnecessary delay > 30 seconds
"""

class AircraftCategory(Enum):
    HEAVY  = "Heavy"
    MEDIUM = "Medium"
    LIGHT  = "Light"

# FAA-mandated minimum separation (NM) based on lead/trail category pair
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
    current_sep:   float = 4.5     # current separation in NM
    elapsed_safe:  float = 0.0     # seconds spent safely separated
    delay_sec:     float = 0.0     # total delay accumulated (seconds)
    violated:      bool  = False

    @property
    def required_sep(self) -> float:
        return REQUIRED_SEPARATION[(self.leading_cat, self.trailing_cat)]

    def step(self, action: str) -> Tuple[float, str]:
        """
        Actions:
          'slow_down_trailing'   — increase gap (adds delay)
          'speed_up_trailing'    — decrease gap (risky)
          'increase_heading_gap' — lateral offset (moderate delay)
          'hold'                 — maintain current vector

        Returns: (reward, description)
        """
        reward = 0.0
        log    = []

        # ── Apply action physics ──────────────────────────────────
        if action == "slow_down_trailing":
            delta = random.uniform(0.3, 0.7)
            self.current_sep += delta
            delay_added = random.uniform(15, 45)
            self.delay_sec += delay_added
            log.append(f"Action: slow_down_trailing → sep +{delta:.2f} NM, delay +{delay_added:.0f}s")

        elif action == "speed_up_trailing":
            delta = random.uniform(0.1, 0.5)
            self.current_sep -= delta
            log.append(f"Action: speed_up_trailing → sep -{delta:.2f} NM")

        elif action == "increase_heading_gap":
            delta = random.uniform(0.5, 1.2)
            self.current_sep += delta
            delay_added = random.uniform(5, 20)
            self.delay_sec += delay_added
            log.append(f"Action: increase_heading_gap → sep +{delta:.2f} NM, delay +{delay_added:.0f}s")

        elif action == "hold":
            drift = random.uniform(-0.15, 0.15)
            self.current_sep += drift
            self.elapsed_safe += 1
            log.append(f"Action: hold → sep drift {drift:+.2f} NM")

        self.current_sep = max(0.5, self.current_sep)

        # ── Reward computation ────────────────────────────────────
        req = self.required_sep

        # Perfect FAA-compliant interval (within 0.5 NM of required)
        if self.current_sep >= req and (self.current_sep - req) <= 0.5:
            reward += 15
            log.append("+15 : Perfect FAA-compliant spacing")
        elif self.current_sep >= req:
            reward += 5
            log.append("+5  : Safe but wider than optimal")

        # Bonus for every safe second accumulated
        if self.elapsed_safe > 0:
            safe_bonus = self.elapsed_safe * 2
            reward += safe_bonus
            log.append(f"+{safe_bonus:.1f} : Safe-time bonus ({self.elapsed_safe:.0f}s)")

        # Loss of separation (< 4 NM hard floor regardless of category pair)
        if self.current_sep < 4.0:
            reward -= 25
            self.violated = True
            log.append(f"-25 : LOSS OF SEPARATION ({self.current_sep:.2f} NM < 4 NM) !!!")

        # Unnecessary delay penalty
        if self.delay_sec > 30:
            extra_delay = self.delay_sec - 30
            penalty = -10 * math.ceil(extra_delay / 30)
            reward += penalty
            log.append(f"{penalty:.0f} : Excessive delay ({self.delay_sec:.0f}s > 30s threshold)")

        description = "\n  ".join(log)
        return round(reward, 2), description


def run_task1():
    print("\n" + "=" * 60)
    print("TASK 1 — WAKE TURBULENCE SEPARATION")
    print("=" * 60)

    scenarios = [
        (AircraftCategory.HEAVY,  AircraftCategory.LIGHT,  3.8, "slow_down_trailing"),
        (AircraftCategory.HEAVY,  AircraftCategory.MEDIUM, 5.2, "hold"),
        (AircraftCategory.MEDIUM, AircraftCategory.LIGHT,  3.2, "increase_heading_gap"),
        (AircraftCategory.HEAVY,  AircraftCategory.HEAVY,  4.0, "speed_up_trailing"),
    ]

    for lead, trail, init_sep, action in scenarios:
        env = WakeTurbulenceEnv(leading_cat=lead, trailing_cat=trail, current_sep=init_sep)
        total_reward = 0.0

        print(f"\n  Lead: {lead.value:<8} | Trail: {trail.value:<8} | "
              f"Required: {env.required_sep} NM | Init Sep: {init_sep} NM")
        print(f"  Strategy: {action}")
        print(f"  {'-' * 50}")

        for step in range(5):
            r, desc = env.step(action)
            total_reward += r
            print(f"  Step {step + 1}: reward={r:+.2f} | sep={env.current_sep:.2f} NM")
            print(f"    {desc}")

        status = "VIOLATED !!!" if env.violated else "SAFE"
        print(f"  -> Total Reward: {total_reward:+.2f} | Status: {status}")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — GO-AROUND PREVENTION SEQUENCING
# ═══════════════════════════════════════════════════════════════════════════════
"""
Reward Structure:
  +12  → Go-around rate < 5%
  +3   → Each plane lands on time without holding
  -20  → Avoidable go-around
  -8   → Each extra minute of holding (~2000 lbs fuel per B737)
"""

@dataclass
class InboundFlight:
    callsign:    str
    eta_min:     float   # minutes from now to runway threshold
    fuel_lbs:    float   # fuel remaining in lbs
    priority:    int     # 1 = highest (emergency/low fuel)
    holding_min: float = 0.0
    landed:      bool  = False
    went_around: bool  = False


def sequence_flights(flights: List[InboundFlight], strategy: str) -> Tuple[float, List[str], dict]:
    """
    Strategies:
      'fcfs'          — first come first served (baseline)
      'fuel_priority' — lowest fuel lands first
      'eta_optimized' — tightest ETA first
      'rl_agent'      — balances ETA, fuel, and priority (RL policy)
    """
    if strategy == "fcfs":
        ordered = sorted(flights, key=lambda f: f.eta_min)

    elif strategy == "fuel_priority":
        ordered = sorted(flights, key=lambda f: f.fuel_lbs)

    elif strategy == "eta_optimized":
        ordered = sorted(flights, key=lambda f: f.eta_min + f.holding_min)

    elif strategy == "rl_agent":
        # RL-derived weighting: urgency (low fuel/high priority) > ETA efficiency
        ordered = sorted(flights, key=lambda f:
            f.eta_min * 0.4
            + (f.fuel_lbs / 5000) * 0.3
            - f.priority * 3.0
        )
    else:
        ordered = flights

    RUNWAY_OCCUPANCY = 2.0   # minutes per landing
    runway_free_at   = 0.0
    total_reward     = 0.0
    go_arounds       = 0
    total_holding    = 0.0
    events           = []

    for flight in ordered:
        earliest = max(flight.eta_min, runway_free_at)
        wait     = max(0.0, earliest - flight.eta_min)
        flight.holding_min = wait
        total_holding += wait

        # Go-around triggers: excessive hold OR critically low fuel with any hold
        goes_around = (wait > 8.0) or (flight.fuel_lbs < 3000 and wait > 3.0)

        if goes_around:
            flight.went_around = True
            go_arounds += 1
            total_reward -= 20
            events.append(
                f"  GO-AROUND {flight.callsign} "
                f"(hold={wait:.1f}min, fuel={flight.fuel_lbs:.0f}lbs)  -> -20"
            )
        else:
            flight.landed = True
            runway_free_at = earliest + RUNWAY_OCCUPANCY

            if wait <= 1.0:
                total_reward += 3
                events.append(f"  ON-TIME {flight.callsign} (hold={wait:.1f}min)  -> +3")
            else:
                events.append(f"  LANDED  {flight.callsign} (hold={wait:.1f}min)")

        # Holding penalty per minute
        if wait > 0:
            hold_penalty = -8 * math.ceil(wait)
            total_reward += hold_penalty
            events.append(
                f"  FUEL    {flight.callsign} holding penalty ({wait:.1f}min)  -> {hold_penalty:.0f}"
            )

    # Go-around rate bonus
    ga_rate = go_arounds / len(flights)
    if ga_rate < 0.05:
        total_reward += 12
        events.append(f"  BONUS   Go-around rate {ga_rate * 100:.1f}% < 5% threshold  -> +12")

    stats = {
        "strategy":      strategy,
        "total_reward":  round(total_reward, 2),
        "go_arounds":    go_arounds,
        "ga_rate_pct":   round(ga_rate * 100, 1),
        "total_holding": round(total_holding, 2),
        "landed":        sum(1 for f in flights if f.landed),
    }
    return total_reward, events, stats


def run_task2():
    print("\n" + "=" * 60)
    print("TASK 2 — GO-AROUND PREVENTION SEQUENCING")
    print("=" * 60)

    random.seed(42)
    callsigns = ["UAL101", "DAL202", "AAL303", "SWA404", "FDX505", "BAW606", "AFR707", "DLH808"]
    flights_template = [
        InboundFlight(cs,
                      round(random.uniform(5, 25), 1),
                      round(random.uniform(2000, 8000), 0),
                      random.randint(1, 3))
        for cs in callsigns
    ]

    print(f"\n  Incoming flights:")
    for f in flights_template:
        print(f"    {f.callsign}: ETA={f.eta_min}min, fuel={f.fuel_lbs:.0f}lbs, priority={f.priority}")

    for strategy in ["fcfs", "fuel_priority", "eta_optimized", "rl_agent"]:
        flights = [
            InboundFlight(f.callsign, f.eta_min, f.fuel_lbs, f.priority)
            for f in flights_template
        ]
        reward, events, stats = sequence_flights(flights, strategy)

        print(f"\n  Strategy: {strategy.upper()}")
        for e in events:
            print(e)
        print(f"  -> Total Reward: {stats['total_reward']:+.2f} | "
              f"Go-arounds: {stats['go_arounds']} ({stats['ga_rate_pct']}%) | "
              f"Total holding: {stats['total_holding']:.1f}min")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3 — EMERGENCY VECTORING
# ═══════════════════════════════════════════════════════════════════════════════
"""
Reward Structure:
  +18  → Conflict-free insertion in under 2 minutes
  +4   → Overall flow stays smooth (throughput drop < 10%)
  -22  → Any safety violation near emergency aircraft
  -12  → Flow breakdown (> 10% drop in throughput)
"""

@dataclass
class Aircraft:
    callsign:  str
    heading:   float   # degrees
    altitude:  float   # feet
    speed:     float   # knots
    lat:       float   # simplified 2D position (1 unit ≈ 1 NM)
    lon:       float


@dataclass
class EmergencyVectorEnv:
    emergency:      Aircraft
    traffic:        List[Aircraft]
    baseline_tput:  float = 10.0   # baseline flights/hr throughput

    def _separation_nm(self, a: Aircraft, b: Aircraft) -> float:
        """Euclidean distance used as NM proxy."""
        return math.hypot(a.lat - b.lat, a.lon - b.lon)

    def insert_emergency(self, new_heading: float, new_altitude: float,
                         insertion_time_min: float) -> Tuple[float, List[str]]:
        """
        Attempt to insert emergency aircraft with given heading/altitude change.
        insertion_time_min: how many minutes the rerouting takes.
        Returns (reward, event_log)
        """
        reward = 0.0
        log    = []

        # Apply heading and altitude to emergency aircraft
        self.emergency.heading  = new_heading
        self.emergency.altitude = new_altitude

        # Recompute position after vectoring (simple kinematic step)
        rad = math.radians(new_heading)
        self.emergency.lat += math.cos(rad) * (self.emergency.speed / 60) * insertion_time_min
        self.emergency.lon += math.sin(rad) * (self.emergency.speed / 60) * insertion_time_min

        # Check conflicts with all traffic (< 3 NM horizontal AND < 1000 ft vertical)
        conflicts = []
        for ac in self.traffic:
            horiz_sep = self._separation_nm(self.emergency, ac)
            vert_sep  = abs(self.emergency.altitude - ac.altitude)
            if horiz_sep < 3.0 and vert_sep < 1000:
                conflicts.append((ac.callsign, horiz_sep, vert_sep))

        # ── Reward: conflict-free fast insertion ─────────────────
        if not conflicts and insertion_time_min <= 2.0:
            reward += 18
            log.append(f"+18 : Conflict-free insertion in {insertion_time_min:.1f} min")
        elif not conflicts:
            reward += 8
            log.append(f"+8  : Conflict-free but slow insertion ({insertion_time_min:.1f} min)")
        else:
            for cs, h, v in conflicts:
                reward -= 22
                log.append(f"-22 : SAFETY VIOLATION with {cs} "
                            f"(horiz={h:.2f}NM, vert={v:.0f}ft) !!!")

        # ── Throughput impact ─────────────────────────────────────
        degradation_pct = len(conflicts) * 15 + max(0, insertion_time_min - 2) * 5
        if degradation_pct < 10:
            reward += 4
            log.append(f"+4  : Flow smooth (throughput drop {degradation_pct:.1f}% < 10%)")
        else:
            reward -= 12
            log.append(f"-12 : FLOW BREAKDOWN (throughput drop {degradation_pct:.1f}% >= 10%)")

        log.append(f"     heading={new_heading}deg, alt={new_altitude}ft, "
                   f"insert_time={insertion_time_min:.1f}min")

        return round(reward, 2), log


def run_task3():
    print("\n" + "=" * 60)
    print("TASK 3 — EMERGENCY VECTORING")
    print("=" * 60)

    traffic = [
        Aircraft("UAL101", 270, 3000, 160, 10.0, 5.0),
        Aircraft("DAL202", 260, 4000, 155, 8.0,  2.0),
        Aircraft("AAL303", 275, 2500, 170, 12.0, 7.0),
    ]

    test_cases = [
        # (new_heading, new_altitude, insert_time_min, label)
        (250, 2000, 1.5,  "Optimal: fast + clear path"),
        (270, 3000, 1.8,  "Good heading but near traffic altitude"),
        (260, 2000, 3.5,  "Safe heading but slow"),
        (270, 3500, 1.0,  "Fast but altitude conflict risk"),
    ]

    for heading, alt, t_min, label in test_cases:
        env = EmergencyVectorEnv(
            emergency=Aircraft("EMER1", 0, 5000, 180, 15.0, 0.0),
            traffic=[
                Aircraft(ac.callsign, ac.heading, ac.altitude, ac.speed, ac.lat, ac.lon)
                for ac in traffic
            ]
        )
        reward, log = env.insert_emergency(heading, alt, t_min)
        print(f"\n  Scenario: {label}")
        for l in log:
            print(f"  {l}")
        print(f"  -> Total Reward: {reward:+.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4 — SIMPLE CONFLICT ALERT SUGGESTER
# ═══════════════════════════════════════════════════════════════════════════════
"""
Reward Structure:
  +10  → Planes safely apart again (> 8 NM) and still on correct heading
  +2   → Every second they stay safe after suggestion
  -5   → Separation gets worse after action
  -15  → Aircraft get too close (< 5 NM)
"""

@dataclass
class ConflictAircraftState:
    callsign:    str
    x:           float   # NM (2D)
    y:           float   # NM
    heading_deg: float   # degrees
    speed_kts:   float   # knots
    target_hdg:  float   # intended destination heading


@dataclass
class ConflictAlertEnv:
    ac1:       ConflictAircraftState
    ac2:       ConflictAircraftState
    time_safe: float = 0.0   # seconds both stayed safely apart

    def _separation(self) -> float:
        return math.hypot(self.ac1.x - self.ac2.x, self.ac1.y - self.ac2.y)

    def _heading_deviation(self, ac: ConflictAircraftState) -> float:
        """Absolute angular difference from target heading."""
        diff = abs(ac.heading_deg - ac.target_hdg) % 360
        return min(diff, 360 - diff)

    def _move(self, ac: ConflictAircraftState, dt_min: float = 0.5):
        """Move aircraft forward dt_min minutes at current heading/speed."""
        rad = math.radians(ac.heading_deg)
        dist = (ac.speed_kts / 60) * dt_min
        ac.x += math.sin(rad) * dist
        ac.y += math.cos(rad) * dist

    def step(self, action: str) -> Tuple[float, str]:
        """
        Actions (applied to ac1, the conflicting aircraft):
          'left_10'   — turn left 10 degrees
          'right_10'  — turn right 10 degrees
          'slow_10'   — reduce speed 10 knots
          'speed_10'  — increase speed 10 knots

        Returns (reward, description)
        """
        reward = 0.0
        log    = []
        sep_before = self._separation()

        # ── Apply action ──────────────────────────────────────────
        if action == "left_10":
            self.ac1.heading_deg = (self.ac1.heading_deg - 10) % 360
            log.append(f"Action: turn {self.ac1.callsign} LEFT 10 -> hdg {self.ac1.heading_deg:.0f}deg")
        elif action == "right_10":
            self.ac1.heading_deg = (self.ac1.heading_deg + 10) % 360
            log.append(f"Action: turn {self.ac1.callsign} RIGHT 10 -> hdg {self.ac1.heading_deg:.0f}deg")
        elif action == "slow_10":
            self.ac1.speed_kts = max(100, self.ac1.speed_kts - 10)
            log.append(f"Action: slow {self.ac1.callsign} 10kts -> {self.ac1.speed_kts:.0f}kts")
        elif action == "speed_10":
            self.ac1.speed_kts += 10
            log.append(f"Action: speed up {self.ac1.callsign} 10kts -> {self.ac1.speed_kts:.0f}kts")

        # Advance both aircraft 0.5 min
        self._move(self.ac1)
        self._move(self.ac2)

        sep_after = self._separation()

        # ── Reward computation ────────────────────────────────────
        dev1      = self._heading_deviation(self.ac1)
        dev2      = self._heading_deviation(self.ac2)
        on_course = (dev1 <= 20) and (dev2 <= 20)

        if sep_after > 8.0 and on_course:
            reward += 10
            log.append(f"+10 : Safely apart ({sep_after:.2f} NM > 8 NM) and on-course")
            self.time_safe += 1
        elif sep_after > 8.0:
            reward += 5
            log.append(f"+5  : Safely apart but off-course (dev={dev1:.0f}deg)")
            self.time_safe += 1
        else:
            self.time_safe = 0

        # Time-safe bonus
        if self.time_safe > 0:
            reward += self.time_safe * 2
            log.append(f"+{self.time_safe * 2:.0f} : {self.time_safe:.0f}s safe-time bonus")

        # Separation getting worse
        if sep_after < sep_before:
            reward -= 5
            log.append(f"-5  : Separation DECREASED ({sep_before:.2f} -> {sep_after:.2f} NM)")

        # Critical proximity
        if sep_after < 5.0:
            reward -= 15
            log.append(f"-15 : CRITICAL PROXIMITY ({sep_after:.2f} NM < 5 NM) !!!")

        desc = "\n  ".join(log)
        return round(reward, 2), desc


def run_task4():
    print("\n" + "=" * 60)
    print("TASK 4 — SIMPLE CONFLICT ALERT SUGGESTER")
    print("=" * 60)

    scenarios = [
        # (ac1_pos, ac2_pos, ac1_hdg, ac2_hdg, speeds, action, label)
        ((0, 0), (6, 0),  90,  270, (250, 250), "left_10",  "Head-on: turn left"),
        ((0, 0), (6, 3),  90,  270, (250, 250), "slow_10",  "Near-converge: slow down"),
        ((0, 0), (5, 5),  45,  225, (280, 280), "right_10", "Converging: turn right"),
        ((0, 0), (7, 0),  90,  270, (300, 300), "speed_10", "Diverging: speed up (check)"),
    ]

    for (x1, y1), (x2, y2), h1, h2, (s1, s2), action, label in scenarios:
        env = ConflictAlertEnv(
            ac1=ConflictAircraftState("AC1", x1, y1, h1, s1, target_hdg=h1),
            ac2=ConflictAircraftState("AC2", x2, y2, h2, s2, target_hdg=h2),
        )
        total = 0.0
        print(f"\n  Scenario: {label}")
        print(f"  Initial separation: {env._separation():.2f} NM | Action: {action}")
        for step in range(4):
            r, desc = env.step(action)
            total += r
            print(f"  Step {step + 1}: reward={r:+.2f} | sep={env._separation():.2f} NM")
            print(f"    {desc}")
        print(f"  -> Total Reward: {total:+.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5 — BASIC GATE / STAND ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════
"""
Reward Structure:
  +15  → Plane reaches gate quickly with a clear path
  +5   → Short taxi distance (saves time/fuel)
  -8   → Plane has to wait or take a detour
  -20  → Suggested gate causes a blockage with another plane
"""

@dataclass
class Gate:
    gate_id:       str
    occupied:      bool
    taxi_dist_min: float          # taxi time from runway exit in minutes
    compatible:    bool           # size-compatible with arriving aircraft
    blocked_by:    Optional[str] = None   # callsign blocking the taxi path


@dataclass
class ArrivingPlane:
    callsign: str
    eta_min:  float   # minutes until landing


@dataclass
class GateAssignmentEnv:
    arriving:      ArrivingPlane
    gates:         List[Gate]
    assigned_gate: Optional[Gate] = None

    def assign(self, gate_id: str) -> Tuple[float, List[str]]:
        """
        Evaluate assigning the arriving plane to a specific gate.
        Returns (reward, event_log)
        """
        gate = next((g for g in self.gates if g.gate_id == gate_id), None)
        reward = 0.0
        log    = []

        if gate is None:
            log.append(f"ERROR: Gate {gate_id} not found.")
            return -20.0, log

        self.assigned_gate = gate

        # ── Penalty: gate already occupied ───────────────────────
        if gate.occupied:
            reward -= 20
            log.append(f"-20 : Gate {gate_id} is OCCUPIED -> blockage !!!")
        else:
            log.append(f"     Gate {gate_id} is free")

        # ── Penalty: gate not compatible with aircraft type ───────
        if not gate.compatible:
            reward -= 20
            log.append(f"-20 : Gate {gate_id} not size-compatible !!!")
        else:
            log.append(f"     Gate {gate_id} is size-compatible")

        # ── Penalty: path blockage by another plane ───────────────
        if gate.blocked_by:
            reward -= 20
            log.append(f"-20 : Taxi path to {gate_id} blocked by {gate.blocked_by} !!!")
        else:
            log.append(f"     Taxi path to {gate_id} is clear")

        # ── Rewards only if gate is fully valid ───────────────────
        valid = (not gate.occupied) and gate.compatible and (not gate.blocked_by)

        if valid:
            # Taxi distance bonus
            if gate.taxi_dist_min <= 5:
                reward += 5
                log.append(f"+5  : Short taxi ({gate.taxi_dist_min:.1f} min) -> fuel saved")
            elif gate.taxi_dist_min > 10:
                reward -= 8
                log.append(f"-8  : Long taxi ({gate.taxi_dist_min:.1f} min) -> detour penalty")
            else:
                log.append(f"     Moderate taxi ({gate.taxi_dist_min:.1f} min)")

            # Quick arrival + close gate = full reward
            if self.arriving.eta_min <= 12 and gate.taxi_dist_min <= 7:
                reward += 15
                log.append(f"+15 : Plane reaches gate quickly "
                            f"(ETA={self.arriving.eta_min:.1f}min, taxi={gate.taxi_dist_min:.1f}min)")
            elif self.arriving.eta_min > 15:
                reward -= 8
                log.append(f"-8  : Long ETA ({self.arriving.eta_min:.1f}min) -> plane must wait")

        return round(reward, 2), log

    def find_best_gate(self) -> str:
        """
        RL scoring policy: evaluate all gates and pick highest score.
        Score penalises bad gates and rewards close + quick assignments.
        """
        best_id    = None
        best_score = float("-inf")

        for gate in self.gates:
            score = 0.0
            if gate.occupied:       score -= 10
            if not gate.compatible: score -= 10
            if gate.blocked_by:     score -= 10
            score -= gate.taxi_dist_min
            if self.arriving.eta_min <= 12 and gate.taxi_dist_min <= 7:
                score += 20
            if score > best_score:
                best_score = score
                best_id    = gate.gate_id

        return best_id


def run_task5():
    print("\n" + "=" * 60)
    print("TASK 5 — GATE / STAND ASSIGNMENT")
    print("=" * 60)

    gate_pool = [
        Gate("A1", occupied=False, taxi_dist_min=4.0,  compatible=True,  blocked_by=None),
        Gate("A2", occupied=True,  taxi_dist_min=3.5,  compatible=True,  blocked_by=None),
        Gate("B1", occupied=False, taxi_dist_min=12.0, compatible=True,  blocked_by=None),
        Gate("B2", occupied=False, taxi_dist_min=6.0,  compatible=False, blocked_by=None),
        Gate("C1", occupied=False, taxi_dist_min=5.5,  compatible=True,  blocked_by="DAL440"),
        Gate("C2", occupied=False, taxi_dist_min=7.5,  compatible=True,  blocked_by=None),
    ]

    arriving_planes = [
        ArrivingPlane("UAL101", eta_min=10.0),
        ArrivingPlane("SWA202", eta_min=18.0),
        ArrivingPlane("AAL303", eta_min=8.0),
    ]

    # Manual test cases to show different reward outcomes
    test_assignments = [
        ("UAL101", "A2"),   # occupied  -> bad
        ("UAL101", "C1"),   # blocked   -> bad
        ("UAL101", "B1"),   # long taxi -> suboptimal
        ("UAL101", "A1"),   # optimal
        ("SWA202", "C2"),   # moderate (long ETA)
        ("AAL303", "A1"),   # optimal for low ETA
    ]

    print("\n  Manual assignments:")
    for callsign, gate_id in test_assignments:
        plane = next(p for p in arriving_planes if p.callsign == callsign)
        env   = GateAssignmentEnv(
            arriving=plane,
            gates=[Gate(g.gate_id, g.occupied, g.taxi_dist_min, g.compatible, g.blocked_by)
                   for g in gate_pool]
        )
        reward, log = env.assign(gate_id)
        print(f"\n  {callsign} -> Gate {gate_id} (ETA={plane.eta_min}min)")
        for l in log:
            print(f"    {l}")
        print(f"  -> Reward: {reward:+.2f}")

    # RL agent auto-assigns
    print("\n  RL Agent auto-assignments:")
    for plane in arriving_planes:
        env = GateAssignmentEnv(
            arriving=plane,
            gates=[Gate(g.gate_id, g.occupied, g.taxi_dist_min, g.compatible, g.blocked_by)
                   for g in gate_pool]
        )
        best   = env.find_best_gate()
        reward, log = env.assign(best)
        print(f"\n  {plane.callsign} (ETA={plane.eta_min}min) -> RL chose Gate {best}")
        for l in log:
            print(f"    {l}")
        print(f"  -> Reward: {reward:+.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  ATC REINFORCEMENT LEARNING SIMULATOR")
    print("  All 5 Tasks — Reward / Penalty Logic")
    print("#" * 60)

    run_task1()
    run_task2()
    run_task3()
    run_task4()
    run_task5()

    print("\n" + "=" * 60)
    print("  All tasks completed.")
    print("=" * 60 + "\n")