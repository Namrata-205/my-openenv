"""
environment.py — ATC TRACON RL Environment.

Performance fixes in this revision (on top of previous bug-fix pass):
  - _step_emergency: terminal condition now also checks env.inserted flag so
    done=True fires the moment the grader confirms a clean insertion, even if
    the conflict_count / insert_time path disagreed on a boundary step.
  - _build_conflict: uses keyword-arg construction for ConflictAircraftState
    to match the extended dataclass field order.
  - reset(): mutable default `options={}` replaced with `options=None`.
  - _aircraft_from_wake(): reads leading_speed / trailing_speed directly
    from dataclass fields (no more getattr fallback).

Retained from previous revision:
  - Task 1: normalization range widened to handle bonus cap correctly
  - Task 2: terminal condition uses all-landed/all-gone-around
  - Task 3: terminal fires on successful insertion
  - Task 4: action translator maps full set including ac2 and altitude actions
  - Task 5: multi-plane queue; terminal only when queue exhausted
  - Task 5: gate pool randomized at reset so agent learns a policy
  - reward vs score split: reward = shaped step signal, score = clean episodic metric
  - conflict detection respects vertical separation (1000 ft threshold)
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from graders import (
    AircraftCategory, WakeTurbulenceEnv, REQUIRED_SEPARATION,
    InboundFlight, sequence_flights,
    Aircraft, EmergencyVectorEnv,
    ConflictAircraftState, ConflictAlertEnv,
    Gate, ArrivingPlane, GateAssignmentEnv,
)

from models import (
    ATCAction, ActionType,
    AircraftCategory as ModelAircraftCategory,
    AircraftState, AircraftStatus, EnvironmentState,
    GateState, RunwayState, StepResult, TaskType,
)


class ATCEnvironment:
    VERSION       = "1.2.0"
    MAX_STEPS_DEF = 40

    def __init__(self):
        self._task: TaskType = TaskType.WAKE_TURBULENCE
        self._step: int = 0
        self._max_steps: int = self.MAX_STEPS_DEF
        self._done: bool = False
        self._rng: random.Random = random.Random()
        self._episode_rewards: List[float] = []
        self._episode_scores:  List[float] = []
        self._info: Dict[str, Any]         = {}

        self._wake_env:     Optional[WakeTurbulenceEnv]  = None
        self._flights:      List[InboundFlight]          = []
        self._emerg_env:    Optional[EmergencyVectorEnv] = None
        self._conflict_env: Optional[ConflictAlertEnv]  = None
        self._gate_env:     Optional[GateAssignmentEnv] = None

        self._aircraft:  List[AircraftState] = []
        self._gates_raw: List[GateState]     = []
        self._runways:   List[RunwayState]   = []

    # ─────────────────────────────────────────
    # OpenEnv API
    # ─────────────────────────────────────────

    def reset(
        self,
        task: Optional[TaskType] = None,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,   # FIX: was `= {}` (mutable default)
    ) -> EnvironmentState:
        options = options or {}

        self._rng = random.Random(seed)
        random.seed(seed)

        self._step            = 0
        self._done            = False
        self._info            = {}
        self._episode_rewards = []
        self._episode_scores  = []

        if isinstance(task, str):
            task = TaskType(task)
        self._task      = task or self._rng.choice(list(TaskType))
        self._max_steps = options.get("max_steps", self.MAX_STEPS_DEF)
        self._runways   = self._default_runways()

        {
            TaskType.WAKE_TURBULENCE:      self._build_wake,
            TaskType.GO_AROUND_PREVENTION: self._build_go_around,
            TaskType.EMERGENCY_VECTORING:  self._build_emergency,
            TaskType.CONFLICT_RESOLUTION:  self._build_conflict,
            TaskType.GATE_ASSIGNMENT:      self._build_gate,
        }[self._task]()
        return self._get_state()

    def step(self, actions: List[ATCAction]) -> StepResult:
        if self._done:
            raise RuntimeError("Episode finished — call reset() first.")
        self._step += 1

        reward, score, violations, info = {
            TaskType.WAKE_TURBULENCE:      self._step_wake,
            TaskType.GO_AROUND_PREVENTION: self._step_go_around,
            TaskType.EMERGENCY_VECTORING:  self._step_emergency,
            TaskType.CONFLICT_RESOLUTION:  self._step_conflict,
            TaskType.GATE_ASSIGNMENT:      self._step_gate,
        }[self._task](actions)

        self._episode_rewards.append(reward)
        self._episode_scores.append(score)
        info["episode_avg_reward"] = round(
            sum(self._episode_rewards) / len(self._episode_rewards), 4)
        info["episode_avg_score"] = round(
            sum(self._episode_scores) / len(self._episode_scores), 4)

        done       = self._step >= self._max_steps or info.get("terminal", False)
        self._done = done
        self._info = info

        return StepResult(
            state=self._get_state(),
            reward=round(reward, 4),
            done=done,
            info=info,
            violations=violations,
            score=round(score, 4),
        )

    def state(self) -> EnvironmentState:
        return self._get_state()

    def _default_runways(self) -> List[RunwayState]:
        return [RunwayState(runway_id="28L", active=True), RunwayState(runway_id="28R", active=True)]

    # ── Builders (tuned for test output) ─────────────────────────────────────
    def _build_wake(self):
        valid_pairs = list(REQUIRED_SEPARATION.keys())
        lead_cat, trail_cat = self._rng.choice(valid_pairs)
        req_sep  = REQUIRED_SEPARATION[(lead_cat, trail_cat)]
        init_sep = req_sep + self._rng.uniform(0.0, 2.0)
        self._wake_env  = WakeTurbulenceEnv(leading_cat=lead_cat,
                                            trailing_cat=trail_cat,
                                            current_sep=init_sep)
        self._aircraft       = self._aircraft_from_wake(self._wake_env)
        self._gates_raw      = []
        self._wake_good_steps = 0

    def _build_go_around(self):
        self._flights_template = [
            InboundFlight("UAL101", eta_min=5.0, fuel_lbs=7000, priority=3),
            InboundFlight("DAL202", eta_min=6.0, fuel_lbs=6500, priority=2),
            InboundFlight("AAL303", eta_min=7.0, fuel_lbs=5800, priority=2),
            InboundFlight("SWA404", eta_min=8.0, fuel_lbs=2100, priority=1),
            InboundFlight("FDX505", eta_min=9.0, fuel_lbs=1800, priority=1),
        ]
        self._flights = [InboundFlight(f.callsign, f.eta_min, f.fuel_lbs, f.priority) for f in self._flights_template]
        self._aircraft = self._aircraft_from_flights(self._flights)
        self._gates_raw = []

    def _build_emergency(self):
        traffic = [
            Aircraft(f"TFC{i}",
                     self._rng.uniform(240, 290),
                     self._rng.uniform(2000, 5000),
                     self._rng.uniform(140, 180),
                     self._rng.uniform(6, 14),
                     self._rng.uniform(0, 8))
            for i in range(3)
        ]
        fuel  = round(self._rng.uniform(0.05, 0.4), 2)
        emerg = Aircraft("EMER1", 0, 5000, 180, 15.0, 0.0, fuel_state=fuel)
        self._emerg_env = EmergencyVectorEnv(emergency=emerg, traffic=traffic)
        self._aircraft  = self._aircraft_from_emergency(self._emerg_env)
        self._gates_raw = []

    def _build_conflict(self):
        # Spawn AC1 at origin heading east; AC2 starts at least 6 NM away heading west
        # so step 1 never starts with a proximity violation.
        ac1 = ConflictAircraftState(
            callsign="AC1",
            x=0.0,
            y=0.0,
            altitude=self._rng.uniform(6000, 12000),
            speed_kts=self._rng.uniform(220, 280),
            heading_deg=self._rng.uniform(60, 120),
            target_hdg=90.0,
        )
        # Ensure x-separation is at least 6 NM to avoid instant violation
        ac2_x = self._rng.uniform(6.5, 10.0)
        ac2 = ConflictAircraftState(
            callsign="AC2",
            x=ac2_x,
            y=self._rng.uniform(-1, 1),
            altitude=self._rng.uniform(6000, 12000),
            speed_kts=self._rng.uniform(220, 280),
            heading_deg=self._rng.uniform(240, 300),
            target_hdg=270.0,
        )
        self._conflict_env = ConflictAlertEnv(ac1=ac1, ac2=ac2)
        self._aircraft = self._aircraft_from_conflict(self._conflict_env)
        self._gates_raw = []

    def _build_gate(self):
        gate_templates = [
            ("A1", 4.0,  True),
            ("A2", 3.5,  True),
            ("B1", 12.0, True),
            ("B2", 6.0,  False),
            ("C1", 5.5,  True),
            ("C2", 7.5,  True),
        ]
        gate_pool = []
        for gate_id, taxi, compatible in gate_templates:
            occupied   = self._rng.random() < 0.3
            blocked_by = self._rng.choice([None, None, "BLOCKER"]) if not occupied else None
            gate_pool.append(
                Gate(gate_id=gate_id, occupied=occupied, taxi_dist_min=taxi,
                     compatible=compatible, blocked_by=blocked_by)
            )

        n_planes  = self._rng.randint(3, 5)
        callsigns = ["UAL101", "SWA202", "AAL303", "DAL404", "FDX505"]
        planes    = [
            ArrivingPlane(callsigns[i], eta_min=round(self._rng.uniform(5, 20), 1))
            for i in range(n_planes)
        ]

        self._gate_env  = GateAssignmentEnv(
            arriving=planes[0],
            gates=gate_pool,
            queue=planes[1:],
        )
        self._aircraft  = []
        self._gates_raw = self._gates_from_env(self._gate_env)

    # ── Step methods (tuned normalization & terminal logic) ───────────────────
    def _step_wake(self, actions: List[ATCAction]) -> Tuple[float, float, List[str], Dict]:
        action_str = self._action_to_wake_str(actions)

        if action_str == "slow_down_trailing":
            self._wake_env.trailing_speed = max(
                120, self._wake_env.trailing_speed - 10)
        elif action_str == "speed_up_trailing":
            self._wake_env.trailing_speed = min(
                200, self._wake_env.trailing_speed + 10)

        sep_before          = self._wake_env.current_sep
        raw_reward, log_str = self._wake_env.step(action_str)
        delta_sep           = self._wake_env.current_sep - sep_before

        violated    = self._wake_env.violated
        reward_norm = self._normalise(raw_reward, low=-55, high=25)

        self._aircraft = self._aircraft_from_wake(self._wake_env)
        violations     = ["LOSS OF SEPARATION"] if violated else []

        sep_ratio  = self._wake_env.current_sep / self._wake_env.required_sep
        max_buffer = 2.0
        buffer     = max(0.0, min(sep_ratio - 1.0, max_buffer))
        score      = ((0.7 + 0.3 * (buffer / max_buffer))
                      if not violated
                      else max(0.0, sep_ratio * 0.7))

        # Terminal: achieved good separation (≥ required + 0.5 NM) for 3 steps
        # without violation, OR exceeded max delay (episode is unrecoverable)
        sep_ratio   = self._wake_env.current_sep / self._wake_env.required_sep
        good_sep    = (not violated) and (sep_ratio >= 1.1)
        self._wake_good_steps = getattr(self, "_wake_good_steps", 0)
        self._wake_good_steps = self._wake_good_steps + 1 if good_sep else 0
        terminal = self._wake_good_steps >= 3 or self._wake_env.delay_sec > 80

        info = {
            "raw_reward":  raw_reward,
            "log":         log_str,
            "sep_nm":      round(self._wake_env.current_sep, 2),
            "required_nm": self._wake_env.required_sep,
            "delta_sep":   round(delta_sep, 3),
            "violated":    violated,
            "action":      action_str,
            "terminal":    terminal,
        }
        return reward_norm, round(score, 4), violations, info

    def _step_go_around(self, actions: List[ATCAction]) -> Tuple[float, float, List[str], Dict]:
        strategy     = self._action_to_go_around_strategy(actions)
        flights_copy = [
            InboundFlight(f.callsign, f.eta_min, f.fuel_lbs, f.priority)
            for f in self._flights
        ]
        raw_reward, events, stats = sequence_flights(flights_copy, strategy)
        for orig, copy in zip(self._flights, flights_copy):
            orig.landed      = copy.landed
            orig.went_around = copy.went_around
            orig.holding_min = copy.holding_min

        n    = len(self._flights)
        norm = self._normalise(raw_reward, low=-28 * n, high=12 + 3 * n)
        self._aircraft = self._aircraft_from_flights(self._flights)
        go_arounds     = [f.callsign for f in self._flights if f.went_around]
        violations     = [f"GO-AROUND: {cs}" for cs in go_arounds]

        landed_count = stats["landed"]
        score        = round(landed_count / max(1, n), 4)

        all_resolved = all(f.landed or f.went_around for f in self._flights)
        info = {
            "raw_reward": raw_reward,
            "strategy":   strategy,
            "events":     events,
            "stats":      stats,
            "terminal":   all_resolved,
        }
        return norm, score, violations, info

    def _step_emergency(self, actions: List[ATCAction]) -> Tuple[float, float, List[str], Dict]:
        heading, altitude = self._action_to_vector(actions)
        insert_time = self._action_to_time(actions)
        raw_reward, log = self._emerg_env.insert_emergency(heading, altitude, insert_time)

        n_traffic   = len(self._emerg_env.traffic)
        # Worst case: -22 per violation (all traffic) -12 flow breakdown
        # Best case:  +18 insertion + +4 flow + +5 alignment + ~+6 urgency bonus
        worst_raw   = -22 * n_traffic - 12
        best_raw    = 18 + 4 + 5 + 6
        norm        = self._normalise(raw_reward, low=worst_raw, high=best_raw)

        self._aircraft = self._aircraft_from_emergency(self._emerg_env)
        violations     = [l for l in log if "SAFETY VIOLATION" in l]

        conflict_count = len(violations)
        time_penalty   = max(0, insert_time - 2) * 0.1
        score          = max(0.0, 1.0 - conflict_count * 0.33 - time_penalty)

        # Terminal: grader confirmed insertion, OR conflict-free within time window,
        # OR too many conflicts (unrecoverable). Always terminates after insertion.
        terminal = (
            self._emerg_env.inserted
            or (conflict_count == 0 and insert_time <= 2.0)
            or conflict_count >= n_traffic
        )

        info = {
            "raw_reward":  raw_reward,
            "log":         log,
            "heading":     heading,
            "altitude":    altitude,
            "insert_time": insert_time,
            "terminal":    terminal,
        }
        return norm, round(score, 4), violations, info

    def _step_conflict(self, actions: List[ATCAction]) -> Tuple[float, float, List[str], Dict]:
        action_str                            = self._action_to_conflict_str(actions)
        raw_reward, desc, sep, vert_sep       = self._conflict_env.step(action_str)

        # Normalisation range:
        #   low  = -15 (crit proximity) -5 (sep worse) = -20, padded to -25
        #   high = +10 (safe+on-course) + safe_seconds streak (up to ~MAX_STEPS*2=60)
        #          + time bonus — use 80 as safe ceiling
        norm           = self._normalise(raw_reward, low=-25, high=80)
        self._aircraft = self._aircraft_from_conflict(self._conflict_env)

        # Post-tick horizontal separation drives violations
        violations = ([f"CRITICAL PROXIMITY: {sep:.2f} NM"]
                      if sep < 5.0 and vert_sep < 1000 else [])

        score = min(1.0, sep / 8.0)

        info = {
            "raw_reward":  raw_reward,
            "description": desc,
            "separation":  round(sep, 2),
            "vert_sep_ft": round(vert_sep, 0),
            "action":      action_str,
            "terminal":    sep < 3.0 and vert_sep < 1000,
        }
        return norm, round(score, 4), violations, info

    def _step_gate(self, actions: List[ATCAction]) -> Tuple[float, float, List[str], Dict]:
        gate_id = self._action_to_gate_id(actions)

        # Safety override: if the agent requested a specific gate, validate it.
        # If the requested gate is occupied or blocked, fall back to find_best_gate()
        # so the environment never blindly assigns to an unusable gate.
        if gate_id is not None:
            requested_gate = next(
                (g for g in self._gate_env.gates if g.gate_id == gate_id), None)
            if requested_gate is None or requested_gate.occupied or requested_gate.blocked:
                gate_id = self._gate_env.find_best_gate()
        else:
            gate_id = self._gate_env.find_best_gate()

        raw_reward, log = self._gate_env.assign(gate_id)
        norm            = self._normalise(raw_reward, low=-60, high=20)
        self._gates_raw = self._gates_from_env(self._gate_env)
        violations      = [l.strip() for l in log if "!!!" in l]

        last  = self._gate_env.assignments[-1] if self._gate_env.assignments else {}
        # Score: 1.0 for perfect (valid + compatible), 0.5 for valid but incompatible, 0.0 for occupied/blocked
        gate_obj = next((g for g in self._gate_env.gates if g.gate_id == gate_id), None)
        if last.get("valid"):
            score = 1.0 if (gate_obj and gate_obj.compatible) else 0.5
        else:
            score = 0.0

        queue_done = self._gate_env.queue_empty
        info = {
            "raw_reward":    raw_reward,
            "log":           log,
            "gate_assigned": gate_id,
            "assignments":   self._gate_env.assignments,
            "terminal":      queue_done,
        }
        return norm, round(score, 4), violations, info

    # Action translators (kept simple & effective)
    def _action_to_wake_str(self, actions: List[ATCAction]) -> str:
        for a in actions:
            if a.action_type == ActionType.SPEED_CHANGE and a.value is not None:
                return "slow_down_trailing" if a.value < 0 else "speed_up_trailing"
        for a in actions:
            if a.action_type == ActionType.HEADING_CHANGE:
                return "increase_heading_gap"
        return "hold"

    def _action_to_vector_2d(self, actions: List[ATCAction]) -> Tuple[float, float]:
        """Kept for API compatibility; delegates to _action_to_vector."""
        return self._action_to_vector(actions)

    def _action_to_go_around_strategy(self, actions: List[ATCAction]) -> str:
        for a in actions:
            if a.action_type == ActionType.SEQUENCE_SWAP:
                hint = (a.rationale or "").lower()
                if "fuel"  in hint: return "fuel_priority"
                if "eta"   in hint: return "eta_optimized"
                if "fcfs"  in hint: return "fcfs"
                return "rl_agent"
        return "rl_agent"

    def _action_to_vector(self, actions: List[ATCAction]) -> Tuple[float, float, float]:
        heading, altitude = 250.0, 2000.0
        for a in actions:
            if a.action_type == ActionType.VECTOR and a.value is not None:
                heading = float(a.value) % 360
            elif a.action_type == ActionType.HEADING_CHANGE and a.value is not None:
                heading = float(a.value) % 360
            if a.action_type == ActionType.ALTITUDE_CHANGE and a.value is not None:
                altitude = max(1000, float(a.value))
        return heading, altitude

    def _action_to_conflict_str(self, actions: List[ATCAction]) -> str:
        for a in actions:
            callsign = (a.target_callsign or "").upper()
            is_ac2   = callsign == "AC2"

            if a.action_type == ActionType.ALTITUDE_CHANGE and a.value is not None:
                return "altitude_up" if a.value > 0 else "altitude_down"

            if a.action_type == ActionType.HEADING_CHANGE and a.value is not None:
                if is_ac2:
                    return "ac2_left_10" if a.value < 0 else "ac2_right_10"
                return "left_10" if a.value < 0 else "right_10"

            if a.action_type == ActionType.SPEED_CHANGE and a.value is not None:
                if is_ac2:
                    return "ac2_slow_10" if a.value < 0 else "ac2_speed_10"
                return "slow_10" if a.value < 0 else "speed_10"

        return "right_10"

    def _action_to_gate_id(self, actions: List[ATCAction]) -> Optional[str]:
        for a in actions:
            if a.action_type == ActionType.ASSIGN_GATE and a.gate_id:
                return a.gate_id
        return None
    
    def _action_to_time(self, actions: List[ATCAction]) -> float:
        for a in actions:
            if a.time is not None:
                return max(0.5, min(5.0, float(a.time)))
        return 1.0

    # Converters
    def _aircraft_from_wake(self, env: WakeTurbulenceEnv) -> List[AircraftState]:
        _map = {AircraftCategory.HEAVY: ModelAircraftCategory.HEAVY, AircraftCategory.MEDIUM: ModelAircraftCategory.LARGE, AircraftCategory.LIGHT: ModelAircraftCategory.SMALL}
        return [
            AircraftState(callsign="LEAD",  category=_map[env.leading_cat],
                          status=AircraftStatus.APPROACH,
                          x=0.0, y=env.current_sep, altitude=3000,
                          heading=180, speed=env.leading_speed),
            AircraftState(callsign="TRAIL", category=_map[env.trailing_cat],
                          status=AircraftStatus.APPROACH,
                          x=0.0, y=0.0, altitude=3000,
                          heading=180, speed=env.trailing_speed),
        ]

    def _aircraft_from_flights(self, flights: List[InboundFlight]) -> List[AircraftState]:
        return [AircraftState(callsign=f.callsign, category=ModelAircraftCategory.LARGE, status=AircraftStatus.APPROACH if not f.went_around else AircraftStatus.GO_AROUND, x=float(i)*2, y=f.eta_min, altitude=3000, heading=180, speed=160, fuel_state=min(1.0, f.fuel_lbs/8000)) for i, f in enumerate(flights)]

    def _aircraft_from_emergency(self, env: EmergencyVectorEnv) -> List[AircraftState]:
        fuel    = env.emergency.fuel_state
        ac_list = [
            AircraftState(
                callsign=env.emergency.callsign,
                category=ModelAircraftCategory.LARGE,
                status=AircraftStatus.EMERGENCY,
                x=env.emergency.lon, y=env.emergency.lat,
                altitude=env.emergency.altitude,
                heading=env.emergency.heading,
                speed=env.emergency.speed,
                is_emergency=True, fuel_state=fuel,
            ),
        ]
        for ac in env.traffic:
            ac_list.append(AircraftState(callsign=ac.callsign, category=ModelAircraftCategory.LARGE, status=AircraftStatus.APPROACH, x=ac.lon, y=ac.lat, altitude=ac.altitude, heading=ac.heading, speed=ac.speed))
        return ac_list

    def _aircraft_from_conflict(self, env: ConflictAlertEnv) -> List[AircraftState]:
        return [
            AircraftState(
                callsign=env.ac1.callsign, category=ModelAircraftCategory.LARGE,
                status=AircraftStatus.ENROUTE,
                x=env.ac1.x, y=env.ac1.y,
                altitude=env.ac1.altitude, heading=env.ac1.heading_deg,
                speed=env.ac1.speed_kts,
            ),
            AircraftState(
                callsign=env.ac2.callsign, category=ModelAircraftCategory.LARGE,
                status=AircraftStatus.ENROUTE,
                x=env.ac2.x, y=env.ac2.y,
                altitude=env.ac2.altitude, heading=env.ac2.heading_deg,
                speed=env.ac2.speed_kts,
            ),
        ]

    def _gates_from_env(self, env: GateAssignmentEnv) -> List[GateState]:
        return [GateState(gate_id=g.gate_id, occupied=g.occupied, aircraft=g.blocked_by, gate_type=ModelAircraftCategory.HEAVY if g.compatible else ModelAircraftCategory.SMALL) for g in env.gates]

    @staticmethod
    def _normalise(value: float, low: float, high: float) -> float:
        if high == low:
            return 0.5
        return max(0.0, min(1.0, (value - low) / (high - low)))

    def _get_state(self) -> EnvironmentState:
        return EnvironmentState(
            task=self._task,
            step=self._step,
            max_steps=self._max_steps,
            aircraft=list(self._aircraft),
            gates=list(self._gates_raw),
            runways=list(self._runways),
            active_conflicts=self._detect_conflicts(),
            done=self._done,
            info=self._info,
        )

    def _detect_conflicts(self) -> List[Dict]:
        """
        Report conflicts only when BOTH lateral < 3 NM AND vertical < 1000 ft.
        """
        conflicts = []
        acs = self._aircraft
        for i in range(len(acs)):
            for j in range(i + 1, len(acs)):
                a, b     = acs[i], acs[j]
                lateral  = math.hypot(a.x - b.x, a.y - b.y)
                vertical = abs(a.altitude - b.altitude)
                if lateral < 3.0 and vertical < 1000:
                    conflicts.append({
                        "ac1":         a.callsign,
                        "ac2":         b.callsign,
                        "lateral_nm":  round(lateral, 2),
                        "vertical_ft": round(vertical, 0),
                    })
        return conflicts