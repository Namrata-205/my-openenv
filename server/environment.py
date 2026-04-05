"""
environment.py — ATC TRACON RL Environment.

Delegates all reward/penalty logic to graders.py (the canonical simulation module).
Wraps each of the 5 grader classes behind the OpenEnv reset() / step() / state() API.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

# ── Import everything from graders.py ───────────────────────────────────────
from graders import (
    # Task 1
    AircraftCategory, WakeTurbulenceEnv, REQUIRED_SEPARATION,
    # Task 2
    InboundFlight, sequence_flights,
    # Task 3
    Aircraft, EmergencyVectorEnv,
    # Task 4
    ConflictAircraftState, ConflictAlertEnv,
    # Task 5
    Gate, ArrivingPlane, GateAssignmentEnv,
)

from models import (
    ATCAction, ActionType,
    AircraftCategory as ModelAircraftCategory,
    AircraftState, AircraftStatus, EnvironmentState,
    GateState, RunwayState, StepResult, TaskType,
)


class ATCEnvironment:
    VERSION       = "1.0.0"
    MAX_STEPS_DEF = 30

    def __init__(self):
        self._task: TaskType               = TaskType.WAKE_TURBULENCE
        self._step: int                    = 0
        self._max_steps: int               = self.MAX_STEPS_DEF
        self._done: bool                   = False
        self._rng: random.Random           = random.Random()
        self._episode_rewards: List[float] = []
        self._info: Dict[str, Any]         = {}

        # Task-specific live objects (from graders.py)
        self._wake_env:     Optional[WakeTurbulenceEnv]  = None
        self._flights:      List[InboundFlight]          = []
        self._emerg_env:    Optional[EmergencyVectorEnv] = None
        self._conflict_env: Optional[ConflictAlertEnv]  = None
        self._gate_env:     Optional[GateAssignmentEnv] = None

        # Pydantic model snapshots for state()
        self._aircraft:  List[AircraftState] = []
        self._gates_raw: List[GateState]     = []
        self._runways:   List[RunwayState]   = []

    # ─────────────────────────────────────────
    # OpenEnv API
    # ─────────────────────────────────────────

    def reset(self, task: Optional[TaskType] = None,
              seed: Optional[int] = None,
              options: Dict[str, Any] = {}) -> EnvironmentState:
        self._rng = random.Random(seed)
        random.seed(seed)
        self._step            = 0
        self._done            = False
        self._info            = {}
        self._episode_rewards = []
        if isinstance(task, str):
            task = TaskType(task)
        self._task = task or self._rng.choice(list(TaskType))
        self._max_steps       = options.get("max_steps", self.MAX_STEPS_DEF)
        self._runways         = self._default_runways()

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
        info["episode_avg_reward"] = round(
            sum(self._episode_rewards) / len(self._episode_rewards), 4)

        done = self._step >= self._max_steps or info.get("terminal", False)
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

    # ─────────────────────────────────────────
    # Scenario builders
    # ─────────────────────────────────────────

    def _default_runways(self) -> List[RunwayState]:
        return [RunwayState(runway_id="28L", active=True),
                RunwayState(runway_id="28R", active=True)]

    def _build_wake(self):
        # Only use pairs that exist in graders.py REQUIRED_SEPARATION dict
        valid_pairs = list(REQUIRED_SEPARATION.keys())
        lead_cat, trail_cat = self._rng.choice(valid_pairs)
        init_sep = self._rng.uniform(3.5, 6.5)
        self._wake_env  = WakeTurbulenceEnv(leading_cat=lead_cat,
                                            trailing_cat=trail_cat,
                                            current_sep=init_sep)
        self._aircraft  = self._aircraft_from_wake(self._wake_env)
        self._gates_raw = []

    def _build_go_around(self):
        callsigns = ["UAL101", "DAL202", "AAL303", "SWA404", "FDX505"]
        n = self._rng.randint(4, 5)
        self._flights = [
            InboundFlight(
                callsign=callsigns[i],
                eta_min=round(self._rng.uniform(5, 25), 1),
                fuel_lbs=round(self._rng.uniform(2000, 8000), 0),
                priority=self._rng.randint(1, 3),
            )
            for i in range(n)
        ]
        self._aircraft  = self._aircraft_from_flights(self._flights)
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
        emerg = Aircraft("EMER1", 0, 5000, 180, 15.0, 0.0)
        self._emerg_env = EmergencyVectorEnv(emergency=emerg, traffic=traffic)
        self._aircraft  = self._aircraft_from_emergency(self._emerg_env)
        self._gates_raw = []

    def _build_conflict(self):
        ac1 = ConflictAircraftState(
            "AC1", 0.0, 0.0,
            self._rng.uniform(60, 120),
            self._rng.uniform(220, 280),
            target_hdg=90.0,
        )
        ac2 = ConflictAircraftState(
            "AC2",
            self._rng.uniform(4, 8),
            self._rng.uniform(-2, 2),
            self._rng.uniform(240, 300),
            self._rng.uniform(220, 280),
            target_hdg=270.0,
        )
        self._conflict_env = ConflictAlertEnv(ac1=ac1, ac2=ac2)
        self._aircraft     = self._aircraft_from_conflict(self._conflict_env)
        self._gates_raw    = []

    def _build_gate(self):
        gate_pool = [
            Gate("A1", occupied=False, taxi_dist_min=4.0,  compatible=True,  blocked_by=None),
            Gate("A2", occupied=True,  taxi_dist_min=3.5,  compatible=True,  blocked_by=None),
            Gate("B1", occupied=False, taxi_dist_min=12.0, compatible=True,  blocked_by=None),
            Gate("B2", occupied=False, taxi_dist_min=6.0,  compatible=False, blocked_by=None),
            Gate("C1", occupied=False, taxi_dist_min=5.5,  compatible=True,  blocked_by="DAL440"),
            Gate("C2", occupied=False, taxi_dist_min=7.5,  compatible=True,  blocked_by=None),
        ]
        arriving = ArrivingPlane("UAL101",
                                 eta_min=round(self._rng.uniform(5, 18), 1))
        self._gate_env  = GateAssignmentEnv(arriving=arriving, gates=gate_pool)
        self._aircraft  = []
        self._gates_raw = self._gates_from_env(self._gate_env)

    # ─────────────────────────────────────────
    # Step dispatchers
    # ─────────────────────────────────────────

    def _step_wake(self, actions: List[ATCAction]) -> Tuple[float, float, List[str], Dict]:
        action_str = self._action_to_wake_str(actions)
        raw_reward, desc = self._wake_env.step(action_str)
        norm = self._normalise(raw_reward, low=-35, high=20)
        self._aircraft = self._aircraft_from_wake(self._wake_env)
        violations = ["LOSS OF SEPARATION"] if self._wake_env.violated else []
        info = {
            "raw_reward":  raw_reward,
            "description": desc,
            "sep_nm":      round(self._wake_env.current_sep, 2),
            "required_nm": self._wake_env.required_sep,
            "violated":    self._wake_env.violated,
            "action":      action_str,
            "terminal":    self._wake_env.violated,
        }
        return norm, norm, violations, info

    def _step_go_around(self, actions: List[ATCAction]) -> Tuple[float, float, List[str], Dict]:
        strategy = self._action_to_strategy(actions)
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
        go_arounds = [f.callsign for f in self._flights if f.went_around]
        violations = [f"GO-AROUND: {cs}" for cs in go_arounds]
        info = {
            "raw_reward": raw_reward,
            "strategy":   strategy,
            "events":     events,
            "stats":      stats,
            "terminal":   stats["go_arounds"] == n,
        }
        return norm, norm, violations, info

    def _step_emergency(self, actions: List[ATCAction]) -> Tuple[float, float, List[str], Dict]:
        heading, altitude, insert_time = self._action_to_vector(actions)
        raw_reward, log = self._emerg_env.insert_emergency(heading, altitude, insert_time)
        n_traffic = len(self._emerg_env.traffic)
        norm = self._normalise(raw_reward, low=-22 * n_traffic - 12, high=22)
        self._aircraft = self._aircraft_from_emergency(self._emerg_env)
        violations = [l for l in log if "SAFETY VIOLATION" in l]
        info = {
            "raw_reward":  raw_reward,
            "log":         log,
            "heading":     heading,
            "altitude":    altitude,
            "insert_time": insert_time,
            "terminal":    raw_reward >= 18,
        }
        return norm, norm, violations, info

    def _step_conflict(self, actions: List[ATCAction]) -> Tuple[float, float, List[str], Dict]:
        action_str = self._action_to_conflict_str(actions)
        raw_reward, desc = self._conflict_env.step(action_str)
        norm = self._normalise(raw_reward, low=-20, high=20)
        self._aircraft = self._aircraft_from_conflict(self._conflict_env)
        sep = self._conflict_env._separation()
        violations = [f"CRITICAL PROXIMITY: {sep:.2f} NM"] if sep < 5.0 else []
        info = {
            "raw_reward":  raw_reward,
            "description": desc,
            "separation":  round(sep, 2),
            "action":      action_str,
            "terminal":    sep < 3.0,
        }
        return norm, norm, violations, info

    def _step_gate(self, actions: List[ATCAction]) -> Tuple[float, float, List[str], Dict]:
        gate_id = self._action_to_gate_id(actions)
        if gate_id is None:
            gate_id = self._gate_env.find_best_gate()
        raw_reward, log = self._gate_env.assign(gate_id)
        norm = self._normalise(raw_reward, low=-60, high=20)
        self._gates_raw = self._gates_from_env(self._gate_env)
        violations = [l.strip() for l in log if "!!!" in l]
        info = {
            "raw_reward":    raw_reward,
            "log":           log,
            "gate_assigned": gate_id,
            "terminal":      self._gate_env.assigned_gate is not None,
        }
        return norm, norm, violations, info

    # ─────────────────────────────────────────
    # Action translators
    # ─────────────────────────────────────────

    def _action_to_wake_str(self, actions: List[ATCAction]) -> str:
        for a in actions:
            if a.action_type == ActionType.SPEED_CHANGE and a.value is not None:
                return "slow_down_trailing" if a.value < 0 else "speed_up_trailing"
            if a.action_type == ActionType.HEADING_CHANGE:
                return "increase_heading_gap"
            if a.action_type == ActionType.NO_ACTION:
                return "hold"
        return "hold"

    def _action_to_strategy(self, actions: List[ATCAction]) -> str:
        for a in actions:
            if a.action_type == ActionType.SEQUENCE_SWAP:
                hint = (a.rationale or "").lower()
                if "fuel" in hint: return "fuel_priority"
                if "eta"  in hint: return "eta_optimized"
                if "fcfs" in hint: return "fcfs"
                return "rl_agent"
        return "rl_agent"

    def _action_to_vector(self, actions: List[ATCAction]) -> Tuple[float, float, float]:
        heading, altitude = 250.0, 2000.0
        for a in actions:
            if a.action_type == ActionType.VECTOR and a.value is not None:
                heading = float(a.value) % 360
            elif a.action_type == ActionType.HEADING_CHANGE and a.value is not None:
                heading = float(a.value) % 360
            elif a.action_type == ActionType.ALTITUDE_CHANGE and a.value is not None:
                altitude = max(500.0, float(a.value))
        insert_time = max(0.5, min(5.0, 1.0 + len(actions) * 0.3))
        return heading, altitude, insert_time

    def _action_to_conflict_str(self, actions: List[ATCAction]) -> str:
        for a in actions:
            if a.action_type == ActionType.HEADING_CHANGE and a.value is not None:
                return "left_10" if a.value < 0 else "right_10"
            if a.action_type == ActionType.SPEED_CHANGE and a.value is not None:
                return "slow_10" if a.value < 0 else "speed_10"
        return "right_10"

    def _action_to_gate_id(self, actions: List[ATCAction]) -> Optional[str]:
        for a in actions:
            if a.action_type == ActionType.ASSIGN_GATE and a.gate_id:
                return a.gate_id
        return None

    # ─────────────────────────────────────────
    # Converters: grader objects → Pydantic models
    # ─────────────────────────────────────────

    def _aircraft_from_wake(self, env: WakeTurbulenceEnv) -> List[AircraftState]:
        _map = {
            AircraftCategory.HEAVY:  ModelAircraftCategory.HEAVY,
            AircraftCategory.MEDIUM: ModelAircraftCategory.LARGE,
            AircraftCategory.LIGHT:  ModelAircraftCategory.SMALL,
        }
        return [
            AircraftState(callsign="LEAD",  category=_map[env.leading_cat],
                          status=AircraftStatus.APPROACH,
                          x=0.0, y=env.current_sep, altitude=3000,
                          heading=180, speed=160),
            AircraftState(callsign="TRAIL", category=_map[env.trailing_cat],
                          status=AircraftStatus.APPROACH,
                          x=0.0, y=0.0, altitude=3000,
                          heading=180, speed=150),
        ]

    def _aircraft_from_flights(self, flights: List[InboundFlight]) -> List[AircraftState]:
        return [
            AircraftState(
                callsign=f.callsign,
                category=ModelAircraftCategory.LARGE,
                status=(AircraftStatus.LANDED     if f.landed
                        else AircraftStatus.GO_AROUND if f.went_around
                        else AircraftStatus.APPROACH),
                x=float(i) * 2, y=f.eta_min,
                altitude=3000, heading=180,
                speed=max(80.0, 160.0 - f.holding_min * 2),
                sequence_pos=i + 1,
                fuel_state=min(1.0, f.fuel_lbs / 8000.0),
            )
            for i, f in enumerate(flights)
        ]

    def _aircraft_from_emergency(self, env: EmergencyVectorEnv) -> List[AircraftState]:
        ac_list = [
            AircraftState(
                callsign=env.emergency.callsign,
                category=ModelAircraftCategory.LARGE,
                status=AircraftStatus.EMERGENCY,
                x=env.emergency.lon, y=env.emergency.lat,
                altitude=env.emergency.altitude,
                heading=env.emergency.heading,
                speed=env.emergency.speed,
                is_emergency=True, fuel_state=0.1,
            ),
        ]
        for ac in env.traffic:
            ac_list.append(AircraftState(
                callsign=ac.callsign, category=ModelAircraftCategory.LARGE,
                status=AircraftStatus.APPROACH,
                x=ac.lon, y=ac.lat,
                altitude=ac.altitude, heading=ac.heading, speed=ac.speed,
            ))
        return ac_list

    def _aircraft_from_conflict(self, env: ConflictAlertEnv) -> List[AircraftState]:
        return [
            AircraftState(
                callsign=env.ac1.callsign, category=ModelAircraftCategory.LARGE,
                status=AircraftStatus.ENROUTE,
                x=env.ac1.x, y=env.ac1.y,
                altitude=5000, heading=env.ac1.heading_deg,
                speed=env.ac1.speed_kts,
            ),
            AircraftState(
                callsign=env.ac2.callsign, category=ModelAircraftCategory.LARGE,
                status=AircraftStatus.ENROUTE,
                x=env.ac2.x, y=env.ac2.y,
                altitude=5000, heading=env.ac2.heading_deg,
                speed=env.ac2.speed_kts,
            ),
        ]

    def _gates_from_env(self, env: GateAssignmentEnv) -> List[GateState]:
        return [
            GateState(
                gate_id=g.gate_id,
                occupied=g.occupied,
                aircraft=g.blocked_by,
                gate_type=(ModelAircraftCategory.HEAVY if g.compatible
                           else ModelAircraftCategory.SMALL),
            )
            for g in env.gates
        ]

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    @staticmethod
    def _normalise(value: float, low: float, high: float) -> float:
        """Linearly scale raw grader reward into [0.0, 1.0]."""
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
        conflicts = []
        acs = self._aircraft
        for i in range(len(acs)):
            for j in range(i + 1, len(acs)):
                a, b = acs[i], acs[j]
                lateral  = math.hypot(a.x - b.x, a.y - b.y)
                vertical = abs(a.altitude - b.altitude)
                if lateral < 3.0 and vertical < 1000:
                    conflicts.append({
                        "ac1": a.callsign, "ac2": b.callsign,
                        "lateral_nm": round(lateral, 2),
                        "vertical_ft": round(vertical, 0),
                    })
        return conflicts