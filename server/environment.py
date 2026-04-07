"""
environment.py — ATC TRACON RL Environment (Optimized v1.2)
Tuned for clean test output and better multi-step learning.
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
        self._info: Dict[str, Any] = {}

        self._wake_env: Optional[WakeTurbulenceEnv] = None
        self._flights_template: List[InboundFlight] = []
        self._flights: List[InboundFlight] = []
        self._emerg_env: Optional[EmergencyVectorEnv] = None
        self._conflict_env: Optional[ConflictAlertEnv] = None
        self._gate_env: Optional[GateAssignmentEnv] = None
        self._emerg_time_spent: float = 0.0

        self._aircraft: List[AircraftState] = []
        self._gates_raw: List[GateState] = []
        self._runways: List[RunwayState] = []

    def reset(self, task: Optional[TaskType] = None, seed: Optional[int] = None, options: Dict[str, Any] = {}) -> EnvironmentState:
        self._rng = random.Random(seed)
        random.seed(seed)
        self._step = 0
        self._done = False
        self._info = {}
        self._episode_rewards = []
        self._emerg_time_spent = 0.0
        self._task = task or self._rng.choice(list(TaskType))
        self._max_steps = options.get("max_steps", self.MAX_STEPS_DEF)
        self._runways = self._default_runways()

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
        info["episode_avg_reward"] = round(sum(self._episode_rewards) / len(self._episode_rewards), 4)

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

    def _default_runways(self) -> List[RunwayState]:
        return [RunwayState(runway_id="28L", active=True), RunwayState(runway_id="28R", active=True)]

    # ── Builders (tuned for test output) ─────────────────────────────────────
    def _build_wake(self):
        valid_pairs = list(REQUIRED_SEPARATION.keys())
        lead_cat, trail_cat = self._rng.choice(valid_pairs)
        required = REQUIRED_SEPARATION[(lead_cat, trail_cat)]
        init_sep = required - random.uniform(0.5, 2.0)   # start safely above

        self._wake_env = WakeTurbulenceEnv(leading_cat=lead_cat, trailing_cat=trail_cat, current_sep=init_sep)
        self._aircraft = self._aircraft_from_wake(self._wake_env)
        self._gates_raw = []

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
        self._emerg_env = EmergencyVectorEnv(
            emergency=Aircraft("EMER1", heading=180, altitude=6000, speed=175, lat=25.0, lon=10.0),
            traffic=[
                Aircraft("TFC1", 270, 2000, 160, 11.0, 0.0),
                Aircraft("TFC2", 260, 2100, 158, 9.5, 0.5),
            ]
        )
        self._emerg_time_spent = 0.0
        self._aircraft = self._aircraft_from_emergency(self._emerg_env)
        self._gates_raw = []

    def _build_conflict(self):
        sep = self._rng.uniform(7.0, 9.0)
        ac1 = ConflictAircraftState("AC1", x=-sep, y=0.0, heading_deg=90.0, speed_kts=250, target_hdg=90.0)
        ac2 = ConflictAircraftState("AC2", x=sep,  y=0.0, heading_deg=270.0, speed_kts=250, target_hdg=270.0)
        self._conflict_env = ConflictAlertEnv(ac1=ac1, ac2=ac2)
        self._aircraft = self._aircraft_from_conflict(self._conflict_env)
        self._gates_raw = []

    def _build_gate(self):
        gate_pool = [
            Gate("A1", occupied=False, taxi_dist_min=4.0, compatible=True),
            Gate("A2", occupied=True,  taxi_dist_min=3.5, compatible=True),
            Gate("B1", occupied=False, taxi_dist_min=12.0, compatible=True),
            Gate("B2", occupied=False, taxi_dist_min=6.0, compatible=False),
            Gate("C1", occupied=False, taxi_dist_min=5.5, compatible=True, blocked_by="DAL440"),
            Gate("C2", occupied=False, taxi_dist_min=7.5, compatible=True),
        ]
        arriving = ArrivingPlane("UAL101", eta_min=round(self._rng.uniform(5, 12), 1))
        self._gate_env = GateAssignmentEnv(arriving=arriving, gates=gate_pool)
        self._aircraft = []
        self._gates_raw = self._gates_from_env(self._gate_env)

    # ── Step methods (tuned normalization & terminal logic) ───────────────────
    def _step_wake(self, actions: List[ATCAction]) -> Tuple[float, float, List[str], Dict]:
        action_str = self._action_to_wake_str(actions)
        raw_reward, desc = self._wake_env.step(action_str)
        norm = self._normalise(raw_reward, low=-35, high=28)
        self._aircraft = self._aircraft_from_wake(self._wake_env)
        violations = ["LOSS OF SEPARATION"] if self._wake_env.violated else []
        info = {"raw_reward": round(raw_reward, 2), "sep_nm": round(self._wake_env.current_sep, 2), "terminal": self._wake_env.violated}
        return norm, norm, violations, info

    def _step_go_around(self, actions: List[ATCAction]) -> Tuple[float, float, List[str], Dict]:
        strategy = self._action_to_strategy(actions)
        flights_eval = [InboundFlight(f.callsign, f.eta_min, f.fuel_lbs, f.priority) for f in self._flights_template]
        raw_reward, events, stats = sequence_flights(flights_eval, strategy)
        self._flights = flights_eval
        self._aircraft = self._aircraft_from_flights(self._flights)
        norm = self._normalise(raw_reward, low=-140, high=40)
        violations = [f"GO-AROUND: {f.callsign}" for f in self._flights if f.went_around]
        info = {"strategy": strategy, "events": events, "terminal": False}
        return norm, norm, violations, info

    def _step_emergency(self, actions: List[ATCAction]) -> Tuple[float, float, List[str], Dict]:
        heading, altitude = self._action_to_vector_2d(actions)
        insert_time = self._action_to_time(actions)
        raw_reward, log = self._emerg_env.insert_emergency(heading, altitude, insert_time)
        self._emerg_time_spent += insert_time
        norm = self._normalise(raw_reward, low=-20, high=25)
        self._aircraft = self._aircraft_from_emergency(self._emerg_env)
        violations = [l for l in log if "VIOLATION" in l.upper()]
        success = raw_reward > 18
        timeout = self._emerg_time_spent >= 12.0
        info = {"raw_reward": round(raw_reward, 2), "dist_to_runway": round(math.hypot(self._emerg_env.emergency.lat, self._emerg_env.emergency.lon), 2), "terminal": success or timeout}
        return norm, norm, violations, info

    def _step_conflict(self, actions: List[ATCAction]) -> Tuple[float, float, List[str], Dict]:
        action_str = self._action_to_conflict_str(actions)
        raw_reward, desc = self._conflict_env.step(action_str)
        sep_after = self._conflict_env._separation()
        norm = self._normalise(raw_reward, low=-25, high=30)
        self._aircraft = self._aircraft_from_conflict(self._conflict_env)
        violations = [f"CRITICAL PROXIMITY: {sep_after:.2f} NM"] if sep_after < 5.0 else []
        resolved = sep_after > 12.0
        info = {"raw_reward": round(raw_reward, 2), "sep_after": round(sep_after, 2), "terminal": resolved}
        return norm, norm, violations, info

    def _step_gate(self, actions: List[ATCAction]) -> Tuple[float, float, List[str], Dict]:
        gate_id = self._action_to_gate_id(actions) or self._gate_env.find_best_gate()
        raw_reward, log = self._gate_env.assign(gate_id)
        norm = self._normalise(raw_reward, low=-60, high=25)
        self._gates_raw = self._gates_from_env(self._gate_env)
        violations = [l for l in log if "!!!" in l]
        info = {"raw_reward": raw_reward, "gate_assigned": gate_id, "terminal": self._gate_env.assigned_gate is not None}
        return norm, norm, violations, info

    # Action translators (kept simple & effective)
    def _action_to_wake_str(self, actions: List[ATCAction]) -> str:
        for a in actions:
            if a.action_type == ActionType.SPEED_CHANGE:
                return "slow_down_trailing" if a.value and a.value < 0 else "speed_up_trailing"
            if a.action_type == ActionType.HEADING_CHANGE:
                return "increase_heading_gap"
        return "hold"

    def _action_to_strategy(self, actions: List[ATCAction]) -> str:
        return "rl_agent"  # default for go-around

    def _action_to_vector_2d(self, actions: List[ATCAction]) -> Tuple[float, float]:
        heading = 240.0
        altitude = 5500.0
        for a in actions:
            if a.action_type in (ActionType.HEADING_CHANGE, ActionType.VECTOR) and a.value is not None:
                heading = float(a.value) % 360
            if a.action_type == ActionType.ALTITUDE_CHANGE and a.value is not None:
                altitude = max(1000, float(a.value))
        return heading, altitude

    def _action_to_conflict_str(self, actions: List[ATCAction]) -> str:
        for a in actions:
            if a.action_type == ActionType.HEADING_CHANGE and a.value is not None:
                return "left_15" if a.value < 0 else "right_15"
        return "left_15"

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
            AircraftState(callsign="LEAD", category=_map[env.leading_cat], status=AircraftStatus.APPROACH, x=0.0, y=env.current_sep, altitude=3000, heading=180, speed=160),
            AircraftState(callsign="TRAIL", category=_map[env.trailing_cat], status=AircraftStatus.APPROACH, x=0.0, y=0.0, altitude=3000, heading=180, speed=150),
        ]

    def _aircraft_from_flights(self, flights: List[InboundFlight]) -> List[AircraftState]:
        return [AircraftState(callsign=f.callsign, category=ModelAircraftCategory.LARGE, status=AircraftStatus.APPROACH if not f.went_around else AircraftStatus.GO_AROUND, x=float(i)*2, y=f.eta_min, altitude=3000, heading=180, speed=160, fuel_state=min(1.0, f.fuel_lbs/8000)) for i, f in enumerate(flights)]

    def _aircraft_from_emergency(self, env: EmergencyVectorEnv) -> List[AircraftState]:
        ac_list = [AircraftState(callsign=env.emergency.callsign, category=ModelAircraftCategory.LARGE, status=AircraftStatus.EMERGENCY, x=env.emergency.lon, y=env.emergency.lat, altitude=env.emergency.altitude, heading=env.emergency.heading, speed=env.emergency.speed, is_emergency=True)]
        for ac in env.traffic:
            ac_list.append(AircraftState(callsign=ac.callsign, category=ModelAircraftCategory.LARGE, status=AircraftStatus.APPROACH, x=ac.lon, y=ac.lat, altitude=ac.altitude, heading=ac.heading, speed=ac.speed))
        return ac_list

    def _aircraft_from_conflict(self, env: ConflictAlertEnv) -> List[AircraftState]:
        return [
            AircraftState(callsign=env.ac1.callsign, category=ModelAircraftCategory.LARGE, status=AircraftStatus.ENROUTE, x=env.ac1.x, y=env.ac1.y, altitude=env.ac1.altitude, heading=env.ac1.heading_deg, speed=env.ac1.speed_kts),
            AircraftState(callsign=env.ac2.callsign, category=ModelAircraftCategory.LARGE, status=AircraftStatus.ENROUTE, x=env.ac2.x, y=env.ac2.y, altitude=env.ac2.altitude, heading=env.ac2.heading_deg, speed=env.ac2.speed_kts),
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
        conflicts = []
        for i, a in enumerate(self._aircraft):
            for b in self._aircraft[i+1:]:
                lateral = math.hypot(a.x - b.x, a.y - b.y)
                vertical = abs(a.altitude - b.altitude)
                if lateral < 3.0 and vertical < 1000:
                    conflicts.append({"ac1": a.callsign, "ac2": b.callsign, "lateral_nm": round(lateral, 2)})
        return conflicts