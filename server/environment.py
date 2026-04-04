"""
ATC TRACON RL Environment — core simulation logic.
Implements reset(), step(), state() following the OpenEnv specification.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from models import (
    ATCAction, ActionType, AircraftCategory, AircraftState, AircraftStatus,
    EnvironmentState, GateState, RunwayState, StepResult, TaskType,
)


# ──────────────────────────────────────────────
# Wake-turbulence separation minimums (NM)
# ──────────────────────────────────────────────
WAKE_SEP: Dict[Tuple[str, str], float] = {
    ("super", "heavy"): 6.0,
    ("super", "large"): 7.0,
    ("super", "small"): 8.0,
    ("heavy", "heavy"): 4.0,
    ("heavy", "large"): 5.0,
    ("heavy", "small"): 6.0,
    ("large", "large"): 3.0,
    ("large", "small"): 4.0,
    ("small", "small"): 3.0,
}

MIN_LATERAL_SEP = 3.0   # NM — standard TRACON separation
MIN_VERTICAL_SEP = 1000  # ft


def _wake_min(leader: AircraftCategory, follower: AircraftCategory) -> float:
    key = (leader.value if hasattr(leader, "value") else leader,
           follower.value if hasattr(follower, "value") else follower)
    return WAKE_SEP.get(key, 3.0)


def _heading_delta(h1: float, h2: float) -> float:
    """Smallest signed angular difference h2-h1 in [-180,180]."""
    d = (h2 - h1) % 360
    return d - 360 if d > 180 else d


class ATCEnvironment:
    """
    Simulates a TRACON airspace supporting 5 tasks:
      1. Wake Turbulence Separation
      2. Go-Around Prevention via Sequencing
      3. Emergency Vectoring
      4. Conflict Alert Resolution
      5. Gate / Stand Assignment
    """

    VERSION = "1.0.0"
    MAX_STEPS_DEFAULT = 30

    def __init__(self):
        self._task: TaskType = TaskType.WAKE_TURBULENCE
        self._step: int = 0
        self._max_steps: int = self.MAX_STEPS_DEFAULT
        self._aircraft: List[AircraftState] = []
        self._gates: List[GateState] = []
        self._runways: List[RunwayState] = []
        self._rng: random.Random = random.Random()
        self._done: bool = False
        self._episode_score: float = 0.0
        self._violations: List[str] = []
        self._info: Dict[str, Any] = {}

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def reset(self, task: Optional[TaskType] = None,
              seed: Optional[int] = None,
              options: Dict[str, Any] = {}) -> EnvironmentState:
        self._rng = random.Random(seed)
        self._step = 0
        self._done = False
        self._violations = []
        self._episode_score = 0.0
        self._info = {}

        if task:
            self._task = task
        else:
            self._task = self._rng.choice(list(TaskType))

        self._max_steps = options.get("max_steps", self.MAX_STEPS_DEFAULT)
        self._build_scenario()
        return self._get_state()

    def step(self, actions: List[ATCAction]) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step += 1
        violations: List[str] = []

        # Apply actions
        for action in actions:
            v = self._apply_action(action)
            violations.extend(v)

        # Simulate physics
        self._simulate_aircraft()

        # Grade the step
        reward, score, step_info = self._grade_step()
        self._episode_score = 0.9 * self._episode_score + 0.1 * score  # EMA

        done = self._step >= self._max_steps or self._check_terminal()
        self._done = done

        info = {**step_info, "episode_score": round(self._episode_score, 4)}
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

    # ──────────────────────────────────────────
    # Scenario builders
    # ──────────────────────────────────────────

    def _build_scenario(self):
        builders = {
            TaskType.WAKE_TURBULENCE:      self._scenario_wake_turbulence,
            TaskType.GO_AROUND_PREVENTION: self._scenario_go_around,
            TaskType.EMERGENCY_VECTORING:  self._scenario_emergency,
            TaskType.CONFLICT_RESOLUTION:  self._scenario_conflict,
            TaskType.GATE_ASSIGNMENT:      self._scenario_gate,
        }
        builders[self._task]()

    def _default_runways(self) -> List[RunwayState]:
        return [
            RunwayState(runway_id="28L", active=True),
            RunwayState(runway_id="28R", active=True),
        ]

    def _default_gates(self) -> List[GateState]:
        gates = []
        for i in range(1, 6):
            gates.append(GateState(gate_id=f"A{i}", gate_type=AircraftCategory.HEAVY))
        for i in range(1, 9):
            gates.append(GateState(gate_id=f"B{i}", gate_type=AircraftCategory.LARGE))
        for i in range(1, 5):
            gates.append(GateState(gate_id=f"C{i}", gate_type=AircraftCategory.SMALL))
        return gates

    def _random_ac(self, callsign: str, cat: AircraftCategory,
                   status: AircraftStatus = AircraftStatus.APPROACH,
                   x_range=(-30, 30), y_range=(-30, 30),
                   alt=3000, speed=180, heading=None,
                   sequence_pos: Optional[int] = None) -> AircraftState:
        return AircraftState(
            callsign=callsign,
            category=cat,
            status=status,
            x=self._rng.uniform(*x_range),
            y=self._rng.uniform(*y_range),
            altitude=alt + self._rng.uniform(-500, 500),
            heading=heading if heading is not None else self._rng.uniform(0, 360),
            speed=speed + self._rng.uniform(-20, 20),
            fuel_state=self._rng.uniform(0.5, 1.0),
            sequence_pos=sequence_pos,
        )

    def _scenario_wake_turbulence(self):
        """Heavy followed closely by a small — agent must increase spacing."""
        self._runways = self._default_runways()
        self._gates = self._default_gates()
        heavy = AircraftState(callsign="UAL100", category=AircraftCategory.HEAVY,
                              status=AircraftStatus.APPROACH,
                              x=0, y=20, altitude=3000, heading=180, speed=160,
                              sequence_pos=1)
        small = AircraftState(callsign="N123AB", category=AircraftCategory.SMALL,
                              status=AircraftStatus.APPROACH,
                              x=0, y=22, altitude=3000, heading=180, speed=140,
                              sequence_pos=2)
        # extra traffic
        extra = [
            self._random_ac(f"EX{i}", AircraftCategory.LARGE, sequence_pos=i+3)
            for i in range(3)
        ]
        self._aircraft = [heavy, small] + extra

    def _scenario_go_around(self):
        """Suboptimal sequencing — agent reorders to prevent go-arounds."""
        self._runways = self._default_runways()
        self._gates = self._default_gates()
        aircraft = []
        cats = [AircraftCategory.LARGE, AircraftCategory.HEAVY,
                AircraftCategory.SMALL, AircraftCategory.LARGE,
                AircraftCategory.HEAVY]
        positions_y = [14, 10, 18, 22, 8]
        speeds      = [160, 140, 180, 155, 145]
        for i, (cat, py, spd) in enumerate(zip(cats, positions_y, speeds)):
            aircraft.append(AircraftState(
                callsign=f"FLT{100+i}",
                category=cat,
                status=AircraftStatus.APPROACH,
                x=self._rng.uniform(-2, 2),
                y=py,
                altitude=3000,
                heading=180,
                speed=spd,
                sequence_pos=i + 1,
                fuel_state=self._rng.uniform(0.4, 0.9),
            ))
        self._aircraft = aircraft

    def _scenario_emergency(self):
        """Emergency aircraft needs priority vectoring to runway."""
        self._runways = self._default_runways()
        self._gates = self._default_gates()
        emerg = AircraftState(
            callsign="MAYDAY1", category=AircraftCategory.LARGE,
            status=AircraftStatus.EMERGENCY,
            x=self._rng.uniform(-20, 20),
            y=self._rng.uniform(-20, 20),
            altitude=4000, heading=self._rng.uniform(0, 360),
            speed=200, is_emergency=True, fuel_state=0.1,
        )
        normal = [
            self._random_ac(f"NRM{i}", self._rng.choice(list(AircraftCategory)),
                            sequence_pos=i + 1)
            for i in range(4)
        ]
        self._aircraft = [emerg] + normal

    def _scenario_conflict(self):
        """Two aircraft on converging paths — agent must resolve before CPA."""
        self._runways = self._default_runways()
        self._gates = self._default_gates()
        ac1 = AircraftState(callsign="AAL201", category=AircraftCategory.LARGE,
                            status=AircraftStatus.ENROUTE,
                            x=-15, y=0, altitude=5000, heading=90, speed=250)
        ac2 = AircraftState(callsign="DAL305", category=AircraftCategory.LARGE,
                            status=AircraftStatus.ENROUTE,
                            x=0, y=-15, altitude=5000, heading=0, speed=250)
        bystanders = [
            self._random_ac(f"BY{i}", AircraftCategory.SMALL,
                            alt=7000, y_range=(10, 30))
            for i in range(3)
        ]
        self._aircraft = [ac1, ac2] + bystanders

    def _scenario_gate(self):
        """Multiple aircraft landing — agent assigns gates efficiently."""
        self._runways = self._default_runways()
        self._gates = self._default_gates()
        cats = [AircraftCategory.HEAVY, AircraftCategory.LARGE,
                AircraftCategory.SMALL, AircraftCategory.HEAVY,
                AircraftCategory.LARGE, AircraftCategory.SMALL]
        aircraft = []
        for i, cat in enumerate(cats):
            aircraft.append(AircraftState(
                callsign=f"ARR{i+1:02d}",
                category=cat,
                status=AircraftStatus.TAXIING,
                x=self._rng.uniform(-2, 2),
                y=self._rng.uniform(-2, 2),
                altitude=0, heading=0, speed=15,
                fuel_state=1.0,
            ))
        self._aircraft = aircraft

    # ──────────────────────────────────────────
    # Action application
    # ──────────────────────────────────────────

    def _apply_action(self, action: ATCAction) -> List[str]:
        violations = []
        ac = self._find_ac(action.target_callsign)
        if ac is None:
            violations.append(f"Unknown callsign: {action.target_callsign}")
            return violations

        atype = action.action_type
        if atype == ActionType.HEADING_CHANGE and action.value is not None:
            delta = max(-30, min(30, action.value))   # cap at ±30° per step
            ac.heading = (ac.heading + delta) % 360

        elif atype == ActionType.SPEED_CHANGE and action.value is not None:
            delta = max(-30, min(30, action.value))
            ac.speed = max(80, min(350, ac.speed + delta))

        elif atype == ActionType.ALTITUDE_CHANGE and action.value is not None:
            delta = max(-2000, min(2000, action.value))
            ac.altitude = max(0, ac.altitude + delta)

        elif atype == ActionType.SEQUENCE_SWAP:
            sec = self._find_ac(action.secondary_target or "")
            if sec:
                p1, p2 = ac.sequence_pos, sec.sequence_pos
                ac.sequence_pos, sec.sequence_pos = p2, p1
            else:
                violations.append(f"sequence_swap: secondary target not found")

        elif atype == ActionType.ASSIGN_GATE and action.gate_id:
            gate = self._find_gate(action.gate_id)
            if gate is None:
                violations.append(f"Unknown gate: {action.gate_id}")
            elif gate.occupied:
                violations.append(f"Gate {action.gate_id} already occupied")
            elif not self._gate_compatible(gate, ac):
                violations.append(f"Gate {action.gate_id} incompatible with {ac.category}")
            else:
                gate.occupied = True
                gate.aircraft = ac.callsign
                ac.assigned_gate = action.gate_id

        elif atype == ActionType.VECTOR and action.value is not None:
            # Direct heading toward runway threshold (value = desired heading)
            ac.heading = action.value % 360

        elif atype == ActionType.NO_ACTION:
            pass

        return violations

    # ──────────────────────────────────────────
    # Physics simulation (simplified)
    # ──────────────────────────────────────────

    def _simulate_aircraft(self):
        dt = 1 / 60.0  # 1-minute timestep in hours
        for ac in self._aircraft:
            if ac.status in (AircraftStatus.PARKED, AircraftStatus.LANDED):
                continue
            rad = math.radians(ac.heading)
            ac.x += ac.speed * math.sin(rad) * dt
            ac.y += ac.speed * math.cos(rad) * dt

            # Approach: descend toward runway
            if ac.status == AircraftStatus.APPROACH:
                dist = math.sqrt(ac.x ** 2 + ac.y ** 2)
                if dist < 1.0:
                    ac.status = AircraftStatus.LANDING
                    ac.altitude = 0
                    ac.speed = 0
                else:
                    target_alt = max(0, dist * 300)
                    ac.altitude += (target_alt - ac.altitude) * 0.1

            # Fuel burn
            if ac.status != AircraftStatus.PARKED:
                ac.fuel_state = max(0.0, ac.fuel_state - 0.005)

    # ──────────────────────────────────────────
    # Graders
    # ──────────────────────────────────────────

    def _grade_step(self) -> Tuple[float, float, Dict]:
        graders = {
            TaskType.WAKE_TURBULENCE:      self._grade_wake,
            TaskType.GO_AROUND_PREVENTION: self._grade_go_around,
            TaskType.EMERGENCY_VECTORING:  self._grade_emergency,
            TaskType.CONFLICT_RESOLUTION:  self._grade_conflict,
            TaskType.GATE_ASSIGNMENT:      self._grade_gate,
        }
        return graders[self._task]()

    def _grade_wake(self) -> Tuple[float, float, Dict]:
        pairs = self._get_consecutive_pairs()
        if not pairs:
            return 1.0, 1.0, {"note": "no pairs"}
        total, perfect = 0.0, 0
        details = []
        for leader, follower in pairs:
            dist = leader.distance_to(follower)
            required = _wake_min(leader.category, follower.category)
            ratio = dist / required
            penalty = max(0.0, 1.0 - ratio)
            score = max(0.0, min(1.0, ratio))
            total += score
            if ratio >= 1.0:
                perfect += 1
            details.append({"leader": leader.callsign,
                             "follower": follower.callsign,
                             "dist_nm": round(dist, 2),
                             "required_nm": required,
                             "score": round(score, 3)})
        avg = total / len(pairs)
        return avg, avg, {"separation_details": details}

    def _grade_go_around(self) -> Tuple[float, float, Dict]:
        """Score = how well speed-ordered aircraft are by sequence position."""
        approach = sorted(
            [a for a in self._aircraft if a.status == AircraftStatus.APPROACH],
            key=lambda a: a.sequence_pos or 999
        )
        if len(approach) < 2:
            return 1.0, 1.0, {}
        inversions = 0
        total_pairs = 0
        for i in range(len(approach) - 1):
            a, b = approach[i], approach[i + 1]
            # Faster aircraft should be ahead (lower seq pos)
            if a.speed < b.speed:   # violation
                inversions += 1
            total_pairs += 1
        score = 1.0 - (inversions / total_pairs)
        # Bonus: no fuel critical aircraft
        fuel_ok = all(a.fuel_state > 0.2 for a in approach)
        score = score * (1.0 if fuel_ok else 0.85)
        return score, score, {"inversions": inversions, "fuel_ok": fuel_ok}

    def _grade_emergency(self) -> Tuple[float, float, Dict]:
        emerg = self._find_emergency()
        if emerg is None:
            return 1.0, 1.0, {"note": "no emergency"}
        dist = math.sqrt(emerg.x ** 2 + emerg.y ** 2)
        max_dist = 40.0
        proximity = 1.0 - min(1.0, dist / max_dist)
        heading_to_runway = math.degrees(math.atan2(-emerg.x, -emerg.y)) % 360
        alignment = 1.0 - abs(_heading_delta(emerg.heading, heading_to_runway)) / 180.0
        fuel_urgency = 1.0 - emerg.fuel_state
        score = 0.4 * proximity + 0.4 * alignment + 0.2 * fuel_urgency
        return score, score, {"dist_nm": round(dist, 2),
                              "alignment": round(alignment, 3),
                              "fuel_remaining": round(emerg.fuel_state, 3)}

    def _grade_conflict(self) -> Tuple[float, float, Dict]:
        conflicts = self._detect_conflicts()
        n_ac = max(1, len(self._aircraft))
        conflict_rate = len(conflicts) / n_ac
        score = max(0.0, 1.0 - conflict_rate)
        return score, score, {"active_conflicts": len(conflicts),
                              "conflict_pairs": conflicts}

    def _grade_gate(self) -> Tuple[float, float, Dict]:
        taxiing = [a for a in self._aircraft if a.status == AircraftStatus.TAXIING]
        if not taxiing:
            return 1.0, 1.0, {}
        assigned = sum(1 for a in taxiing if a.assigned_gate)
        valid    = sum(1 for a in taxiing if self._valid_gate_assignment(a))
        assign_score = assigned / len(taxiing)
        valid_score  = valid / max(1, assigned)
        score = 0.5 * assign_score + 0.5 * valid_score
        return score, score, {"assigned": assigned,
                              "valid": valid,
                              "total": len(taxiing)}

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _get_state(self) -> EnvironmentState:
        return EnvironmentState(
            task=self._task,
            step=self._step,
            max_steps=self._max_steps,
            aircraft=list(self._aircraft),
            gates=list(self._gates),
            runways=list(self._runways),
            active_conflicts=self._detect_conflicts(),
            done=self._done,
            info=self._info,
        )

    def _find_ac(self, callsign: str) -> Optional[AircraftState]:
        for ac in self._aircraft:
            if ac.callsign == callsign:
                return ac
        return None

    def _find_gate(self, gate_id: str) -> Optional[GateState]:
        for g in self._gates:
            if g.gate_id == gate_id:
                return g
        return None

    def _find_emergency(self) -> Optional[AircraftState]:
        for ac in self._aircraft:
            if ac.is_emergency:
                return ac
        return None

    def _get_consecutive_pairs(self):
        approach = sorted(
            [a for a in self._aircraft if a.status == AircraftStatus.APPROACH
             and a.sequence_pos is not None],
            key=lambda a: a.sequence_pos,
        )
        return [(approach[i], approach[i + 1]) for i in range(len(approach) - 1)]

    def _detect_conflicts(self) -> List[Dict]:
        conflicts = []
        acs = self._aircraft
        for i in range(len(acs)):
            for j in range(i + 1, len(acs)):
                a, b = acs[i], acs[j]
                lateral = a.distance_to(b)
                vertical = abs(a.altitude - b.altitude)
                if lateral < MIN_LATERAL_SEP and vertical < MIN_VERTICAL_SEP:
                    conflicts.append({
                        "ac1": a.callsign,
                        "ac2": b.callsign,
                        "lateral_nm": round(lateral, 2),
                        "vertical_ft": round(vertical, 0),
                    })
        return conflicts

    def _gate_compatible(self, gate: GateState, ac: AircraftState) -> bool:
        order = [AircraftCategory.SMALL, AircraftCategory.LARGE,
                 AircraftCategory.HEAVY, AircraftCategory.SUPER]
        ac_rank   = order.index(ac.category)
        gate_rank = order.index(gate.gate_type)
        return ac_rank <= gate_rank

    def _valid_gate_assignment(self, ac: AircraftState) -> bool:
        if not ac.assigned_gate:
            return False
        gate = self._find_gate(ac.assigned_gate)
        if gate is None:
            return False
        return self._gate_compatible(gate, ac)

    def _check_terminal(self) -> bool:
        """Early termination conditions."""
        if self._task == TaskType.EMERGENCY_VECTORING:
            emerg = self._find_emergency()
            if emerg and emerg.status in (AircraftStatus.LANDING, AircraftStatus.LANDED):
                return True
        if self._task == TaskType.GATE_ASSIGNMENT:
            taxiing = [a for a in self._aircraft if a.status == AircraftStatus.TAXIING]
            if all(a.assigned_gate for a in taxiing):
                return True
        return False