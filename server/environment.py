"""
MonsoonFloodGateEnv — OpenEnv-compliant environment
=====================================================
Wraps the FloodSimulation with the standard OpenEnv interface:
  reset() → Observation
  step(action) → (Observation, Reward, done, info)
  state() → GlobalState

Multi-agent: this class handles a single agent (one basin).
The server spawns one instance per basin for multi-agent runs.
"""

from __future__ import annotations

from models import Action, GlobalState, Observation, Reward
from simulation import FloodSimulation
from graders import grade, TASKS


class MonsoonFloodGateEnv:
    """
    OpenEnv environment for Monsoon Flood Gate Control.

    Parameters
    ----------
    task_id : str
        One of "task_easy", "task_medium", "task_hard"
    basin_idx : int
        Which sub-basin this agent controls (0, 1, or 2)
    storm_seed : int
        Random seed for storm generation (reproducibility)
    """

    def __init__(self, task_id: str = "task_easy", basin_idx: int = 0, storm_seed: int | None = None):
        if task_id not in TASKS:
            raise ValueError(f"task_id must be one of {list(TASKS.keys())}")
        if basin_idx not in (0, 1, 2):
            raise ValueError("basin_idx must be 0, 1, or 2")

        task_meta = TASKS[task_id]
        self.task_id = task_id
        self.basin_idx = basin_idx
        self.storm_seed = storm_seed if storm_seed is not None else task_meta["storm_seed"]
        self._sim = FloodSimulation(task_id=task_id, storm_seed=self.storm_seed)
        self._global_state: GlobalState | None = None
        self._done: bool = False
        self._step_count: int = 0

    # ─── OpenEnv Interface ────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset environment. Returns initial observation for this agent's basin."""
        self._global_state = self._sim.reset()
        self._done = False
        self._step_count = 0
        return self._sim.get_observation(self.basin_idx)

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """
        Apply action for this basin and advance simulation.

        In multi-agent mode, neighbouring basins use a default heuristic
        (proportional control based on water level).
        Returns observation for this basin only.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        # Build actions for all 3 basins
        all_actions = self._build_all_actions(action)

        # Step simulation
        global_state, rewards, done, info = self._sim.step(all_actions)
        self._global_state = global_state
        self._done = done
        self._step_count += 1

        obs = self._sim.get_observation(self.basin_idx)
        reward = rewards[self.basin_idx]

        # Attach final grader score on episode end
        if done:
            info["final_score"] = grade(global_state)
            info["task_id"] = self.task_id

        return obs, reward, done, info

    def state(self) -> GlobalState:
        """Return full global state (for graders and debugging)."""
        if self._global_state is None:
            raise RuntimeError("Call reset() first.")
        return self._global_state

    def grade_episode(self) -> float:
        """Run the programmatic grader. Returns score in [0.0, 1.0]."""
        if self._global_state is None:
            raise RuntimeError("No episode data. Call reset() and run steps first.")
        return grade(self._global_state)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _build_all_actions(self, own_action: Action) -> list[Action]:
        """
        Fill in actions for other basins using a simple heuristic
        (proportional-control fallback for single-agent mode).
        """
        actions = []
        for i in range(3):
            if i == self.basin_idx:
                actions.append(own_action)
            else:
                # Heuristic: open gates proportional to water level
                obs_i = self._sim.get_observation(i)
                wf = obs_i.water_level_fraction
                gate_open = min(1.0, max(0.0, (wf - 0.4) * 2))  # open above 40% fill
                pump_on = min(1.0, max(0.0, wf - 0.7))           # pump above 70% fill
                actions.append(Action(
                    gate_positions=[gate_open, gate_open, gate_open * 0.5],
                    pump_speeds=[pump_on, pump_on * 0.8],
                ))
        return actions

    # ─── Properties ───────────────────────────────────────────────────────────

    @property
    def observation_space(self) -> dict:
        return {
            "basin_id": "int",
            "water_level_m": "float [0, capacity_m]",
            "water_level_fraction": "float [0.0, 1.0+]",
            "rainfall_mm_per_hr": "float [0, 250]",
            "river_level_m": "float [0.3, 3.5]",
            "gate_positions": "list[float] x3 [0.0, 1.0]",
            "pump_speeds": "list[float] x2 [0.0, 1.0]",
            "is_flooding": "bool",
            "neighbors": "list[NeighborObservation] (partial, noisy)",
            "timestep": "int [0, 72]",
            "elapsed_minutes": "float [0, 360]",
            "storm_phase": "str {pre_monsoon, peak, recession, clear}",
            "rainfall_forecast": "list[float] x3 (noisy 3-step lookahead)",
        }

    @property
    def action_space(self) -> dict:
        return {
            "gate_positions": "list[float] x3 — each in [0.0, 1.0]",
            "pump_speeds": "list[float] x2 — each in [0.0, 1.0]",
        }
