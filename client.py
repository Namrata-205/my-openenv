"""
Python client for the ATC TRACON RL Environment.
Provides a clean interface for interacting with the REST API.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

from models import (
    ATCAction, EnvironmentState, ResetRequest, StepRequest,
    StepResult, TaskType,
)


class ATCClient:
    """
    HTTP client for the ATC TRACON RL Environment.

    Usage
    -----
    client = ATCClient("http://localhost:7860")
    state  = client.reset(task=TaskType.CONFLICT_RESOLUTION)
    result = client.step([ATCAction(action_type="heading_change",
                                    target_callsign="AAL201",
                                    value=30)])
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        self.base_url = (base_url or os.getenv("API_BASE_URL", "http://localhost:7860")).rstrip("/")
        self.timeout  = timeout
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ──────────────────────────────────────────
    # OpenEnv API
    # ──────────────────────────────────────────

    def reset(self,
              task: Optional[TaskType] = None,
              seed: Optional[int] = None,
              options: Dict[str, Any] = {}) -> EnvironmentState:
        payload = ResetRequest(task=task, seed=seed, options=options)
        resp = self._post("/reset", payload.dict(exclude_none=True))
        return EnvironmentState(**resp)

    def step(self, actions: List[ATCAction]) -> StepResult:
        payload = StepRequest(actions=actions)
        resp = self._post("/step", payload.dict())
        return StepResult(**resp)

    def state(self) -> EnvironmentState:
        resp = self._get("/state")
        return EnvironmentState(**resp)

    # ──────────────────────────────────────────
    # Convenience helpers
    # ──────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    def list_tasks(self) -> List[str]:
        return self._get("/tasks")["tasks"]

    def task_info(self, task_name: str) -> Dict[str, Any]:
        return self._get(f"/tasks/{task_name}")

    # ──────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────

    def _get(self, path: str) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = self._session.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = self._session.post(url, json=data, timeout=self.timeout)
        r.raise_for_status()
        return r.json()