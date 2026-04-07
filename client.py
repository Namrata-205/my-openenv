"""
client.py — Python HTTP client for the ATC TRACON RL Environment.

Usage
-----
    from my_env.client import ATCClient

    client = ATCClient("http://localhost:7860")
    state  = client.reset(task="conflict_resolution", seed=42)

    while not state["done"]:
        actions = [{"action_type": "no_action", "target_callsign": "AC1"}]
        result  = client.step(actions)
        state   = result["state"]

    client.close()
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests


class ATCClient:
    """
    Thin HTTP client wrapping the ATC TRACON RL FastAPI server.

    Parameters
    ----------
    base_url:   Root URL of the running server (e.g. ``http://localhost:7860``).
    timeout:    Per-request timeout in seconds (default 30).
    max_retries: Number of retry attempts on transient 5xx errors (default 3).
    retry_delay: Base delay in seconds between retries — doubles each attempt (default 1.0).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.base_url    = base_url.rstrip("/")
        self.timeout     = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._session    = requests.Session()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _post(self, path: str, payload: Dict) -> Dict:
        url   = f"{self.base_url}{path}"
        delay = self.retry_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._session.post(url, json=payload, timeout=self.timeout)
                if r.ok:
                    return r.json()
                if r.status_code < 500:
                    r.raise_for_status()
                print(f"[ATCClient] POST {path} attempt {attempt}/{self.max_retries} got {r.status_code} — retrying in {delay:.1f}s")
            except requests.RequestException as exc:
                print(f"[ATCClient] POST {path} attempt {attempt}/{self.max_retries} error: {exc} — retrying in {delay:.1f}s")
            time.sleep(delay)
            delay *= 2
        raise RuntimeError(f"POST {path} failed after {self.max_retries} attempts")

    def _get(self, path: str) -> Dict:
        url = f"{self.base_url}{path}"
        r   = self._session.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ── Public API ─────────────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """Return server health info (version, tasks list)."""
        return self._get("/health")

    def reset(
        self,
        task: Optional[str] = None,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Reset the environment and return the initial EnvironmentState dict.

        Parameters
        ----------
        task:    One of ``wake_turbulence``, ``go_around_prevention``,
                 ``emergency_vectoring``, ``conflict_resolution``, ``gate_assignment``.
                 Pass ``None`` to let the server choose randomly.
        seed:    Optional integer seed for reproducibility.
        options: Optional dict forwarded to the environment (e.g. ``max_steps``).
        """
        payload: Dict[str, Any] = {}
        if task    is not None: payload["task"]    = task
        if seed    is not None: payload["seed"]    = seed
        if options is not None: payload["options"] = options
        return self._post("/reset", payload)

    def step(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Advance the environment by one step.

        Parameters
        ----------
        actions: List of action dicts. Each dict must contain at minimum
                 ``action_type`` and ``target_callsign``.

        Returns
        -------
        StepResult dict with keys: ``state``, ``reward``, ``done``,
        ``info``, ``violations``, ``score``.
        """
        return self._post("/step", {"actions": actions})

    def state(self) -> Dict[str, Any]:
        """Return the current EnvironmentState without advancing the episode."""
        return self._get("/state")

    def tasks(self) -> List[str]:
        """Return the list of supported task names."""
        return self._get("/tasks")["tasks"]

    def task_info(self, task_name: str) -> Dict[str, Any]:
        """Return description and reward structure for a specific task."""
        return self._get(f"/tasks/{task_name}")

    def close(self) -> None:
        """Close the underlying requests Session."""
        self._session.close()

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "ATCClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()