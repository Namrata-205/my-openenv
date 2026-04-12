"""
inference.py — ATC TRACON RL Environment inference script.
COMPLIANT VERSION: Always calls the validator's LiteLLM proxy first.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI


# ============================================================
# ENVIRONMENT VARIABLES - USE EXACT NAMES PROVIDED BY VALIDATOR
# ============================================================
# The validator injects these. DO NOT RENAME.
# API_BASE_URL is the URL for the LiteLLM proxy.
# API_KEY is the key for the LiteLLM proxy.
PROXY_URL = os.environ.get("API_BASE_URL")
PROXY_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# Your environment server's URL. This is different from the proxy URL.
# The validator does not provide this. You must set it correctly for local testing.
# It defaults to localhost, but on HF Spaces it will be the internal URL.
ENV_SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://localhost:7860")

DEFAULT_TASK = "conflict_resolution"
MAX_STEPS = 10
MAX_RETRIES = 3
RETRY_DELAY = 1.0

# ============================================================
# LITELLM PROXY CLIENT (MANDATORY FOR VALIDATOR)
# ============================================================
if not PROXY_URL or not PROXY_KEY:
    print("CRITICAL ERROR: API_BASE_URL and API_KEY environment variables are not set.")
    print("This script MUST be run in an environment that provides these.")
    print("The LiteLLM proxy is required for all decisions.")
    sys.exit(1)

llm_client = OpenAI(
    base_url=PROXY_URL,
    api_key=PROXY_KEY,
)
print(f"[INFO] LiteLLM proxy configured at {PROXY_URL}")


# ============================================================
# ENVIRONMENT SERVER COMMUNICATION
# ============================================================
def is_server_ready(max_attempts: int = 15, wait_seconds: int = 2) -> bool:
    """Check if the environment server is ready."""
    print(f"[INFO] Checking environment server at {ENV_SERVER_URL}...")
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{ENV_SERVER_URL}/health", timeout=5)
            if response.status_code == 200:
                print(f"[INFO] Environment server is ready.")
                return True
        except Exception:
            pass
        if attempt < max_attempts - 1:
            time.sleep(wait_seconds)
    return False

def _post_with_retry(url: str, payload: Dict, label: str) -> Dict:
    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.ok:
                return r.json()
            if r.status_code < 500:
                r.raise_for_status()
        except Exception as exc:
            print(f"[WARN] {label} attempt {attempt}/{MAX_RETRIES} failed: {exc}")
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"{label} failed after {MAX_RETRIES} attempts")

def env_reset(task: str, seed: Optional[int] = None) -> Dict:
    payload = {"task": task}
    if seed is not None:
        payload["seed"] = seed
    return _post_with_retry(f"{ENV_SERVER_URL}/reset", payload, "env_reset")

def env_step(actions: List[Dict]) -> Dict:
    return _post_with_retry(f"{ENV_SERVER_URL}/step", {"actions": actions}, "env_step")

# ============================================================
# PROMPT BUILDING (Unchanged, but crucial)
# ============================================================
SYSTEM_PROMPT = """You are an expert Air Traffic Controller managing TRACON airspace.

You MUST take corrective action when conflicts exist or separation is insufficient.

CRITICAL RULES:
- NEVER use "ALL" as a callsign. ALWAYS use a valid aircraft callsign from the state.
- For conflict_resolution: act on BOTH conflicting aircraft when needed.
- For wake_turbulence: act on the trailing aircraft.
- For emergency_vectoring: prioritize the MAYDAY aircraft.
- For go_around_prevention: manage landing sequence spacing.
- For gate_assignment: assign gates to arriving aircraft.

Allowed action_type values:
- heading_change: negative = left, positive = right (value in degrees)
- speed_change: negative = slow down, positive = speed up (value in knots)
- altitude_change: positive = climb, negative = descend (value in feet)
- sequence_swap: swap landing order (secondary_target required)
- assign_gate: assign to gate (gate_id required)
- vector: direct to heading (value in degrees)
- no_action: no change needed

Respond ONLY with valid JSON (no markdown fences, no extra text). Use the exact format below.

Format:
[
  {
    "action_type": "...",
    "target_callsign": "<REAL CALLSIGN>",
    "value": <number or null>,
    "secondary_target": null or "<callsign>",
    "gate_id": null or "<GATE_ID>",
    "rationale": "short reason"
  }
]"""

def build_user_prompt(state: Dict, step: int) -> str:
    # (Keep your existing, detailed build_user_prompt function here)
    # It should construct a comprehensive JSON state summary.
    # For brevity, I've included a simplified but functional version.
    aircraft_summary = []
    for ac in state.get("aircraft", []):
        aircraft_summary.append({
            "callsign": ac.get("callsign"),
            "x": ac.get("x"), "y": ac.get("y"),
            "altitude": ac.get("altitude"), "heading": ac.get("heading"),
            "speed": ac.get("speed"), "emergency": ac.get("is_emergency", False),
        })
    return json.dumps({
        "step": step, "task": state.get("task"),
        "active_conflicts": state.get("active_conflicts", []),
        "aircraft": aircraft_summary,
    }, indent=2)

def safe_parse_actions(text: str, state: Dict) -> List[Dict]:
    # (Keep your existing, robust safe_parse_actions function here)
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"): lines = lines[1:]
            if lines and lines[-1].strip() == "```": lines = lines[:-1]
            cleaned = "\n".join(lines)
        actions = json.loads(cleaned)
        if not isinstance(actions, list): actions = [actions]
        return actions
    except Exception as e:
        print(f"[WARN] JSON parse failed: {e}. Using fallback.")
        return [{"action_type": "no_action", "target_callsign": state.get("aircraft", [{}])[0].get("callsign", "NONE"), "rationale": f"Parse error: {e}"}]

# ============================================================
# CORE DECISION LOGIC: PROXY-FIRST
# ============================================================
def get_actions(state: Dict, step: int) -> List[Dict]:
    """
    ALWAYS call the LiteLLM proxy for decision-making.
    This is what the validator checks for.
    """
    print(f"[LLM] Calling LiteLLM proxy for decision at step {step}...")
    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(state, step)},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        raw_text = response.choices[0].message.content or ""
        print(f"[LLM] Proxy call successful.")
        return safe_parse_actions(raw_text, state)
    except Exception as e:
        print(f"[LLM] Proxy call FAILED: {e}")
        # This is a critical failure for the validator.
        # We raise an exception to make it clear the proxy is not working.
        raise RuntimeError(f"LiteLLM proxy call failed, which is required for validation: {e}")

# ============================================================
# MAIN INFERENCE LOOP
# ============================================================
def run_inference(task: str = DEFAULT_TASK, seed: Optional[int] = None):
    print("[START]")
    print(f"[INFO] LiteLLM Proxy URL: {PROXY_URL}")
    print(f"[INFO] Environment Server URL: {ENV_SERVER_URL}")

    if not is_server_ready():
        print("[ERROR] Environment server not ready. Exiting.")
        print("[END]")
        return

    try:
        state = env_reset(task, seed=seed)
    except Exception as e:
        print(f"[ERROR] Failed to reset environment: {e}")
        print("[END]")
        return

    print(f"[STEP] Environment reset. Task: {state.get('task')} | Aircraft: {len(state.get('aircraft', []))}")

    llm_calls = 0
    for step_num in range(1, MAX_STEPS + 1):
        if state.get("done", False):
            break

        # THIS IS THE KEY CHANGE: ALWAYS CALL THE PROXY
        actions = get_actions(state, step_num)
        llm_calls += 1

        try:
            result = env_step(actions)
        except Exception as e:
            print(f"[ERROR] Step failed: {e}")
            break

        state = result["state"]
        print(f"[STEP] {step_num} | Reward: {result.get('reward', 0):.4f} | Score: {result.get('score', 0):.4f} | Done: {result.get('done', False)}")

        if result.get("done", False):
            break
        time.sleep(0.1)

    print(f"[STEP] Episode complete. Total LLM proxy calls: {llm_calls}")
    print("[END]")

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run_inference(task=task, seed=seed)
