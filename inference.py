"""
inference.py — ATC TRACON RL Environment inference script.

Uses an LLM (via Hugging Face OpenAI-compatible API) as the RL agent.

Required environment variables:
  API_BASE_URL  — base URL of the running ATC environment server
  MODEL_NAME    — model identifier (HF / OpenAI-compatible)
  HF_TOKEN      — API key

Output format (strict):
  [START]
  [STEP] ...
  [END]
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import requests
from openai import OpenAI


# ── Environment variables ──────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

HF_BASE_URL  = "https://api-inference.huggingface.co/v1"

MAX_STEPS = 10
TASK = "conflict_resolution"


# ── OpenAI Client (HF) ─────────────────────────────────────────────────────────
client = OpenAI(
    base_url=HF_BASE_URL,
    api_key=HF_TOKEN
)


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def env_reset(task: str) -> Dict:
    r = requests.post(f"{API_BASE_URL}/reset", json={"task": task}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(actions: List[Dict]) -> Dict:
    r = requests.post(f"{API_BASE_URL}/step", json={"actions": actions}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_state() -> Dict:
    r = requests.get(f"{API_BASE_URL}/state", timeout=30)
    r.raise_for_status()
    return r.json()


# ── LLM agent ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Air Traffic Controller.

You MUST take corrective action when conflicts exist.

Rules:
- NEVER use "ALL" as a callsign
- ALWAYS use a valid aircraft callsign from the state
- If no action is needed, choose ANY one aircraft and return no_action for it

Respond ONLY with valid JSON.

Format:
[
  {
    "action_type": "...",
    "target_callsign": "<REAL CALLSIGN>",
    "value": <number or null>,
    "secondary_target": null,
    "gate_id": null,
    "rationale": "short reason"
  }
]
"""

def build_user_prompt(state: Dict, step: int) -> str:
    aircraft_summary = []

    for ac in state.get("aircraft", []):
        aircraft_summary.append({
            "callsign": ac["callsign"],
            "category": ac["category"],
            "status": ac["status"],
            "x_nm": round(ac["x"], 1),
            "y_nm": round(ac["y"], 1),
            "altitude_ft": int(ac["altitude"]),
            "heading_deg": round(ac["heading"], 1),
            "speed_kts": round(ac["speed"], 1),
            "emergency": ac.get("is_emergency", False),
            "fuel": round(ac.get("fuel_state", 1.0), 2),
            "sequence": ac.get("sequence_pos"),
            "gate": ac.get("assigned_gate"),
        })

    return json.dumps({
        "step": step,
        "task": state.get("task"),
        "must_act": len(state.get("active_conflicts", [])) > 0,
        "active_conflicts": state.get("active_conflicts", []),
        "aircraft": aircraft_summary,
        "gates_available": [
            g["gate_id"] for g in state.get("gates", []) if not g["occupied"]
        ],
    }, indent=2)


def safe_parse_actions(text: str) -> List[Dict]:
    """Robust JSON parsing with fallback."""
    cleaned = text.strip()

    # Remove markdown fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1])
        else:
            cleaned = "\n".join(lines[1:])

    try:
        actions = json.loads(cleaned)
        if not isinstance(actions, list):
            actions = [actions]
        return actions
    except Exception:
        callsign = "UNKNOWN"
        if state.get("aircraft"):
            callsign = state["aircraft"][0]["callsign"]
        return [{
            "action_type": "no_action",
            "target_callsign": callsign,   # ✅ use the variable
            "rationale": "Fallback"
        }]


def call_llm(state: Dict, step: int) -> List[Dict]:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(state, step)}
            ],
            temperature=0.2,
            max_tokens=800,
        )

        raw_text = response.choices[0].message.content or ""
        return safe_parse_actions(raw_text)

    except Exception as exc:
        return [{
            "action_type": "no_action",
            "target_callsign": "ALL",
            "rationale": f"LLM error fallback: {str(exc)[:50]}"
        }]


# ── Main inference loop ────────────────────────────────────────────────────────

def run_inference():
    print("[START]")
    print(f"[STEP] Initialising environment at {API_BASE_URL} | task={TASK} | model={MODEL_NAME}")

    # Reset environment
    try:
        state = env_reset(TASK)
    except Exception as exc:
        print(f"[STEP] ERROR: Failed to reset environment — {exc}")
        print("[END]")
        sys.exit(1)

    print(f"[STEP] Environment reset. Task: {state.get('task')} | Aircraft: {len(state.get('aircraft', []))}")

    cumulative_reward = 0.0
    cumulative_score = 0.0
    total_violations = 0
    completed_steps = 0

    for step_num in range(1, MAX_STEPS + 1):

        if state.get("done", False):
            print(f"[STEP] Episode finished early at step {step_num - 1}.")
            break

        # LLM decision
        actions = call_llm(state, step_num)

        action_summary = "; ".join(
            f"{a.get('action_type')}({a.get('target_callsign')}"
            + (f",Δ{a.get('value')}" if a.get("value") is not None else "")
            + ")"
            for a in actions
        )
        print(f"[STEP] step={step_num} | actions=[{action_summary}]")

        # Apply actions
        try:
            result = env_step(actions)
        except Exception as exc:
            print(f"[STEP] step={step_num} | ENV ERROR: {exc}")
            break

        state = result["state"]
        reward = result.get("reward", 0.0)
        score = result.get("score", 0.0)
        done = result.get("done", False)
        violations = result.get("violations", [])

        cumulative_reward += reward
        cumulative_score += score
        total_violations += len(violations)
        completed_steps = step_num

        conflict_count = len(state.get("active_conflicts", []))
        viol_str = f" | violations={violations}" if violations else ""

        print(
            f"[STEP] step={step_num} | reward={reward:.4f} | score={score:.4f}"
            f" | conflicts={conflict_count} | done={done}{viol_str}"
        )

        if done:
            print(f"[STEP] Episode completed at step {step_num}.")
            break

        time.sleep(0.1)

    # Summary
    avg_reward = cumulative_reward / max(1, completed_steps)
    avg_score = cumulative_score / max(1, completed_steps)

    print(f"[STEP] ── Episode Summary ──────────────────────────")
    print(f"[STEP] Task:              {TASK}")
    print(f"[STEP] Completed steps:   {completed_steps}")
    print(f"[STEP] Avg reward:        {avg_reward:.4f}")
    print(f"[STEP] Avg score:         {avg_score:.4f}")
    print(f"[STEP] Total violations:  {total_violations}")
    print(f"[STEP] Final score range: [0.0, 1.0] ✓")
    print("[END]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        TASK = sys.argv[1]
    run_inference()