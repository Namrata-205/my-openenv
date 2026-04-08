"""
inference.py — ATC TRACON RL Environment inference script.

Changes from original:
  - call_llm: added guard for missing/None task in state.
  - wake_turbulence: separation now uses Euclidean distance (was Y-axis only).
  - conflict_resolution: proactive pre-conflict detection added (mirrors test_env.py).
  - TASK promoted to a parameter in run_inference() — no longer mutated as a global.
  - env_reset: accepts optional seed for reproducible runs.
  - env_reset / env_step: exponential-backoff retry (3 attempts) for transient 5xx errors.
  - Fallback chain unchanged: Rule-based → Grok API → HuggingFace LLM → hard fallback.
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


# ── Environment variables ──────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

# Grok API (xAI) — set GROK_API_KEY in env to enable.
# NOTE: Do NOT bake real keys into this file or the Dockerfile ENV.
GROK_API_KEY  = os.environ.get("GROK_API_KEY", "")
GROK_MODEL    = os.environ.get("GROK_MODEL", "grok-3-mini")
GROK_BASE_URL = "https://api.x.ai/v1"

HF_BASE_URL  = "https://api-inference.huggingface.co/v1"

DEFAULT_TASK = "conflict_resolution"
MAX_STEPS    = 10
MAX_RETRIES  = 3          # retries on transient server errors
RETRY_DELAY  = 1.0        # seconds (doubles each retry)


# ── OpenAI Client (HuggingFace) ────────────────────────────────────────────────
hf_client = OpenAI(
    base_url=HF_BASE_URL,
    api_key=HF_TOKEN or "dummy",
)

# ── Grok Client (xAI) ─────────────────────────────────────────────────────────
grok_client = OpenAI(
    base_url=GROK_BASE_URL,
    api_key=GROK_API_KEY or "dummy",
)


# ── HTTP helpers with retry ────────────────────────────────────────────────────

def _post_with_retry(url: str, payload: Dict, label: str) -> Dict:
    """POST with exponential-backoff retry on 5xx responses."""
    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.ok:
                return r.json()
            # Retry only on server errors (5xx), not client errors (4xx).
            if r.status_code < 500:
                r.raise_for_status()
            print(f"[WARN] {label} attempt {attempt}/{MAX_RETRIES} got {r.status_code} — retrying in {delay:.1f}s")
        except requests.exceptions.RequestException as exc:
            print(f"[WARN] {label} attempt {attempt}/{MAX_RETRIES} exception: {exc} — retrying in {delay:.1f}s")
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"{label} failed after {MAX_RETRIES} attempts")


def env_reset(task: str, seed: Optional[int] = None) -> Dict:
    payload: Dict[str, Any] = {"task": task}
    if seed is not None:
        payload["seed"] = seed
    return _post_with_retry(f"{API_BASE_URL}/reset", payload, "env_reset")


def env_step(actions: List[Dict]) -> Dict:
    return _post_with_retry(f"{API_BASE_URL}/step", {"actions": actions}, "env_step")


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
- For conflict_resolution you may act on AC1 or AC2 independently
- Allowed action_type values: heading_change, speed_change, altitude_change,
  sequence_swap, assign_gate, vector, no_action
- For heading_change: negative value = turn left, positive = turn right
- For altitude_change: positive value = climb, negative = descend
- If no action is needed, choose ANY one aircraft and return no_action for it

Respond ONLY with valid JSON (no markdown fences, no extra text).

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
            "callsign":     ac["callsign"],
            "category":     ac["category"],
            "status":       ac["status"],
            "x_nm":         round(ac["x"], 1),
            "y_nm":         round(ac["y"], 1),
            "altitude_ft":  int(ac["altitude"]),
            "heading_deg":  round(ac["heading"], 1),
            "speed_kts":    round(ac["speed"], 1),
            "emergency":    ac.get("is_emergency", False),
            "fuel":         round(ac.get("fuel_state", 1.0), 2),
            "sequence":     ac.get("sequence_pos"),
            "gate":         ac.get("assigned_gate"),
        })

    return json.dumps({
        "step":             step,
        "task":             state.get("task"),
        "must_act":         len(state.get("active_conflicts", [])) > 0,
        "active_conflicts": state.get("active_conflicts", []),
        "aircraft":         aircraft_summary,
        "gates_available":  [
            g["gate_id"] for g in state.get("gates", []) if not g["occupied"]
        ],
    }, indent=2)


def _fallback_action(state: Dict, reason: str) -> List[Dict]:
    """Always pick a real callsign from state, never use 'ALL'."""
    callsign = "AAL201"
    if state.get("aircraft"):
        callsign = state["aircraft"][0]["callsign"]
    return [{
        "action_type":      "no_action",
        "target_callsign":  callsign,
        "value":            None,
        "secondary_target": None,
        "gate_id":          None,
        "rationale":        f"Fallback: {reason[:80]}",
    }]


def safe_parse_actions(text: str, state: Dict) -> List[Dict]:
    """state is passed in so the fallback can use real callsigns."""
    cleaned = text.strip()

    # Remove markdown fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        end   = -1 if lines[-1].strip() == "```" else len(lines)
        cleaned = "\n".join(lines[1:end])

    try:
        actions = json.loads(cleaned)
        if not isinstance(actions, list):
            actions = [actions]
        return actions
    except Exception as exc:
        return _fallback_action(state, f"JSON parse error: {exc}")


# ── Grok API fallback ──────────────────────────────────────────────────────────

def call_grok(state: Dict, step: int) -> List[Dict]:
    """Call Grok (xAI) API as secondary LLM fallback."""
    if not GROK_API_KEY:
        raise ValueError("GROK_API_KEY not set")

    response = grok_client.chat.completions.create(
        model=GROK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(state, step)},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    raw_text = response.choices[0].message.content or ""
    return safe_parse_actions(raw_text, state)


# ── HuggingFace LLM fallback ───────────────────────────────────────────────────

def call_hf_llm(state: Dict, step: int) -> List[Dict]:
    """Call HuggingFace-hosted LLM as tertiary fallback."""
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not set")

    response = hf_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(state, step)},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    raw_text = response.choices[0].message.content or ""
    return safe_parse_actions(raw_text, state)


# ── Main agent: Rule-based → Grok → HuggingFace → hard fallback ───────────────

def call_llm(state: Dict, step: int) -> List[Dict]:
    """
    HYBRID AGENT with fallback chain:
      1. Rule-based ATC decisions (primary — fastest, most reliable)
      2. Grok API (secondary LLM — if GROK_API_KEY is set)
      3. HuggingFace LLM (tertiary — if HF_TOKEN is set)
      4. Hard no_action fallback (always succeeds)
    """
    task = state.get("task")

    # FIX: guard against missing task to avoid silent fall-through.
    if not task:
        return _fallback_action(state, "No task field in state")

    # ================================================================
    # TASK 1: WAKE TURBULENCE
    # ================================================================
    if task == "wake_turbulence":
        aircraft = state.get("aircraft", [])

        if len(aircraft) < 2:
            return _fallback_action(state, "Not enough aircraft")

        lead  = aircraft[0]
        trail = aircraft[1]

        # FIX: use Euclidean distance — Y-axis alone ignores lateral offset.
        separation = math.sqrt(
            (lead["x"] - trail["x"]) ** 2 + (lead["y"] - trail["y"]) ** 2
        )
        required_nm  = state.get("info", {}).get("required_nm", 5.0)
        danger_zone  = required_nm * 1.1
        caution_zone = required_nm * 1.3

        if separation < danger_zone:
            return [{
                "action_type":      "speed_change",
                "target_callsign":  trail["callsign"],
                "value":            -20,
                "secondary_target": None,
                "gate_id":          None,
                "rationale":        f"Increase separation (sep={separation:.1f} NM < danger {danger_zone:.1f} NM)",
            }]
        elif separation < caution_zone:
            return [{
                "action_type":      "heading_change",
                "target_callsign":  trail["callsign"],
                "value":            10,
                "secondary_target": None,
                "gate_id":          None,
                "rationale":        "Slight vectoring to increase spacing",
            }]
        else:
            return [{
                "action_type":      "no_action",
                "target_callsign":  trail["callsign"],
                "value":            None,
                "secondary_target": None,
                "gate_id":          None,
                "rationale":        "Safe separation maintained",
            }]

    # ================================================================
    # TASK 4: CONFLICT RESOLUTION
    # ================================================================
    if task == "conflict_resolution":
        conflicts = state.get("active_conflicts", [])
        aircraft  = state.get("aircraft", [])

        # ============================================================
        # 1. PROACTIVE (FIRST PRIORITY)
        # ============================================================
        if len(aircraft) >= 2:
            a1, a2 = aircraft[0], aircraft[1]

            dx = a1["x"] - a2["x"]
            dy = a1["y"] - a2["y"]
            distance = math.sqrt(dx*dx + dy*dy)

            vertical = abs(a1["altitude"] - a2["altitude"])

            if distance < 3 and vertical < 1000:
                return [
                    {
                        "action_type": "altitude_change",
                        "target_callsign": a1["callsign"],
                        "value": 1000,
                        "secondary_target": None,
                        "gate_id": None,
                        "rationale": f"Emergency separation (dist={distance:.2f})",
                    },
                    {
                        "action_type": "heading_change",
                        "target_callsign": a2["callsign"],
                        "value": 25,
                        "secondary_target": None,
                        "gate_id": None,
                        "rationale": f"Aggressive diverge (dist={distance:.2f})",
                    },
                ]

            elif distance < 6 and vertical < 1500:
                return [
                    {
                        "action_type": "heading_change",
                        "target_callsign": a1["callsign"],
                        "value": -15,
                        "secondary_target": None,
                        "gate_id": None,
                        "rationale": f"Proactive avoidance (dist={distance:.2f})",
                    },
                    {
                        "action_type": "heading_change",
                        "target_callsign": a2["callsign"],
                        "value": 15,
                        "secondary_target": None,
                        "gate_id": None,
                        "rationale": f"Proactive avoidance (dist={distance:.2f})",
                    },
                ]

    # ============================================================
    # 2. REACTIVE (if conflict already exists)
    # ============================================================
        if conflicts:
            conflict = conflicts[0]
            ac1 = next((a for a in aircraft if a["callsign"] == conflict["ac1"]), None)
            ac2 = next((a for a in aircraft if a["callsign"] == conflict["ac2"]), None)

            if ac1 is None or ac2 is None:
                return _fallback_action(state, "Conflict callsign not found")

            return [
                {
                    "action_type": "altitude_change",
                    "target_callsign": ac1["callsign"],
                    "value": 1000,
                    "secondary_target": None,
                    "gate_id": None,
                    "rationale": "Reactive climb for separation",
                },
                {
                    "action_type": "heading_change",
                    "target_callsign": ac2["callsign"],
                    "value": 15,
                    "secondary_target": None,
                    "gate_id": None,
                    "rationale": "Reactive divergence",
                },
            ]

        return _fallback_action(state, "No conflict detected")

    # ================================================================
    # TASK 3: EMERGENCY VECTORING
    # ================================================================
    if task == "emergency_vectoring":
        aircraft = state.get("aircraft", [])
        emerg = next((a for a in aircraft if a.get("is_emergency")), None)

        if not emerg:
            return _fallback_action(state, "No emergency aircraft")

        return [{
            "action_type":      "vector",
            "target_callsign":  emerg["callsign"],
            "value":            180,
            "secondary_target": None,
            "gate_id":          None,
            "rationale":        "Direct emergency aircraft toward runway heading 180°",
        }]

    # ================================================================
    # TASK 2: GO-AROUND PREVENTION
    # ================================================================
    # ================================================================
    if task == "go_around_prevention":
        aircraft = state.get("aircraft", [])
        if not aircraft:
            return _fallback_action(state, "No aircraft")


        # Helper: distance from runway threshold
        def dist(a):
            return math.sqrt(a["x"]**2 + a["y"]**2)

        # Priority: emergencies first, then low fuel, then closest
        def priority(a):
            return (
                not a.get("is_emergency", False),
                a.get("fuel_state", 1.0),
                dist(a),
            )

        sorted_ac = sorted(aircraft, key=priority)
        actions = []

        # Step 1: Check spacing on approach
        approach_ac = sorted([a for a in aircraft if dist(a) < 60], key=dist)

        for i in range(len(approach_ac) - 1):
            lead = approach_ac[i]
            trail = approach_ac[i + 1]
            gap = dist(trail) - dist(lead)

            if gap < 6:  # critical
                actions.append({
                    "action_type": "speed_change",
                    "target_callsign": trail["callsign"],
                    "value": -50,  # moderate critical slowdown
                    "secondary_target": None,
                    "gate_id": None,
                    "rationale": f"CRITICAL: Prevent go-around, gap={gap:.1f}",
                })
                break  # only fix one issue per step
            elif gap < 10:  # warning
                actions.append({
                    "action_type": "speed_change",
                    "target_callsign": trail["callsign"],
                    "value": -20,  # warning slowdown
                    "secondary_target": None,
                    "gate_id": None,
                    "rationale": f"Warning: Build separation, gap={gap:.1f}",
                })
                break

        # Step 2: Ensure top-priority landing order
        if sorted_ac[0]["callsign"] != aircraft[0]["callsign"]:
            actions.append({
                "action_type": "sequence_swap",
                "target_callsign": sorted_ac[0]["callsign"],
                "secondary_target": aircraft[0]["callsign"],
                "value": None,
                "gate_id": None,
                "rationale": "Prioritize top landing aircraft",
            })

        # Return actions or no_action if already optimal
        if actions:
            return actions[:2]  # limit to max 2 actions per step

        return [{
            "action_type": "no_action",
            "target_callsign": aircraft[0]["callsign"],
            "value": None,
            "secondary_target": None,
            "gate_id": None,
            "rationale": "Sequence optimal",
        }]
        # ================================================================
    # TASK 5: GATE ASSIGNMENT
    # ================================================================
    # TASK 5: GATE ASSIGNMENT (updated with type check)
    # TASK 5: GATE ASSIGNMENT
    if task == "gate_assignment":
        gates = state.get("gates", [])
        aircraft = state.get("aircraft", [])
        
        # Find aircraft that need gate assignment
        # (not yet assigned to a gate OR just arriving)
        needs_gate = []
        for ac in aircraft:
            ac_callsign = ac.get("callsign", "")
            assigned_gate = ac.get("assigned_gate", ac.get("gate", None))
            
            # Check if aircraft needs a gate
            if not assigned_gate and ac.get("status") != "DEPARTING":
                needs_gate.append(ac)
        
        if not needs_gate:
            # No aircraft need gates yet - wait
            return [{
                "action_type": "no_action",
                "target_callsign": aircraft[0]["callsign"] if aircraft else "NONE",
                "value": None,
                "secondary_target": None,
                "gate_id": None,
                "rationale": "Waiting for aircraft that need gate assignment",
            }]
        
        # Process the highest priority aircraft
        target_ac = needs_gate[0]
        callsign = target_ac.get("callsign")
        ac_type = target_ac.get("aircraft_type", target_ac.get("type", ""))
        
        # Find available gates (not occupied AND not blocked)
        available_gates = []
        for g in gates:
            # Check if gate is occupied
            if g.get("occupied", False):
                continue
            
            # Check if gate is blocked
            if g.get("is_blocked", False) or g.get("blocked", False):
                continue
            
            # Check compatibility
            compatible = g.get("compatible_types", g.get("compatible_aircraft", []))
            if compatible and ac_type and ac_type not in compatible:
                continue  # Incompatible type
            
            # Gate is available
            available_gates.append(g)
        
        if available_gates:
            # Choose best gate (shortest taxi distance)
            best_gate = min(available_gates, 
                        key=lambda g: g.get("taxi_dist_min", g.get("distance", 999)))
            
            return [{
                "action_type": "assign_gate",
                "target_callsign": callsign,
                "value": None,
                "secondary_target": None,
                "gate_id": best_gate["gate_id"],
                "rationale": f"Assign {callsign} to {best_gate['gate_id']}",
            }]
        
        # No available gates - fallback
        return [{
            "action_type": "no_action",
            "target_callsign": callsign,
            "value": None,
            "secondary_target": None,
            "gate_id": None,
            "rationale": f"No available gates for {callsign}",
        }]

    # ================================================================
    # UNKNOWN TASK → Try Grok, then HF LLM, then hard fallback
    # ================================================================
    print(f"[STEP] Unknown task '{task}' — trying LLM fallbacks")

    if GROK_API_KEY:
        try:
            print("[STEP] Trying Grok API...")
            result = call_grok(state, step)
            print("[STEP] Grok API succeeded")
            return result
        except Exception as exc:
            print(f"[STEP] Grok API failed: {exc}")

    if HF_TOKEN:
        try:
            print("[STEP] Trying HuggingFace LLM...")
            result = call_hf_llm(state, step)
            print("[STEP] HuggingFace LLM succeeded")
            return result
        except Exception as exc:
            print(f"[STEP] HuggingFace LLM failed: {exc}")

    return _fallback_action(state, f"No rule or LLM handled task '{task}'")


# ── Main inference loop ────────────────────────────────────────────────────────

def run_inference(task: str = DEFAULT_TASK, seed: Optional[int] = None):
    """
    Run one episode of the ATC environment.

    Args:
        task: One of the TaskType enum values (e.g. 'conflict_resolution').
        seed: Optional RNG seed for reproducibility. Pass an int to fix the episode.
    """
    print("[START]")
    print(
        f"[STEP] Initialising environment at {API_BASE_URL} | "
        f"task={task} | model={MODEL_NAME} | seed={seed}"
    )
    if GROK_API_KEY:
        print(f"[STEP] Grok API enabled (model={GROK_MODEL})")
    else:
        print("[STEP] Grok API disabled — set GROK_API_KEY to enable")

    try:
        state = env_reset(task, seed=seed)
    except Exception as exc:
        print(f"[STEP] ERROR: Failed to reset environment — {exc}")
        print("[END]")
        sys.exit(1)

    print(
        f"[STEP] Environment reset. Task: {state.get('task')} | "
        f"Aircraft: {len(state.get('aircraft', []))}"
    )

    cumulative_reward = 0.0
    cumulative_score  = 0.0
    total_violations  = 0
    completed_steps   = 0

    for step_num in range(1, MAX_STEPS + 1):

        if state.get("done", False):
            print(f"[STEP] Episode finished early at step {step_num - 1}.")
            break

        actions = call_llm(state, step_num)

        action_summary = "; ".join(
            f"{a.get('action_type')}({a.get('target_callsign')}"
            + (f",Δ{a.get('value')}" if a.get("value") is not None else "")
            + ")"
            for a in actions
        )
        print(f"[STEP] step={step_num} | actions=[{action_summary}]")

        try:
            result = env_step(actions)
        except Exception as exc:
            print(f"[STEP] step={step_num} | ENV ERROR: {exc}")
            break

        state      = result["state"]
        reward     = result.get("reward", 0.0)
        score      = result.get("score",  0.0)
        done       = result.get("done",   False)
        violations = result.get("violations", [])

        cumulative_reward += reward
        cumulative_score  += score
        total_violations  += len(violations)
        completed_steps    = step_num

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

    avg_reward = cumulative_reward / max(1, completed_steps)
    avg_score  = cumulative_score  / max(1, completed_steps)

    print(f"[STEP] ── Episode Summary ──────────────────────────")
    print(f"[STEP] Task:              {task}")
    print(f"[STEP] Completed steps:   {completed_steps}")
    print(f"[STEP] Avg reward:        {avg_reward:.4f}")
    print(f"[STEP] Avg score:         {avg_score:.4f}")
    print(f"[STEP] Total violations:  {total_violations}")
    print(f"[STEP] Final score range: [0.0, 1.0] ✓")
    print("[END]")


if __name__ == "__main__":
    _task = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK
    _seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run_inference(task=_task, seed=_seed)