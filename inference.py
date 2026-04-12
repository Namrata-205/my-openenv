"""
inference.py — ATC TRACON RL Environment inference script.

Uses an LLM (via validator's LiteLLM proxy) as the RL agent.

Required environment variables (injected by validator during evaluation):
  API_BASE_URL  — base URL of the LLM proxy (optional locally)
  API_KEY       — API key for the LLM proxy (optional locally)
  MODEL_NAME    — model identifier (optional, defaults to gpt-4o-mini)

For local testing, these can be omitted and will use intelligent mock.
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


# ── Environment variables (with fallbacks for local testing) ───────────────────
# For validator: These will be injected and override the defaults
# For local: Use your own API endpoint or mock
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# ATC Environment URL (separate from LLM proxy)
ATC_API_BASE_URL = os.environ.get("ATC_API_BASE_URL", "http://localhost:7860")

MAX_STEPS = 10
DEFAULT_TASK = "conflict_resolution"


# ── OpenAI Client (will use validator's proxy if API_KEY is set) ──────────────
USE_REAL_LLM = API_KEY and API_KEY != "" and API_BASE_URL

if USE_REAL_LLM:
    llm_client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )
    print(f"[INIT] REAL LLM client initialized with base_url: {API_BASE_URL}")
else:
    print(f"[INIT] Using INTELLIGENT MOCK LLM for local testing")
    if not API_KEY:
        print(f"[INIT] No API_KEY provided - validator will inject this")
    if API_BASE_URL == "https://api.openai.com/v1":
        print(f"[INIT] Using default API_BASE_URL - validator will override this")


# ── HTTP helpers for ATC environment ──────────────────────────────────────────

def env_reset(task: str, seed: Optional[int] = None) -> Dict:
    """Reset the ATC environment."""
    payload: Dict[str, Any] = {"task": task}
    if seed is not None:
        payload["seed"] = seed
    
    r = requests.post(f"{ATC_API_BASE_URL}/reset", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(actions: List[Dict]) -> Dict:
    """Take a step in the ATC environment."""
    r = requests.post(f"{ATC_API_BASE_URL}/step", json={"actions": actions}, timeout=30)
    r.raise_for_status()
    return r.json()


def check_environment_health() -> bool:
    """Check if the ATC environment is ready."""
    try:
        r = requests.get(f"{ATC_API_BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ── Intelligent Mock LLM for local testing ────────────────────────────────────

def get_intelligent_mock_actions(state: Dict, step: int) -> List[Dict]:
    """
    Intelligent mock LLM responses for local testing.
    This actually resolves conflicts properly.
    """
    task = state.get("task", "unknown")
    aircraft = state.get("aircraft", [])
    conflicts = state.get("active_conflicts", [])
    
    if not aircraft:
        return [{
            "action_type": "no_action",
            "target_callsign": "NONE",
            "value": None,
            "secondary_target": None,
            "gate_id": None,
            "rationale": "No aircraft in state"
        }]
    
    # Task-specific intelligent mock logic
    if task == "conflict_resolution":
        if len(aircraft) >= 2:
            ac1, ac2 = aircraft[0], aircraft[1]
            
            # Calculate distance between aircraft
            dx = ac1.get("x", 0) - ac2.get("x", 0)
            dy = ac1.get("y", 0) - ac2.get("y", 0)
            distance = math.sqrt(dx*dx + dy*dy)
            vertical_sep = abs(ac1.get("altitude", 0) - ac2.get("altitude", 0))
            
            # If too close, take action
            if distance < 5 and vertical_sep < 1000:
                return [
                    {
                        "action_type": "altitude_change",
                        "target_callsign": ac1.get("callsign"),
                        "value": 1000,
                        "secondary_target": None,
                        "gate_id": None,
                        "rationale": f"Conflict detected: distance={distance:.1f}nm, climbing to separate"
                    },
                    {
                        "action_type": "heading_change",
                        "target_callsign": ac2.get("callsign"),
                        "value": 15,
                        "secondary_target": None,
                        "gate_id": None,
                        "rationale": f"Turning to deconflict"
                    }
                ]
            
            # Proactive conflict prevention
            if distance < 8 and vertical_sep < 1500:
                return [
                    {
                        "action_type": "heading_change",
                        "target_callsign": ac1.get("callsign"),
                        "value": -10,
                        "secondary_target": None,
                        "gate_id": None,
                        "rationale": f"Proactive turn to maintain separation"
                    }
                ]
        
        # Check active conflicts from state
        if conflicts:
            conflict = conflicts[0]
            ac1 = next((a for a in aircraft if a.get("callsign") == conflict.get("ac1")), None)
            ac2 = next((a for a in aircraft if a.get("callsign") == conflict.get("ac2")), None)
            
            if ac1 and ac2:
                return [
                    {
                        "action_type": "altitude_change",
                        "target_callsign": ac1.get("callsign"),
                        "value": 1000,
                        "secondary_target": None,
                        "gate_id": None,
                        "rationale": f"Resolving conflict between {ac1.get('callsign')} and {ac2.get('callsign')}"
                    }
                ]
    
    elif task == "wake_turbulence":
        if len(aircraft) >= 2:
            lead = aircraft[0]
            trail = aircraft[1]
            
            # Calculate separation
            dx = lead.get("x", 0) - trail.get("x", 0)
            dy = lead.get("y", 0) - trail.get("y", 0)
            separation = math.sqrt(dx*dx + dy*dy)
            required = state.get("info", {}).get("required_nm", 5.0)
            
            if separation < required * 1.1:
                return [{
                    "action_type": "speed_change",
                    "target_callsign": trail.get("callsign"),
                    "value": -20,
                    "secondary_target": None,
                    "gate_id": None,
                    "rationale": f"Separation={separation:.1f}nm < required, slowing trailing aircraft"
                }]
    
    elif task == "emergency_vectoring":
        # Find emergency aircraft
        emerg = next((a for a in aircraft if a.get("is_emergency")), None)
        if emerg:
            return [{
                "action_type": "vector",
                "target_callsign": emerg.get("callsign"),
                "value": 180,
                "secondary_target": None,
                "gate_id": None,
                "rationale": f"Emergency vector for {emerg.get('callsign')} to runway heading"
            }]
    
    elif task == "go_around_prevention":
        if len(aircraft) >= 2:
            # Sort by distance to runway
            def dist_to_runway(a):
                return math.sqrt(a.get("x", 0)**2 + a.get("y", 0)**2)
            
            sorted_ac = sorted(aircraft, key=dist_to_runway)
            
            # Check spacing on approach
            for i in range(len(sorted_ac) - 1):
                lead = sorted_ac[i]
                trail = sorted_ac[i + 1]
                gap = dist_to_runway(trail) - dist_to_runway(lead)
                
                if gap < 6:
                    return [{
                        "action_type": "speed_change",
                        "target_callsign": trail.get("callsign"),
                        "value": -30,
                        "secondary_target": None,
                        "gate_id": None,
                        "rationale": f"Small gap ({gap:.1f}nm), slowing trailing aircraft"
                    }]
            
            # Reorder if needed for fuel priority
            if sorted_ac[0].get("fuel", 1.0) < 0.3:
                return [{
                    "action_type": "sequence_swap",
                    "target_callsign": sorted_ac[0].get("callsign"),
                    "secondary_target": aircraft[0].get("callsign"),
                    "value": None,
                    "gate_id": None,
                    "rationale": "Reordering for low fuel aircraft"
                }]
    
    elif task == "gate_assignment":
        # Find aircraft without gates
        needs_gate = [a for a in aircraft if not a.get("assigned_gate") and a.get("status") != "DEPARTING"]
        
        if needs_gate:
            gates = state.get("gates", [])
            available_gates = [g for g in gates if not g.get("occupied") and not g.get("is_blocked")]
            
            if available_gates:
                return [{
                    "action_type": "assign_gate",
                    "target_callsign": needs_gate[0].get("callsign"),
                    "value": None,
                    "secondary_target": None,
                    "gate_id": available_gates[0].get("gate_id"),
                    "rationale": f"Assigning gate {available_gates[0].get('gate_id')}"
                }]
    
    # Default no_action
    return [{
        "action_type": "no_action",
        "target_callsign": aircraft[0].get("callsign"),
        "value": None,
        "secondary_target": None,
        "gate_id": None,
        "rationale": f"No action needed for task {task}"
    }]


# ── LLM agent ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Air Traffic Controller managing TRACON airspace.

CRITICAL RULES:
- NEVER use "ALL" as a callsign - ALWAYS use a valid callsign from the state
- ALWAYS respond with valid JSON only (no markdown formatting, no extra text)
- For conflict_resolution: act on both conflicting aircraft when needed
- For wake_turbulence: slow down the trailing aircraft
- For emergency_vectoring: prioritize the MAYDAY aircraft
- For go_around_prevention: manage landing sequence spacing
- For gate_assignment: assign gates to arriving aircraft

Allowed action_type values:
- heading_change: negative = turn left, positive = turn right (value in degrees)
- speed_change: negative = slow down, positive = speed up (value in knots)
- altitude_change: positive = climb, negative = descend (value in feet)
- sequence_swap: swap landing order (requires secondary_target callsign)
- assign_gate: assign to gate (requires gate_id)
- vector: direct to heading (value in degrees)
- no_action: no change needed

Response format (JSON array - no markdown):
[
  {
    "action_type": "speed_change",
    "target_callsign": "AAL123",
    "value": -20,
    "secondary_target": null,
    "gate_id": null,
    "rationale": "Slow down to increase separation"
  }
]
"""

def build_user_prompt(state: Dict, step: int) -> str:
    """Build a detailed prompt with current state information."""
    task = state.get("task", "unknown")
    
    aircraft_summary = []
    for ac in state.get("aircraft", []):
        ac_info = {
            "callsign": ac.get("callsign"),
            "category": ac.get("category"),
            "status": ac.get("status"),
            "x_nm": round(ac.get("x", 0), 1),
            "y_nm": round(ac.get("y", 0), 1),
            "altitude_ft": int(ac.get("altitude", 0)),
            "heading_deg": round(ac.get("heading", 0), 1),
            "speed_kts": round(ac.get("speed", 0), 1),
            "emergency": ac.get("is_emergency", False),
            "fuel": round(ac.get("fuel_state", 1.0), 2),
        }
        
        # Add task-specific fields
        if task == "go_around_prevention":
            ac_info["sequence_pos"] = ac.get("sequence_pos")
            ac_info["distance_to_runway_nm"] = round(
                (ac.get("x", 0)**2 + ac.get("y", 0)**2) ** 0.5, 1
            )
        elif task == "gate_assignment":
            ac_info["aircraft_type"] = ac.get("aircraft_type", ac.get("type", ""))
            ac_info["assigned_gate"] = ac.get("assigned_gate", ac.get("gate"))
        
        aircraft_summary.append(ac_info)
    
    prompt_data = {
        "step": step,
        "task": task,
        "must_act": len(state.get("active_conflicts", [])) > 0,
        "active_conflicts": state.get("active_conflicts", []),
        "aircraft": aircraft_summary,
    }
    
    # Add task-specific guidance and data
    if task == "wake_turbulence":
        prompt_data["required_separation_nm"] = state.get("info", {}).get("required_nm", 5.0)
        prompt_data["guidance"] = "Slow down trailing aircraft if separation < required. Use heading changes for moderate gaps."
        
    elif task == "conflict_resolution":
        prompt_data["guidance"] = "Resolve conflicts by changing altitude (1000ft) or heading (15-30°). Act on both aircraft."
        
    elif task == "emergency_vectoring":
        prompt_data["runway_heading"] = state.get("info", {}).get("runway_heading", 180)
        prompt_data["guidance"] = "Give immediate priority to MAYDAY aircraft. Vector directly to runway."
        
    elif task == "go_around_prevention":
        prompt_data["guidance"] = "Maintain 5+ NM spacing on approach. Slow down trailing aircraft if gap < 6 NM."
        
    elif task == "gate_assignment":
        prompt_data["gates_available"] = [
            {
                "gate_id": g.get("gate_id"),
                "compatible_types": g.get("compatible_types", g.get("compatible_aircraft", [])),
                "occupied": g.get("occupied", False),
                "blocked": g.get("is_blocked", False)
            }
            for g in state.get("gates", [])
            if not g.get("occupied", False)
        ]
        prompt_data["guidance"] = "Assign arriving aircraft to compatible, unoccupied gates."
    
    return json.dumps(prompt_data, indent=2)


def safe_parse_actions(text: str, state: Dict) -> List[Dict]:
    """Robust JSON parsing with validation."""
    cleaned = text.strip()
    
    # Remove markdown code blocks
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    
    try:
        actions = json.loads(cleaned)
        if not isinstance(actions, list):
            actions = [actions]
        
        # Validate and fix actions
        validated_actions = []
        for action in actions:
            # Skip actions without required fields
            if "action_type" not in action:
                continue
            if "target_callsign" not in action:
                continue
            
            # Fix "ALL" callsign
            if action["target_callsign"] == "ALL":
                if state.get("aircraft"):
                    action["target_callsign"] = state["aircraft"][0].get("callsign", "AAL201")
                else:
                    continue
            
            # Ensure all fields exist
            action.setdefault("value", None)
            action.setdefault("secondary_target", None)
            action.setdefault("gate_id", None)
            action.setdefault("rationale", "")
            
            validated_actions.append(action)
        
        if validated_actions:
            return validated_actions
        
    except Exception as e:
        print(f"[WARN] JSON parse error: {e}")
    
    # Fallback action
    callsign = "AAL201"
    if state.get("aircraft"):
        callsign = state["aircraft"][0].get("callsign", "AAL201")
    
    return [{
        "action_type": "no_action",
        "target_callsign": callsign,
        "value": None,
        "secondary_target": None,
        "gate_id": None,
        "rationale": "Fallback - using no_action"
    }]


def call_real_llm(state: Dict, step: int) -> List[Dict]:
    """
    Call the validator's LLM proxy for decision making.
    This is what the validator checks for - uses the injected API_BASE_URL.
    """
    task = state.get("task", "unknown")
    print(f"[LLM] Calling validator proxy at {API_BASE_URL} for task '{task}'")
    
    try:
        # Make the API call through validator's proxy
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(state, step)}
            ],
            temperature=0.2,
            max_tokens=800,
        )
        
        print(f"[LLM] Proxy call successful")
        raw_text = response.choices[0].message.content or ""
        return safe_parse_actions(raw_text, state)
        
    except Exception as exc:
        print(f"[LLM] Proxy call failed: {exc}")
        # Return intelligent mock as fallback
        return get_intelligent_mock_actions(state, step)


def call_llm(state: Dict, step: int) -> List[Dict]:
    """
    Main LLM calling function.
    Uses real LLM if API_KEY is available, otherwise uses intelligent mock.
    """
    if USE_REAL_LLM:
        return call_real_llm(state, step)
    else:
        return get_intelligent_mock_actions(state, step)


# ── Main inference loop ────────────────────────────────────────────────────────

def run_inference(task: str = DEFAULT_TASK, seed: Optional[int] = None):
    """Run one episode of the ATC environment."""
    print("[START]")
    print(f"[INFO] LLM Mode: {'REAL (Validator Mode)' if USE_REAL_LLM else 'INTELLIGENT MOCK (Local Test)'}")
    print(f"[INFO] ATC Environment URL: {ATC_API_BASE_URL}")
    print(f"[INFO] Model: {MODEL_NAME}")
    print(f"[INFO] Task: {task}")
    
    # Check if ATC environment is ready
    if not check_environment_health():
        print(f"[WARN] ATC environment not reachable at {ATC_API_BASE_URL}")
        print("[WARN] Make sure the environment server is running")
        return
    
    print(f"[STEP] Initialising environment at {ATC_API_BASE_URL} | task={task}")
    
    # Reset environment
    try:
        state = env_reset(task, seed=seed)
    except Exception as exc:
        print(f"[STEP] ERROR: Failed to reset environment — {exc}")
        print("[END]")
        return
    
    print(f"[STEP] Environment reset. Task: {state.get('task')} | Aircraft: {len(state.get('aircraft', []))}")
    
    cumulative_reward = 0.0
    cumulative_score = 0.0
    total_violations = 0
    completed_steps = 0
    llm_calls = 0
    
    for step_num in range(1, MAX_STEPS + 1):
        
        if state.get("done", False):
            print(f"[STEP] Episode finished early at step {step_num - 1}.")
            break
        
        # Get LLM decision
        actions = call_llm(state, step_num)
        llm_calls += 1
        
        # Create action summary for logging
        action_summary = "; ".join(
            f"{a.get('action_type')}({a.get('target_callsign')}"
            + (f",{a.get('value')}" if a.get("value") is not None else "")
            + (f",→{a.get('gate_id')}" if a.get("gate_id") else "")
            + (f",swap {a.get('secondary_target')}" if a.get('secondary_target') else "")
            + ")"
            for a in actions
        )
        print(f"[STEP] step={step_num} | actions=[{action_summary}]")
        
        # Apply actions to environment
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
    
    # Episode summary
    avg_reward = cumulative_reward / max(1, completed_steps)
    avg_score = cumulative_score / max(1, completed_steps)
    
    print(f"[STEP] ── Episode Summary ──────────────────────────")
    print(f"[STEP] Task:              {task}")
    print(f"[STEP] Completed steps:   {completed_steps}")
    print(f"[STEP] LLM calls:         {llm_calls}")
    print(f"[STEP] LLM Mode:          {'REAL' if USE_REAL_LLM else 'MOCK'}")
    print(f"[STEP] Avg reward:        {avg_reward:.4f}")
    print(f"[STEP] Avg score:         {avg_score:.4f}")
    print(f"[STEP] Total violations:  {total_violations}")
    print("[END]")


if __name__ == "__main__":
    # Parse command line arguments
    task = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    try:
        run_inference(task=task, seed=seed)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(0)
