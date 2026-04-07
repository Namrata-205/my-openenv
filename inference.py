"""
inference.py — ATC TRACON RL Inference (Fixed + Stable)

Key Fixes:
- Supports heading + altitude + time
- Strict JSON parsing
- Prevents multi-action spam
- Better prompt for dynamic control
"""

import json
import os
import sys
import time
from typing import Dict, List

import requests
from openai import OpenAI

# ====================== CONFIG ======================
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

MAX_STEPS = 12
TASK = sys.argv[1] if len(sys.argv) > 1 else "emergency_vectoring"

if not HF_TOKEN:
    print("ERROR: HF_TOKEN is not set.")
    sys.exit(1)

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)

# ====================== PROMPT ======================
SYSTEM_PROMPT = """You are an expert Air Traffic Controller.

STRICT RULES:
- Output ONLY valid JSON
- EXACTLY ONE action in a list
- NO explanations outside JSON

Task: emergency_vectoring

You must control:
- heading (0–359 degrees)
- altitude (1000–40000 ft)
- time (0.5–5 minutes)

Goals:
- Avoid conflicts with other aircraft
- Move emergency aircraft closer to runway (0,0)
- Do NOT repeat same action every step

Format:
[
  {
    "action_type": "heading_change",
    "target_callsign": "EMER1",
    "value": 250,
    "altitude": 6000,
    "time": 2.0,
    "rationale": "short reason"
  }
]
"""

def build_user_prompt(state: Dict, step: int) -> str:
    aircraft = []
    for ac in state.get("aircraft", []):
        aircraft.append({
            "callsign": ac.get("callsign"),
            "x": ac.get("x"),
            "y": ac.get("y"),
            "heading": ac.get("heading"),
            "altitude": ac.get("altitude"),
            "is_emergency": ac.get("is_emergency", False)
        })

    return json.dumps({
        "step": step,
        "task": state.get("task"),
        "aircraft": aircraft
    }, indent=2)


# ====================== SAFE PARSER ======================
def safe_parse_actions(raw_text: str, valid_callsigns: List[str]) -> List[Dict]:
    fallback_callsign = valid_callsigns[0] if valid_callsigns else "EMER1"
    

    try:
        text = raw_text.strip()

        # remove markdown
        if text.startswith("```"):
            text = "\n".join(text.splitlines()[1:-1])

        data = json.loads(text)
        if not isinstance(data, list):
            data = [data]

        action = data[0]

        # ✅ Extract safely
        heading = int(action.get("value", 240)) % 360
        altitude = int(action.get("altitude", 5000))
        time_val = float(action.get("time", 1.5))

        altitude = max(1000, min(40000, altitude))
        time_val = max(0.5, min(5.0, time_val))

        return [{
            "action_type": "heading_change",
            "target_callsign": action.get("target_callsign", fallback_callsign),
            "value": heading,
            "altitude": altitude,
            "time": time_val
        }]

    except Exception as e:
        print("⚠️ PARSE ERROR:", str(e))
        return [{
            "action_type": "heading_change",
            "target_callsign": fallback_callsign,
            "value": 240,
            "altitude": 5000,
            "time": 1.5
        }]


# ====================== LLM CALL ======================
def call_llm(state: Dict, step: int) -> List[Dict]:
    valid_callsigns = [ac.get("callsign") for ac in state.get("aircraft", [])]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(state, step)}
            ],
            temperature=0.6,
            max_tokens=300,
        )

        raw = response.choices[0].message.content or ""

        print("RAW:", raw)  # ✅ DEBUG

        return safe_parse_actions(raw, valid_callsigns)

    except Exception as e:
        print("❌ LLM ERROR:", str(e))

        fallback = valid_callsigns[0] if valid_callsigns else "EMER1"
        return [{
            "action_type": "heading_change",
            "target_callsign": fallback,
            "value": 240,
            "altitude": 5000,
            "time": 1.5
        }]


# ====================== ENV ======================
def env_reset(task: str) -> Dict:
    r = requests.post(f"{API_BASE_URL}/reset", json={"task": task}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(actions: List[Dict]) -> Dict:
    r = requests.post(f"{API_BASE_URL}/step", json={"actions": actions}, timeout=30)
    r.raise_for_status()
    return r.json()


# ====================== MAIN LOOP ======================
def run_inference():
    print(f"[START] Task: {TASK} | Model: {MODEL_NAME}")

    state = env_reset(TASK)
    print(f"Reset OK | Aircraft: {len(state.get('aircraft', []))}\n")

    cumulative_reward = 0.0
    prev_action = None

    for step_num in range(1, MAX_STEPS + 1):
        if state.get("done", False):
            break

        actions = call_llm(state, step_num)

        # detect repetition
        if actions == prev_action:
            print("⚠️ SAME ACTION REPEATED")
        prev_action = actions

        action_str = ", ".join(
            f"{a['value']}° alt={a['altitude']} time={a['time']}"
            for a in actions
        )

        print(f"[STEP {step_num}] Action: {action_str}")

        result = env_step(actions)

        state = result.get("state", {})
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        print("RAW REWARD:", result.get("info", {}).get("raw_reward"))

        cumulative_reward += reward

        print(f"[STEP {step_num}] Reward: {reward:.4f} | Done: {done}\n")

        if done:
            break

        time.sleep(0.4)

    print(f"=== SUMMARY ===")
    print(f"Average reward: {cumulative_reward / max(1, step_num):.4f}")
    print("=== END ===")


if __name__ == "__main__":
    run_inference()