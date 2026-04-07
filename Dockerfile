"""
inference.py — ATC TRACON RL Inference (Hugging Face Router - Working Version)
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

# Use a model that is more likely to work on HF Router free tier
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"   # Smaller, faster, more reliable on router
# Alternative: "Qwen/Qwen2-7B-Instruct"

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

print("=== TOKEN DEBUG ===")
print(f"Length : {len(HF_TOKEN)}")
print(f"Starts with: {HF_TOKEN[:20] if HF_TOKEN else 'EMPTY'}...")
print("===================\n")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN is not set.")
    print(r'Run: $env:HF_TOKEN = "hf_your_token_here"')
    sys.exit(1)

MAX_STEPS = 12
TASK = sys.argv[1] if len(sys.argv) > 1 else "emergency_vectoring"

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)

# ====================== PROMPT ======================
def get_system_prompt(task: str):
    if task == "wake_turbulence":
        return """You are an expert Air Traffic Controller.

Task: Maintain safe wake turbulence separation.

Aircraft:
- LEAD (front aircraft)
- TRAIL (behind aircraft)

Valid actions:
- speed_change (value negative → slow down TRAIL, positive → speed up)
- heading_change (increase separation)
- no_action

Goal:
- Keep separation close to required minimum (NOT too far, NOT too close)
- Avoid loss of separation (< 4 NM)

Output ONLY JSON:
[
  {"action_type": "...", "target_callsign": "TRAIL", "value": -10}
]
"""

    elif task == "emergency_vectoring":
        return """You are an expert ATC.

Vector EMER1 safely to runway.

Actions:
- heading_change
- altitude_change

Goal:
- Avoid traffic
- Minimize distance to runway
"""

    return "You are an ATC agent."

def build_user_prompt(state: Dict, step: int) -> str:
    aircraft = [{"callsign": ac.get("callsign"), "is_emergency": ac.get("is_emergency", False)} 
                for ac in state.get("aircraft", [])]
    return json.dumps({"step": step, "task": state.get("task"), "aircraft": aircraft}, indent=2)


def safe_parse_actions(raw_text: str, valid_callsigns: List[str]) -> List[Dict]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1]) if text.endswith("```") else text[3:]

    try:
        data = json.loads(text)
        if not isinstance(data, list):
            data = [data]
        for a in data:
            if a.get("target_callsign") not in valid_callsigns:
                a["target_callsign"] = valid_callsigns[0] if valid_callsigns else "EMER1"
        return data
    except:
        fallback = valid_callsigns[0] if valid_callsigns else "EMER1"
        return [{"action_type": "heading_change", "target_callsign": fallback, "value": 240, "rationale": "fallback"}]


def call_llm(state: Dict, step: int) -> List[Dict]:
    valid_callsigns = [ac.get("callsign") for ac in state.get("aircraft", [])]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": get_system_prompt(state.get("task"))},
                {"role": "user", "content": build_user_prompt(state, step)}
            ],
            temperature=0.3,
            max_tokens=600,
        )
        raw = response.choices[0].message.content or ""
        return safe_parse_actions(raw, valid_callsigns)
    except Exception as e:
        print(f"LLM Error: {str(e)[:180]}")
        fallback = valid_callsigns[0] if valid_callsigns else "EMER1"
        return [{"action_type": "heading_change", "target_callsign": fallback, "value": 240, "rationale": "error"}]


def env_reset(task: str) -> Dict:
    r = requests.post(f"{API_BASE_URL}/reset", json={"task": task}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(actions: List[Dict]) -> Dict:
    r = requests.post(f"{API_BASE_URL}/step", json={"actions": actions}, timeout=30)
    r.raise_for_status()
    return r.json()


def run_inference():
    print(f"[START] Task: {TASK} | Model: {MODEL_NAME}")

    state = env_reset(TASK)
    print(f"Reset OK | Aircraft: {len(state.get('aircraft', []))}\n")

    cumulative_reward = 0.0

    for step_num in range(1, MAX_STEPS + 1):
        if state.get("done", False):
            break

        actions = call_llm(state, step_num)
        action_str = "; ".join(f"{a.get('action_type')}({a.get('target_callsign')})" for a in actions)
        print(f"[STEP {step_num}] Action: {action_str}")

        result = env_step(actions)
        state = result.get("state", {})
        reward = result.get("reward", 0.0)
        done = result.get("done", False)

        cumulative_reward += reward
        print(f"[STEP {step_num}] Reward: {reward:.4f} | Done: {done}\n")

        if done:
            break

        time.sleep(0.4)

    print(f"=== SUMMARY ===\nAverage reward: {cumulative_reward / max(1, step_num):.4f}\n=== END ===")


if __name__ == "__main__":
    run_inference()