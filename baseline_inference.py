"""
Baseline Inference Script — Monsoon Flood Gate Control
=======================================================
Runs three agent strategies against the environment via the REST API.
Uses the OpenAI client (pointed at HF router or local server).

Usage:
  # Against local Docker container
  python baseline_inference.py --base-url http://localhost:8000

  # Against deployed HF Space
  python baseline_inference.py --base-url https://<your-space>.hf.space

Environment Variables:
  OPENAI_API_KEY  — Required for the LLM agent strategy
  OPENENV_BASE_URL — Overrides --base-url

Strategies:
  1. Rule-Based Heuristic  (no LLM, deterministic)
  2. LLM Agent             (GPT/Claude-compatible via OpenAI client)
  3. Reactive Threshold    (simple bang-bang controller)

Produces reproducible baseline scores on all 3 tasks.
"""

import os
import json
import time
import argparse
import requests
from typing import Any

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = os.environ.get("OPENENV_BASE_URL", "http://localhost:8000")
TASKS = ["task_easy", "task_medium", "task_hard"]
NUM_BASINS = 3
MAX_STEPS = 72
STORM_SEED = None  # None = use task default


# ─── HTTP Helpers ─────────────────────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str, basin_idx: int) -> dict:
        r = requests.post(f"{self.base_url}/reset", json={
            "task_id": task_id, "basin_idx": basin_idx
        })
        r.raise_for_status()
        return r.json()

    def step(self, task_id: str, basin_idx: int, action: dict) -> dict:
        r = requests.post(f"{self.base_url}/step", json={
            "task_id": task_id, "basin_idx": basin_idx, "action": action
        })
        r.raise_for_status()
        return r.json()

    def grade(self, task_id: str, basin_idx: int) -> float:
        r = requests.get(f"{self.base_url}/grade", params={
            "task_id": task_id, "basin_idx": basin_idx
        })
        r.raise_for_status()
        return r.json()["score"]

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


# ─── Strategy 1: Rule-Based Heuristic ────────────────────────────────────────

def heuristic_action(obs: dict) -> dict:
    """
    Proportional control:
    - Open gates proportional to (water_level - 40% threshold)
    - Run pumps when > 70% full
    - Boost if neighbor is also flooding
    """
    wf = obs["water_level_fraction"]
    neighbors_flooding = any(n["is_flooding"] for n in obs.get("neighbors", []))

    # Gate openings: ramp up from 40% fill
    if wf < 0.4:
        gate = 0.0
    elif wf < 0.7:
        gate = (wf - 0.4) / 0.3  # 0 to 1
    else:
        gate = 1.0

    # Boost if neighbors flooding too (shared drainage pressure)
    if neighbors_flooding:
        gate = min(1.0, gate + 0.2)

    # Pumps: engage above 70%
    pump = max(0.0, min(1.0, (wf - 0.7) / 0.3)) if wf > 0.7 else 0.0

    # Forecast bonus: pre-open if big rain coming
    forecast = obs.get("rainfall_forecast", [])
    current_rain = obs.get("rainfall_mm_per_hr", 0)
    if forecast and forecast[0] > current_rain * 1.4 and wf < 0.6:
        gate = min(1.0, gate + 0.3)

    return {
        "gate_positions": [gate, gate, gate * 0.7],
        "pump_speeds": [pump, pump * 0.8],
    }


def run_heuristic(client: EnvClient, task_id: str) -> float:
    scores = []
    for basin_idx in range(NUM_BASINS):
        obs = client.reset(task_id, basin_idx)
        done = False
        step_count = 0
        while not done and step_count < MAX_STEPS:
            action = heuristic_action(obs)
            result = client.step(task_id, basin_idx, action)
            obs = result["observation"]
            done = result["done"]
            step_count += 1
        score = client.grade(task_id, basin_idx)
        scores.append(score)
    return sum(scores) / len(scores)


# ─── Strategy 2: LLM Agent (OpenAI-compatible) ────────────────────────────────

LLM_SYSTEM_PROMPT = """You are a flood control AI for Mumbai's stormwater management system.
You control gates and pumps for a stormwater basin during monsoon events.

Your goal: Minimize flooding (water level exceeding capacity) while conserving energy.

At each step you receive a JSON observation and must output a JSON action:
{
  "gate_positions": [g1, g2, g3],  // each 0.0 (closed) to 1.0 (fully open)
  "pump_speeds": [p1, p2]          // each 0.0 (off) to 1.0 (full speed)
}

Rules:
- Open gates when water level > 50% (gravity drainage, free)
- Use pumps sparingly when > 70% (costs energy)
- Pre-open gates when rainfall_forecast shows spike incoming
- If neighbor basins are flooding, avoid adding outflow pressure

Respond ONLY with valid JSON action. No explanation.
"""


def llm_action(client_llm: Any, obs: dict) -> dict:
    """Query LLM for an action given observation."""
    obs_str = json.dumps(obs, indent=2)
    try:
        response = client_llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": f"Current observation:\n{obs_str}\n\nOutput action JSON:"},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        # Strip code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        action = json.loads(raw)
        # Validate structure
        assert "gate_positions" in action and "pump_speeds" in action
        return action
    except Exception:
        # Fallback to heuristic on LLM parse error
        return heuristic_action(obs)


def run_llm_agent(client: EnvClient, task_id: str, api_key: str) -> float:
    if not HAS_OPENAI:
        print("  [!] openai package not installed. Falling back to heuristic.")
        return run_heuristic(client, task_id)

    llm = OpenAI(api_key=api_key)
    scores = []
    for basin_idx in range(NUM_BASINS):
        obs = client.reset(task_id, basin_idx)
        done = False
        step_count = 0
        while not done and step_count < MAX_STEPS:
            action = llm_action(llm, obs)
            result = client.step(task_id, basin_idx, action)
            obs = result["observation"]
            done = result["done"]
            step_count += 1
        score = client.grade(task_id, basin_idx)
        scores.append(score)
    return sum(scores) / len(scores)


# ─── Strategy 3: Reactive Threshold (Bang-Bang) ───────────────────────────────

def reactive_action(obs: dict) -> dict:
    """Simple bang-bang: fully open/close based on threshold."""
    wf = obs["water_level_fraction"]
    rain = obs["rainfall_mm_per_hr"]

    if wf > 0.75 or rain > 80:
        return {"gate_positions": [1.0, 1.0, 1.0], "pump_speeds": [1.0, 1.0]}
    elif wf > 0.5:
        return {"gate_positions": [0.5, 0.5, 0.3], "pump_speeds": [0.0, 0.0]}
    else:
        return {"gate_positions": [0.0, 0.0, 0.0], "pump_speeds": [0.0, 0.0]}


def run_reactive(client: EnvClient, task_id: str) -> float:
    scores = []
    for basin_idx in range(NUM_BASINS):
        obs = client.reset(task_id, basin_idx)
        done = False
        step_count = 0
        while not done and step_count < MAX_STEPS:
            action = reactive_action(obs)
            result = client.step(task_id, basin_idx, action)
            obs = result["observation"]
            done = result["done"]
            step_count += 1
        score = client.grade(task_id, basin_idx)
        scores.append(score)
    return sum(scores) / len(scores)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Monsoon Flood Gate Control — Baseline Inference")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--strategy", choices=["heuristic", "llm", "reactive", "all"], default="all")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    args = parser.parse_args()

    client = EnvClient(args.base_url)

    print(f"🌧️  Monsoon Flood Gate Control — Baseline Inference")
    print(f"   Server: {args.base_url}")

    # Health check
    if not client.health():
        print(f"❌ Server not reachable at {args.base_url}. Start the Docker container first.")
        return

    print(f"   Server: ✅ healthy\n")

    strategies = {
        "heuristic": ("Rule-Based Heuristic", lambda t: run_heuristic(client, t)),
        "reactive":  ("Reactive Threshold",   lambda t: run_reactive(client, t)),
        "llm":       ("LLM Agent (GPT-4o-mini)", lambda t: run_llm_agent(client, t, args.api_key)),
    }

    selected = list(strategies.keys()) if args.strategy == "all" else [args.strategy]

    results = {}
    for strat_key in selected:
        name, run_fn = strategies[strat_key]
        print(f"{'='*55}")
        print(f"Strategy: {name}")
        print(f"{'='*55}")

        task_scores = {}
        for task_id in TASKS:
            print(f"  Running {task_id}... ", end="", flush=True)
            t0 = time.time()
            score = run_fn(task_id)
            elapsed = time.time() - t0
            print(f"score={score:.4f}  ({elapsed:.1f}s)")
            task_scores[task_id] = score

        avg = sum(task_scores.values()) / len(task_scores)
        print(f"  ─────────────────────────────────")
        print(f"  Average score: {avg:.4f}")
        results[strat_key] = {"tasks": task_scores, "average": avg}

    print(f"\n{'='*55}")
    print("BASELINE SCORE SUMMARY")
    print(f"{'='*55}")
    print(f"{'Strategy':<30} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Avg':>8}")
    print("-" * 55)
    for key, data in results.items():
        name = strategies[key][0][:28]
        ts = data["tasks"]
        print(f"{name:<30} {ts.get('task_easy',0):>8.4f} {ts.get('task_medium',0):>8.4f} "
              f"{ts.get('task_hard',0):>8.4f} {data['average']:>8.4f}")

    print(f"\n✅ Baseline complete. Results saved to baseline_results.json")
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
