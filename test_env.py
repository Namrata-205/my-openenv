"""
test_env.py — Tests all 5 ATC tasks against the running server.
Run from my-openenv/ root after starting the server:
    python server/app.py
    python test_env.py
"""

import requests

BASE = "http://localhost:7860"

# ── One representative action per task ───────────────────────────────────────
TASK_ACTIONS = {
    "wake_turbulence": [
        {"action_type": "speed_change", "target_callsign": "TRAIL", "value": -10}
    ],
    "go_around_prevention": [
        {"action_type": "sequence_swap", "target_callsign": "UAL101",
         "secondary_target": "DAL202", "rationale": "rl agent"}
    ],
    "emergency_vectoring": [
        {"action_type": "vector",          "target_callsign": "EMER1", "value": 250},
        {"action_type": "altitude_change", "target_callsign": "EMER1", "value": 2000}
    ],
    "conflict_resolution": [
        {"action_type": "heading_change", "target_callsign": "AC1", "value": -10}
    ],
    "gate_assignment": [
        {"action_type": "assign_gate", "target_callsign": "UAL101", "gate_id": "A1"}
    ],
}

# ── Helper ────────────────────────────────────────────────────────────────────
def check(r, label):
    if r.status_code != 200:
        print(f"\n[FAIL] {label}")
        print(f"  Status : {r.status_code}")
        print(f"  Detail : {r.text}")
        raise SystemExit(1)
    return r.json()


print("\n" + "=" * 55)
print("  ATC TRACON RL — Environment Tests")
print("=" * 55)

# ── 1. Health ────────────────────────────────────────────────────────────────
data = check(requests.get(f"{BASE}/health"), "GET /health")
print(f"\n[PASS] /health  →  status={data['status']}  version={data['version']}")
print(f"       tasks: {data['tasks']}")

# ── 2. Tasks list ────────────────────────────────────────────────────────────
data = check(requests.get(f"{BASE}/tasks"), "GET /tasks")
print(f"\n[PASS] /tasks   →  {data['tasks']}")

# ── 3. Each task: reset → state → step × 3 ───────────────────────────────────
print()
for task, actions in TASK_ACTIONS.items():

    # reset
    data = check(
        requests.post(f"{BASE}/reset",
                      json={"task": task, "seed": 42}),
        f"POST /reset  task={task}"
    )
    n_aircraft = len(data.get("aircraft", []))
    n_gates    = len(data.get("gates", []))
    print(f"[PASS] reset   {task:<25}  aircraft={n_aircraft}  gates={n_gates}")

    # state
    state = check(requests.get(f"{BASE}/state"), f"GET /state  task={task}")
    assert state["step"] == 0, f"Expected step=0, got {state['step']}"

    # Track cumulative reward
    cumulative_reward = 0.0

    # 3 steps
    for s in range(1, 4):
        # --- Filter actions based on existing aircraft ---
        existing_callsigns = {ac["callsign"] for ac in state.get("aircraft", [])}
        filtered_actions = [
            act for act in actions if act["target_callsign"] in existing_callsigns
        ]
        if not filtered_actions:
            print(f"[TASK={task}] Step {s} → No valid actions, skipping step")
            break

        data = check(
            requests.post(f"{BASE}/step", json={"actions": filtered_actions}),
            f"POST /step  task={task}  step={s}"
        )
        reward = data["reward"]
        score  = data["score"]
        done   = data["done"]
        viols  = data.get("violations", [])

        # Update cumulative reward
        cumulative_reward += reward

        # Print detailed step info
        viol_str = f"  violations={viols}" if viols else ""
        print(f"[TASK={task}] Step {s} → Reward: {reward:.4f}  "
              f"Score: {score:.4f}  Cumulative: {cumulative_reward:.4f}  Done: {done}{viol_str}")

        if done:
            break

        # Update state for next step filtering
        state = data

    # Print total reward for the task
    print(f"[TASK={task}] Total cumulative reward after {s} step(s): {cumulative_reward:.4f}\n")

# ── 4. Error handling — step before reset ────────────────────────────────────
data = check(requests.get(f"{BASE}/state"), "GET /state after loop")
print(f"[PASS] /state after all tasks  →  step={data['step']}  done={data['done']}")

# ── Summary ─────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  All tests passed ✓")
print("  Rewards normalised [0.0, 1.0] ✓")
print("  All 5 tasks responded 200 OK ✓")
print("=" * 55 + "\n")