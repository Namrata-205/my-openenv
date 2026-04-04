"""
HAVENT CHANGED THIS FILE: PLS NOTEE

Task Graders for Monsoon Flood Gate Control
============================================
Three tasks of increasing difficulty with programmatic scorers.

All tasks are scored 0.0–1.0 with partial credit.
Graders are deterministic given the same GlobalState.
"""

from __future__ import annotations

from models import GlobalState


# ─── Task Metadata ────────────────────────────────────────────────────────────

TASKS = {
    "task_easy": {
        "name": "Single-Peak Monsoon Response",
        "description": (
            "A single, predictable monsoon burst hits over 6 hours. "
            "The agent must open gates proactively to prevent flooding, "
            "while minimising pump energy. Storm profile is smooth and "
            "telegraphed via the rainfall forecast."
        ),
        "difficulty": "easy",
        "storm_seed": 42,
        "max_steps": 72,
        "target_flood_volume_m3": 500.0,      # tolerable leakage
        "target_energy_kwh": 8.0,             # energy budget
    },
    "task_medium": {
        "name": "Double-Burst Storm Coordination",
        "description": (
            "Two storm bursts separated by a lull. Agents must drain water "
            "during the lull to create buffer capacity for the second burst. "
            "Requires coordination between sub-basins. Forecast is noisy."
        ),
        "difficulty": "medium",
        "storm_seed": 137,
        "max_steps": 72,
        "target_flood_volume_m3": 1500.0,
        "target_energy_kwh": 15.0,
    },
    "task_hard": {
        "name": "Erratic Extreme Event (Dharavi Worst-Case)",
        "description": (
            "Simulates an erratic Mumbai extreme event: sudden rainfall spikes "
            "up to 200mm/hr, unpredictable drops, and rising downstream river. "
            "Agents must react with minimal forecast information. "
            "Partial observability is most challenging here."
        ),
        "difficulty": "hard",
        "storm_seed": 999,
        "max_steps": 72,
        "target_flood_volume_m3": 5000.0,
        "target_energy_kwh": 25.0,
    },
}


# ─── Grader Functions ─────────────────────────────────────────────────────────

def grade_task_easy(state: GlobalState) -> float:
    """
    Score for task_easy.

    Criteria:
    - Primary (70%): How much of the episode was flood-free across all basins
    - Secondary (20%): Energy efficiency (stayed under budget)
    - Bonus (10%): No single basin flooded more than 10 minutes
    """
    task = TASKS["task_easy"]

    # Primary: flood volume vs. budget
    flood_vol = state.episode_flood_volume_m3
    target = task["target_flood_volume_m3"]
    if flood_vol <= 0:
        flood_score = 1.0
    elif flood_vol <= target:
        flood_score = 1.0 - 0.3 * (flood_vol / target)
    else:
        # Significant penalty beyond target
        overflow_ratio = min(5.0, flood_vol / target)
        flood_score = max(0.0, 0.7 - 0.15 * (overflow_ratio - 1))
    flood_score = max(0.0, flood_score)

    # Secondary: energy efficiency
    energy = state.episode_energy_kwh
    e_budget = task["target_energy_kwh"]
    if energy <= e_budget:
        energy_score = 1.0
    else:
        energy_score = max(0.0, 1.0 - 0.5 * (energy - e_budget) / e_budget)

    # Bonus: no single basin flooded excessively
    max_flood_minutes = max(b.flood_minutes for b in state.basins)
    bonus = 1.0 if max_flood_minutes <= 10 else max(0.0, 1.0 - (max_flood_minutes - 10) / 50)

    score = 0.70 * flood_score + 0.20 * energy_score + 0.10 * bonus
    return round(min(1.0, max(0.0, score)), 4)


def grade_task_medium(state: GlobalState) -> float:
    """
    Score for task_medium.

    Criteria:
    - Primary (60%): Flood volume control across both storm bursts
    - Secondary (25%): Coordination quality (no basin significantly worse than others)
    - Tertiary (15%): Energy efficiency
    """
    task = TASKS["task_medium"]

    # Primary: flood volume
    flood_vol = state.episode_flood_volume_m3
    target = task["target_flood_volume_m3"]
    if flood_vol <= 0:
        flood_score = 1.0
    elif flood_vol <= target:
        flood_score = 0.7 + 0.3 * (1.0 - flood_vol / target)
    else:
        excess_ratio = min(4.0, flood_vol / target)
        flood_score = max(0.0, 0.7 - 0.2 * (excess_ratio - 1))

    # Secondary: inter-basin coordination (std dev of flood minutes)
    flood_mins = [b.flood_minutes for b in state.basins]
    mean_fm = sum(flood_mins) / len(flood_mins)
    std_fm = (sum((f - mean_fm) ** 2 for f in flood_mins) / len(flood_mins)) ** 0.5
    # Low std = good coordination
    coord_score = max(0.0, 1.0 - std_fm / max(1, mean_fm + 1))

    # Tertiary: energy
    e_budget = task["target_energy_kwh"]
    energy_score = max(0.0, 1.0 - max(0, state.episode_energy_kwh - e_budget) / e_budget)

    score = 0.60 * flood_score + 0.25 * coord_score + 0.15 * energy_score
    return round(min(1.0, max(0.0, score)), 4)


def grade_task_hard(state: GlobalState) -> float:
    """
    Score for task_hard (Erratic Extreme Event).

    Criteria:
    - Primary (50%): Total flood volume vs. (generous) budget
    - Secondary (30%): Worst-basin performance (resilience to worst failure)
    - Tertiary (20%): Reaction speed — did agents drain before spikes?
      (Proxied by: total outflow capacity utilised during peak steps)
    """
    task = TASKS["task_hard"]

    # Primary: flood volume (generous budget for hard task)
    flood_vol = state.episode_flood_volume_m3
    target = task["target_flood_volume_m3"]
    if flood_vol <= 0:
        flood_score = 1.0
    elif flood_vol <= target:
        flood_score = 0.5 + 0.5 * (1.0 - flood_vol / target)
    else:
        ratio = min(3.0, flood_vol / target)
        flood_score = max(0.0, 0.5 - 0.2 * (ratio - 1))

    # Secondary: worst basin (max flood_minutes)
    max_flood_min = max(b.flood_minutes for b in state.basins)
    # 60 min of flooding = very bad; 0 = perfect
    worst_score = max(0.0, 1.0 - max_flood_min / 60.0)

    # Tertiary: average pump utilisation (proxy for responsiveness)
    # We compute this from total energy vs. max possible
    max_energy = task["target_energy_kwh"] * 2
    actual_energy = state.episode_energy_kwh
    # Reward moderate energy use (too little = not responding, too much = wasteful)
    optimal_energy = task["target_energy_kwh"]
    e_ratio = actual_energy / max(0.1, optimal_energy)
    if 0.5 <= e_ratio <= 1.5:
        responsiveness = 1.0
    elif e_ratio < 0.5:
        responsiveness = e_ratio / 0.5  # under-responding
    else:
        responsiveness = max(0.0, 1.0 - (e_ratio - 1.5) / 1.5)

    score = 0.50 * flood_score + 0.30 * worst_score + 0.20 * responsiveness
    return round(min(1.0, max(0.0, score)), 4)


# ─── Dispatcher ───────────────────────────────────────────────────────────────

GRADER_MAP = {
    "task_easy": grade_task_easy,
    "task_medium": grade_task_medium,
    "task_hard": grade_task_hard,
}


def grade(state: GlobalState) -> float:
    """Grade the episode. Returns score in [0.0, 1.0]."""
    grader = GRADER_MAP.get(state.task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id: {state.task_id}")
    return grader(state)
