"""Tunable parameters for Astar Island — SINGLE SOURCE OF TRUTH.

All tunable hyperparameters live here. Both train.py (LORO evaluation)
and model.py (inference/submission) import from this file to stay in sync.

The autoresearch swarm (Gemini agent) edits THIS file.
"""

import numpy as np

# ──────────────────────────────────────────────────────────────
# TUNABLE PARAMETERS — edit these to experiment
# ──────────────────────────────────────────────────────────────

# GBT blend weight: how much to trust XGBoost vs heuristic (0-1)
BLEND_WEIGHT = 0.35
# Per-terrain blend overrides (None = use BLEND_WEIGHT)
BLEND_TERRAIN = {"plains": 1.00, "forest": 1.00, "settl": 1.00}

# Smoothing: blend final prediction with uniform prior to reduce overconfidence
SMOOTH_WEIGHT = 0.0  # disabled — hurts score

# L7 observation-ratio correction strengths per class:
# [empty, settlement, port, ruin, forest, mountain]
# Safe L7: zero for rare classes (port, ruin) — prevents catastrophic KL
L7_STRENGTHS = np.array([1.40, 1.00, 0.0, 0.0, 1.50, 0.0])
L7_MIN_OBS = np.array([200, 80, 60, 20, 100, 0])  # min obs count per class
L7_ADJ_MIN = 0.80  # hard safety clamp
L7_ADJ_MAX = 1.25

# ── Round-type adaptive parameters ──
# Thresholds for round-type classification from obs_settl_rate
ROUND_TYPE_THRESHOLDS = {"extinction": 0.04, "expansion": 0.12}
# Per-round-type overrides for L7 strengths and clamps
ROUND_TYPE_L7 = {
    "extinction": {
        "strengths": np.array([1.60, 0.60, 0.0, 0.0, 1.70, 0.0]),
        "adj_min": np.array([0.70, 0.20, 1.00, 1.00, 0.70, 1.00]),
        "adj_max": np.array([1.40, 1.20, 1.00, 1.00, 1.40, 1.00]),
    },
    "expansion": {
        "strengths": np.array([1.30, 1.50, 0.0, 0.0, 1.60, 0.0]),
        "adj_min": np.array([0.95, 0.95, 1.00, 1.00, 0.95, 1.00]),
        "adj_max": np.array([1.25, 1.35, 1.00, 1.00, 1.25, 1.00]),
    },
    "normal": {
        "strengths": np.array([1.40, 1.00, 0.0, 0.0, 1.50, 0.0]),
        "adj_min": np.array([0.75, 0.35, 1.00, 1.00, 0.75, 1.00]),
        "adj_max": np.array([1.30, 1.35, 1.00, 1.00, 1.30, 1.00]),
    },
}

# Empirical blend weights (Layer 6.7) — round-type-aware
EMP_BLEND_EXPANSION = 0.35
EMP_BLEND_NORMAL = 0.50

# Spatial smoothing sigma for settlement and forest channels
SPATIAL_SMOOTH_SIGMA = 0.5
SPATIAL_SMOOTH_BLEND = 0.15  # blend smoothed with original

# Per-terrain XGBoost hyperparameters
XGB_HPARAMS = {
    "plains": dict(n_estimators=600, max_depth=5, learning_rate=0.08,
                   reg_alpha=0.1, reg_lambda=2.0, subsample=0.9,
                   colsample_bytree=0.55, min_child_weight=3),
    "forest": dict(n_estimators=600, max_depth=5, learning_rate=0.08,
                   reg_alpha=0.1, reg_lambda=2.0, subsample=0.9,
                   colsample_bytree=0.55, min_child_weight=3),
    "settl":  dict(n_estimators=600, max_depth=5, learning_rate=0.08,
                   reg_alpha=0.1, reg_lambda=2.0, subsample=0.9,
                   colsample_bytree=0.55, min_child_weight=3),
}

# ──────────────────────────────────────────────────────────────
# Round data
# ──────────────────────────────────────────────────────────────

ROUNDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    6: "ae78003a-4efe-425a-881a-d16a39bca0ad",
    7: "36e581f1-73f8-453f-ab98-cbe3052b701b",
    8: "c5cdf100-a876-4fb7-b5d8-757162c97989",
    9: "2a341ace-0f57-4309-9b89-e59fe0f09179",
    10: "75e625c3-60cb-4392-af3e-c86a98bde8c2",
    11: "324fde07-1670-4202-b199-7aa92ecb40ee",
    12: "795bfb1f-54bd-4f39-a526-9868b36f7ebd",
    13: "7b4bda99-6165-4221-97cc-27880f5e6d95",
    14: "d0a2c894-2162-4d49-86cf-435b9013f3b8",
    15: "cc5442dd-bc5d-418b-911b-7eb960cb0390",
    16: "8f664aed-8839-4c85-bed0-77a2cac7c6f5",
    17: "3eb0c25d-28fa-48ca-b8e1-fc249e3918e9",
    18: "b0f9d1bf-4b71-4e6e-816c-19c718d29056",
    19: "597e60cf-d1a1-4627-ac4d-2a61da68b6df",
    21: "b3a0be6b-b48b-419d-916a-b7a77fa58c4d",
    22: "a8be24e1-bd48-49bb-aa46-c5593da79f6f",
}

ROUND_WEIGHTS = {1: 1.0, 2: 1.05, 4: 1.05**3, 5: 1.05**4, 6: 1.05**5, 7: 1.05**6, 8: 1.05**7, 9: 1.05**8, 10: 1.05**9, 11: 1.05**10, 13: 1.05**12, 14: 1.05**13, 15: 1.05**14, 16: 1.05**15, 17: 1.05**16, 18: 1.05**17, 19: 1.05**18, 21: 1.05**20, 22: 1.05**21}

# Rounds used for training/LORO (R12 excluded: 0 observations pollute XGBoost)
TRAINING_ROUNDS = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22]


# ──────────────────────────────────────────────────────────────
# Round-type classification
# ──────────────────────────────────────────────────────────────

def classify_round_type(obs_stats):
    """Classify round type from observation stats.
    obs_stats layout: base[0:7] + [min_food, max_pop, food_std, min_defense, defense_std,
      n_factions_norm, obs_settl_rate, ...] -> obs_settl_rate is index 13.
    """
    if obs_stats is None:
        return "normal"
    obs_settl_rate = float(obs_stats[13])
    if obs_settl_rate < ROUND_TYPE_THRESHOLDS["extinction"]:
        return "extinction"
    elif obs_settl_rate > ROUND_TYPE_THRESHOLDS["expansion"]:
        return "expansion"
    return "normal"
