#!/usr/bin/env python3
"""Deep analysis of observation data to find hidden patterns.

Cross-references observation snapshots (mid-simulation) with ground truth
(final state after 50 years) to discover what predicts settlement survival,
expansion, collapse, and other outcomes.

Outputs:
  - Settlement survival analysis (what features predict alive vs dead)
  - Observation feature correlations with GT outcomes
  - Candidate features for XGBoost ranked by predictive power
  - Per-round hidden parameter estimates
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from dtos import TERRAIN_TO_CLASS, NUM_CLASSES
from utils import load_observations

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

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
}


def load_round(rnum):
    with open(os.path.join(DATA_DIR, f"round{rnum}_initial.json")) as f:
        info = json.load(f)
    gts = []
    for seed in range(5):
        gts.append(np.load(os.path.join(DATA_DIR, f"gt_r{rnum}_seed{seed}.npy")))
    obs = load_observations(ROUNDS[rnum])
    return info, gts, obs


def extract_settlement_features(obs_list):
    """Extract all settlement features from observations for a round."""
    records = []
    for obs in obs_list:
        seed = obs.get("seed_index", 0)
        for s in obs.get("settlements", []):
            records.append({
                "seed": seed,
                "x": s.get("x", 0),
                "y": s.get("y", 0),
                "population": s.get("population", 0),
                "food": s.get("food", 0),
                "wealth": s.get("wealth", 0),
                "defense": s.get("defense", 0),
                "alive": s.get("alive", True),
                "has_port": s.get("has_port", False),
                "owner_id": s.get("owner_id", -1),
                "tech_level": s.get("tech_level", 0),
                "has_longship": s.get("has_longship", False),
            })
    return records


def analyze_settlement_survival(info, gts, settlement_records):
    """Correlate observed settlement features with GT outcomes."""
    results = {"alive_became": defaultdict(int), "dead_became": defaultdict(int)}
    feature_vs_outcome = []

    cls_names = ['empty', 'settlement', 'port', 'ruin', 'forest', 'mountain']

    for rec in settlement_records:
        seed = rec["seed"]
        x, y = rec["x"], rec["y"]
        gt = gts[seed]
        h, w, _ = gt.shape
        if not (0 <= y < h and 0 <= x < w):
            continue

        gt_probs = gt[y, x]
        gt_dominant = cls_names[np.argmax(gt_probs)]
        gt_settl_prob = gt_probs[1]  # P(settlement)
        gt_port_prob = gt_probs[2]
        gt_ruin_prob = gt_probs[3]
        gt_survived = gt_settl_prob + gt_port_prob  # P(still a settlement or port)

        status = "alive" if rec["alive"] else "dead"
        results[f"{status}_became"][gt_dominant] += 1

        feature_vs_outcome.append({
            **rec,
            "gt_dominant": gt_dominant,
            "gt_settl_prob": gt_settl_prob,
            "gt_port_prob": gt_port_prob,
            "gt_ruin_prob": gt_ruin_prob,
            "gt_survived": gt_survived,
            "gt_empty_prob": gt_probs[0],
            "gt_forest_prob": gt_probs[4],
        })

    return results, feature_vs_outcome


def compute_round_stats(obs_list):
    """Compute round-level statistics from observations."""
    pops, foods, wealths, defenses, techs = [], [], [], [], []
    factions = set()
    alive, dead, ports, longships = 0, 0, 0, 0
    settl_per_viewport = []
    terrain_counts = defaultdict(int)

    for obs in obs_list:
        vp_settl = 0
        for s in obs.get("settlements", []):
            if s.get("alive", True):
                alive += 1
                pops.append(s.get("population", 0))
                foods.append(s.get("food", 0))
                wealths.append(s.get("wealth", 0))
                defenses.append(s.get("defense", 0))
                techs.append(s.get("tech_level", 0))
                factions.add(s.get("owner_id", -1))
                if s.get("has_port"):
                    ports += 1
                if s.get("has_longship"):
                    longships += 1
            else:
                dead += 1

        for row in obs.get("grid", []):
            for code in row:
                terrain_counts[code] += 1
                if code in {1, 2}:
                    vp_settl += 1
        settl_per_viewport.append(vp_settl)

    total_cells = sum(terrain_counts.values())
    return {
        "avg_pop": np.mean(pops) if pops else 0,
        "med_pop": np.median(pops) if pops else 0,
        "max_pop": max(pops) if pops else 0,
        "min_pop": min(pops) if pops else 0,
        "std_pop": np.std(pops) if pops else 0,
        "avg_food": np.mean(foods) if foods else 0,
        "min_food": min(foods) if foods else 0,
        "max_food": max(foods) if foods else 0,
        "std_food": np.std(foods) if foods else 0,
        "avg_wealth": np.mean(wealths) if wealths else 0,
        "max_wealth": max(wealths) if wealths else 0,
        "avg_defense": np.mean(defenses) if defenses else 0,
        "min_defense": min(defenses) if defenses else 0,
        "avg_tech": np.mean(techs) if techs else 0,
        "max_tech": max(techs) if techs else 0,
        "alive_count": alive,
        "dead_count": dead,
        "alive_rate": alive / max(alive + dead, 1),
        "port_count": ports,
        "port_rate": ports / max(alive, 1),
        "longship_count": longships,
        "longship_rate": longships / max(alive, 1),
        "n_factions": len(factions),
        "settl_per_obs": np.mean(settl_per_viewport) if settl_per_viewport else 0,
        "settl_per_obs_std": np.std(settl_per_viewport) if settl_per_viewport else 0,
        # Terrain proportions in observed area
        "obs_empty_rate": (terrain_counts.get(11, 0) + terrain_counts.get(0, 0)) / max(total_cells, 1),
        "obs_settl_rate": (terrain_counts.get(1, 0) + terrain_counts.get(2, 0)) / max(total_cells, 1),
        "obs_forest_rate": terrain_counts.get(4, 0) / max(total_cells, 1),
        "obs_ruin_rate": terrain_counts.get(3, 0) / max(total_cells, 1),
    }


def compute_feature_correlations(feature_data):
    """Compute correlations between settlement features and GT outcomes."""
    if not feature_data:
        return {}

    features = ["population", "food", "wealth", "defense", "tech_level"]
    outcomes = ["gt_survived", "gt_settl_prob", "gt_port_prob", "gt_ruin_prob",
                "gt_empty_prob", "gt_forest_prob"]

    correlations = {}
    for feat in features:
        for out in outcomes:
            vals_f = [d[feat] for d in feature_data if d["alive"]]
            vals_o = [d[out] for d in feature_data if d["alive"]]
            if len(vals_f) > 5 and np.std(vals_f) > 0:
                corr = np.corrcoef(vals_f, vals_o)[0, 1]
                correlations[f"{feat}_vs_{out}"] = corr

    return correlations


def analyze_expansion_patterns(info, gts):
    """Analyze how far settlements expand from initial positions."""
    results = []
    for seed in range(5):
        grid = info["initial_states"][seed]["grid"]
        settls = info["initial_states"][seed]["settlements"]
        gt = gts[seed]
        h, w = len(grid), len(grid[0])

        # Initial settlement positions
        init_pos = set()
        for s in settls:
            x, y = s.get("x", 0), s.get("y", 0)
            init_pos.add((y, x))

        # Count new settlements in GT (cells that became settlement/port but weren't initially)
        new_settl = 0
        new_port = 0
        new_ruin = 0
        survived = 0
        for y in range(h):
            for x in range(w):
                if grid[y][x] in {10, 5}:
                    continue
                dominant = np.argmax(gt[y, x])
                if (y, x) in init_pos:
                    # Initial settlement cell
                    if dominant == 1:
                        survived += 1
                    elif dominant == 2:
                        survived += 1  # became port
                elif dominant == 1:
                    new_settl += 1
                elif dominant == 2:
                    new_port += 1
                elif dominant == 3:
                    new_ruin += 1

        results.append({
            "seed": seed,
            "initial_settl": len(init_pos),
            "survived": survived,
            "new_settlements": new_settl,
            "new_ports": new_port,
            "new_ruins": new_ruin,
            "survival_rate": survived / max(len(init_pos), 1),
            "expansion_rate": new_settl / max(len(init_pos), 1),
        })
    return results


def analyze_per_cell_obs_features(info, gts, obs_list):
    """For each cell with observations, compute features and correlate with GT.

    This is the key analysis: what observation-derived cell features predict GT?
    """
    records = []

    for seed in range(5):
        grid = info["initial_states"][seed]["grid"]
        settls = info["initial_states"][seed]["settlements"]
        gt = gts[seed]
        h, w = len(grid), len(grid[0])

        # Build settlement position set
        settl_pos = [(s.get("x", 0), s.get("y", 0)) for s in settls]

        # Collect per-cell observation data
        cell_obs = {}  # (y, x) -> list of observed terrain codes
        cell_settl_stats = {}  # (y, x) -> list of settlement stat dicts

        seed_obs = [o for o in obs_list if o.get("seed_index") == seed]
        for obs in seed_obs:
            vp = obs.get("viewport", {})
            vy, vx = vp.get("y", 0), vp.get("x", 0)
            obs_grid = obs.get("grid", [])

            for dy in range(len(obs_grid)):
                for dx in range(len(obs_grid[0]) if obs_grid else 0):
                    ay, ax = vy + dy, vx + dx
                    if 0 <= ay < h and 0 <= ax < w:
                        key = (ay, ax)
                        if key not in cell_obs:
                            cell_obs[key] = []
                        cell_obs[key].append(obs_grid[dy][dx])

            # Map settlement stats to positions
            for s in obs.get("settlements", []):
                sx, sy = s.get("x", 0), s.get("y", 0)
                key = (sy, sx)
                if key not in cell_settl_stats:
                    cell_settl_stats[key] = []
                cell_settl_stats[key].append(s)

        # Now analyze each observed cell
        for (y, x), obs_codes in cell_obs.items():
            code = grid[y][x]
            if code in {10, 5}:
                continue

            gt_probs = gt[y, x]
            n_obs = len(obs_codes)

            # Observation-derived features
            obs_settl_count = sum(1 for c in obs_codes if c in {1, 2})
            obs_empty_count = sum(1 for c in obs_codes if c in {0, 11, 10})
            obs_forest_count = sum(1 for c in obs_codes if c == 4)
            obs_ruin_count = sum(1 for c in obs_codes if c == 3)

            obs_settl_rate = obs_settl_count / n_obs
            obs_ruin_rate = obs_ruin_count / n_obs

            # Distance to nearest initial settlement
            dist = min((abs(y - sy) + abs(x - sx) for sx, sy in settl_pos), default=99)

            # Settlement stats for this cell (if it's a settlement)
            stats = cell_settl_stats.get((y, x), [])
            avg_pop = np.mean([s.get("population", 0) for s in stats]) if stats else 0
            avg_food = np.mean([s.get("food", 0) for s in stats]) if stats else 0
            avg_wealth = np.mean([s.get("wealth", 0) for s in stats]) if stats else 0
            avg_defense = np.mean([s.get("defense", 0) for s in stats]) if stats else 0
            alive_rate = np.mean([1 if s.get("alive", True) else 0 for s in stats]) if stats else 1

            records.append({
                "initial_code": code,
                "dist_to_settl": dist,
                "n_obs": n_obs,
                "obs_settl_rate": obs_settl_rate,
                "obs_ruin_rate": obs_ruin_rate,
                "has_settl_stats": len(stats) > 0,
                "avg_pop": avg_pop,
                "avg_food": avg_food,
                "avg_wealth": avg_wealth,
                "avg_defense": avg_defense,
                "alive_rate": alive_rate,
                "gt_empty": gt_probs[0],
                "gt_settl": gt_probs[1],
                "gt_port": gt_probs[2],
                "gt_ruin": gt_probs[3],
                "gt_forest": gt_probs[4],
            })

    return records


def main():
    print("=" * 70)
    print("DEEP OBSERVATION ANALYSIS — Astar Island")
    print("=" * 70)

    # ── 1. Round-level statistics ──
    print("\n\n### 1. ROUND-LEVEL OBSERVATION STATISTICS ###\n")
    header = f"{'Round':<6} {'Pop':>6} {'Food':>6} {'Wlth':>6} {'Def':>6} {'Tech':>5} {'Alive%':>7} {'Port%':>6} {'Ship%':>6} {'Fctns':>5} {'S/obs':>6} {'Ruin%':>6}"
    print(header)
    print("-" * len(header))

    round_stats_all = {}
    for rnum in sorted(ROUNDS.keys()):
        try:
            _, _, obs = load_round(rnum)
        except FileNotFoundError:
            continue
        stats = compute_round_stats(obs)
        round_stats_all[rnum] = stats
        print(f"R{rnum:<5} {stats['avg_pop']:>6.1f} {stats['avg_food']:>6.1f} "
              f"{stats['avg_wealth']:>6.1f} {stats['avg_defense']:>6.1f} "
              f"{stats['avg_tech']:>5.1f} {stats['alive_rate']*100:>6.1f}% "
              f"{stats['port_rate']*100:>5.1f}% {stats['longship_rate']*100:>5.1f}% "
              f"{stats['n_factions']:>5} {stats['settl_per_obs']:>6.1f} "
              f"{stats['obs_ruin_rate']*100:>5.2f}%")

    # ── 2. Expansion patterns from GT ──
    print("\n\n### 2. EXPANSION PATTERNS (from Ground Truth) ###\n")
    print(f"{'Round':<6} {'Init':>5} {'Surv':>5} {'SurvR':>6} {'NewS':>5} {'NewP':>5} {'Ruins':>5} {'ExpR':>6}")
    print("-" * 50)

    for rnum in sorted(ROUNDS.keys()):
        try:
            info, gts, _ = load_round(rnum)
        except FileNotFoundError:
            continue
        exp = analyze_expansion_patterns(info, gts)
        for e in exp[:1]:  # Just seed 0 summary
            pass
        # Average across seeds
        avg_init = np.mean([e["initial_settl"] for e in exp])
        avg_surv = np.mean([e["survived"] for e in exp])
        avg_survr = np.mean([e["survival_rate"] for e in exp])
        avg_new = np.mean([e["new_settlements"] for e in exp])
        avg_port = np.mean([e["new_ports"] for e in exp])
        avg_ruin = np.mean([e["new_ruins"] for e in exp])
        avg_expr = np.mean([e["expansion_rate"] for e in exp])
        print(f"R{rnum:<5} {avg_init:>5.0f} {avg_surv:>5.0f} {avg_survr:>5.1%} "
              f"{avg_new:>5.0f} {avg_port:>5.0f} {avg_ruin:>5.0f} {avg_expr:>5.1%}")

    # ── 3. Settlement survival analysis ──
    print("\n\n### 3. SETTLEMENT SURVIVAL ANALYSIS ###\n")
    print("What happens to observed settlements in GT?\n")

    all_feature_data = []
    for rnum in sorted(ROUNDS.keys()):
        try:
            info, gts, obs = load_round(rnum)
        except FileNotFoundError:
            continue
        srecs = extract_settlement_features(obs)
        survival, fdata = analyze_settlement_survival(info, gts, srecs)

        print(f"R{rnum}:")
        for status in ["alive_became", "dead_became"]:
            counts = survival[status]
            total = sum(counts.values())
            if total > 0:
                parts = " ".join(f"{k}={v}({v/total:.0%})" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
                print(f"  {status}: n={total} → {parts}")

        all_feature_data.extend(fdata)

    # ── 4. Feature correlations ──
    print("\n\n### 4. SETTLEMENT FEATURE → GT OUTCOME CORRELATIONS ###\n")
    print("(Only alive settlements, correlation with GT probabilities)\n")

    corrs = compute_feature_correlations(all_feature_data)
    sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, corr in sorted_corrs[:20]:
        direction = "+" if corr > 0 else "-"
        bar = "#" * int(abs(corr) * 50)
        print(f"  {name:<40} r={corr:>+.3f} {bar}")

    # ── 5. Per-cell observation feature analysis ──
    print("\n\n### 5. PER-CELL OBSERVATION FEATURES vs GT ###\n")
    print("Analyzing what cell-level obs features predict GT outcomes...\n")

    all_cell_data = []
    for rnum in sorted(ROUNDS.keys()):
        try:
            info, gts, obs = load_round(rnum)
        except FileNotFoundError:
            continue
        cells = analyze_per_cell_obs_features(info, gts, obs)
        all_cell_data.extend(cells)

    # Group by initial terrain and analyze
    for code_name, codes in [("Plains", {11, 0}), ("Forest", {4,}), ("Settlement", {1, 2})]:
        subset = [d for d in all_cell_data if d["initial_code"] in codes]
        if not subset:
            continue

        print(f"\n{code_name} cells (n={len(subset)}):")

        # Correlation between observation features and GT
        feats = ["obs_settl_rate", "obs_ruin_rate", "avg_pop", "avg_food",
                 "avg_wealth", "avg_defense", "alive_rate", "dist_to_settl"]
        outcomes = ["gt_settl", "gt_empty", "gt_forest", "gt_ruin"]

        for feat in feats:
            vals_f = [d[feat] for d in subset]
            if np.std(vals_f) < 1e-10:
                continue
            for out in outcomes:
                vals_o = [d[out] for d in subset]
                if np.std(vals_o) < 1e-10:
                    continue
                corr = np.corrcoef(vals_f, vals_o)[0, 1]
                if abs(corr) > 0.05:
                    print(f"  {feat:<20} vs {out:<12} r={corr:>+.3f}")

    # ── 6. Round parameter clustering ──
    print("\n\n### 6. HIDDEN PARAMETER ESTIMATES ###\n")
    print("Grouping rounds by observation signatures...\n")

    features_per_round = {}
    for rnum, stats in round_stats_all.items():
        features_per_round[rnum] = [
            stats["avg_pop"], stats["avg_food"], stats["avg_wealth"],
            stats["avg_defense"], stats["alive_rate"], stats["port_rate"],
            stats["settl_per_obs"], stats["n_factions"], stats["obs_ruin_rate"],
        ]

    feat_names = ["pop", "food", "wealth", "defense", "alive%", "port%",
                  "settl/obs", "factions", "ruin%"]

    # Compute z-scores to identify outlier rounds
    all_vals = np.array(list(features_per_round.values()))
    means = all_vals.mean(axis=0)
    stds = all_vals.std(axis=0)
    stds[stds < 1e-10] = 1

    print(f"{'Round':<6} " + " ".join(f"{n:>8}" for n in feat_names))
    print("-" * 90)
    for rnum in sorted(features_per_round.keys()):
        zscores = (np.array(features_per_round[rnum]) - means) / stds
        zscore_str = " ".join(f"{z:>+8.1f}" for z in zscores)
        print(f"R{rnum:<5} {zscore_str}")

    print("\nOutlier rounds (|z| > 1.5):")
    for rnum in sorted(features_per_round.keys()):
        zscores = (np.array(features_per_round[rnum]) - means) / stds
        outliers = [(feat_names[i], zscores[i]) for i in range(len(zscores)) if abs(zscores[i]) > 1.5]
        if outliers:
            parts = ", ".join(f"{name}={z:+.1f}σ" for name, z in outliers)
            print(f"  R{rnum}: {parts}")

    # ── 7. Feature importance ranking for new XGBoost features ──
    print("\n\n### 7. CANDIDATE NEW FEATURES FOR XGBOOST ###\n")
    print("Based on correlation analysis, these observation-derived features")
    print("could improve predictions if added to XGBoost:\n")

    candidates = [
        ("obs_settl_rate", "Fraction of observations where cell was settlement",
         "Round-level: how often settlements appear in viewports"),
        ("obs_ruin_rate", "Fraction of observations where cell was ruin",
         "Round-level: ruin frequency indicates collapse rate"),
        ("avg_pop_zscore", "Normalized average population across observations",
         "High pop = strong expansion, low pop = extinction"),
        ("food_deficit", "avg_food - avg_pop (surplus/deficit)",
         "Negative = starvation risk, positive = growth potential"),
        ("wealth_per_capita", "avg_wealth / max(avg_pop, 1)",
         "Trade activity indicator"),
        ("defense_pop_ratio", "avg_defense / max(avg_pop, 1)",
         "Militarization level"),
        ("faction_diversity", "n_factions / n_observations",
         "More factions = more conflict, less expansion"),
        ("dead_rate", "1 - alive_rate",
         "Direct extinction signal"),
        ("min_food_signal", "min observed food value",
         "Harsh winter indicator (low min food = severe winters)"),
        ("max_pop_signal", "max observed population",
         "Expansion ceiling indicator"),
    ]

    for i, (name, desc, rationale) in enumerate(candidates, 1):
        print(f"  {i}. {name}")
        print(f"     {desc}")
        print(f"     Rationale: {rationale}")
        print()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
