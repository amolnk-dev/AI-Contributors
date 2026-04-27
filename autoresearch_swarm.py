"""Karpathy-style autoresearch agent using Gemini 3.1 Pro.

Exact same protocol as autoresearch-mlx/program.md:
  1. Read the code. Understand the architecture.
  2. Form a hypothesis. Edit train.py.
  3. git commit. Run. Read val_metric.
  4. Keep if improved (amend commit with results.tsv).
  5. Discard if worse (git reset --hard).
  6. NEVER STOP.

The LLM (Gemini) does its own research — reads full files,
analyzes results, reasons about what to try next.

Usage:
    VM_ID=s1 VM_FOCUS=r7_adaptive GOOGLE_API_KEY=... \
      python3 -u autoresearch_swarm.py

Environment:
    VM_ID          — unique per VM
    VM_FOCUS       — research specialization (see FOCUS_PROMPTS)
    GOOGLE_API_KEY — Gemini API key
    MODEL_ID       — Gemini model (default: gemini-3.1-pro-preview)
    PARALLEL_XGB   — parallel workers (default: 18)
"""

import hashlib
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

TASK_DIR = Path(__file__).parent.resolve()
VM_ID = os.environ.get("VM_ID", "vm0")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("MODEL_ID", "gemini-3.1-flash-lite-preview")
VM_FOCUS = os.environ.get("VM_FOCUS", "general")

TRAIN_PY = TASK_DIR / "train.py"
PARAMS_PY = TASK_DIR / "params.py"
MODEL_PY = TASK_DIR / "model.py"
RESULTS_TSV = TASK_DIR / "results.tsv"

SEED = int(hashlib.md5(VM_ID.encode()).hexdigest()[:8], 16) % (2**31)

# ─── VM Specialization ──────────────────────────────────────────

FOCUS_PROMPTS = {
    "r7_adaptive": "FOCUS: Fix R7=72.5 (extreme expansion round). Try round-type detection from obs_stats, per-type blend/L7, adaptive distance tables.",
    "distance_decay": "FOCUS: Non-linear distance decay. Current tables use linear buckets 1-8. Try quadratic, exponential, or obs-rate-scaled decay.",
    "expansion_features": "FOCUS: New features for expansion pressure. Try settlement cluster density, expansion front detection, food pressure ratio.",
    "faction_analysis": "FOCUS: Faction/owner_id features from observations. Try faction diversity r3, conflict zone detection, conquest events.",
    "port_trade": "FOCUS: Port and trade corridor features. Try port-to-port BFS, coastal path length, trade route indicators.",
    "winter_raiding": "FOCUS: Winter/raiding signals. Try defense variance, wealth depletion rate, population crash proxy.",
    "terrain_interaction": "FOCUS: Terrain boundary features. Try terrain transition counts, biome edge detection, mountain blocking.",
    "directional": "FOCUS: Directional expansion features. Try expansion vectors, coastal direction encoding, asymmetric distances.",
    "l7_tuning": "FOCUS: L7 observation correction. Try per-class strengths, round-type-adaptive L7, dynamic clamp bounds.",
    "xgb_tuning": "FOCUS: XGB hyperparameters. Try per-terrain depth/lr/trees, colsample sweep, regularization tuning.",
    "general": "FOCUS: Creative exploration. Try anything — new features, new blend strategies, architectural changes.",
}


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{VM_ID}] {msg}", flush=True)


# ─── Git operations ─────────────────────────────────────────────

def git(cmd):
    """Run a git command in TASK_DIR."""
    r = subprocess.run(
        f"git {cmd}", shell=True, capture_output=True, text=True,
        cwd=str(TASK_DIR), timeout=30
    )
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def git_setup():
    """Initialize git repo and branch for this VM."""
    if not (TASK_DIR / ".git").exists():
        git("init")
        git("config user.email 'autoresearch@vm'")
        git("config user.name 'autoresearch'")
        git(f"checkout -b autoresearch/{VM_ID}")
        git("add -A")
        git("commit -m 'initial: baseline'")
    log(f"Git branch: autoresearch/{VM_ID}")


def git_commit(message):
    git("add params.py train.py results.tsv")
    git(f'commit -m "{message}"')


def git_amend():
    git("add results.tsv")
    git("commit --amend --no-edit")


def git_reset_hard():
    """Revert to last kept commit."""
    git("checkout -- params.py train.py")


def git_log_oneline(n=10):
    _, out, _ = git(f"log --oneline -{n}")
    return out


def git_diff_stat():
    _, out, _ = git("diff --stat HEAD~1 -- train.py 2>/dev/null")
    return out


# ─── Results TSV (exact Karpathy format) ─────────────────────────

TSV_HEADER = "commit\tval_metric\tmemory_gb\tdiff_lines\tstatus\tdescription\treject_reason\n"


def init_results():
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(TSV_HEADER)


def get_commit_hash():
    _, out, _ = git("rev-parse --short HEAD")
    return out[:7] if out else "0000000"


def count_diff_lines():
    _, out, _ = git("diff HEAD -- params.py")
    return len([l for l in out.split("\n") if l.startswith("+") or l.startswith("-")]) if out else 0


def append_result(val_metric, status, description, reject_reason="-"):
    commit = get_commit_hash()
    diff_lines = count_diff_lines()
    row = f"{commit}\t{val_metric:.6f}\t0.0\t{diff_lines}\t{status}\t{description}\t{reject_reason}\n"
    with open(RESULTS_TSV, "a") as f:
        f.write(row)


def get_results_history():
    if not RESULTS_TSV.exists():
        return "No experiments yet."
    return RESULTS_TSV.read_text()


def get_best_metric():
    if not RESULTS_TSV.exists():
        return 0.0
    best = 0.0
    for line in RESULTS_TSV.read_text().strip().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 5 and parts[4] == "keep":
            try:
                best = max(best, float(parts[1]))
            except ValueError:
                pass
    return best


# ─── Run experiment ──────────────────────────────────────────────

def run_train():
    """Run train.py and return (val_metric, peak_vram_mb, output)."""
    try:
        r = subprocess.run(
            [sys.executable, str(TRAIN_PY)],
            capture_output=True, text=True, timeout=900,
            cwd=str(TASK_DIR),
        )
        output = r.stdout
        val_metric = 0.0
        peak_vram = 0.0
        for line in output.split("\n"):
            if line.startswith("val_metric:"):
                val_metric = float(line.split(":")[1].strip())
            if line.startswith("peak_vram_mb:"):
                peak_vram = float(line.split(":")[1].strip())

        if val_metric == 0.0 and r.returncode != 0:
            log(f"CRASH. Exit code: {r.returncode}")
            log(f"stderr (last 300): {r.stderr[-300:]}")

        return val_metric, peak_vram, output
    except subprocess.TimeoutExpired:
        log("TIMEOUT (>15 min)")
        return 0.0, 0.0, "TIMEOUT"
    except Exception as e:
        log(f"ERROR: {e}")
        return 0.0, 0.0, str(e)


# ─── Gemini: The Researcher ─────────────────────────────────────

def ask_gemini(params_py_content, train_py_content, model_py_summary, results_history, run_idx):
    """Ask Gemini to analyze and suggest ONE code change to params.py.

    Follows the Karpathy protocol: the LLM reads the full code,
    understands the architecture, reasons about what to try,
    and generates a precise edit to params.py (the tunable parameters file).
    """
    if not GOOGLE_API_KEY:
        return None

    focus = FOCUS_PROMPTS.get(VM_FOCUS, FOCUS_PROMPTS["general"])

    prompt = f"""You are an autonomous ML researcher running Karpathy-style autoresearch.
You are optimizing a probabilistic terrain prediction model for a Norse civilization simulator.

## Your Task
Read the code below. Understand the architecture. Form a hypothesis.
Generate ONE edit to params.py that you believe will improve val_metric.

## The Metric
val_metric = weighted average LORO score (higher is better, max 100).
Score formula: 100 * exp(-3 * entropy_weighted_KL_divergence).
KL divergence between your prediction and Monte Carlo ground truth.
CRITICAL: Never let any probability be 0.0 (KL → infinity).

## Per-Round Scores (your weaknesses)
R7=72.5 (extreme expansion — BIGGEST opportunity, 17 pts below mean)
R1=85.5, R14=85.7, R5=86.5, R6=87.6, R11=87.7 (expansion rounds)
R2=92.1, R15=92.4, R10=93.0, R9=93.1, R13=93.7, R4=94.3, R8=95.2 (strong)
R16=88.8, R17=93.0 (recent)
Pattern: STRONG on extinction, WEAK on expansion.

## Your Specialization
{focus}

## Experiment History (results.tsv)
```
{results_history[-3000:]}
```

## Git Log (recent commits)
```
{git_log_oneline(15)}
```

## params.py (THE FILE YOU EDIT — all tunable parameters)
```python
{params_py_content}
```

## train.py (READ ONLY — evaluation pipeline, uses params from params.py)
```python
{train_py_content}
```

## model.py (READ ONLY — key functions for context)
{model_py_summary}

## Rules
1. Generate exactly ONE change to params.py. Small and isolated.
2. Reason about WHY this change should help before writing code.
3. Don't repeat experiments that already failed in results.tsv.
4. Simplicity criterion: don't add ugly complexity for tiny gains.
5. If 3+ consecutive experiments failed, try a completely different direction.
6. NEVER modify imports or the classify_round_type function signature.
7. You can change any parameter values, add new parameters, or modify thresholds.

## What has NOT worked (proven failures — do NOT try these)
- KNN round matching, probability sharpening, LightGBM
- Cross-seed transfer (Layer 4), Layer 2 Bayesian obs
- Higher PROB_FLOOR, simulator blend, voronoi features
- Grid-level features (constant across cells = useless for XGB)
- Plains in r3 (too correlated with existing features)

## Output Format
Respond with ONLY this JSON:
{{
  "reasoning": "2-3 sentences explaining your hypothesis and why this should work",
  "description": "short description for results.tsv (max 60 chars)",
  "search_replace": [
    {{"old": "exact string to find in params.py", "new": "replacement string"}}
  ]
}}

Think step by step. What is the model getting wrong? Why? What change addresses the root cause?"""

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GOOGLE_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.8,
                "maxOutputTokens": 4000,
            }
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())

        text = body["candidates"][0]["content"]["parts"][0]["text"].strip()
        # Strip markdown fences
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        result = json.loads(text)
        return result

    except Exception as e:
        log(f"Gemini error: {e}")
        return None


def get_model_py_summary():
    """Read model.py and extract key function signatures + architecture summary."""
    if not MODEL_PY.exists():
        return "model.py not available"

    content = MODEL_PY.read_text()
    # Extract function defs and docstrings
    lines = content.split("\n")
    summary_lines = []
    for i, line in enumerate(lines):
        if line.strip().startswith("def ") or line.strip().startswith("class "):
            summary_lines.append(line)
            # Get docstring if present
            if i + 1 < len(lines) and '"""' in lines[i + 1]:
                summary_lines.append(lines[i + 1])
        elif "EMPIRICAL" in line or "BLEND" in line or "NUM_CLASSES" in line:
            summary_lines.append(line)

    summary = "\n".join(summary_lines[:80])
    return f"```python\n# model.py key functions (READ ONLY — do not edit):\n{summary}\n```"


def apply_edit(patch):
    """Apply search_replace edit to params.py. Returns True if successful."""
    source = PARAMS_PY.read_text()

    for sr in patch.get("search_replace", []):
        old = sr.get("old", "")
        new = sr.get("new", "")
        if not old:
            log("  Empty search string")
            return False
        if old not in source:
            log(f"  Search string not found: {old[:60]}...")
            return False
        source = source.replace(old, new, 1)

    PARAMS_PY.write_text(source)
    return True


# ─── Main: The Karpathy Loop ────────────────────────────────────

def main():
    rng = random.Random(SEED)

    log("=" * 60)
    log("AUTORESEARCH — Karpathy Protocol")
    log(f"VM={VM_ID} | Focus={VM_FOCUS}")
    log(f"Model={GEMINI_MODEL}")
    log(f"Parallel XGB: {os.environ.get('PARALLEL_XGB', 'default')}")
    log("=" * 60)

    # Setup
    git_setup()
    init_results()

    # Step 1: Establish baseline
    best_metric = get_best_metric()
    if best_metric == 0.0:
        log("Running baseline...")
        val_metric, _, output = run_train()
        if val_metric > 0:
            best_metric = val_metric
            append_result(val_metric, "keep", "baseline")
            git_amend()
            log(f"Baseline: val_metric={val_metric:.6f}")
        else:
            log("Baseline CRASHED. Check train.py.")
            return

    log(f"Starting from best={best_metric:.6f}")
    log("NEVER STOP — running until interrupted")
    log("")

    # Step 2: Loop forever
    run_idx = 0
    consecutive_failures = 0

    while True:
        run_idx += 1
        log(f"{'=' * 60}")
        log(f"Experiment {run_idx} | Best: {best_metric:.6f}")
        log(f"{'=' * 60}")

        # Sync new round data from GCS (if available)
        try:
            subprocess.run(
                ["gsutil", "-q", "rsync", "gs://ainm-astar-data/", str(TASK_DIR / "data") + "/"],
                capture_output=True, timeout=30
            )
        except Exception:
            pass

        # Read current state
        params_py_content = PARAMS_PY.read_text()
        train_py_content = TRAIN_PY.read_text()
        model_py_summary = get_model_py_summary()
        results_history = get_results_history()

        # Ask Gemini for a hypothesis + edit to params.py
        patch = ask_gemini(params_py_content, train_py_content, model_py_summary, results_history, run_idx)

        if patch is None:
            log("Gemini returned nothing. Sleeping 30s and retrying.")
            time.sleep(30)
            continue

        reasoning = patch.get("reasoning", "no reasoning")
        description = patch.get("description", "unknown change")[:60]

        log(f"Hypothesis: {reasoning}")
        log(f"Edit: {description}")

        # Apply the edit
        if not apply_edit(patch):
            log("Edit failed to apply. Discarding.")
            git("checkout -- params.py")
            append_result(0.0, "crash", description, "edit failed to apply")
            consecutive_failures += 1
            if consecutive_failures >= 5:
                log("5 consecutive failures. Sleeping 60s for Gemini to cool down.")
                time.sleep(60)
                consecutive_failures = 0
            continue

        # Git commit
        git_commit(f"experiment: {description}")

        # Run
        log("Running train.py...")
        t0 = time.time()
        val_metric, peak_vram, output = run_train()
        duration = time.time() - t0

        if val_metric <= 0:
            # Crash
            log(f"CRASH ({duration:.0f}s)")
            # Try to read error
            for line in output.split("\n")[-10:]:
                if line.strip():
                    log(f"  {line.strip()}")
            append_result(0.0, "crash", description, "crash/no metric")
            git_reset_hard()
            git("checkout -- train.py")
            consecutive_failures += 1
            continue

        consecutive_failures = 0

        # Decide: keep or discard
        if val_metric > best_metric:
            delta = val_metric - best_metric
            best_metric = val_metric
            append_result(val_metric, "keep", description)
            git_amend()  # include results.tsv in the commit
            log(f"KEEP! val_metric={val_metric:.6f} (+{delta:.6f})")

            # Push best score to GCS leaderboard
            try:
                import socket
                hostname = socket.gethostname()
                zone_guess = subprocess.run(
                    ["curl", "-s", "-H", "Metadata-Flavor: Google",
                     "http://metadata.google.internal/computeMetadata/v1/instance/zone"],
                    capture_output=True, text=True, timeout=3
                ).stdout.strip().split("/")[-1] if True else "unknown"
                leaderboard = f"{val_metric:.6f} {hostname} {zone_guess} {description[:50]}\n"
                lb_file = Path(f"/tmp/best_{VM_ID}.txt")
                lb_file.write_text(leaderboard)
                subprocess.run(
                    ["gsutil", "cp", str(lb_file), f"gs://ainm-astar-results/best_{VM_ID}.txt"],
                    capture_output=True, timeout=10
                )
                log(f"  Pushed to GCS leaderboard")
            except Exception as e:
                log(f"  GCS push failed: {e}")

            # Log per-round scores
            for line in output.split("\n"):
                if line.startswith("round_") and "_score:" in line:
                    log(f"  {line.strip()}")
        else:
            delta = val_metric - best_metric
            append_result(val_metric, "discard", description, f"val_metric {delta:+.6f}")
            git_reset_hard()
            git("checkout -- params.py")
            log(f"DISCARD. val_metric={val_metric:.6f} ({delta:+.6f})")

        log(f"Duration: {duration:.0f}s")
        time.sleep(5)


if __name__ == "__main__":
    main()
