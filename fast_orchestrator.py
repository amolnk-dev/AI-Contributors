#!/usr/bin/env python3
"""Fast orchestrator — reads GCS leaderboard, pulls winner, retrains, resubmits.

No local LORO. Trusts VM scores. 2 min per improvement.

Usage:
    ASTAR_TOKEN=... python fast_orchestrator.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

TASK_DIR = Path(__file__).parent.resolve()
REPO_DIR = TASK_DIR.parent.parent
PROJECT = "ai-nm26osl-1823"
ASTAR_TOKEN = os.environ.get("ASTAR_TOKEN", "")
BASELINE = 89.74  # current production baseline
THRESHOLD = 0.2   # only pull if improvement > this

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def read_gcs_leaderboard():
    """Read best scores from all VMs via GCS. Returns (score, vm, zone) or None."""
    try:
        r = subprocess.run(
            ["gsutil", "cat", "gs://ainm-astar-results/best_*.txt"],
            capture_output=True, text=True, timeout=15
        )
        if r.returncode != 0:
            return None

        best_score, best_vm, best_zone = 0, "", ""
        for line in r.stdout.strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    score = float(parts[0])
                    if score > best_score:
                        best_score = score
                        best_vm = parts[1]
                        best_zone = parts[2]
                except ValueError:
                    pass

        return (best_score, best_vm, best_zone) if best_score > 0 else None
    except Exception as e:
        log(f"GCS read error: {e}")
        return None

def pull_and_integrate(vm, zone, score):
    """Download winning train.py, retrain, resubmit. ~2 min total."""
    global BASELINE

    log(f"Pulling from {vm} (score={score:.4f})...")

    # Download train.py
    best_file = TASK_DIR / "train.py.winner"
    rc = subprocess.run(
        ["gcloud", "compute", "scp", f"{vm}:/tmp/astar/train.py", str(best_file),
         f"--zone={zone}", f"--project={PROJECT}"],
        capture_output=True, timeout=30
    ).returncode

    if rc != 0:
        log("  Download failed")
        return False

    # Swap in
    backup = TASK_DIR / "train.py.pre_winner"
    backup.write_text((TASK_DIR / "train.py").read_text())
    (TASK_DIR / "train.py").write_text(best_file.read_text())

    # Retrain
    log("  Retraining GBT...")
    r = subprocess.run(
        [sys.executable, str(TASK_DIR / "retrain_gbt.py")],
        capture_output=True, text=True, timeout=300, cwd=str(TASK_DIR)
    )
    if r.returncode != 0:
        log(f"  Retrain failed: {r.stderr[-200:]}")
        (TASK_DIR / "train.py").write_text(backup.read_text())
        return False

    # Resubmit
    if ASTAR_TOKEN:
        log("  Resubmitting...")
        env = os.environ.copy()
        env["ASTAR_TOKEN"] = ASTAR_TOKEN
        r = subprocess.run(
            [sys.executable, str(TASK_DIR / "run.py"), "--resume"],
            capture_output=True, text=True, timeout=600, cwd=str(TASK_DIR), env=env
        )
        if "accepted" in r.stdout:
            log(f"  RESUBMITTED! New baseline: {score:.4f}")
            BASELINE = score

            # Upload new train.py to GCS for other VMs
            subprocess.run(
                ["gsutil", "cp", str(TASK_DIR / "train.py"), "gs://ainm-astar-data/train_latest.py"],
                capture_output=True, timeout=15
            )
            return True
        else:
            log(f"  Submit failed")

    return False

def check_rounds():
    """Check for active/completed rounds."""
    if not ASTAR_TOKEN:
        return None, None
    try:
        sys.path.insert(0, str(TASK_DIR))
        from client import AstarClient
        c = AstarClient(token=ASTAR_TOKEN)
        rounds = c.get_rounds()
        active = next((r for r in rounds if r.status == "active"), None)
        completed = sorted(
            [r for r in rounds if r.status == "completed"],
            key=lambda r: r.round_number, reverse=True
        )
        latest_completed = completed[0] if completed else None
        return active, latest_completed
    except Exception as e:
        log(f"Round check error: {e}")
        return None, None

def main():
    log("=" * 50)
    log("FAST ORCHESTRATOR")
    log(f"Baseline: {BASELINE} | Threshold: +{THRESHOLD}")
    log(f"Token: {'SET' if ASTAR_TOKEN else 'MISSING'}")
    log("=" * 50)

    last_round = None

    while True:
        # 1. Check GCS leaderboard
        result = read_gcs_leaderboard()
        if result:
            score, vm, zone = result
            delta = score - BASELINE
            if delta > THRESHOLD:
                log(f"IMPROVEMENT: {score:.4f} (+{delta:.4f}) on {vm}")
                pull_and_integrate(vm, zone, score)
            else:
                log(f"Best: {score:.4f} ({delta:+.4f}) — below threshold")
        else:
            log("No GCS results yet")

        # 2. Check rounds
        active, completed = check_rounds()
        if active:
            log(f"Active: R{active.round_number}")
        if completed and (last_round is None or completed.round_number > last_round):
            log(f"New completed round: R{completed.round_number}")
            # TODO: download GT, upload to GCS
            last_round = completed.round_number

        time.sleep(60)

if __name__ == "__main__":
    main()
