#!/usr/bin/env python3
"""Live orchestrator — continuously collects results, verifies, resubmits, deploys new data.

Runs forever on your Mac. Every cycle:
1. Collect best results from VMs across all regions
2. If improvement found: verify locally with full LORO
3. If verified + active round: auto-resubmit
4. If round completed: download GT, update model, deploy to all VMs

Usage:
    ASTAR_TOKEN=... python orchestrator_live.py
    ASTAR_TOKEN=... python orchestrator_live.py --interval 300  # check every 5 min
"""

import argparse
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
LOG_FILE = TASK_DIR / "orchestrator_live.log"

# Track state
BASELINE_WAVG = 89.47
LAST_ROUND_CHECKED = None
BEST_SWARM_METRIC = 0.0
RESUBMIT_COUNT = 0

# All regions where we have VMs
REGIONS = [
    "europe-west4-a",
    "europe-west1-b",
    "europe-west3-c",
    "us-central1-a",
    "us-east1-b",
    "us-east4-c",
]


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_cmd(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


def gcloud_ssh(vm, zone, command, timeout=15):
    cmd = f'gcloud compute ssh {vm} --zone={zone} --project={PROJECT} --ssh-flag="-o StrictHostKeyChecking=no" --command=\'{command}\' 2>/dev/null'
    return run_cmd(cmd, timeout=timeout)


# ─── Collect results from all VMs ────────────────────────────────

def collect_results():
    """Collect results.tsv from VMs across all regions. Returns best metric + VM info."""
    log("Collecting results from all regions...")

    # Find all astar VMs
    rc, out, _ = run_cmd(
        f'gcloud compute instances list --project={PROJECT} --filter="name~ainm-astar AND status:RUNNING" --format="csv[no-heading](name,zone)"',
        timeout=30
    )
    if rc != 0 or not out:
        log("  Could not list VMs")
        return None

    vms = []
    for line in out.strip().split("\n"):
        parts = line.split(",")
        if len(parts) == 2:
            vms.append((parts[0], parts[1]))

    log(f"  Found {len(vms)} running VMs")

    # Sample ~20 VMs across regions (don't SSH to all 500+)
    import random
    sample = random.sample(vms, min(20, len(vms)))

    best_metric = 0.0
    best_vm = None
    best_zone = None
    total_experiments = 0

    for vm, zone in sample:
        rc, out, _ = gcloud_ssh(vm, zone, 'grep "keep" /tmp/astar/results.tsv 2>/dev/null | sort -t"\t" -k2 -rn | head -1')
        if rc == 0 and out:
            parts = out.split("\t")
            if len(parts) >= 2:
                try:
                    metric = float(parts[1])
                    if metric > best_metric:
                        best_metric = metric
                        best_vm = vm
                        best_zone = zone
                except ValueError:
                    pass

        # Count experiments
        rc2, out2, _ = gcloud_ssh(vm, zone, 'wc -l < /tmp/astar/results.tsv 2>/dev/null || echo 0')
        if rc2 == 0:
            try:
                total_experiments += int(out2.strip()) - 1  # subtract header
            except ValueError:
                pass

    # Extrapolate total experiments from sample
    if len(sample) > 0:
        total_experiments = int(total_experiments * len(vms) / len(sample))

    log(f"  Best metric: {best_metric:.4f} (on {best_vm})")
    log(f"  Estimated total experiments: {total_experiments}")

    return {
        "best_metric": best_metric,
        "best_vm": best_vm,
        "best_zone": best_zone,
        "total_vms": len(vms),
        "total_experiments": total_experiments,
    }


# ─── Verify improvement locally ─────────────────────────────────

def verify_improvement(vm, zone):
    """Download best train.py from VM and verify with full LORO locally."""
    log(f"Downloading best train.py from {vm}...")

    best_file = TASK_DIR / "train.py.swarm_best"
    rc, _, err = run_cmd(
        f'gcloud compute scp {vm}:/tmp/astar/train.py.best {best_file} --zone={zone} --project={PROJECT} 2>&1',
        timeout=30
    )
    if rc != 0:
        # Try train.py directly (some VMs might not have .best)
        rc, _, _ = run_cmd(
            f'gcloud compute scp {vm}:/tmp/astar/train.py {best_file} --zone={zone} --project={PROJECT} 2>&1',
            timeout=30
        )
    if rc != 0:
        log("  Download failed")
        return None

    # Backup current train.py
    backup = TASK_DIR / "train.py.orchestrator_backup"
    backup.write_text((TASK_DIR / "train.py").read_text())

    # Swap in swarm best
    (TASK_DIR / "train.py").write_text(best_file.read_text())

    # Run full LORO
    log("  Running full LORO verification...")
    rc, out, err = run_cmd(
        f'cd {TASK_DIR} && source {REPO_DIR}/.venv/bin/activate && python train.py 2>&1',
        timeout=900
    )

    # Parse val_metric
    val_metric = 0.0
    for line in out.split("\n"):
        if line.startswith("val_metric:"):
            val_metric = float(line.split(":")[1].strip())

    # Restore original
    (TASK_DIR / "train.py").write_text(backup.read_text())

    if val_metric > 0:
        log(f"  Verified WAVG: {val_metric:.4f}")
    else:
        log(f"  LORO failed: {err[-200:]}")

    return val_metric


# ─── Resubmit to active round ───────────────────────────────────

def resubmit(train_py_path):
    """Retrain GBT and resubmit to active round."""
    global RESUBMIT_COUNT

    if not ASTAR_TOKEN:
        log("  No ASTAR_TOKEN — cannot resubmit")
        return False

    log("  Retraining GBT models...")
    # Swap in the improved train.py
    backup = TASK_DIR / "train.py.pre_resubmit"
    backup.write_text((TASK_DIR / "train.py").read_text())
    (TASK_DIR / "train.py").write_text(train_py_path.read_text())

    rc, out, _ = run_cmd(
        f'cd {TASK_DIR} && source {REPO_DIR}/.venv/bin/activate && python retrain_gbt.py 2>&1',
        timeout=300
    )

    if rc != 0:
        log("  Retrain failed")
        (TASK_DIR / "train.py").write_text(backup.read_text())
        return False

    log("  Submitting to active round...")
    rc, out, _ = run_cmd(
        f'cd {TASK_DIR} && source {REPO_DIR}/.venv/bin/activate && ASTAR_TOKEN={ASTAR_TOKEN} python run.py --resume 2>&1',
        timeout=600
    )

    # Restore
    (TASK_DIR / "train.py").write_text(backup.read_text())

    if "accepted" in out:
        RESUBMIT_COUNT += 1
        log(f"  RESUBMITTED! (#{RESUBMIT_COUNT})")
        return True
    else:
        log(f"  Submit failed: {out[-200:]}")
        return False


# ─── Check for new completed round ──────────────────────────────

def check_round_status():
    """Check if there's a new completed round to download GT from."""
    if not ASTAR_TOKEN:
        return None, None

    sys.path.insert(0, str(TASK_DIR))
    try:
        from client import AstarClient
        c = AstarClient(token=ASTAR_TOKEN)
        rounds = c.get_rounds()

        active = None
        latest_completed = None
        for r in rounds:
            if r.status == "active":
                active = r
            if r.status == "completed":
                if latest_completed is None or r.round_number > latest_completed.round_number:
                    latest_completed = r

        return active, latest_completed
    except Exception as e:
        log(f"  Round check error: {e}")
        return None, None


def download_and_update_round(round_info):
    """Download GT for a completed round and update the model."""
    global LAST_ROUND_CHECKED

    rnum = round_info.round_number
    if LAST_ROUND_CHECKED == rnum:
        return  # Already processed

    log(f"New completed round: R{rnum}")

    try:
        from client import AstarClient
        import numpy as np

        c = AstarClient(token=ASTAR_TOKEN)
        detail = c.get_round_detail(round_info.id)

        # Save initial states
        states = {"initial_states": []}
        for seed in range(5):
            s = detail.initial_states[seed]
            states["initial_states"].append({
                "grid": s.grid,
                "settlements": [x.model_dump() for x in s.settlements]
            })
        with open(TASK_DIR / "data" / f"round{rnum}_initial.json", "w") as f:
            json.dump(states, f)

        # Save GT
        for seed in range(5):
            a = c.get_analysis(round_info.id, seed)
            np.save(TASK_DIR / "data" / f"gt_r{rnum}_seed{seed}.npy", np.array(a.ground_truth))
            log(f"  Seed {seed}: score={a.score:.2f}")

        LAST_ROUND_CHECKED = rnum
        log(f"  R{rnum} GT saved. TODO: update ROUNDS dict + retrain manually.")

    except Exception as e:
        log(f"  GT download error: {e}")


# ─── Main loop ──────────────────────────────────────────────────

def main():
    global BASELINE_WAVG, BEST_SWARM_METRIC

    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    args = parser.parse_args()

    if not ASTAR_TOKEN:
        log("WARNING: No ASTAR_TOKEN — will collect results but cannot resubmit")

    log("=" * 60)
    log("LIVE ORCHESTRATOR — Astar Island")
    log(f"Baseline: {BASELINE_WAVG}")
    log(f"Interval: {args.interval}s")
    log(f"Token: {'SET' if ASTAR_TOKEN else 'MISSING'}")
    log("=" * 60)

    cycle = 0
    while True:
        cycle += 1
        log(f"\n{'='*60}")
        log(f"Cycle {cycle}")
        log(f"{'='*60}")

        # 1. Collect results
        results = collect_results()

        if results and results["best_metric"] > BEST_SWARM_METRIC:
            BEST_SWARM_METRIC = results["best_metric"]

            # 2. Check if improvement over baseline
            if results["best_metric"] > BASELINE_WAVG + 0.01:
                log(f"IMPROVEMENT FOUND! {results['best_metric']:.4f} > {BASELINE_WAVG:.4f}")

                # 3. Verify locally
                verified = verify_improvement(results["best_vm"], results["best_zone"])

                if verified is not None and verified > BASELINE_WAVG:
                    log(f"VERIFIED! Local WAVG: {verified:.4f}")
                    BASELINE_WAVG = verified

                    # 4. Check for active round and resubmit
                    active, _ = check_round_status()
                    if active:
                        log(f"Active round: R{active.round_number} — resubmitting...")
                        resubmit(TASK_DIR / "train.py.swarm_best")
                    else:
                        log("No active round — improvement saved for next round")
                elif verified is not None:
                    log(f"Not verified locally ({verified:.4f} vs {BASELINE_WAVG:.4f})")
                else:
                    log("Verification failed (LORO crashed or download failed)")

        # 5. Check for completed rounds (new GT to download)
        active, completed = check_round_status()
        if completed:
            download_and_update_round(completed)

        if active:
            log(f"Active round: R{active.round_number}")

        # Status
        if results:
            log(f"Fleet: {results['total_vms']} VMs | ~{results['total_experiments']} experiments | best={BEST_SWARM_METRIC:.4f} | baseline={BASELINE_WAVG:.4f} | resubmits={RESUBMIT_COUNT}")

        log(f"Sleeping {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
