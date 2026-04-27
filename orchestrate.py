#!/usr/bin/env python3
"""Autoresearch orchestrator — monitors agents, tests stacking, reports progress.

Usage:
    python orchestrate.py --status       # Dashboard: show agent progress
    python orchestrate.py --deploy       # Deploy autoresearch to GCP VMs
    python orchestrate.py --collect      # Collect results from GCP VMs
    python orchestrate.py --stack        # Test if agent improvements stack
    python orchestrate.py --final        # Full integration test

Monitors 3 experiment streams:
    1. GCP VM experiments (fleet sweep on ainm-astar-sweep + other VMs)
    2. Local worktree agents (if running)
    3. Combined stacking tests
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(TASK_DIR, "..", ".."))
LOG_FILE = os.path.join(TASK_DIR, "orchestrator.log")
RESULTS_DIR = os.path.join(TASK_DIR, "gcp_results")

# GCP VMs to use for autoresearch (CPU-only, astar doesn't need GPU)
SWEEP_VM = "ainm-astar-sweep"
SWEEP_ZONE = "europe-west4-a"

# Additional VMs that can run experiments
EXPERIMENT_VMS = [
    ("ainm-norgesgruppen-exp4", "europe-west1-b"),
    ("ainm-norgesgruppen-exp5", "us-central1-a"),
    ("ainm-norgesgruppen-exp6", "us-central1-a"),
    ("ainm-norgesgruppen-exp7", "us-central1-a"),
    ("ainm-norgesgruppen-exp8", "us-central1-a"),
    ("ainm-norgesgruppen-exp9", "us-central1-a"),
    ("ainm-norgesgruppen-exp10", "us-central1-a"),
    ("ainm-norgesgruppen-exp11", "us-central1-a"),
    ("ainm-norgesgruppen-exp12", "us-central1-a"),
    ("ainm-norgesgruppen-exp13", "us-central1-a"),
    ("ainm-norgesgruppen-ar2", "europe-west1-b"),
]

# Files needed on each VM
DEPLOY_FILES = [
    "train.py", "model.py", "evaluate.py", "dtos.py", "utils.py",
    "simulator.py", "client.py",
]
DATA_FILES_PATTERNS = [
    "data/gt_r*_seed*.npy",
    "data/round*_initial.json",
    "data/obs_*.jsonl",
    "data/gbt_models.pkl",
    "data/calibration.json",
]

BASELINE_WAVG = 89.41  # Current baseline


def log(msg):
    """Append to orchestrator log."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_cmd(cmd, timeout=120):
    """Run a shell command and return (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"


def gcloud_ssh(vm, zone, command, timeout=30):
    """Run a command on a GCP VM via gcloud ssh."""
    cmd = f"gcloud compute ssh {vm} --zone={zone} --project=ai-nm26osl-1823 --command='{command}' 2>/dev/null"
    return run_cmd(cmd, timeout=timeout)


def deploy_to_vm(vm, zone):
    """Deploy astar-island code + data to a VM."""
    log(f"Deploying to {vm} ({zone})...")
    remote_dir = "/tmp/astar"

    # Create remote directory
    gcloud_ssh(vm, zone, f"mkdir -p {remote_dir}/data {remote_dir}/experiments")

    # Copy code files
    code_files = " ".join(os.path.join(TASK_DIR, f) for f in DEPLOY_FILES if os.path.exists(os.path.join(TASK_DIR, f)))
    cmd = f"gcloud compute scp {code_files} {vm}:{remote_dir}/ --zone={zone} --project=ai-nm26osl-1823 2>/dev/null"
    rc, out, err = run_cmd(cmd, timeout=60)
    if rc != 0:
        log(f"  ERROR copying code to {vm}: {err}")
        return False

    # Copy data files
    import glob
    data_files = []
    for pattern in DATA_FILES_PATTERNS:
        data_files.extend(glob.glob(os.path.join(TASK_DIR, pattern)))
    if data_files:
        data_str = " ".join(data_files)
        cmd = f"gcloud compute scp {data_str} {vm}:{remote_dir}/data/ --zone={zone} --project=ai-nm26osl-1823 2>/dev/null"
        rc, out, err = run_cmd(cmd, timeout=120)
        if rc != 0:
            log(f"  ERROR copying data to {vm}: {err}")
            return False

    # Copy experiment variants if they exist
    exp_dir = os.path.join(TASK_DIR, "experiments")
    if os.path.exists(exp_dir):
        exp_files = " ".join(
            os.path.join(exp_dir, f) for f in os.listdir(exp_dir)
            if f.endswith(".py") or f == "manifest.txt"
        )
        if exp_files:
            cmd = f"gcloud compute scp {exp_files} {vm}:{remote_dir}/experiments/ --zone={zone} --project=ai-nm26osl-1823 2>/dev/null"
            run_cmd(cmd, timeout=60)

    # Install dependencies
    gcloud_ssh(vm, zone, f"cd {remote_dir} && pip install -q xgboost scikit-learn scipy numpy pydantic 2>/dev/null", timeout=120)

    log(f"  Deployed to {vm}")
    return True


def run_experiment_on_vm(vm, zone, experiment_name="baseline", train_file=None):
    """Run a single experiment on a VM."""
    remote_dir = "/tmp/astar"

    if train_file:
        # Copy custom train.py
        cmd = f"gcloud compute scp {train_file} {vm}:{remote_dir}/train.py --zone={zone} --project=ai-nm26osl-1823 2>/dev/null"
        run_cmd(cmd, timeout=30)

    # Run experiment
    log(f"  Running {experiment_name} on {vm}...")
    rc, out, err = gcloud_ssh(
        vm, zone,
        f"cd {remote_dir} && PYTHONUNBUFFERED=1 python3 train.py 2>&1",
        timeout=900  # 15 min max
    )

    # Parse val_metric
    val_metric = None
    per_round = {}
    for line in out.split("\n"):
        if line.startswith("val_metric:"):
            try:
                val_metric = float(line.split(":")[1].strip())
            except ValueError:
                pass
        if line.startswith("round_") and "_score:" in line:
            parts = line.split(":")
            round_num = int(parts[0].replace("round_", "").replace("_score", ""))
            score = float(parts[1].strip())
            per_round[round_num] = score

    return {
        "vm": vm,
        "experiment": experiment_name,
        "val_metric": val_metric,
        "per_round": per_round,
        "returncode": rc,
        "output": out[-500:] if out else "",
        "error": err[-200:] if err else "",
    }


def deploy_cmd(args):
    """Deploy code to all experiment VMs."""
    vms = [(SWEEP_VM, SWEEP_ZONE)] + EXPERIMENT_VMS[:args.num_vms - 1]
    log(f"Deploying to {len(vms)} VMs...")

    for vm, zone in vms:
        deploy_to_vm(vm, zone)

    log("Deployment complete.")


def run_sweep_cmd(args):
    """Run experiment sweep on GCP VMs."""
    # Read manifest
    manifest_path = os.path.join(TASK_DIR, "experiments", "manifest.txt")
    if not os.path.exists(manifest_path):
        log("No manifest.txt found. Run 'python gen_experiments.py' first.")
        return

    experiments = []
    with open(manifest_path) as f:
        for line in f:
            if line.startswith("name|") or not line.strip():
                continue
            parts = line.strip().split("|")
            if len(parts) >= 3:
                experiments.append({"name": parts[0], "file": parts[1], "desc": parts[2]})

    vms = [(SWEEP_VM, SWEEP_ZONE)] + EXPERIMENT_VMS[:len(experiments) - 1]
    log(f"Running {len(experiments)} experiments on {len(vms)} VMs...")

    results = []
    for i, exp in enumerate(experiments):
        vm, zone = vms[i % len(vms)]
        train_file = os.path.join(TASK_DIR, "experiments", exp["file"])
        if not os.path.exists(train_file):
            log(f"  SKIP {exp['name']}: {train_file} not found")
            continue
        result = run_experiment_on_vm(vm, zone, exp["name"], train_file)
        results.append(result)
        if result["val_metric"]:
            delta = result["val_metric"] - BASELINE_WAVG
            sign = "+" if delta >= 0 else ""
            log(f"  {exp['name']}: WAVG={result['val_metric']:.4f} ({sign}{delta:.4f}) — {exp['desc']}")
        else:
            log(f"  {exp['name']}: FAILED — {result['error'][:100]}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(RESULTS_DIR, f"sweep_{ts}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {results_path}")

    # Summary
    valid = [r for r in results if r["val_metric"] is not None]
    if valid:
        best = max(valid, key=lambda r: r["val_metric"])
        log(f"\nBest: {best['experiment']} WAVG={best['val_metric']:.4f}")


def status_cmd(args):
    """Show orchestrator dashboard."""
    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║  AUTORESEARCH DASHBOARD — {datetime.now().strftime('%Y-%m-%d %H:%M')}              ║")
    print("╠══════════════════════════════════════════════════════════╣")

    # Check GCP sweep VM
    rc, out, _ = gcloud_ssh(SWEEP_VM, SWEEP_ZONE, "pgrep -f 'python.*train' && echo RUNNING || echo IDLE", timeout=10)
    sweep_status = "RUNNING" if "RUNNING" in out else "IDLE"
    print(f"║  GCP Sweep VM ({SWEEP_VM}): {sweep_status:<20}    ║")

    # Check for latest results
    if os.path.exists(RESULTS_DIR):
        result_files = sorted(
            [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")],
            reverse=True
        )
        if result_files:
            latest = os.path.join(RESULTS_DIR, result_files[0])
            with open(latest) as f:
                results = json.load(f)
            valid = [r for r in results if r.get("val_metric") is not None]
            if valid:
                best = max(valid, key=lambda r: r["val_metric"])
                delta = best["val_metric"] - BASELINE_WAVG
                sign = "+" if delta >= 0 else ""
                print(f"║  Latest sweep: {len(valid)} experiments                       ║")
                print(f"║  Best: {best['experiment']:<15} WAVG={best['val_metric']:.4f} ({sign}{delta:.4f})  ║")
    print("╠══════════════════════════════════════════════════════════╣")

    # Check local worktrees
    worktrees = [
        ("worktree-expansion", "Agent A (R7 specialist)"),
        ("worktree-features", "Agent B (features)"),
        ("worktree-params", "Agent C (params)"),
    ]
    for wt_dir, label in worktrees:
        wt_path = os.path.join(REPO_DIR, "..", wt_dir)
        if os.path.exists(wt_path):
            # Count commits
            rc, out, _ = run_cmd(f"cd {wt_path} && git log --oneline autoresearch/r16.. 2>/dev/null")
            commits = len(out.strip().split("\n")) if out.strip() else 0
            # Check if running
            rc2, out2, _ = run_cmd(f"ls {wt_path}/tasks/astar-island/run.log 2>/dev/null")
            status = "ACTIVE" if rc2 == 0 else "SETUP"
            print(f"║  {label:<30} {status:<8} {commits} experiments  ║")
        else:
            print(f"║  {label:<30} NOT CREATED              ║")

    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Baseline WAVG: {BASELINE_WAVG:.4f}                              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Show recent log entries
    if os.path.exists(LOG_FILE):
        print(f"\nRecent log ({LOG_FILE}):")
        with open(LOG_FILE) as f:
            lines = f.readlines()
        for line in lines[-10:]:
            print(f"  {line.rstrip()}")


def collect_cmd(args):
    """Collect latest results from GCP VM."""
    rc, out, _ = gcloud_ssh(
        SWEEP_VM, SWEEP_ZONE,
        "cat /tmp/astar/run.log 2>/dev/null | grep -E '^(round_|val_metric|weighted_avg|training_seconds)' | tail -20",
        timeout=15
    )
    if out:
        print(f"Latest results from {SWEEP_VM}:")
        print(out)
    else:
        print(f"No results on {SWEEP_VM} yet.")


def main():
    parser = argparse.ArgumentParser(description="Autoresearch orchestrator")
    parser.add_argument("--status", action="store_true", help="Show dashboard")
    parser.add_argument("--deploy", action="store_true", help="Deploy to GCP VMs")
    parser.add_argument("--sweep", action="store_true", help="Run experiment sweep")
    parser.add_argument("--collect", action="store_true", help="Collect GCP results")
    parser.add_argument("--stack", action="store_true", help="Test stacking")
    parser.add_argument("--final", action="store_true", help="Full integration")
    parser.add_argument("--num-vms", type=int, default=3, help="Number of VMs to use")
    args = parser.parse_args()

    if args.status:
        status_cmd(args)
    elif args.deploy:
        deploy_cmd(args)
    elif args.sweep:
        run_sweep_cmd(args)
    elif args.collect:
        collect_cmd(args)
    elif args.stack:
        log("Stacking test not yet implemented — run manually with combined train.py")
    elif args.final:
        log("Final integration not yet implemented — collect + stack + validate")
    else:
        status_cmd(args)


if __name__ == "__main__":
    main()
