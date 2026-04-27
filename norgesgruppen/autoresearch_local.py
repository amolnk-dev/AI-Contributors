"""Autoresearch: local overnight optimization on M3 MacBook Pro.

Runs YOLOv8 experiments sequentially in background.
Each experiment trains a model variant and evaluates with eval_local.py.
Keeps best model, logs everything to results.tsv.

Usage:
    nohup python autoresearch_local.py > autoresearch.log 2>&1 &
    # or:
    python autoresearch_local.py &

Monitor:
    tail -f tasks/norgesgruppen/autoresearch.log
    cat tasks/norgesgruppen/results.tsv | column -t -s $'\t'
"""

import json
import subprocess
import time
import shutil
from datetime import datetime
from pathlib import Path

TASK_DIR = Path(__file__).parent.resolve()
DATA_YAML = TASK_DIR / "data" / "yolo" / "data.yaml"
MODELS_DIR = TASK_DIR / "models"
RESULTS_TSV = TASK_DIR / "results.tsv"
VENV_PYTHON = Path(__file__).parent.parent.parent / ".venv" / "bin" / "python"

# Experiments to run — ordered by expected impact
# Each is (name, train.py args dict, timeout_minutes)
EXPERIMENTS = [
    # 1. Larger model at same resolution
    ("yolov8l-1280", {
        "--model": "yolov8l.pt",
        "--imgsz": "1280",
        "--epochs": "80",
        "--batch": "2",
        "--patience": "15",
        "--device": "mps",
    }, 120),

    # 2. More augmentation on current best (m)
    ("yolov8m-1280-aug", {
        "--model": "yolov8m.pt",
        "--imgsz": "1280",
        "--epochs": "120",
        "--batch": "2",
        "--patience": "25",
        "--device": "mps",
    }, 120),

    # 3. Higher resolution
    ("yolov8m-1600", {
        "--model": "yolov8m.pt",
        "--imgsz": "1600",
        "--epochs": "80",
        "--batch": "1",
        "--patience": "15",
        "--device": "mps",
    }, 150),

    # 4. Small model (fast baseline, useful for ensemble)
    ("yolov8s-1280", {
        "--model": "yolov8s.pt",
        "--imgsz": "1280",
        "--epochs": "100",
        "--batch": "4",
        "--patience": "20",
        "--device": "mps",
    }, 60),

    # 5. XL model (if weights fit in 420MB)
    ("yolov8x-1280", {
        "--model": "yolov8x.pt",
        "--imgsz": "1280",
        "--epochs": "60",
        "--batch": "1",
        "--patience": "15",
        "--device": "mps",
    }, 180),

    # 6. RT-DETR (transformer-based, in ultralytics 8.1.0)
    ("rtdetr-l-1280", {
        "--model": "rtdetr-l.pt",
        "--imgsz": "1280",
        "--epochs": "60",
        "--batch": "1",
        "--patience": "15",
        "--device": "mps",
    }, 150),

    # 7. Resume best with longer training
    ("yolov8m-1280-long", {
        "--model": "yolov8m.pt",
        "--imgsz": "1280",
        "--epochs": "200",
        "--batch": "2",
        "--patience": "30",
        "--device": "mps",
    }, 180),
]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def init_results():
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(
            "timestamp\tname\tval_metric\tmodel_size_mb\tduration_min\tstatus\tnotes\n"
        )


def append_result(name, val_metric, model_size_mb, duration_min, status, notes=""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    row = f"{ts}\t{name}\t{val_metric:.4f}\t{model_size_mb:.1f}\t{duration_min:.1f}\t{status}\t{notes}\n"
    with open(RESULTS_TSV, "a") as f:
        f.write(row)


def get_best_metric():
    """Read best val_metric from results.tsv."""
    if not RESULTS_TSV.exists():
        return 0.0
    best = 0.0
    for line in RESULTS_TSV.read_text().strip().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 3 and parts[5] == "kept":
            try:
                best = max(best, float(parts[2]))
            except ValueError:
                pass
    return best


def run_experiment(name, args, timeout_minutes):
    log(f"=== Starting experiment: {name} ===")
    start = time.time()

    # Build command
    cmd = [str(VENV_PYTHON), str(TASK_DIR / "train.py")]
    for k, v in args.items():
        cmd.extend([k, v])
    # Use relative project path (wandb chokes on absolute paths)
    cmd.extend(["--project", "runs"])
    cmd.extend(["--name", name])

    log(f"Command: {' '.join(cmd)}")

    # Propagate wandb-disable env vars to subprocess
    import os
    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"

    try:
        result = subprocess.run(
            cmd,
            cwd=str(TASK_DIR),
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60,
            env=env,
        )

        duration_min = (time.time() - start) / 60
        log_file = TASK_DIR / f"{name}.log"
        log_file.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr)

        if result.returncode != 0:
            log(f"FAILED (exit {result.returncode})")
            # Show last few lines of error
            for line in result.stderr.split("\n")[-5:]:
                if line.strip():
                    log(f"  stderr: {line}")
            append_result(name, 0.0, 0.0, duration_min, "failed", f"exit {result.returncode}")
            return None

        # Parse val_metric from output
        val_metric = None
        for line in result.stdout.split("\n"):
            if line.startswith("val_metric:"):
                try:
                    val_metric = float(line.split(":")[1].strip())
                except ValueError:
                    pass

        if val_metric is None:
            log("FAILED: no val_metric in output")
            append_result(name, 0.0, 0.0, duration_min, "failed", "no val_metric")
            return None

        # Check model size
        best_pt = TASK_DIR / "runs" / name / "weights" / "best.pt"
        model_size_mb = best_pt.stat().st_size / (1024 * 1024) if best_pt.exists() else 0.0

        log(f"Result: val_metric={val_metric:.4f}, size={model_size_mb:.1f}MB, time={duration_min:.1f}min")

        # Check if model fits in 420MB submission limit
        if model_size_mb > 400:
            log(f"WARNING: Model {model_size_mb:.0f}MB may not fit in 420MB zip")
            append_result(name, val_metric, model_size_mb, duration_min, "too_large",
                         f">{model_size_mb:.0f}MB")
            return val_metric

        # Compare with current best
        current_best = get_best_metric()
        if val_metric > current_best:
            log(f"NEW BEST! {val_metric:.4f} > {current_best:.4f}")
            # Save as new best model
            if best_pt.exists():
                shutil.copy2(best_pt, MODELS_DIR / "best.pt")
                log(f"Saved to models/best.pt")
            append_result(name, val_metric, model_size_mb, duration_min, "kept",
                         f"new best (prev {current_best:.4f})")
        else:
            log(f"Not improved: {val_metric:.4f} <= {current_best:.4f}")
            append_result(name, val_metric, model_size_mb, duration_min, "rejected",
                         f"below best {current_best:.4f}")

        return val_metric

    except subprocess.TimeoutExpired:
        duration_min = (time.time() - start) / 60
        log(f"TIMEOUT after {duration_min:.0f} min")
        append_result(name, 0.0, 0.0, duration_min, "timeout", f">{timeout_minutes}min limit")
        return None


def main():
    log("Autoresearch starting")
    log(f"Task dir: {TASK_DIR}")
    log(f"Python: {VENV_PYTHON}")
    log(f"Data: {DATA_YAML}")
    log(f"Experiments planned: {len(EXPERIMENTS)}")

    init_results()

    # Record baseline
    baseline_pt = MODELS_DIR / "best.pt"
    if baseline_pt.exists():
        size = baseline_pt.stat().st_size / (1024 * 1024)
        log(f"Baseline model: {size:.1f}MB")
        # Check if we have a baseline in results already
        if get_best_metric() == 0.0:
            append_result("baseline-yolov8m", 0.6894, size, 0, "kept", "initial submission score")

    for i, (name, args, timeout) in enumerate(EXPERIMENTS):
        log(f"\n{'='*60}")
        log(f"Experiment {i+1}/{len(EXPERIMENTS)}: {name}")
        log(f"{'='*60}")

        val_metric = run_experiment(name, args, timeout)

        # Brief pause between experiments to let system cool
        if i < len(EXPERIMENTS) - 1:
            log("Cooling 30s before next experiment...")
            time.sleep(30)

    log("\n" + "="*60)
    log("Autoresearch complete!")
    log(f"Best metric: {get_best_metric():.4f}")
    log("Results saved to results.tsv")
    log("="*60)


if __name__ == "__main__":
    main()
