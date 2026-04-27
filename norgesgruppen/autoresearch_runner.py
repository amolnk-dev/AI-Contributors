"""Autonomous autoresearch runner for NorgesGruppen object detection.

Runs a series of experiments, logs results to results.tsv, and keeps
the best model. Based on the autoresearch-mlx protocol.

Usage:
    python autoresearch_runner.py  # runs all experiments sequentially

Each experiment:
    1. Modify train.py args via CLI overrides
    2. Run training
    3. Read val_metric from output
    4. Log to results.tsv
    5. If improved → save as new best, else discard
"""

import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

TASK_DIR = Path(__file__).parent
DATA_YAML = str(TASK_DIR / "data" / "yolo" / "data.yaml")
RESULTS_TSV = TASK_DIR / "results.tsv"
BEST_MODEL = TASK_DIR / "models" / "best_overall.pt"
BEST_METRIC = 0.0

# Experiment definitions: (name, description, train.py args)
EXPERIMENTS = [
    # 1. Baseline: YOLOv8l 100 epochs (already done, but re-run to establish baseline in results.tsv)
    ("baseline_l", "YOLOv8l 100ep 1280px baseline", {
        "--model": "yolov8l.pt", "--imgsz": "1280", "--epochs": "100",
        "--batch": "2", "--name": "exp_baseline_l",
    }),
    # 2. More epochs — let it converge further
    ("longer_200ep", "YOLOv8l 200ep 1280px longer training", {
        "--model": "yolov8l.pt", "--imgsz": "1280", "--epochs": "200",
        "--batch": "2", "--patience": "30", "--name": "exp_longer",
    }),
    # 3. Resume from best with lower LR
    ("finetune_lr001", "finetune best_overall.pt lr=0.001 50ep", {
        "--model": "models/best_overall.pt", "--imgsz": "1280", "--epochs": "50",
        "--batch": "2", "--patience": "15", "--name": "exp_finetune_lr",
    }),
    # 4. YOLOv8x — even bigger model
    ("yolov8x", "YOLOv8x 100ep 1280px largest model", {
        "--model": "yolov8x.pt", "--imgsz": "1280", "--epochs": "100",
        "--batch": "1", "--name": "exp_x",
    }),
    # 5. Higher resolution
    ("res_1600", "YOLOv8l 100ep 1600px higher resolution", {
        "--model": "yolov8l.pt", "--imgsz": "1600", "--epochs": "100",
        "--batch": "1", "--name": "exp_res1600",
    }),
    # 6. More augmentation
    ("aug_heavy", "YOLOv8l 100ep heavy augmentation", {
        "--model": "yolov8l.pt", "--imgsz": "1280", "--epochs": "100",
        "--batch": "2", "--name": "exp_aug_heavy",
    }),
    # 7. Less augmentation
    ("aug_light", "YOLOv8l 100ep light augmentation", {
        "--model": "yolov8l.pt", "--imgsz": "1280", "--epochs": "100",
        "--batch": "2", "--name": "exp_aug_light",
    }),
    # 8. RT-DETR — transformer-based detector
    ("rtdetr_l", "RT-DETR-l 100ep 1280px transformer", {
        "--model": "rtdetr-l.pt", "--imgsz": "1280", "--epochs": "100",
        "--batch": "2", "--name": "exp_rtdetr",
    }),
    # 9. Resume best with cosine LR schedule, long patience
    ("finetune_cos", "finetune best_overall.pt cosine LR 100ep", {
        "--model": "models/best_overall.pt", "--imgsz": "1280", "--epochs": "100",
        "--batch": "2", "--patience": "30", "--name": "exp_finetune_cos",
    }),
    # 10. YOLOv8l at 960px (faster training, might be enough resolution)
    ("res_960", "YOLOv8l 100ep 960px lower resolution", {
        "--model": "yolov8l.pt", "--imgsz": "960", "--epochs": "100",
        "--batch": "4", "--name": "exp_res960",
    }),
    # 11. Resume best for 50 more epochs — final polish
    ("final_polish", "finetune best_overall.pt 50ep final polish", {
        "--model": "models/best_overall.pt", "--imgsz": "1280", "--epochs": "50",
        "--batch": "2", "--patience": "20", "--name": "exp_final",
    }),
]

# Augmentation overrides for specific experiments
AUG_OVERRIDES = {
    "aug_heavy": {"mosaic": 1.0, "mixup": 0.3, "copy_paste": 0.3, "degrees": 15.0, "scale": 0.7, "erasing": 0.5},
    "aug_light": {"mosaic": 0.5, "mixup": 0.0, "copy_paste": 0.0, "degrees": 5.0, "scale": 0.3, "erasing": 0.2},
    "finetune_lr001": {"lr0": 0.001, "lrf": 0.01, "warmup_epochs": 1.0},
    "finetune_cos": {"lr0": 0.005, "lrf": 0.001, "cos_lr": True, "warmup_epochs": 2.0},
    "final_polish": {"lr0": 0.001, "lrf": 0.001, "warmup_epochs": 1.0, "mosaic": 0.5, "mixup": 0.05},
}


def init_results():
    """Initialize results.tsv if it doesn't exist."""
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(
            "timestamp\tname\tval_metric\tmodel_size_mb\tstatus\tdescription\n"
        )


def log_result(name, val_metric, model_size_mb, status, description):
    """Append a result to results.tsv."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{ts}\t{name}\t{val_metric:.6f}\t{model_size_mb:.1f}\t{status}\t{description}\n")


def run_experiment(name, description, args, aug_overrides=None):
    """Run a single training experiment."""
    global BEST_METRIC

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"Description: {description}")
    print(f"Args: {args}")
    if aug_overrides:
        print(f"Augmentation overrides: {aug_overrides}")
    print(f"{'='*60}\n")

    # Build command
    cmd = ["python3", "train.py", "--data", DATA_YAML, "--device", "0"]
    for k, v in args.items():
        cmd.extend([k, str(v)])

    # For augmentation overrides, we need to modify train.py temporarily
    # Instead, we'll write a wrapper that patches the train call
    if aug_overrides:
        wrapper = TASK_DIR / f"_run_{name}.py"
        wrapper_code = f'''
import sys, os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
_orig = torch.load
def _patched(*a, **kw):
    if "weights_only" not in kw: kw["weights_only"] = False
    return _orig(*a, **kw)
torch.load = _patched
from ultralytics import YOLO
from pathlib import Path
import shutil

model = YOLO("{args.get("--model", "yolov8l.pt")}")
results = model.train(
    data="{DATA_YAML}",
    imgsz={args.get("--imgsz", "1280")},
    epochs={args.get("--epochs", "100")},
    batch={args.get("--batch", "2")},
    patience={args.get("--patience", "20")},
    device=0,
    project="runs",
    name="{args.get("--name", "exp")}",
    exist_ok=True,
    seed=42,
    mosaic={aug_overrides.get("mosaic", 1.0)},
    mixup={aug_overrides.get("mixup", 0.1)},
    copy_paste={aug_overrides.get("copy_paste", 0.1)},
    degrees={aug_overrides.get("degrees", 10.0)},
    scale={aug_overrides.get("scale", 0.5)},
    fliplr=0.5, flipud=0.0,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
    box=7.5, cls=0.5, dfl=1.5,
    save=True, save_period=-1, plots=True, verbose=True,
    lr0={aug_overrides.get("lr0", 0.01)},
    lrf={aug_overrides.get("lrf", 0.01)},
    warmup_epochs={aug_overrides.get("warmup_epochs", 3.0)},
    cos_lr={aug_overrides.get("cos_lr", False)},
    erasing={aug_overrides.get("erasing", 0.4)},
)
metrics = results.results_dict
map50 = metrics.get("metrics/mAP50(B)", 0.0)
best_pt = Path("runs") / "{args.get("--name", "exp")}" / "weights" / "best.pt"
if best_pt.exists():
    size_mb = best_pt.stat().st_size / (1024 * 1024)
    print(f"Model size: {{size_mb:.1f}} MB")
print(f"val_metric: {{map50}}")
try:
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"peak_vram_mb: {{peak:.0f}}")
except: pass
'''
        wrapper.write_text(wrapper_code)
        cmd = ["python3", str(wrapper)]

    # Run with timeout (2 hours max per experiment)
    log_path = TASK_DIR / f"run_{name}.log"
    start = time.time()

    try:
        env = dict(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        result = subprocess.run(
            cmd,
            cwd=str(TASK_DIR),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
            env=env,
        )
        log_path.write_text(result.stdout + "\n" + result.stderr)
        duration = time.time() - start

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after 2 hours")
        log_result(name, 0.0, 0.0, "timeout", description)
        return
    except Exception as e:
        print(f"CRASH: {e}")
        log_result(name, 0.0, 0.0, "crash", description)
        return

    # Parse val_metric
    val_metric = 0.0
    model_size = 0.0
    for line in result.stdout.splitlines():
        if line.startswith("val_metric:"):
            try:
                val_metric = float(line.split(":")[1].strip())
            except ValueError:
                pass
        if line.startswith("Model size:"):
            try:
                model_size = float(line.split(":")[1].strip().replace("MB", "").strip())
            except ValueError:
                pass

    if result.returncode != 0 or val_metric == 0.0:
        print(f"FAILED: returncode={result.returncode}, val_metric={val_metric}")
        # Check last 20 lines for error
        lines = (result.stderr or result.stdout).strip().splitlines()
        for l in lines[-20:]:
            print(f"  {l}")
        log_result(name, val_metric, model_size, "crash", description)
        return

    print(f"val_metric: {val_metric:.6f} (duration: {duration/60:.1f} min)")

    # Compare with best
    exp_name = args.get("--name", "exp")
    best_pt = TASK_DIR / "runs" / exp_name / "weights" / "best.pt"

    if val_metric > BEST_METRIC:
        print(f"NEW BEST! {val_metric:.6f} > {BEST_METRIC:.6f}")
        BEST_METRIC = val_metric
        if best_pt.exists():
            BEST_MODEL.parent.mkdir(exist_ok=True)
            shutil.copy2(best_pt, BEST_MODEL)
            print(f"Saved to {BEST_MODEL}")
        log_result(name, val_metric, model_size, "keep", description)
    else:
        print(f"No improvement: {val_metric:.6f} <= {BEST_METRIC:.6f}")
        log_result(name, val_metric, model_size, "discard", description)


import os

def main():
    global BEST_METRIC

    init_results()

    # Initialize best model from existing best if available
    existing_best = TASK_DIR / "models" / "best.pt"
    if existing_best.exists() and not BEST_MODEL.exists():
        shutil.copy2(existing_best, BEST_MODEL)
        print(f"Initialized best_overall.pt from existing best.pt")

    # Set initial best metric from previous run
    BEST_METRIC = 0.6454  # YOLOv8l baseline from previous training

    print(f"\n{'#'*60}")
    print(f"AUTORESEARCH: NorgesGruppen Object Detection")
    print(f"Starting at: {datetime.now()}")
    print(f"Initial best metric: {BEST_METRIC}")
    print(f"Experiments planned: {len(EXPERIMENTS)}")
    print(f"{'#'*60}\n")

    for name, description, args in EXPERIMENTS:
        aug = AUG_OVERRIDES.get(name)
        try:
            run_experiment(name, description, args, aug)
        except Exception as e:
            print(f"UNEXPECTED ERROR in {name}: {e}")
            log_result(name, 0.0, 0.0, "crash", f"{description} - {e}")

        # Print running summary
        print(f"\n--- Running Best: {BEST_METRIC:.6f} ---\n")

    print(f"\n{'#'*60}")
    print(f"AUTORESEARCH COMPLETE")
    print(f"Final best metric: {BEST_METRIC}")
    print(f"Best model: {BEST_MODEL}")
    print(f"Results: {RESULTS_TSV}")
    print(f"Finished at: {datetime.now()}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
