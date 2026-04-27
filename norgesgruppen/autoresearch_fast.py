"""Fast autoresearch: 5-minute experiment loops on M3 MacBook Pro.

Strategy: fine-tune from our already-trained best.pt with short bursts.
Each loop tweaks one hyperparameter, trains 5-10 epochs, keeps or reverts.

Usage:
    nohup python autoresearch_fast.py > autoresearch.log 2>&1 &

Monitor:
    tail -f autoresearch.log
    cat results.tsv | column -t -s $'\t'
"""

import json
import subprocess
import time
import shutil
import os
from datetime import datetime
from pathlib import Path

TASK_DIR = Path(__file__).parent.resolve()
DATA_YAML = TASK_DIR / "data" / "yolo" / "data.yaml"
MODELS_DIR = TASK_DIR / "models"
RESULTS_TSV = TASK_DIR / "results.tsv"
VENV_PYTHON = TASK_DIR.parent.parent / ".venv" / "bin" / "python"
BEST_PT = MODELS_DIR / "best.pt"

# Each experiment: (name, extra_args_for_model.train(), description)
# These fine-tune FROM best.pt with short epoch bursts
EXPERIMENTS = [
    # --- Phase 1: Quick augmentation sweeps (fine-tune best.pt, 8 epochs each) ---
    ("ft-mosaic0.5", {"mosaic": 0.5, "epochs": 8}, "reduce mosaic for small dataset"),
    ("ft-mosaic0", {"mosaic": 0.0, "epochs": 8}, "no mosaic — pure images"),
    ("ft-mixup0.3", {"mixup": 0.3, "epochs": 8}, "more mixup"),
    ("ft-mixup0", {"mixup": 0.0, "epochs": 8}, "no mixup"),
    ("ft-copypaste0.3", {"copy_paste": 0.3, "epochs": 8}, "more copy-paste"),
    ("ft-copypaste0", {"copy_paste": 0.0, "epochs": 8}, "no copy-paste"),
    ("ft-scale0.9", {"scale": 0.9, "epochs": 8}, "aggressive scale aug"),
    ("ft-degrees20", {"degrees": 20.0, "epochs": 8}, "more rotation"),
    ("ft-hsv-high", {"hsv_h": 0.03, "hsv_s": 0.7, "hsv_v": 0.5, "epochs": 8}, "stronger color aug"),

    # --- Phase 2: Loss weight sweeps ---
    ("ft-box10", {"box": 10.0, "epochs": 8}, "higher box loss weight"),
    ("ft-cls1.0", {"cls": 1.0, "epochs": 8}, "higher cls loss — boost classification"),
    ("ft-cls2.0", {"cls": 2.0, "epochs": 8}, "much higher cls loss"),
    ("ft-dfl2.0", {"dfl": 2.0, "epochs": 8}, "higher dfl loss"),

    # --- Phase 3: Learning rate / optimizer ---
    ("ft-lr0.001", {"lr0": 0.001, "epochs": 10}, "lower initial lr"),
    ("ft-lr0.05", {"lr0": 0.05, "epochs": 10}, "higher initial lr"),
    ("ft-warmup0", {"warmup_epochs": 0.0, "epochs": 8}, "no warmup"),
    ("ft-cosine", {"cos_lr": True, "epochs": 12}, "cosine lr schedule"),

    # --- Phase 4: Longer fine-tune with winning config ---
    ("ft-long-20ep", {"epochs": 20, "patience": 10}, "longer fine-tune from best"),
    ("ft-long-30ep", {"epochs": 30, "patience": 15}, "even longer fine-tune"),

    # --- Phase 5: Resolution experiments ---
    ("ft-imgsz960", {"imgsz": 960, "epochs": 10}, "lower res — faster, may lose small objs"),
    ("ft-imgsz1600", {"imgsz": 1600, "batch": 1, "epochs": 8}, "higher res for small products"),

    # --- Phase 6: Close mosaic timing ---
    ("ft-closemosaic5", {"close_mosaic": 5, "epochs": 10}, "close mosaic earlier"),
    ("ft-closemosaic0", {"close_mosaic": 0, "epochs": 10}, "no mosaic close"),
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
    if not RESULTS_TSV.exists():
        return 0.0
    best = 0.0
    for line in RESULTS_TSV.read_text().strip().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 6 and parts[5] == "kept":
            try:
                best = max(best, float(parts[2]))
            except ValueError:
                pass
    return best


def run_experiment(name, train_kwargs, description, timeout_minutes=8):
    log(f"--- {name}: {description} ---")
    start = time.time()

    # Write a temp training script that fine-tunes from best.pt
    # This avoids argument parsing complexity — we directly call model.train()
    imgsz = train_kwargs.pop("imgsz", 1280)
    epochs = train_kwargs.pop("epochs", 8)
    batch = train_kwargs.pop("batch", 2)
    patience = train_kwargs.pop("patience", epochs)  # default: no early stop within short runs

    train_script = TASK_DIR / f"_tmp_train_{name}.py"
    train_script.write_text(f'''
import torch
_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{{**kw, "weights_only": False}})

from ultralytics import YOLO
from pathlib import Path

model = YOLO("{BEST_PT}")
results = model.train(
    data="{DATA_YAML}",
    imgsz={imgsz},
    epochs={epochs},
    batch={batch},
    patience={patience},
    device="mps",
    project="runs",
    name="{name}",
    exist_ok=True,
    seed=42,
    save=True,
    save_period=-1,
    plots=False,
    verbose=False,
    # defaults that can be overridden
    mosaic={train_kwargs.get("mosaic", 1.0)},
    mixup={train_kwargs.get("mixup", 0.1)},
    copy_paste={train_kwargs.get("copy_paste", 0.1)},
    degrees={train_kwargs.get("degrees", 10.0)},
    scale={train_kwargs.get("scale", 0.5)},
    fliplr=0.5,
    flipud=0.0,
    hsv_h={train_kwargs.get("hsv_h", 0.015)},
    hsv_s={train_kwargs.get("hsv_s", 0.5)},
    hsv_v={train_kwargs.get("hsv_v", 0.3)},
    box={train_kwargs.get("box", 7.5)},
    cls={train_kwargs.get("cls", 0.5)},
    dfl={train_kwargs.get("dfl", 1.5)},
    close_mosaic={train_kwargs.get("close_mosaic", 10)},
    lr0={train_kwargs.get("lr0", 0.01)},
    warmup_epochs={train_kwargs.get("warmup_epochs", 3.0)},
    cos_lr={train_kwargs.get("cos_lr", False)},
)

metrics = results.results_dict
map50 = metrics.get("metrics/mAP50(B)", 0.0)
print(f"val_metric: {{map50}}")

best_pt = Path("runs") / "{name}" / "weights" / "best.pt"
if best_pt.exists():
    size_mb = best_pt.stat().st_size / (1024 * 1024)
    print(f"model_size_mb: {{size_mb:.1f}}")
''')

    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"

    try:
        result = subprocess.run(
            [str(VENV_PYTHON), str(train_script)],
            cwd=str(TASK_DIR),
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60,
            env=env,
        )

        duration_min = (time.time() - start) / 60

        # Clean up temp script
        train_script.unlink(missing_ok=True)

        # Save full log
        log_file = TASK_DIR / f"{name}.log"
        log_file.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr)

        if result.returncode != 0:
            err_lines = [l for l in result.stderr.split("\n") if l.strip()][-3:]
            log(f"FAILED ({duration_min:.1f}min): {'; '.join(err_lines)}")
            append_result(name, 0.0, 0.0, duration_min, "failed", f"exit {result.returncode}")
            return None

        # Parse metrics
        val_metric = None
        model_size = 0.0
        for line in result.stdout.split("\n"):
            if line.startswith("val_metric:"):
                try:
                    val_metric = float(line.split(":")[1].strip())
                except ValueError:
                    pass
            if line.startswith("model_size_mb:"):
                try:
                    model_size = float(line.split(":")[1].strip())
                except ValueError:
                    pass

        if val_metric is None:
            log(f"NO METRIC ({duration_min:.1f}min)")
            append_result(name, 0.0, 0.0, duration_min, "failed", "no val_metric")
            return None

        current_best = get_best_metric()
        if val_metric > current_best:
            log(f"NEW BEST: {val_metric:.4f} > {current_best:.4f} ({duration_min:.1f}min)")
            best_pt_path = TASK_DIR / "runs" / name / "weights" / "best.pt"
            if best_pt_path.exists():
                shutil.copy2(best_pt_path, BEST_PT)
            append_result(name, val_metric, model_size, duration_min, "kept",
                         f"beat {current_best:.4f}")
        else:
            log(f"NO GAIN: {val_metric:.4f} <= {current_best:.4f} ({duration_min:.1f}min)")
            append_result(name, val_metric, model_size, duration_min, "rejected",
                         f"below {current_best:.4f}")

        return val_metric

    except subprocess.TimeoutExpired:
        train_script.unlink(missing_ok=True)
        duration_min = (time.time() - start) / 60
        log(f"TIMEOUT ({duration_min:.0f}min)")
        append_result(name, 0.0, 0.0, duration_min, "timeout", "")
        return None


def main():
    log(f"Fast autoresearch: {len(EXPERIMENTS)} experiments, ~5min each")
    log(f"Baseline: {get_best_metric():.4f}")
    init_results()

    for i, (name, kwargs, desc) in enumerate(EXPERIMENTS):
        log(f"\n[{i+1}/{len(EXPERIMENTS)}]")
        run_experiment(name, kwargs.copy(), desc)
        time.sleep(5)  # brief cooldown

    log(f"\nDone! Best: {get_best_metric():.4f}")
    log(f"See results.tsv for full log")


if __name__ == "__main__":
    main()
