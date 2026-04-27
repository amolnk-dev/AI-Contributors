"""NorgesGruppen Object Detection — YOLOv8 Training Script.

Autoresearch-compatible: prints `val_metric: <mAP50>` on completion.

Usage:
    python train.py                           # defaults
    python train.py --model yolov8l.pt --imgsz 1280 --epochs 100
    python train.py --model yolov8x.pt --imgsz 960 --batch 4

Prerequisites:
    pip install ultralytics==8.1.0
    python convert_coco.py  # creates data/yolo/data.yaml
"""

import argparse
import sys
import os

# Add project root for shared imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pathlib import Path

# Disable wandb to avoid project name validation issues
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

# Fix PyTorch 2.6+ weights_only=True default breaking ultralytics 8.1.0
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8m.pt",
                        help="Base model: yolov8n/s/m/l/x.pt or path to resume")
    parser.add_argument("--data", default=None,
                        help="Path to data.yaml (auto-detected if not set)")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Training image size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size (-1 for auto)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--device", default="0",
                        help="CUDA device (0, 0,1, cpu)")
    parser.add_argument("--project", default="runs",
                        help="Save results to project/name")
    parser.add_argument("--name", default="train",
                        help="Experiment name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="Final LR factor (lr0 * lrf)")
    parser.add_argument("--optimizer", default="auto",
                        help="Optimizer: auto, SGD, Adam, AdamW")
    parser.add_argument("--cos-lr", action="store_true",
                        help="Use cosine LR scheduler")
    parser.add_argument("--warmup-epochs", type=float, default=3.0,
                        help="Warmup epochs")
    parser.add_argument("--close-mosaic", type=int, default=10,
                        help="Disable mosaic for last N epochs")
    parser.add_argument("--freeze", type=int, default=None,
                        help="Freeze first N layers")
    parser.add_argument("--cls-weight", type=float, default=0.5,
                        help="Classification loss weight (default: 0.5)")
    args = parser.parse_args()

    # Auto-detect data.yaml
    task_dir = Path(__file__).parent
    if args.data is None:
        data_yaml = task_dir / "data" / "yolo" / "data.yaml"
        if not data_yaml.exists():
            print("ERROR: data.yaml not found. Run convert_coco.py first.")
            sys.exit(1)
        args.data = str(data_yaml.resolve())

    print(f"Training config:")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch}")
    print(f"  Patience: {args.patience}")
    print(f"  Seed: {args.seed}")
    print(f"  LR: {args.lr0} → {args.lrf}, cos={args.cos_lr}, warmup={args.warmup_epochs}")
    print(f"  Optimizer: {args.optimizer}, freeze={args.freeze}")
    print(f"  Loss weights: box=7.5, cls={args.cls_weight}, dfl=1.5")

    model = YOLO(args.model)

    train_kwargs = dict(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=args.seed,
        # LR and optimizer (configurable via CLI)
        lr0=args.lr0,
        lrf=args.lrf,
        optimizer=args.optimizer,
        cos_lr=args.cos_lr,
        warmup_epochs=args.warmup_epochs,
        close_mosaic=args.close_mosaic,
        # Augmentation defaults are good for small datasets
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=10.0,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,       # shelves are not upside down
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        # Detection-specific
        box=7.5,
        cls=args.cls_weight,
        dfl=1.5,
        # Save
        save=True,
        save_period=-1,    # only save best + last
        plots=True,
        verbose=True,
    )
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze

    results = model.train(**train_kwargs)

    # Extract mAP50 from results for autoresearch
    # results.results_dict has metrics from the last validation
    metrics = results.results_dict
    map50 = metrics.get("metrics/mAP50(B)", 0.0)

    # Copy best.pt to models/
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    models_dir = task_dir / "models"
    models_dir.mkdir(exist_ok=True)
    if best_pt.exists():
        import shutil
        shutil.copy2(best_pt, models_dir / "best.pt")
        print(f"Saved best.pt to {models_dir / 'best.pt'}")
        # Also report file size
        size_mb = best_pt.stat().st_size / (1024 * 1024)
        print(f"Model size: {size_mb:.1f} MB")

    # Autoresearch-compatible metric output
    print(f"val_metric: {map50}")

    # Also report VRAM usage if available
    try:
        import torch
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"peak_vram_mb: {peak_mb:.0f}")
    except Exception:
        pass


if __name__ == "__main__":
    train()
