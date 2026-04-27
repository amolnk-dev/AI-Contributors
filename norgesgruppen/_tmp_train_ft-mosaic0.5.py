
import torch
_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

from ultralytics import YOLO
from pathlib import Path

model = YOLO("/Users/walgermo/Utvikling/ai-nm-2026/tasks/norgesgruppen/models/best.pt")
results = model.train(
    data="/Users/walgermo/Utvikling/ai-nm-2026/tasks/norgesgruppen/data/yolo/data.yaml",
    imgsz=1280,
    epochs=8,
    batch=2,
    patience=8,
    device="mps",
    project="runs",
    name="ft-mosaic0.5",
    exist_ok=True,
    seed=42,
    save=True,
    save_period=-1,
    plots=False,
    verbose=False,
    # defaults that can be overridden
    mosaic=0.5,
    mixup=0.1,
    copy_paste=0.1,
    degrees=10.0,
    scale=0.5,
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    close_mosaic=10,
    lr0=0.01,
    warmup_epochs=3.0,
    cos_lr=False,
)

metrics = results.results_dict
map50 = metrics.get("metrics/mAP50(B)", 0.0)
print(f"val_metric: {map50}")

best_pt = Path("runs") / "ft-mosaic0.5" / "weights" / "best.pt"
if best_pt.exists():
    size_mb = best_pt.stat().st_size / (1024 * 1024)
    print(f"model_size_mb: {size_mb:.1f}")
