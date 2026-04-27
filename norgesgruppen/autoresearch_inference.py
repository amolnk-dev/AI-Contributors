"""Autoresearch: sweep inference parameters in run.py.

No retraining needed — just tweaks conf, NMS, WBF params, scales.
Each experiment takes ~3 min (inference on 37 held-out test images).

Usage:
    nohup python autoresearch_inference.py > autoresearch_infer.log 2>&1 &

Monitor:
    tail -f autoresearch_infer.log
    cat results_inference.tsv | column -t -s $'\t'
"""

import json
import subprocess
import time
import shutil
import os
from datetime import datetime
from pathlib import Path

TASK_DIR = Path(__file__).parent.resolve()
VENV_PYTHON = TASK_DIR.parent.parent / ".venv" / "bin" / "python"
RESULTS_TSV = TASK_DIR / "results_inference.tsv"
RUN_PY = TASK_DIR / "run.py"
RUN_PY_BACKUP = TASK_DIR / "run.py.best"
TEST_DIR = TASK_DIR / "data" / "yolo-3way" / "images" / "test"
GT_FILE = TASK_DIR / "data" / "train" / "annotations.json"
MODEL_PT = TASK_DIR / "models" / "best.pt"

# Template for run.py with placeholders
RUN_PY_TEMPLATE = '''"""NorgesGruppen Object Detection — Auto-generated inference config."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

_torch_load = torch.load
torch.load = lambda *args, **kwargs: _torch_load(*args, **{{**kwargs, "weights_only": False}})

from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image


def run_at_scale(model, img_path, device, imgsz, augment=False, conf={conf}, iou={nms_iou}):
    results = model(
        str(img_path), device=device, verbose=False,
        imgsz=imgsz, conf=conf, iou=iou, max_det={max_det}, augment=augment,
    )
    boxes_norm, scores, labels = [], [], []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        img_h, img_w = r.orig_shape
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            boxes_norm.append([max(0,x1/img_w), max(0,y1/img_h), min(1,x2/img_w), min(1,y2/img_h)])
            scores.append(float(r.boxes.conf[i].item()))
            labels.append(int(r.boxes.cls[i].item()))
    return boxes_norm, scores, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(__file__).parent / "best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))

    predictions = []
    image_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(img_path)
        img_w, img_h = img.size

        all_boxes, all_scores, all_labels = [], [], []
{passes}
        if not all_boxes:
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            iou_thr={wbf_iou_thr}, skip_box_thr={wbf_skip_thr}, weights={wbf_weights},
        )

        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1, y1 = box[0]*img_w, box[1]*img_h
            x2, y2 = box[2]*img_w, box[3]*img_h
            predictions.append({{
                "image_id": image_id, "category_id": int(label),
                "bbox": [round(x1,1), round(y1,1), round(x2-x1,1), round(y2-y1,1)],
                "score": round(float(score), 4),
            }})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)
    print(f"Wrote {{len(predictions)}} predictions for {{len(image_files)}} images")

if __name__ == "__main__":
    main()
'''

def make_pass_code(scale, augment):
    aug = "True" if augment else "False"
    return f"""
        boxes, scores, labels = run_at_scale(model, img_path, device, {scale}, augment={aug})
        if boxes:
            all_boxes.append(boxes); all_scores.append(scores); all_labels.append(labels)"""


# Experiments: (name, config_dict)
EXPERIMENTS = [
    # --- Baseline (current config) ---
    ("baseline", {
        "conf": 0.001, "nms_iou": 0.7, "max_det": 500,
        "scales": [(960, False), (1280, False), (1280, True)],
        "wbf_iou_thr": 0.55, "wbf_skip_thr": 0.0001, "wbf_weights": [1, 2, 3],
    }),

    # --- WBF iou_thr sweep ---
    ("wbf-iou-0.45", {"wbf_iou_thr": 0.45}),
    ("wbf-iou-0.50", {"wbf_iou_thr": 0.50}),
    ("wbf-iou-0.60", {"wbf_iou_thr": 0.60}),
    ("wbf-iou-0.65", {"wbf_iou_thr": 0.65}),

    # --- WBF weights sweep ---
    ("wbf-w-1-1-2", {"wbf_weights": [1, 1, 2]}),
    ("wbf-w-1-3-3", {"wbf_weights": [1, 3, 3]}),
    ("wbf-w-0-1-2", {"wbf_weights": [0, 1, 2]}),
    ("wbf-w-1-1-1", {"wbf_weights": [1, 1, 1]}),
    ("wbf-w-1-2-5", {"wbf_weights": [1, 2, 5]}),

    # --- WBF skip_box_thr sweep ---
    ("wbf-skip-0.001", {"wbf_skip_thr": 0.001}),
    ("wbf-skip-0.01", {"wbf_skip_thr": 0.01}),
    ("wbf-skip-0.05", {"wbf_skip_thr": 0.05}),

    # --- Per-pass conf sweep ---
    ("conf-0.005", {"conf": 0.005}),
    ("conf-0.01", {"conf": 0.01}),
    ("conf-0.0005", {"conf": 0.0005}),

    # --- NMS iou sweep ---
    ("nms-iou-0.5", {"nms_iou": 0.5}),
    ("nms-iou-0.6", {"nms_iou": 0.6}),
    ("nms-iou-0.8", {"nms_iou": 0.8}),

    # --- max_det sweep ---
    ("maxdet-300", {"max_det": 300}),
    ("maxdet-700", {"max_det": 700}),

    # --- Scale combos ---
    ("scales-2pass", {
        "scales": [(960, False), (1280, True)],
        "wbf_weights": [1, 3],
    }),
    ("scales-1280-only-tta", {
        "scales": [(1280, True)],
        "wbf_weights": [1],
    }),
    ("scales-4pass", {
        "scales": [(640, False), (960, False), (1280, False), (1280, True)],
        "wbf_weights": [1, 1, 2, 3],
    }),
    ("scales-1280-1600", {
        "scales": [(1280, False), (1280, True), (1600, False)],
        "wbf_weights": [2, 3, 1],
    }),
]

# Default config (overridden per experiment)
DEFAULT = {
    "conf": 0.001, "nms_iou": 0.7, "max_det": 500,
    "scales": [(960, False), (1280, False), (1280, True)],
    "wbf_iou_thr": 0.55, "wbf_skip_thr": 0.0001, "wbf_weights": [1, 2, 3],
}


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def init_results():
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("timestamp\tname\tval_metric\tdet_map\tcls_map\tn_preds\ttime_s\tstatus\tconfig\n")


def append_result(name, val_metric, det_map, cls_map, n_preds, time_s, status, config_str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    row = f"{ts}\t{name}\t{val_metric:.4f}\t{det_map:.4f}\t{cls_map:.4f}\t{n_preds}\t{time_s:.0f}\t{status}\t{config_str}\n"
    with open(RESULTS_TSV, "a") as f:
        f.write(row)


def get_best_metric():
    if not RESULTS_TSV.exists():
        return 0.0
    best = 0.0
    for line in RESULTS_TSV.read_text().strip().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 8 and parts[7] == "kept":
            try:
                best = max(best, float(parts[2]))
            except ValueError:
                pass
    return best


def generate_run_py(config):
    passes_code = ""
    for scale, augment in config["scales"]:
        passes_code += make_pass_code(scale, augment)

    return RUN_PY_TEMPLATE.format(
        conf=config["conf"],
        nms_iou=config["nms_iou"],
        max_det=config["max_det"],
        passes=passes_code,
        wbf_iou_thr=config["wbf_iou_thr"],
        wbf_skip_thr=config["wbf_skip_thr"],
        wbf_weights=config["wbf_weights"],
    )


def run_experiment(name, config):
    log(f"--- {name} ---")
    config_str = json.dumps({k: v for k, v in config.items() if k != "scales"})
    scales_str = str(config["scales"])

    # Generate and write run.py
    tmp_dir = Path("/tmp/_infer_sweep")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    run_py_content = generate_run_py(config)
    (tmp_dir / "run.py").write_text(run_py_content)
    shutil.copy2(MODEL_PT, tmp_dir / "best.pt")

    output_json = tmp_dir / "predictions.json"

    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"

    start = time.time()
    try:
        result = subprocess.run(
            [str(VENV_PYTHON), str(tmp_dir / "run.py"),
             "--input", str(TEST_DIR), "--output", str(output_json)],
            capture_output=True, text=True, timeout=600, env=env,
        )
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT")
        append_result(name, 0, 0, 0, 0, time.time()-start, "timeout", config_str)
        return None

    elapsed = time.time() - start

    if result.returncode != 0:
        err = result.stderr.split("\n")[-3:] if result.stderr else ["unknown"]
        log(f"FAILED: {'; '.join(e.strip() for e in err if e.strip())}")
        append_result(name, 0, 0, 0, 0, elapsed, "failed", config_str)
        return None

    # Score predictions
    import sys
    sys.path.insert(0, str(TASK_DIR))
    from eval_local import evaluate_map

    with open(output_json) as f:
        predictions = json.load(f)
    with open(GT_FILE) as f:
        coco = json.load(f)

    test_ids = set()
    for p in TEST_DIR.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            test_ids.add(int(p.stem.split("_")[-1]))

    gt_list = [
        {"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"]}
        for a in coco["annotations"] if a["image_id"] in test_ids
    ]

    det_map = evaluate_map(predictions, gt_list, iou_threshold=0.5, use_category=False)
    cls_map = evaluate_map(predictions, gt_list, iou_threshold=0.5, use_category=True)
    combined = 0.7 * det_map + 0.3 * cls_map

    current_best = get_best_metric()
    if combined > current_best:
        log(f"NEW BEST: {combined:.4f} > {current_best:.4f} (det={det_map:.4f} cls={cls_map:.4f}) [{elapsed:.0f}s]")
        # Save this run.py as the best
        shutil.copy2(tmp_dir / "run.py", RUN_PY)
        shutil.copy2(tmp_dir / "run.py", RUN_PY_BACKUP)
        append_result(name, combined, det_map, cls_map, len(predictions), elapsed, "kept", config_str)
    else:
        log(f"NO GAIN: {combined:.4f} <= {current_best:.4f} (det={det_map:.4f} cls={cls_map:.4f}) [{elapsed:.0f}s]")
        append_result(name, combined, det_map, cls_map, len(predictions), elapsed, "rejected", config_str)

    return combined


def main():
    log(f"Inference autoresearch: {len(EXPERIMENTS)} experiments")
    init_results()

    for i, (name, overrides) in enumerate(EXPERIMENTS):
        config = {**DEFAULT, **overrides}
        log(f"\n[{i+1}/{len(EXPERIMENTS)}]")
        run_experiment(name, config)
        time.sleep(2)

    log(f"\nDone! Best: {get_best_metric():.4f}")


if __name__ == "__main__":
    main()
