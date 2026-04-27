"""Final sweep: WBF conf_type, fusion methods, and edge optimizations.

Tests ideas that the param sweep didn't cover.
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
TEST_DIR = TASK_DIR / "data" / "yolo-3way" / "images" / "test"
GT_FILE = TASK_DIR / "data" / "train" / "annotations.json"
MODEL_PT = TASK_DIR / "models" / "best.pt"

TEMPLATE = '''"""Auto-generated run.py for sweep."""
import argparse
import json
from pathlib import Path
import torch

_torch_load = torch.load
torch.load = lambda *args, **kwargs: _torch_load(*args, **{{**kwargs, "weights_only": False}})

from ultralytics import YOLO
{fusion_import}

def run_at_scale(model, img_path, device, imgsz, augment=False):
    results = model(
        str(img_path), device=device, verbose=False,
        imgsz=imgsz, conf={conf}, iou={nms_iou}, max_det={max_det}, augment=augment,
    )
    boxes_norm, scores, labels = [], [], []
    img_h, img_w = 0, 0
    for r in results:
        img_h, img_w = r.orig_shape
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            boxes_norm.append([max(0,x1/img_w), max(0,y1/img_h), min(1,x2/img_w), min(1,y2/img_h)])
            scores.append(float(r.boxes.conf[i].item()))
            labels.append(int(r.boxes.cls[i].item()))
    return boxes_norm, scores, labels, img_w, img_h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    model = YOLO(str(Path(__file__).parent / "best.pt"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictions = []
    image_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png"))

    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])
        all_boxes, all_scores, all_labels = [], [], []
        img_w, img_h = 0, 0
{passes}
        if not all_boxes:
            continue
{fusion_call}
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1, y1 = box[0]*img_w, box[1]*img_h
            x2, y2 = box[2]*img_w, box[3]*img_h
            predictions.append({{
                "image_id": image_id, "category_id": int(label),
                "bbox": [round(x1,{round_precision}), round(y1,{round_precision}),
                         round(x2-x1,{round_precision}), round(y2-y1,{round_precision})],
                "score": round(float(score), {score_precision}),
            }})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)
    print(f"Wrote {{len(predictions)}} predictions for {{len(image_files)}} images")

if __name__ == "__main__":
    main()
'''

def make_pass(scale, augment):
    aug = "True" if augment else "False"
    return f"""
        boxes, scores, labels, img_w, img_h = run_at_scale(model, img_path, device, {scale}, augment={aug})
        if boxes:
            all_boxes.append(boxes); all_scores.append(scores); all_labels.append(labels)"""


PASSES_3 = make_pass(960, False) + make_pass(1280, False) + make_pass(1280, True)
PASSES_2 = make_pass(1280, False) + make_pass(1280, True)

WBF_CALL = """
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            iou_thr={iou_thr}, skip_box_thr={skip_thr}, weights={weights}, conf_type='{conf_type}',
        )"""

NMW_CALL = """
        fused_boxes, fused_scores, fused_labels = non_maximum_weighted(
            all_boxes, all_scores, all_labels,
            iou_thr={iou_thr}, skip_box_thr={skip_thr}, weights={weights},
        )"""

EXPERIMENTS = [
    # Baseline (current best)
    ("baseline", {
        "fusion_import": "from ensemble_boxes import weighted_boxes_fusion",
        "passes": PASSES_3, "conf": 0.01, "nms_iou": 0.7, "max_det": 500,
        "fusion_call": WBF_CALL.format(iou_thr=0.55, skip_thr=0.001, weights=[1,2,3], conf_type="avg"),
        "round_precision": 1, "score_precision": 4,
    }),

    # --- conf_type variants ---
    ("conf-max", {
        "fusion_call": WBF_CALL.format(iou_thr=0.55, skip_thr=0.001, weights=[1,2,3], conf_type="max"),
    }),
    ("conf-box_and_model_avg", {
        "fusion_call": WBF_CALL.format(iou_thr=0.55, skip_thr=0.001, weights=[1,2,3], conf_type="box_and_model_avg"),
    }),
    ("conf-absent_aware", {
        "fusion_call": WBF_CALL.format(iou_thr=0.55, skip_thr=0.001, weights=[1,2,3], conf_type="absent_model_aware_avg"),
    }),

    # --- Alternative fusion: Non-Maximum Weighted ---
    ("nmw", {
        "fusion_import": "from ensemble_boxes import non_maximum_weighted",
        "fusion_call": NMW_CALL.format(iou_thr=0.55, skip_thr=0.001, weights=[1,2,3]),
    }),

    # --- 2-pass (drop 960, save time) ---
    ("2pass-wbf", {
        "passes": PASSES_2,
        "fusion_call": WBF_CALL.format(iou_thr=0.55, skip_thr=0.001, weights=[1,2], conf_type="avg"),
    }),
    ("2pass-absent", {
        "passes": PASSES_2,
        "fusion_call": WBF_CALL.format(iou_thr=0.55, skip_thr=0.001, weights=[1,2], conf_type="absent_model_aware_avg"),
    }),

    # --- Precision experiments ---
    ("precision-high", {
        "round_precision": 2, "score_precision": 6,
    }),

    # --- Best combo from earlier + conf_type ---
    ("iou065-absent", {
        "fusion_call": WBF_CALL.format(iou_thr=0.65, skip_thr=0.001, weights=[1,2,3], conf_type="absent_model_aware_avg"),
    }),
    ("iou065-max", {
        "fusion_call": WBF_CALL.format(iou_thr=0.65, skip_thr=0.001, weights=[1,2,3], conf_type="max"),
    }),
]

DEFAULT = {
    "fusion_import": "from ensemble_boxes import weighted_boxes_fusion",
    "passes": PASSES_3, "conf": 0.01, "nms_iou": 0.7, "max_det": 500,
    "fusion_call": WBF_CALL.format(iou_thr=0.55, skip_thr=0.001, weights=[1,2,3], conf_type="avg"),
    "round_precision": 1, "score_precision": 4,
}


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_experiment(name, config):
    log(f"--- {name} ---")

    tmp_dir = Path("/tmp/_sweep_final")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    code = TEMPLATE.format(**config)
    (tmp_dir / "run.py").write_text(code)
    shutil.copy2(MODEL_PT, tmp_dir / "best.pt")

    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"

    start = time.time()
    try:
        result = subprocess.run(
            [str(VENV_PYTHON), str(tmp_dir / "run.py"),
             "--input", str(TEST_DIR), "--output", str(tmp_dir / "pred.json")],
            capture_output=True, text=True, timeout=600, env=env,
        )
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT"); return None

    elapsed = time.time() - start
    if result.returncode != 0:
        log(f"  FAILED: {result.stderr[-200:]}"); return None

    import sys
    sys.path.insert(0, str(TASK_DIR))
    from eval_local import evaluate_map

    with open(tmp_dir / "pred.json") as f:
        preds = json.load(f)
    with open(GT_FILE) as f:
        coco = json.load(f)

    test_ids = set()
    for p in TEST_DIR.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            test_ids.add(int(p.stem.split("_")[-1]))
    gt_list = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"]}
               for a in coco["annotations"] if a["image_id"] in test_ids]

    det = evaluate_map(preds, gt_list, 0.5, False)
    cls = evaluate_map(preds, gt_list, 0.5, True)
    combined = 0.7 * det + 0.3 * cls

    log(f"  {combined:.4f} (det={det:.4f} cls={cls:.4f}) [{elapsed:.0f}s] [{len(preds)} preds]")
    return combined


def main():
    log(f"Final sweep: {len(EXPERIMENTS)} experiments")
    results = []

    for i, (name, overrides) in enumerate(EXPERIMENTS):
        config = {**DEFAULT, **overrides}
        log(f"\n[{i+1}/{len(EXPERIMENTS)}]")
        score = run_experiment(name, config)
        results.append((name, score))
        time.sleep(2)

    log("\n=== RESULTS ===")
    for name, score in sorted(results, key=lambda x: x[1] or 0, reverse=True):
        log(f"  {name:30s} {score:.4f}" if score else f"  {name:30s} FAILED")


if __name__ == "__main__":
    main()
