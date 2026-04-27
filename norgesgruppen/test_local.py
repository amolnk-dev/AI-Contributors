"""Local sandbox simulator — test run.py exactly as the competition does.

Simulates: python run.py --input <images> --output <predictions.json>
Then scores with eval_local.py formula: 0.7 * det_mAP + 0.3 * cls_mAP

Usage:
    # Quick test on sample image (smoke test)
    python test_local.py --quick

    # Full eval on val set (matches competition scoring)
    python test_local.py

    # Test a specific model
    python test_local.py --model models/best.pt

    # Test the submission zip directly
    python test_local.py --zip submission.zip

Prints val_metric for autoresearch compatibility.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

# Patch torch.load for ultralytics 8.1.0 compat
_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

TASK_DIR = Path(__file__).parent.resolve()
VAL_IMAGES = TASK_DIR / "data" / "yolo" / "images" / "val"
SAMPLE_IMAGES = TASK_DIR / "sample-img"
GT_FILE = TASK_DIR / "data" / "train" / "annotations.json"
MODELS_DIR = TASK_DIR / "models"


def run_inference(run_py, model_pt, input_dir, output_json):
    """Execute run.py as subprocess, exactly like the sandbox."""
    # Create a temp dir with run.py + model
    tmp = Path("/tmp/_sandbox_sim")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    shutil.copy2(run_py, tmp / "run.py")
    shutil.copy2(model_pt, tmp / "best.pt")

    output_json.parent.mkdir(parents=True, exist_ok=True)

    import os
    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(tmp / "run.py"),
         "--input", str(input_dir),
         "--output", str(output_json)],
        capture_output=True, text=True, timeout=300, env=env,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"ERROR: run.py exited with code {result.returncode}")
        print(result.stderr[-500:] if result.stderr else result.stdout[-500:])
        return None, elapsed

    print(result.stdout.strip())
    return output_json, elapsed


def evaluate(predictions_path, gt_path, val_image_dir):
    """Score predictions using pycocotools (matches competition scoring)."""
    import numpy as np
    import tempfile
    import os
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with open(predictions_path) as f:
        predictions = json.load(f)
    with open(gt_path) as f:
        gt_data = json.load(f)

    # Filter to eval images only
    eval_ids = set()
    for p in Path(val_image_dir).iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            eval_ids.add(int(p.stem.split("_")[-1]))

    predictions = [p for p in predictions if p["image_id"] in eval_ids]
    eval_images = [img for img in gt_data["images"] if img["id"] in eval_ids]

    # --- Detection mAP (category ignored → all category_id=1) ---
    det_anns = [
        {**ann, "category_id": 1}
        for ann in gt_data["annotations"] if ann["image_id"] in eval_ids
    ]
    det_gt = {
        "images": eval_images,
        "annotations": det_anns,
        "categories": [{"id": 1, "name": "product"}],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(det_gt, f)
        det_gt_f = f.name

    det_preds = [{**p, "category_id": 1} for p in predictions]
    coco_gt = COCO(det_gt_f)
    coco_dt = coco_gt.loadRes(det_preds)
    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.params.iouThrs = np.array([0.5])
    e.params.maxDets = [1, 10, 500]
    e.evaluate()
    e.accumulate()
    det_map = float(np.mean(e.eval["precision"][0, :, :, 0, 2]))
    os.unlink(det_gt_f)

    # --- Classification mAP (per-category, matches competition) ---
    cls_anns = [ann for ann in gt_data["annotations"] if ann["image_id"] in eval_ids]
    cls_gt = {
        "images": eval_images,
        "annotations": cls_anns,
        "categories": gt_data["categories"],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cls_gt, f)
        cls_gt_f = f.name

    coco_gt2 = COCO(cls_gt_f)
    coco_dt2 = coco_gt2.loadRes(predictions)
    e2 = COCOeval(coco_gt2, coco_dt2, "bbox")
    e2.params.iouThrs = np.array([0.5])
    e2.params.maxDets = [1, 10, 500]
    e2.evaluate()
    e2.accumulate()
    prec = e2.eval["precision"][0, :, :, 0, 2]
    cls_map = float(np.mean(prec[prec > -1]))
    os.unlink(cls_gt_f)

    combined = 0.7 * det_map + 0.3 * cls_map
    n_gt = len(cls_anns)

    return det_map, cls_map, combined, len(predictions), n_gt


def test_from_zip(zip_path, input_dir, output_json):
    """Extract zip and run."""
    import zipfile
    tmp = Path("/tmp/_sandbox_sim")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)

    output_json.parent.mkdir(parents=True, exist_ok=True)

    import os
    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(tmp / "run.py"),
         "--input", str(input_dir),
         "--output", str(output_json)],
        capture_output=True, text=True, timeout=300, env=env,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"ERROR: run.py exited with code {result.returncode}")
        print(result.stderr[-500:] if result.stderr else result.stdout[-500:])
        return None, elapsed

    print(result.stdout.strip())
    return output_json, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Smoke test on sample image only")
    parser.add_argument("--model", default=None, help="Path to .pt model (default: models/best.pt)")
    parser.add_argument("--zip", default=None, help="Test a submission zip directly")
    parser.add_argument("--data", default=None,
                        help="YOLO data dir with test/ split (e.g. data/yolo-3way). "
                             "Uses held-out test images for honest scoring.")
    parser.add_argument("--run-py", default=None,
                        help="Path to run.py variant (default: run.py)")
    args = parser.parse_args()

    output_json = Path("/tmp/_sandbox_sim_output/predictions.json")

    if args.quick:
        # Copy sample images to temp dir with proper naming (img_XXXXX.ext)
        tmp_input = Path("/tmp/_sandbox_sim_input")
        if tmp_input.exists():
            shutil.rmtree(tmp_input)
        tmp_input.mkdir()
        idx = 1
        for img in sorted(SAMPLE_IMAGES.iterdir()):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                dst = tmp_input / f"img_{idx:05d}{img.suffix.lower()}"
                shutil.copy2(img, dst)
                idx += 1
        input_dir = tmp_input
        print(f"Quick smoke test on {idx-1} sample image(s)")
    elif args.data:
        # Use held-out test split from 3-way data
        data_dir = TASK_DIR / args.data
        input_dir = data_dir / "images" / "test"
        if not input_dir.exists():
            print(f"ERROR: No test split at {input_dir}")
            print("Run: python convert_coco_3way.py first")
            return
        print(f"Held-out test eval on {input_dir} ({len(list(input_dir.iterdir()))} images)")
    else:
        input_dir = VAL_IMAGES
        print(f"Full eval on {input_dir} ({len(list(input_dir.iterdir()))} images)")

    # Run inference
    if args.zip:
        print(f"Testing zip: {args.zip}")
        pred_path, elapsed = test_from_zip(args.zip, input_dir, output_json)
    else:
        model_pt = Path(args.model) if args.model else MODELS_DIR / "best.pt"
        run_py = Path(args.run_py) if args.run_py else TASK_DIR / "run.py"
        print(f"Model: {model_pt} ({model_pt.stat().st_size/1e6:.1f}MB)")
        pred_path, elapsed = run_inference(run_py, model_pt, input_dir, output_json)

    if pred_path is None:
        print("FAILED — run.py crashed")
        return

    print(f"Inference time: {elapsed:.1f}s")

    # Load and check predictions
    with open(pred_path) as f:
        preds = json.load(f)
    print(f"Predictions: {len(preds)}")

    if args.quick:
        # Just show predictions, no scoring (no GT for sample images)
        print(f"\nSmoke test PASSED — {len(preds)} detections in {elapsed:.1f}s")
        if preds:
            cats = set(p["category_id"] for p in preds)
            scores = [p["score"] for p in preds]
            print(f"Categories: {len(cats)} unique")
            print(f"Scores: min={min(scores):.4f} max={max(scores):.4f} mean={sum(scores)/len(scores):.4f}")
        print(f"\nval_metric: -1")  # no GT available
        return

    # Full eval
    if not GT_FILE.exists():
        print(f"WARNING: No GT file at {GT_FILE} — cannot score")
        return

    det_map, cls_map, combined, n_pred, n_gt = evaluate(pred_path, GT_FILE, input_dir)

    print(f"\n{'='*50}")
    print(f"Detection mAP@0.5:       {det_map:.4f} (weight: 70%)")
    print(f"Classification mAP@0.5:  {cls_map:.4f} (weight: 30%)")
    print(f"{'='*50}")
    print(f"Combined score:          {combined:.4f}")
    print(f"{'='*50}")
    print(f"Predictions: {n_pred}, GT: {n_gt}")
    print(f"Time: {elapsed:.1f}s (limit: 300s)")
    print(f"\nval_metric: {combined}")


if __name__ == "__main__":
    main()
