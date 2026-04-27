"""Local evaluation matching the competition scoring formula.

Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5

Usage:
    python eval_local.py --predictions predictions.json --gt data/train/annotations.json
    python eval_local.py --model models/best.pt --images data/yolo/images/val --gt data/train/annotations.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def compute_iou(box1, box2):
    """Compute IoU between two COCO [x, y, w, h] boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter

    return inter / union if union > 0 else 0.0


def compute_ap(recalls, precisions):
    """Compute AP using 101-point interpolation (COCO style)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # Monotonically decreasing precision
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point interpolation
    recall_points = np.linspace(0, 1, 101)
    ap = 0.0
    for t in recall_points:
        prec_at_t = mpre[mrec >= t]
        ap += (prec_at_t.max() if len(prec_at_t) > 0 else 0.0)
    ap /= 101.0
    return ap


def evaluate_map(predictions, ground_truth, iou_threshold=0.5, use_category=False):
    """Compute mAP@0.5 (detection or classification).

    Args:
        predictions: list of {image_id, category_id, bbox, score}
        ground_truth: list of {image_id, category_id, bbox}
        iou_threshold: IoU threshold for matching
        use_category: if True, require category match (classification mAP)
    """
    # Group GT by image
    gt_by_image = {}
    for gt in ground_truth:
        img_id = gt["image_id"]
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        gt_by_image[img_id].append(gt)

    # Sort predictions by confidence (descending)
    preds_sorted = sorted(predictions, key=lambda x: x["score"], reverse=True)

    # Track which GT boxes have been matched
    matched = set()
    tp_list = []
    fp_list = []
    total_gt = len(ground_truth)

    for pred in preds_sorted:
        img_id = pred["image_id"]
        gts = gt_by_image.get(img_id, [])

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gts):
            key = (img_id, gt_idx)
            if key in matched:
                continue

            # Category check for classification mAP
            if use_category and pred["category_id"] != gt["category_id"]:
                continue

            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched.add((img_id, best_gt_idx))
            tp_list.append(1)
            fp_list.append(0)
        else:
            tp_list.append(0)
            fp_list.append(1)

    if total_gt == 0:
        return 0.0

    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    return compute_ap(recalls, precisions)


def run_model_inference(model_path, images_dir):
    """Run YOLOv8 inference and return predictions list."""
    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    predictions = []

    image_files = sorted(
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])
        results = model(str(img_path), device=device, verbose=False, imgsz=1280, conf=0.01, max_det=300)

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(r.boxes.cls[i].item()),
                    "bbox": [round(x1, 1), round(y1, 1),
                             round(x2 - x1, 1), round(y2 - y1, 1)],
                    "score": round(float(r.boxes.conf[i].item()), 4),
                })

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", help="Path to predictions.json")
    parser.add_argument("--model", help="Path to .pt model (runs inference)")
    parser.add_argument("--images", help="Path to image directory (with --model)")
    parser.add_argument("--gt", required=True, help="Path to COCO annotations.json")
    parser.add_argument("--val-only", action="store_true",
                        help="Only evaluate on val split images")
    args = parser.parse_args()

    # Load ground truth
    with open(args.gt) as f:
        coco = json.load(f)

    gt_list = []
    image_id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    for ann in coco["annotations"]:
        gt_list.append({
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
        })

    # Get predictions
    if args.predictions:
        with open(args.predictions) as f:
            predictions = json.load(f)
    elif args.model and args.images:
        print(f"Running inference with {args.model} on {args.images}...")
        predictions = run_model_inference(args.model, args.images)

        # Filter GT to only include images in the val set
        val_image_ids = set()
        for p in Path(args.images).iterdir():
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                val_image_ids.add(int(p.stem.split("_")[-1]))
        gt_list = [g for g in gt_list if g["image_id"] in val_image_ids]
        print(f"Evaluating on {len(val_image_ids)} images, {len(gt_list)} GT boxes")
    else:
        print("Provide --predictions or --model + --images")
        return

    # Compute scores
    det_map = evaluate_map(predictions, gt_list, iou_threshold=0.5, use_category=False)
    cls_map = evaluate_map(predictions, gt_list, iou_threshold=0.5, use_category=True)
    combined = 0.7 * det_map + 0.3 * cls_map

    print(f"\n{'='*50}")
    print(f"Detection mAP@0.5:       {det_map:.4f} (weight: 70%)")
    print(f"Classification mAP@0.5:  {cls_map:.4f} (weight: 30%)")
    print(f"{'='*50}")
    print(f"Combined score:          {combined:.4f}")
    print(f"{'='*50}")
    print(f"\nPredictions: {len(predictions)}, GT: {len(gt_list)}")


if __name__ == "__main__":
    main()
