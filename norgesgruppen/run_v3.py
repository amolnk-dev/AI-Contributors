"""NorgesGruppen — WBF-tuned v3 (same 3 passes, tuned params).

Same passes as v1 (960 + 1280 + 1280TTA) but with optimized WBF params:
- Higher weight on TTA pass (1,2,5 vs 1,2,3)
- Stricter iou_thr=0.5 (vs 0.55)
- Higher skip_box_thr=0.005 (vs 0.001) — less noise
- Higher conf=0.02 on non-TTA passes

Safe on timing (same as proven v1).

Executed as: python run.py --input /data/images --output /output/predictions.json
"""

import argparse
import json
from pathlib import Path

import torch

_torch_load = torch.load
torch.load = lambda *args, **kwargs: _torch_load(*args, **{**kwargs, "weights_only": False})

from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion


def run_at_scale(model, img_path, device, imgsz, augment=False, conf=0.01):
    results = model(
        str(img_path), device=device, verbose=False,
        imgsz=imgsz, conf=conf, iou=0.7, max_det=300, augment=augment,
    )
    boxes_norm, scores, labels = [], [], []
    img_h, img_w = 0, 0
    for r in results:
        img_h, img_w = r.orig_shape
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            boxes_norm.append([max(0, x1 / img_w), max(0, y1 / img_h),
                              min(1, x2 / img_w), min(1, y2 / img_h)])
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
    model_path = Path(__file__).parent / "best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))

    predictions = []
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])

        all_boxes, all_scores, all_labels = [], [], []
        img_w, img_h = 0, 0

        # Pass 1: 960 no TTA (higher conf to reduce noise)
        boxes, scores, labels, img_w, img_h = run_at_scale(model, img_path, device, 960, conf=0.02)
        if boxes:
            all_boxes.append(boxes); all_scores.append(scores); all_labels.append(labels)

        # Pass 2: 1280 no TTA
        boxes, scores, labels, img_w, img_h = run_at_scale(model, img_path, device, 1280, conf=0.01)
        if boxes:
            all_boxes.append(boxes); all_scores.append(scores); all_labels.append(labels)

        # Pass 3: 1280 + TTA (best, heavily weighted)
        boxes, scores, labels, _, _ = run_at_scale(model, img_path, device, 1280, augment=True, conf=0.01)
        if boxes:
            all_boxes.append(boxes); all_scores.append(scores); all_labels.append(labels)

        if not all_boxes:
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            iou_thr=0.5,
            skip_box_thr=0.005,
            weights=[1, 2, 5],
        )

        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1 = box[0] * img_w
            y1 = box[1] * img_h
            x2 = box[2] * img_w
            y2 = box[3] * img_h
            predictions.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [round(x1, 2), round(y1, 2),
                         round(x2 - x1, 2), round(y2 - y1, 2)],
                "score": round(float(score), 6),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)
    print(f"Wrote {len(predictions)} predictions for {len(image_files)} images")


if __name__ == "__main__":
    main()
