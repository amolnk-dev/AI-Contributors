"""NorgesGruppen Object Detection — Multi-scale WBF + TTA Inference.

3-pass inference (960, 1280, 1280+TTA) with Weighted Boxes Fusion.
Optimized params from inference autoresearch sweep (0.8823 held-out test).

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


def run_at_scale(model, img_path, device, imgsz, augment=False):
    """Run inference at a specific scale, return normalized boxes, scores, labels, and original image dims."""
    results = model(
        str(img_path), device=device, verbose=False,
        imgsz=imgsz, conf=0.01, iou=0.7, max_det=500, augment=augment,
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

        # Pass 1: 960 no TTA (fast, catches large products)
        boxes, scores, labels, img_w, img_h = run_at_scale(model, img_path, device, 960)
        if boxes:
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        # Pass 2: 1280 no TTA (training scale, clean signal)
        boxes, scores, labels, img_w, img_h = run_at_scale(model, img_path, device, 1280)
        if boxes:
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        # Pass 3: 1280 + TTA (training scale, augmented)
        boxes, scores, labels, _, _ = run_at_scale(model, img_path, device, 1280, augment=True)
        if boxes:
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        if not all_boxes:
            continue

        # Weighted Boxes Fusion — variant: absent_model_aware_avg
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            iou_thr=0.65,
            skip_box_thr=0.01,
            weights=[1, 2, 3],
            conf_type='absent_model_aware_avg',
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
