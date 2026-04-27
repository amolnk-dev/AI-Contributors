"""NorgesGruppen — Two-stage inference: YOLO detection + EfficientNet classification.

Stage 1: YOLO detects bounding boxes (high detection mAP)
Stage 2: EfficientNet classifies each crop (improves classification mAP)

Executed as: python run.py --input /data/images --output /output/predictions.json
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

_torch_load = torch.load
torch.load = lambda *args, **kwargs: _torch_load(*args, **{**kwargs, "weights_only": False})

from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion


def load_classifier(model_path, device):
    """Load the EfficientNet classifier."""
    import timm
    ckpt = torch.load(str(model_path), map_location=device)
    num_classes = ckpt["num_classes"]
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: int(k) for k, v in class_to_idx.items()}

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return model, transform, idx_to_class


def run_at_scale(model, img_path, device, imgsz, augment=False):
    """Run YOLO inference at a specific scale."""
    results = model(
        str(img_path), device=device, verbose=False,
        imgsz=imgsz, conf=0.01, iou=0.7, max_det=500, augment=augment,
    )
    boxes_norm, scores, labels = [], [], []
    boxes_pixel = []
    img_h, img_w = 0, 0
    for r in results:
        img_h, img_w = r.orig_shape
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            boxes_norm.append([max(0, x1 / img_w), max(0, y1 / img_h),
                              min(1, x2 / img_w), min(1, y2 / img_h)])
            boxes_pixel.append([x1, y1, x2, y2])
            scores.append(float(r.boxes.conf[i].item()))
            labels.append(int(r.boxes.cls[i].item()))
    return boxes_norm, scores, labels, boxes_pixel, img_w, img_h


def classify_crops(image, boxes_pixel, classifier, transform, idx_to_class, device,
                   yolo_labels, yolo_scores, blend_weight=0.7):
    """Classify cropped detections and optionally override YOLO labels.

    Args:
        blend_weight: how much to trust classifier vs YOLO (0=all YOLO, 1=all classifier)
    """
    from PIL import Image as PILImage

    if not boxes_pixel:
        return yolo_labels

    new_labels = list(yolo_labels)
    img = PILImage.open(image).convert("RGB")
    img_w, img_h = img.size

    for i, (x1, y1, x2, y2) in enumerate(boxes_pixel):
        # Add padding
        w, h = x2 - x1, y2 - y1
        pad_x = int(w * 0.1)
        pad_y = int(h * 0.1)
        crop = img.crop((
            max(0, int(x1) - pad_x),
            max(0, int(y1) - pad_y),
            min(img_w, int(x2) + pad_x),
            min(img_h, int(y2) + pad_y),
        ))

        # Classify
        inp = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = classifier(inp)
            probs = F.softmax(logits, dim=1)
            cls_conf, cls_idx = probs.max(1)
            cls_conf = cls_conf.item()
            cls_idx = cls_idx.item()

        # Map classifier index to category ID
        cat_id = idx_to_class.get(cls_idx, yolo_labels[i])

        # Blend: if classifier is confident AND disagrees with YOLO, override
        if cls_conf > blend_weight:
            new_labels[i] = cat_id

    return new_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    script_dir = Path(__file__).parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load YOLO detector
    yolo = YOLO(str(script_dir / "best.pt"))

    # Load classifier (if available)
    classifier_path = script_dir / "classifier.pt"
    classifier, cls_transform, idx_to_class = None, None, None
    if classifier_path.exists():
        classifier, cls_transform, idx_to_class = load_classifier(classifier_path, device)
        print("Two-stage mode: YOLO + EfficientNet classifier")
    else:
        print("Single-stage mode: YOLO only (no classifier.pt found)")

    predictions = []
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])

        all_boxes, all_scores, all_labels = [], [], []
        all_boxes_pixel = []
        img_w, img_h = 0, 0

        # Multi-scale detection with WBF
        for imgsz, augment, weight in [(960, False, 1), (1280, False, 2), (1280, True, 3)]:
            boxes, scores, labels, boxes_px, img_w, img_h = run_at_scale(
                yolo, img_path, device, imgsz, augment
            )
            if boxes:
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

        if not all_boxes:
            continue

        # WBF fusion
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            iou_thr=0.55, skip_box_thr=0.001, weights=[1, 2, 3],
        )

        # Convert fused boxes back to pixel coords for classifier
        fused_pixel = []
        for box in fused_boxes:
            fused_pixel.append([
                box[0] * img_w, box[1] * img_h,
                box[2] * img_w, box[3] * img_h
            ])

        # Stage 2: Reclassify with EfficientNet (if available)
        final_labels = list(fused_labels)
        if classifier is not None:
            final_labels = classify_crops(
                img_path, fused_pixel, classifier, cls_transform, idx_to_class,
                device, list(fused_labels), list(fused_scores), blend_weight=0.7
            )

        for box, score, label in zip(fused_boxes, fused_scores, final_labels):
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
