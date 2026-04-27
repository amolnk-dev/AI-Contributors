"""NorgesGruppen — Fast Two-Stage: YOLO detection + ConvNeXt reclassification.

Single YOLO pass at 1280 + TTA, then ConvNeXt reclassifies ALL crops.
No multi-scale WBF (saves ~60% inference time vs 3-pass WBF).

Target: <300s for 200+ images on L4 GPU.
Inference budget per image: ~1.2s (YOLO 0.5s + batch classify 0.7s)

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


def load_classifier(model_path, device):
    """Load ConvNeXt-Tiny classifier."""
    import timm
    ckpt = torch.load(str(model_path), map_location=device)
    num_classes = ckpt["num_classes"]
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: int(k) for k, v in class_to_idx.items()}

    model = timm.create_model(ckpt.get("model_name", "convnext_tiny"),
                               pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, transform, idx_to_class


def batch_classify(image_pil, boxes_xyxy, classifier, transform, idx_to_class, device,
                   yolo_labels, yolo_scores, conf_threshold=0.5):
    """Batch-classify all crops at once (much faster than one-by-one).

    Only overrides YOLO label when classifier confidence > conf_threshold.
    """
    if not boxes_xyxy:
        return yolo_labels

    img_w, img_h = image_pil.size
    crops = []

    for x1, y1, x2, y2 in boxes_xyxy:
        w, h = x2 - x1, y2 - y1
        px, py = int(w * 0.1), int(h * 0.1)
        crop = image_pil.crop((
            max(0, int(x1) - px), max(0, int(y1) - py),
            min(img_w, int(x2) + px), min(img_h, int(y2) + py),
        ))
        crops.append(transform(crop))

    # Batch inference — all crops at once
    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        logits = classifier(batch)
        probs = F.softmax(logits, dim=1)
        cls_confs, cls_idxs = probs.max(1)

    new_labels = list(yolo_labels)
    for i in range(len(boxes_xyxy)):
        cls_conf = cls_confs[i].item()
        if cls_conf > conf_threshold:
            new_labels[i] = idx_to_class.get(cls_idxs[i].item(), yolo_labels[i])

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

    # Load YOLO
    yolo = YOLO(str(script_dir / "best.pt"))

    # Load classifier
    classifier_path = script_dir / "classifier.pt"
    classifier, cls_tf, idx_to_class = None, None, None
    if classifier_path.exists():
        classifier, cls_tf, idx_to_class = load_classifier(classifier_path, device)

    from PIL import Image as PILImage

    predictions = []
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])

        # Single YOLO pass at 1280 + TTA (fast, ~0.5s on L4)
        results = yolo(
            str(img_path), device=device, verbose=False,
            imgsz=1280, conf=0.01, iou=0.7, max_det=300, augment=True,
        )

        boxes_xyxy = []
        scores = []
        labels = []
        img_h, img_w = 0, 0

        for r in results:
            img_h, img_w = r.orig_shape
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                boxes_xyxy.append([x1, y1, x2, y2])
                scores.append(float(r.boxes.conf[i].item()))
                labels.append(int(r.boxes.cls[i].item()))

        # Reclassify with ConvNeXt
        if classifier is not None and boxes_xyxy:
            img_pil = PILImage.open(img_path).convert("RGB")
            labels = batch_classify(
                img_pil, boxes_xyxy, classifier, cls_tf, idx_to_class,
                device, labels, scores, conf_threshold=0.5
            )

        for (x1, y1, x2, y2), score, label in zip(boxes_xyxy, scores, labels):
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
