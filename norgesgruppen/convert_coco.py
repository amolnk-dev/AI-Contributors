"""Convert COCO annotations to YOLO format for ultralytics training.

Usage:
    python convert_coco.py --coco-dir data/train --output-dir data/yolo --val-ratio 0.2

Creates:
    data/yolo/
    ├── images/
    │   ├── train/  (symlinks or copies)
    │   └── val/
    ├── labels/
    │   ├── train/  (one .txt per image)
    │   └── val/
    └── data.yaml
"""

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def validate_annotation(ann: dict, img_w: int, img_h: int) -> bool:
    """Filter out bad annotations: zero-area, out-of-bounds, negative."""
    x, y, w, h = ann["bbox"]
    if w <= 0 or h <= 0:
        return False
    if x < 0 or y < 0:
        return False
    if x + w > img_w + 1 or y + h > img_h + 1:  # +1 for rounding tolerance
        return False
    return True


def coco_to_yolo_bbox(bbox: list, img_w: int, img_h: int) -> list:
    """Convert COCO [x, y, w, h] pixels → YOLO [cx, cy, w, h] normalized."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return [cx, cy, nw, nh]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-dir", default="data/train",
                        help="Directory with annotations.json and images/")
    parser.add_argument("--output-dir", default="data/yolo",
                        help="Output directory for YOLO format")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output_dir)
    ann_file = coco_dir / "annotations.json"

    print(f"Loading annotations from {ann_file}")
    with open(ann_file) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat for cat in coco["categories"]}
    nc = len(categories)
    print(f"Found {len(images)} images, {len(coco['annotations'])} annotations, {nc} categories")

    # Group annotations by image
    anns_by_image = defaultdict(list)
    skipped = 0
    for ann in coco["annotations"]:
        img = images[ann["image_id"]]
        if validate_annotation(ann, img["width"], img["height"]):
            anns_by_image[ann["image_id"]].append(ann)
        else:
            skipped += 1
    if skipped > 0:
        print(f"Skipped {skipped} invalid annotations")

    # Train/val split — by image, with fixed seed
    random.seed(args.seed)
    image_ids = sorted(anns_by_image.keys())
    random.shuffle(image_ids)
    val_count = int(len(image_ids) * args.val_ratio)
    if args.val_ratio == 0:
        val_ids = set()
        train_ids = set(image_ids)
    else:
        val_count = max(1, val_count)
        val_ids = set(image_ids[:val_count])
        train_ids = set(image_ids[val_count:])
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val")

    # Create output dirs
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Convert and copy
    stats = Counter()
    for img_id in image_ids:
        img = images[img_id]
        split = "val" if img_id in val_ids else "train"
        img_w, img_h = img["width"], img["height"]

        # Copy image
        src_img = coco_dir / "images" / img["file_name"]
        dst_img = output_dir / "images" / split / img["file_name"]
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
        else:
            print(f"WARNING: Missing image {src_img}")
            stats["missing_images"] += 1
            continue

        # Write YOLO label file
        label_name = Path(img["file_name"]).stem + ".txt"
        label_path = output_dir / "labels" / split / label_name

        lines = []
        for ann in anns_by_image[img_id]:
            cx, cy, nw, nh = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
            cat_id = ann["category_id"]
            lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats[f"{split}_annotations"] += 1

        label_path.write_text("\n".join(lines) + "\n" if lines else "")
        stats[f"{split}_images"] += 1

    # Generate data.yaml
    category_names = [categories[i]["name"] for i in range(nc)]
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": nc,
        "names": category_names,
    }

    yaml_path = output_dir / "data.yaml"
    # Write YAML manually to avoid pyyaml dependency
    with open(yaml_path, "w") as f:
        f.write(f"path: {data_yaml['path']}\n")
        f.write(f"train: {data_yaml['train']}\n")
        f.write(f"val: {data_yaml['val']}\n")
        f.write(f"nc: {data_yaml['nc']}\n")
        f.write("names:\n")
        for i, name in enumerate(category_names):
            # Escape quotes in names
            safe_name = name.replace("'", "''")
            f.write(f"  {i}: '{safe_name}'\n")

    print(f"\nConversion complete:")
    print(f"  Train: {stats['train_images']} images, {stats['train_annotations']} annotations")
    print(f"  Val: {stats['val_images']} images, {stats['val_annotations']} annotations")
    if stats["missing_images"]:
        print(f"  WARNING: {stats['missing_images']} missing images")
    print(f"  data.yaml: {yaml_path}")
    print(f"  Categories: {nc} (IDs 0-{nc-1})")


if __name__ == "__main__":
    main()
