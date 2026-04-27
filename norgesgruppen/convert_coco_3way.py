"""Convert COCO annotations to YOLO format with 3-way split (train/val/test).

The test split is NEVER used during training — it gives an honest estimate
of competition score before submitting.

Usage:
    python convert_coco_3way.py
    python convert_coco_3way.py --val-ratio 0.15 --test-ratio 0.15
    python convert_coco_3way.py --output-dir data/yolo-3way

Creates:
    data/yolo-3way/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml       (train + val only — test is invisible to ultralytics)
"""

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def validate_annotation(ann, img_w, img_h):
    x, y, w, h = ann["bbox"]
    if w <= 0 or h <= 0:
        return False
    if x < 0 or y < 0:
        return False
    if x + w > img_w + 1 or y + h > img_h + 1:
        return False
    return True


def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = max(0, min(1, (x + w / 2) / img_w))
    cy = max(0, min(1, (y + h / 2) / img_h))
    nw = max(0, min(1, w / img_w))
    nh = max(0, min(1, h / img_h))
    return [cx, cy, nw, nh]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-dir", default="data/train")
    parser.add_argument("--output-dir", default="data/yolo-3way")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
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

    # 3-way split with fixed seed
    random.seed(args.seed)
    image_ids = sorted(anns_by_image.keys())
    random.shuffle(image_ids)

    n = len(image_ids)
    test_count = max(1, int(n * args.test_ratio))
    val_count = max(1, int(n * args.val_ratio))

    test_ids = set(image_ids[:test_count])
    val_ids = set(image_ids[test_count:test_count + val_count])
    train_ids = set(image_ids[test_count + val_count:])

    print(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    # Count annotations per split for sanity check
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        ann_count = sum(len(anns_by_image[i]) for i in ids)
        print(f"  {name}: {len(ids)} images, {ann_count} annotations")

    # Create output dirs
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Convert and copy
    stats = Counter()
    for img_id in image_ids:
        img = images[img_id]
        if img_id in test_ids:
            split = "test"
        elif img_id in val_ids:
            split = "val"
        else:
            split = "train"

        img_w, img_h = img["width"], img["height"]

        src_img = coco_dir / "images" / img["file_name"]
        dst_img = output_dir / "images" / split / img["file_name"]
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
        else:
            print(f"WARNING: Missing image {src_img}")
            stats["missing_images"] += 1
            continue

        label_name = Path(img["file_name"]).stem + ".txt"
        label_path = output_dir / "labels" / split / label_name

        lines = []
        for ann in anns_by_image[img_id]:
            cx, cy, nw, nh = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
            lines.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats[f"{split}_annotations"] += 1

        label_path.write_text("\n".join(lines) + "\n" if lines else "")
        stats[f"{split}_images"] += 1

    # data.yaml — ONLY train + val (test is invisible to ultralytics)
    category_names = [categories[i]["name"] for i in range(nc)]
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {nc}\n")
        f.write("names:\n")
        for i, name in enumerate(category_names):
            safe_name = name.replace("'", "''")
            f.write(f"  {i}: '{safe_name}'\n")

    # Also save test image IDs for eval_local
    test_meta = {
        "test_image_ids": sorted(test_ids),
        "test_dir": "images/test",
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
    }
    with open(output_dir / "test_meta.json", "w") as f:
        json.dump(test_meta, f, indent=2)

    print(f"\nConversion complete:")
    print(f"  Train: {stats['train_images']} images, {stats['train_annotations']} annotations")
    print(f"  Val: {stats['val_images']} images, {stats['val_annotations']} annotations")
    print(f"  Test: {stats['test_images']} images, {stats['test_annotations']} annotations")
    print(f"  data.yaml: {yaml_path} (train+val only)")
    print(f"  test_meta.json: image IDs for held-out eval")


if __name__ == "__main__":
    main()
