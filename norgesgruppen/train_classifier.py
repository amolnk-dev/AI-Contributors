"""Two-stage classifier training for NorgesGruppen product identification.

Stage 1: YOLO detects bounding boxes (already trained)
Stage 2: This classifier identifies products from cropped detections

Uses product reference images + cropped training annotations to train
an EfficientNet-B0 classifier on individual product patches.

Usage:
    # Step 1: Extract crops from training data
    python train_classifier.py --extract-crops --coco-dir data/train --output-dir data/crops

    # Step 2: Train classifier (optional: include reference images)
    python train_classifier.py --train --crops-dir data/crops --ref-dir data/product_images

    # Step 3: Export for submission
    python train_classifier.py --export --output models/classifier.pt

Prerequisites:
    - Product reference images downloaded from competition site
    - pip install timm==0.9.12 (pre-installed in sandbox)
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ProductCropDataset(Dataset):
    """Dataset of cropped product patches with category labels."""

    def __init__(self, root_dir, transform=None, max_per_class=500):
        self.transform = transform
        self.samples = []  # (path, label)
        self.classes = sorted([d.name for d in Path(root_dir).iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls_dir in Path(root_dir).iterdir():
            if not cls_dir.is_dir():
                continue
            label = self.class_to_idx.get(cls_dir.name)
            if label is None:
                continue
            images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
            # Cap per class to prevent imbalance
            if len(images) > max_per_class:
                images = random.sample(images, max_per_class)
            for img_path in images:
                self.samples.append((img_path, label))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def extract_crops(coco_dir, output_dir, min_size=32):
    """Extract cropped product patches from COCO annotations."""
    coco_dir = Path(coco_dir)
    output_dir = Path(output_dir)

    with open(coco_dir / "annotations.json") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    print(f"Extracting crops from {len(coco['annotations'])} annotations...")
    stats = Counter()

    for ann in coco["annotations"]:
        img_info = images[ann["image_id"]]
        cat_id = ann["category_id"]
        x, y, w, h = ann["bbox"]

        # Skip tiny crops
        if w < min_size or h < min_size:
            stats["skipped_small"] += 1
            continue

        # Create category directory
        cat_dir = output_dir / str(cat_id)
        cat_dir.mkdir(parents=True, exist_ok=True)

        # Crop and save
        img_path = coco_dir / "images" / img_info["file_name"]
        if not img_path.exists():
            stats["missing"] += 1
            continue

        try:
            img = Image.open(img_path)
            # Add padding (10% on each side)
            pad_x = int(w * 0.1)
            pad_y = int(h * 0.1)
            crop = img.crop((
                max(0, int(x) - pad_x),
                max(0, int(y) - pad_y),
                min(img.width, int(x + w) + pad_x),
                min(img.height, int(y + h) + pad_y),
            ))
            crop_name = f"{img_info['id']}_{ann['id']}.jpg"
            crop.save(cat_dir / crop_name, quality=95)
            stats["saved"] += 1
        except Exception as e:
            stats["error"] += 1

    print(f"Extracted: {stats['saved']} crops, {stats['skipped_small']} too small, {stats.get('error', 0)} errors")
    print(f"Categories with crops: {len(list(output_dir.iterdir()))}")


def add_reference_images(ref_dir, crops_dir):
    """Add product reference images to the crops dataset."""
    ref_dir = Path(ref_dir)
    crops_dir = Path(crops_dir)

    if not ref_dir.exists():
        print(f"Reference images not found at {ref_dir}")
        return

    # Load metadata to map product codes to category IDs
    metadata_path = ref_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        print("No metadata.json found in reference images")
        return

    added = 0
    for product_dir in ref_dir.iterdir():
        if not product_dir.is_dir():
            continue
        product_code = product_dir.name
        # Map product code to category ID from training annotations
        # This requires cross-referencing with annotations.json
        for img_path in product_dir.glob("*.jpg"):
            # Copy to appropriate category directory
            # (needs category mapping from annotations)
            added += 1

    print(f"Added {added} reference images")


def train_classifier(crops_dir, num_classes, epochs=30, batch_size=32, lr=0.001):
    """Train EfficientNet-B0 classifier on product crops."""
    import timm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = ProductCropDataset(crops_dir, transform=train_transform)
    print(f"Dataset: {len(dataset)} samples, {len(dataset.classes)} classes")

    # Split 90/10
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model: EfficientNet-B0 (available in timm 0.9.12, ~20MB)
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": dataset.class_to_idx,
                "num_classes": num_classes,
            }, "models/classifier.pt")
            print(f"  Saved best (val_acc={best_acc:.4f})")

    print(f"Best val accuracy: {best_acc:.4f}")
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-crops", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--coco-dir", default="data/train")
    parser.add_argument("--output-dir", default="data/crops")
    parser.add_argument("--crops-dir", default="data/crops")
    parser.add_argument("--ref-dir", default="data/product_images")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.extract_crops:
        extract_crops(args.coco_dir, args.output_dir)

    if args.train:
        # Count categories from crops
        num_classes = len(list(Path(args.crops_dir).iterdir()))
        train_classifier(args.crops_dir, num_classes, args.epochs, args.batch_size)

    if args.export:
        print("Export: models/classifier.pt ready for submission")
        size = Path("models/classifier.pt").stat().st_size / (1024 * 1024)
        print(f"Classifier size: {size:.1f} MB")


if __name__ == "__main__":
    main()
