"""Train ConvNeXt-Tiny classifier for two-stage product identification.

Extracts crops from training annotations + adds reference images,
then trains ConvNeXt-Tiny (28MB, in timm 0.9.12 / sandbox).

Usage:
    python train_convnext_cls.py

Produces: models/classifier.pt (~28MB)
"""

import json
import random
import shutil
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ProductDataset(Dataset):
    def __init__(self, root, transform, max_per_class=500):
        self.transform = transform
        self.samples = []
        self.classes = sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
        self.c2i = {c: i for i, c in enumerate(self.classes)}
        for d in Path(root).iterdir():
            if not d.is_dir():
                continue
            imgs = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            if len(imgs) > max_per_class:
                imgs = random.sample(imgs, max_per_class)
            for img in imgs:
                self.samples.append((img, self.c2i[d.name]))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.transform(img), label


def extract_crops_and_refs(coco_dir, ref_zip, mapping_file, output_dir):
    """Extract training crops + reference images into category directories."""
    coco_dir = Path(coco_dir)
    output_dir = Path(output_dir)

    # Extract crops from annotations
    with open(coco_dir / "annotations.json") as f:
        coco = json.load(f)
    images = {img["id"]: img for img in coco["images"]}
    n_crops = 0
    for ann in coco["annotations"]:
        info = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        if w < 32 or h < 32:
            continue
        cat_dir = output_dir / str(ann["category_id"])
        cat_dir.mkdir(parents=True, exist_ok=True)
        img_path = coco_dir / "images" / info["file_name"]
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path)
            px, py = int(w * 0.1), int(h * 0.1)
            crop = img.crop((
                max(0, int(x) - px), max(0, int(y) - py),
                min(img.width, int(x + w) + px), min(img.height, int(y + h) + py)
            ))
            crop.save(cat_dir / f"{info['id']}_{ann['id']}.jpg", quality=95)
            n_crops += 1
        except Exception:
            pass
    print(f"Extracted {n_crops} training crops")

    # Add reference images
    if Path(ref_zip).exists() and Path(mapping_file).exists():
        import zipfile
        with open(mapping_file) as f:
            mapping = json.load(f)
        ref_tmp = Path("/tmp/product_images")
        if not ref_tmp.exists():
            with zipfile.ZipFile(ref_zip) as z:
                z.extractall(ref_tmp)
        n_refs = 0
        for barcode, cat_id in mapping.items():
            src = ref_tmp / barcode
            if not src.exists():
                continue
            dst = output_dir / str(cat_id)
            dst.mkdir(parents=True, exist_ok=True)
            for img_path in src.glob("*.jpg"):
                shutil.copy2(img_path, dst / f"ref_{barcode}_{img_path.stem}.jpg")
                n_refs += 1
        print(f"Added {n_refs} reference images")

    total = sum(len(list(d.glob("*.jpg"))) for d in output_dir.iterdir() if d.is_dir())
    print(f"Total: {total} images in {len(list(output_dir.iterdir()))} categories")


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Augmentation — simulate shelf photo conditions
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds = ProductDataset("data/crops", train_tf)
    n_val = max(1, len(ds) // 10)
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    # Override val transform
    val_ds.dataset = ProductDataset("data/crops", val_tf)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, num_workers=4, pin_memory=True)

    nc = len(ds.classes)
    print(f"Training ConvNeXt-Tiny: {len(ds)} samples, {nc} classes, device={device}")

    # ConvNeXt-Tiny: ~28MB, strong classifier, available in timm 0.9.12
    model = timm.create_model("convnext_tiny", pretrained=True, num_classes=nc)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    for epoch in range(40):
        model.train()
        correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)
        scheduler.step()

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vc += model(x).argmax(1).eq(y).sum().item()
                vt += y.size(0)
        va = vc / vt if vt else 0
        print(f"Ep {epoch+1}/40: train={correct/total:.4f} val={va:.4f}")

        if va > best_acc:
            best_acc = va
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": ds.c2i,
                "num_classes": nc,
                "model_name": "convnext_tiny",
            }, "models/classifier.pt")
            print(f"  Saved best ({best_acc:.4f})")

    print(f"Best val accuracy: {best_acc:.4f}")
    size = Path("models/classifier.pt").stat().st_size / 1024 / 1024
    print(f"Classifier size: {size:.1f} MB")


if __name__ == "__main__":
    crops_dir = Path("data/crops")
    if not crops_dir.exists() or len(list(crops_dir.iterdir())) < 100:
        print("Extracting crops + reference images...")
        extract_crops_and_refs(
            "data/train",
            "NM_NGD_product_images.zip",
            "barcode_to_category.json",
            "data/crops",
        )
    train()
