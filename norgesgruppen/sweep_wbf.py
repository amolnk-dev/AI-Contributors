"""Sweep WBF inference params on the proven 0.9208 models.

Tests combinations of iou_thr, skip_box_thr, conf_type, weights,
number of scales, and per-pass conf/iou thresholds.

Usage: python sweep_wbf.py
"""
import json, itertools, subprocess, sys, shutil, time
from pathlib import Path

TASK_DIR = Path(__file__).parent.resolve()
SUBMISSION_ZIP = TASK_DIR / "submission_overnight.zip"  # The 0.9208 submission
VAL_IMAGES = TASK_DIR / "data" / "yolo" / "images" / "val"
GT_FILE = TASK_DIR / "data" / "train" / "annotations.json"

# Param grid
CONFIGS = []

# Core WBF params
for iou_thr in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    for skip_box in [0.001, 0.005, 0.01, 0.02]:
        for conf_type in ['avg', 'max', 'box_and_model_avg']:
            for weights in [[1,2,3,2,1], [1,3,5,3,1], [2,3,4,3,2], [1,1,2,1,1]]:
                CONFIGS.append({
                    'iou_thr': iou_thr,
                    'skip_box_thr': skip_box,
                    'conf_type': conf_type,
                    'weights': weights,
                })

print(f"Total configs to sweep: {len(CONFIGS)}")

# Generate run.py for each config
TEMPLATE = '''
import argparse, json
from pathlib import Path
import torch
_torch_load = torch.load
torch.load = lambda *args, **kwargs: _torch_load(*args, **{{**kwargs, "weights_only": False}})
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

def run_model(model, img_path, device, imgsz, augment=False):
    results = model(str(img_path), device=device, verbose=False, imgsz=imgsz, conf=0.01, iou=0.7, max_det=500, augment=augment)
    boxes_norm, scores, labels = [], [], []
    img_h, img_w = 0, 0
    for r in results:
        img_h, img_w = r.orig_shape
        if r.boxes is None or len(r.boxes) == 0: continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            boxes_norm.append([max(0, x1/img_w), max(0, y1/img_h), min(1, x2/img_w), min(1, y2/img_h)])
            scores.append(float(r.boxes.conf[i].item()))
            labels.append(int(r.boxes.cls[i].item()))
    return boxes_norm, scores, labels, img_w, img_h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_dir, output_path = Path(args.input), Path(args.output)
    script_dir = Path(__file__).parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [YOLO(str(script_dir / "model_a.pt")), YOLO(str(script_dir / "model_b.pt")), YOLO(str(script_dir / "model_c.pt"))]
    predictions = []
    image_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])
        all_boxes, all_scores, all_labels = [], [], []
        img_w, img_h = 0, 0
        for imgsz, aug in [(960, False), (1280, False), (1280, True)]:
            boxes, scores, labels, img_w, img_h = run_model(models[0], img_path, device, imgsz, aug)
            if boxes: all_boxes.append(boxes); all_scores.append(scores); all_labels.append(labels)
        for m in models[1:]:
            boxes, scores, labels, _, _ = run_model(m, img_path, device, 1280, augment=True)
            if boxes: all_boxes.append(boxes); all_scores.append(scores); all_labels.append(labels)
        if not all_boxes: continue
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            iou_thr={iou_thr}, skip_box_thr={skip_box_thr}, weights={weights}, conf_type='{conf_type}',
        )
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1, y1 = box[0]*img_w, box[1]*img_h
            x2, y2 = box[2]*img_w, box[3]*img_h
            predictions.append({{"image_id": image_id, "category_id": int(label),
                "bbox": [round(x1,2), round(y1,2), round(x2-x1,2), round(y2-y1,2)], "score": round(float(score),6)}})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f: json.dump(predictions, f)
    print(f"Wrote {{len(predictions)}} predictions")

if __name__ == "__main__": main()
'''

# Run sweep locally (uses the val set for ranking)
results = []
best_score = 0
best_config = None

for i, cfg in enumerate(CONFIGS):
    # Generate run.py with these params
    run_code = TEMPLATE.format(**cfg)
    
    tmp = Path("/tmp/_sweep_sandbox")
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir()
    
    (tmp / "run.py").write_text(run_code)
    
    # Extract models from submission zip
    import zipfile
    with zipfile.ZipFile(SUBMISSION_ZIP) as zf:
        for name in ['model_a.pt', 'model_b.pt', 'model_c.pt']:
            zf.extract(name, tmp)
    
    output_json = tmp / "predictions.json"
    
    import os
    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(tmp / "run.py"), "--input", str(VAL_IMAGES), "--output", str(output_json)],
        capture_output=True, text=True, timeout=300, env=env,
    )
    elapsed = time.time() - start
    
    if result.returncode != 0 or not output_json.exists():
        print(f"[{i+1}/{len(CONFIGS)}] FAILED: {cfg}")
        continue
    
    # Score using pycocotools
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        
        coco_gt = COCO(str(GT_FILE))
        
        # Filter GT to only val images
        val_img_ids = set()
        for p in VAL_IMAGES.iterdir():
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                val_img_ids.add(int(p.stem.split('_')[-1]))
        
        preds = json.loads(output_json.read_text())
        preds = [p for p in preds if p['image_id'] in val_img_ids]
        
        if not preds:
            print(f"[{i+1}/{len(CONFIGS)}] 0 preds")
            continue
        
        # Write filtered preds
        filtered_path = tmp / "filtered_preds.json"
        filtered_path.write_text(json.dumps(preds))
        
        coco_dt = coco_gt.loadRes(str(filtered_path))
        
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = list(val_img_ids)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        map50 = coco_eval.stats[1]  # mAP@0.5
        
        results.append((map50, cfg, elapsed))
        
        if map50 > best_score:
            best_score = map50
            best_config = cfg
            print(f"[{i+1}/{len(CONFIGS)}] NEW BEST: {map50:.4f} | {cfg} ({elapsed:.1f}s)")
        elif (i+1) % 20 == 0:
            print(f"[{i+1}/{len(CONFIGS)}] {map50:.4f} | best so far: {best_score:.4f}")
    except Exception as e:
        print(f"[{i+1}/{len(CONFIGS)}] Eval error: {e}")
        continue

# Print final results
print("\n" + "="*60)
print(f"SWEEP COMPLETE: {len(results)} configs tested")
print(f"BEST: {best_score:.4f}")
print(f"CONFIG: {best_config}")
print("\nTop 10:")
results.sort(key=lambda x: -x[0])
for score, cfg, elapsed in results[:10]:
    print(f"  {score:.4f} | iou={cfg['iou_thr']} skip={cfg['skip_box_thr']} type={cfg['conf_type']} w={cfg['weights']} ({elapsed:.0f}s)")

