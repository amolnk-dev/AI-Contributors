# VM Fleet Overview — NorgesGruppen Autoresearch

**Last updated:** 2026-03-21 15:00 UTC
**Compute:** Sponsored (AI Championship) — free runs
**Best competition score:** 0.9221 (single-stage, multi-scale WBF + TTA)
**Fleet:** 31 GPUs (10x A100 40GB + 21x L4 24GB)

## Current Status

**Overnight autoresearch running** — `autoresearch_overnight.py` deployed to 31 VMs. Each VM samples random hyperparameter configs from a search space (70% exploitation / 30% exploration), trains YOLOv8l from scratch with 80/20 split for honest eval, and saves the best model per VM. Expected ~150+ experiments by morning.

### A100 Fleet (10x, autoresearch)

| VM | Zone | Role | Status |
|----|------|------|--------|
| a0 | us-central1-f | overnight HPO | training |
| a1 | us-central1-f | overnight HPO | training |
| a2 | us-central1-f | overnight HPO | training |
| a3 | us-central1-f | overnight HPO | training |
| a4 | us-central1-f | overnight HPO | training |
| a100 | us-central1-f | overnight HPO | training |
| a100-2 | us-central1-f | overnight HPO | training |
| a100-3 | us-central1-f | overnight HPO | training |
| a100-6 | us-central1-f | overnight HPO | training |
| a100-9 | us-central1-b | overnight HPO | training |

### L4 Fleet (21x, mixed)

| VM | Zone | Role | Status |
|----|------|------|--------|
| l0-l8 | mixed | overnight HPO | training |
| ar2 | europe-west1-b | overnight HPO | training |
| exp4 | europe-west1-b | overnight HPO | training |
| exp5-exp13 | us-central1-a | cls=1.0 seed sweep | training |
| cls | us-central1-a | overnight HPO | training |

## Competition Submissions

| # | Model | Val mAP50 | Score | Date | Notes |
|---|-------|-----------|-------|------|-------|
| 1 | YOLOv8m 100ep (199 imgs) | 0.53 | 0.6894 | Mar 19 | First baseline |
| 2 | YOLOv8l finetune-cos (199 imgs) | 0.7224 | 0.8966 | Mar 20 | Cosine LR finetune |
| 3 | YOLOv8l SGD finetune (199 imgs) | 0.7316 | 0.9007 | Mar 20 | SGD + cosine LR |
| 4 | YOLOv8l full-dataset 235ep | inflated | 0.9040 | Mar 20 | Full 248 imgs |
| 5 | YOLOv8l full-dataset 300ep (a100-6) | inflated | 0.9055 | Mar 20 | 300ep complete |
| 6 | YOLOv8l full-dataset 300ep (a100-2) | inflated | **0.9221** | Mar 20 | **BEST** |
| 7 | Fast two-stage (YOLO + ConvNeXt) | - | 0.8941 | Mar 20 | Classifier HURT score |

## Key Lessons

1. **Single-stage YOLO > Two-stage**: ConvNeXt classifier (98.8% val) actually lowered competition score by 3%
2. **Full dataset training is the #1 improvement**: 199→248 images gave biggest competition score gains
3. **Multi-scale WBF + TTA matters**: The 3-pass inference (960+1280+1280TTA) is critical
4. **Val mAP50 is inflated on full-dataset**: Val images are in training set, can't trust val_metric
5. **More epochs helps but plateaus**: 200→300ep gave diminishing returns
6. **Classifier trained on same crops as YOLO**: It can't improve what YOLO already knows — needs truly new signal (embedding fallback, shelf context)

## Key Findings

1. **Full dataset (248 imgs) >> 199 imgs**: mAP 0.727 at ep194 vs 0.645 at ep200 with 199 imgs
2. **Cosine LR finetune works**: Train 200ep → finetune with cos_lr SGD → best results
3. **But repeated finetuning degrades generalization**: Local eval drops while val_metric rises (overfitting to val set)
4. **Resolution 1280px optimal**: 960px and 1600px both worse
5. **YOLOv8l > YOLOv8x**: Bigger model early stops at lower mAP
6. **Batch=2 > batch=8 for SGD finetune**: Small batch noise helps SGD
7. **TTA + multi-scale WBF in run.py gives huge boost**: val 0.73 → competition 0.90
8. **Don't run classifier alongside YOLO on A100**: OOM crash (shared VRAM)

## Available Submissions (ready to upload)

| File | Size | Model | Notes |
|------|------|-------|-------|
| submission.zip | 78MB | best_0.7316 (SGD finetune) | Scored 0.9007 |
| submission_swa.zip | 78MB | SWA average of top 3 | Untested |
| submission_ensemble.zip | 233MB | 3-model WBF ensemble | Local eval slightly worse |
| models/best_fulldataset_slim.pt | 84MB | Full-dataset ep206 (stripped) | Not yet packaged |

## Pending Work (Morning March 22)

1. **Collect overnight results** — `bash scripts/gcp/collect-overnight.sh`
2. **Identify top 5 configs** by val_metric, select diverse top 3 for ensemble
3. **Retrain top 3 on full dataset** (`data/yolo_full/data.yaml`) on A100s for 300ep
4. **SWA top 3-5 models** as additional candidate
5. **Strip + package** with `strip_model.py` + `package_ensemble.sh`
6. **Submit best combos** (6 submissions available)

## Overnight Autoresearch

### How it works
`autoresearch_overnight.py` runs autonomously on each VM:
1. Sample random config from search space (VM_ID-seeded RNG → different per VM)
2. Generate inline training script, run as subprocess
3. Parse `val_metric` from stdout
4. If improved → save model as `best_overnight_{VM_ID}.pt`
5. Log all params + result to `overnight_results_{VM_ID}.tsv`
6. Cleanup runs dir (saves disk), repeat forever

### Search space (anchored at proven-best)
- **Fixed:** model=yolov8l.pt, imgsz=1280, SGD, cos_lr, lr0=0.005, lrf=0.01, batch=-1, patience=50
- **Varied:** cls [0.8-1.2], box [5-10], dfl [1-2], mosaic [0.8-1.0], mixup [0.05-0.2], copy_paste [0.05-0.15], degrees [5-15], scale [0.3-0.7], epochs [280-350], close_mosaic [10-30], warmup [3-5], freeze [None/5/10], label_smoothing [0-0.1]
- **Data:** `data/yolo/data.yaml` (80/20 split) for honest relative ranking

### Commands
```bash
# Deploy overnight to all VMs
bash scripts/gcp/deploy-overnight.sh

# Collect results + best models
bash scripts/gcp/collect-overnight.sh

# Monitor a VM
gcloud compute ssh <VM> --zone=<Z> --command='tail -50 ~/task/overnight.log'

# Top results across all VMs
cat tasks/norgesgruppen/overnight_results_*.tsv | sort -t$'\t' -k3 -rn | head -20
```

### Start on a single VM
```bash
systemd-run --unit=ainm-overnight --remain-after-exit bash -c \
  'cd /root/task && VM_ID=<id> GPU_TYPE=<a100|l4> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u autoresearch_overnight.py > overnight.log 2>&1'
```

## VM Setup Runbook

### Quick Start
```bash
# Create VM
bash scripts/gcp/create-vm.sh norgesgruppen medium --name=<name>   # L4
bash scripts/gcp/create-vm.sh norgesgruppen heavy --name=<name>    # A100

# Deploy data (VM→VM via SSH)
# Source A100 has tarball at /tmp/task_overnight.tar.gz
gcloud compute ssh <source> --zone=<z> --command="
cat /tmp/task_overnight.tar.gz | ssh -o StrictHostKeyChecking=no root@<NEW_IP> 'mkdir -p /root/task && cd /root/task && tar xzf -'
"

# Install deps (ORDER MATTERS — numpy FIRST, then opencv!)
pip install --break-system-packages ultralytics==8.1.0
pip install --break-system-packages --force-reinstall 'numpy==1.26.4'
pip install --break-system-packages 'opencv-python-headless==4.8.1.78'
find /usr/local/lib -name 'cv2*.so' -path '*/opencv_python/*' -delete

# Fix data path
sed -i 's|^path: .*|path: /root/task/data/yolo|' data/yolo/data.yaml
```

### Common Issues
| Issue | Fix |
|-------|-----|
| numpy.core.multiarray ImportError | `pip install --force-reinstall numpy==1.26.4` (BEFORE opencv!) |
| libGL.so.1 not found | `pip install opencv-python-headless==4.8.1.78` + delete cv2*.so |
| opencv pulls numpy 2.x back | Install numpy AFTER opencv, or pin `opencv-python-headless==4.8.1.78` |
| SyntaxError: keyword repeated | Use CLI args (`--lr0`, `--optimizer`), NOT sed |
| OOM on A100 | Don't share GPU between YOLO + classifier |
| nohup process dies on SSH disconnect | Use `systemd-run --unit=name --remain-after-exit` instead |

### Infrastructure
- **SSH Keys:** ainm-norgesgruppen-a100 has keys to all VMs (added via gcloud)
- **Data transfer:** VM→VM via tar pipe through SSH
- **All VMs:** Non-spot, systemd-run for persistence
- **Direct SSH:** `ssh -i ~/.ssh/google_compute_engine root@<IP>`
- **Tarball location:** `/tmp/task_overnight.tar.gz` on ainm-norgesgruppen-a100 (2.3GB, includes data + scripts)
