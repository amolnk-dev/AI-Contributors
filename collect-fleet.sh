# GCP Fleet Experiments for Astar Island

Run 12 experiments in parallel across idle GCP VMs. Each VM runs one config — results in ~3 min instead of ~36 min locally.

## Quick Start

```bash
# 1. Deploy code to all VMs (first time or after code changes)
bash scripts/gcp/fleet-experiment.sh deploy

# 2. Run experiments
bash scripts/gcp/fleet-experiment.sh sweep

# 3. Collect results
bash scripts/gcp/fleet-experiment.sh collect

# Or all at once:
bash scripts/gcp/fleet-experiment.sh all
```

## How It Works

```
Local machine                    12 GCP VMs (g2-standard-8)
┌──────────┐                    ┌─────────────────────┐
│ train.py │──scp──────────────▶│ /tmp/astar/train.py │ × 12
│ model.py │                    │ /tmp/astar/model.py  │
│ data/*   │                    │ /tmp/astar/data/*    │
└──────────┘                    └──────────┬──────────┘
                                           │
                                    sed (modify param)
                                           │
                                    python3 train.py
                                           │
                                    /tmp/result.log
                                           │
                                ◀──grep val_metric──
```

## Adding New Experiments

Edit the `experiments` array in `fleet-experiment.sh`:

```bash
local experiments=(
    "exp5|sed -i 's/EMP_BLEND = 0.45/EMP_BLEND = 0.50/' train.py|EMP=0.50"
    "exp6|sed -i 's/OLD_VALUE/NEW_VALUE/' train.py|DESCRIPTION"
    ...
)
```

Format: `VM_SUFFIX|SED_COMMAND|LABEL`

## Known Issues & Fixes

### Problem: `sed` doesn't change values inside functions
The `sed` command only works for top-level variables. Values inside function bodies (like `EMP_BLEND = 0.45` inside `evaluate_loro()`) need the full line context:
```bash
# BAD — matches first occurrence only
sed -i 's/EMP_BLEND = 0.45/EMP_BLEND = 0.50/' train.py

# GOOD — match the indented version
sed -i 's/                EMP_BLEND = 0.45/                EMP_BLEND = 0.50/' train.py
```

### Problem: Missing dependencies
VMs need: `pip3 install --break-system-packages pydantic xgboost numpy scipy scikit-learn httpx`
The deploy script installs these, but if you add new imports, update the install list.

### Problem: SSH timeout on long runs
LORO takes ~2 min. The `wait` in the sweep script handles this. If SSH disconnects, results will still be in `/tmp/result.log` on the VM.

## Better Approach: Pre-built Experiment Scripts

Instead of `sed` (fragile), create separate experiment files:

```bash
# Create experiment variants locally
for emp in 0.35 0.40 0.45 0.50 0.55 0.60; do
    cp train.py train_emp${emp}.py
    # Use Python to modify (more reliable than sed):
    python3 -c "
import re
with open('train_emp${emp}.py') as f: s = f.read()
s = re.sub(r'EMP_BLEND = [0-9.]+', 'EMP_BLEND = ${emp}', s)
with open('train_emp${emp}.py', 'w') as f: f.write(s)
"
done

# Deploy all variants
gcloud compute scp train_emp*.py VM:/tmp/astar/ --zone=...

# Run specific variant per VM
gcloud compute ssh VM --command="cd /tmp/astar && python3 train_emp0.50.py > /tmp/result.log 2>&1"
```

## Available VMs

| VM | Zone | Type | CPUs |
|----|------|------|------|
| ainm-norgesgruppen-exp5 to exp13 | us-central1-a | g2-standard-8 | 8 |
| ainm-norgesgruppen-cls | us-central1-a | g2-standard-8 | 8 |
| ainm-norgesgruppen-ar2 | europe-west1-b | g2-standard-8 | 8 |
| ainm-norgesgruppen-exp4 | europe-west1-b | g2-standard-8 | 8 |
| ainm-astar-sweep | europe-west4-a | c2-standard-30 | 30 |

**Always check VMs are idle before using:** `bash scripts/gcp/fleet-experiment.sh check`

## Best Practice for Experiments

1. **Test locally first** — run `python train.py` to confirm baseline works
2. **Create experiment files with Python** — not sed (avoids regex bugs)
3. **Deploy all at once** — `fleet-experiment.sh deploy`
4. **Run one per VM** — each VM runs a different config
5. **Collect results** — `fleet-experiment.sh collect`
6. **Apply winner to model.py** — only if it beats baseline
7. **Retrain GBT** — `python retrain_gbt.py` after config changes
8. **Backtest production** — run train.py with `test_rounds` including all available data
