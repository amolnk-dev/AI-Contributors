# Astar Island — Live Round Runbook

Step-by-step for every round. Follow this exactly.

## When a Round Opens

### Step 1: Submit immediately (~2 min)
Don't wait — submit with the current production model first.

```bash
cd tasks/astar-island
ASTAR_TOKEN=<token> python run.py
```

This uses 50 queries, builds predictions, and submits all 5 seeds.
You can always resubmit later with a better model — only the last submission counts.

### Step 2: Check the swarm for improvements (~5 min)
While the round is open, check if the autoresearch swarm found anything better.

```bash
# Dashboard (if running)
open http://localhost:8050

# Or pull results manually
bash scripts/gcp/collect-astar.sh
```

Look for experiments with val_metric > current baseline (89.47).

### Step 3: Integrate best swarm result (if improved)

```bash
# Download best train.py from winning VM
bash scripts/gcp/collect-astar.sh --best

# Verify locally with full LORO
cp tasks/astar-island/train.py tasks/astar-island/train.py.backup
cp tasks/astar-island/train.py.best_from_swarm tasks/astar-island/train.py
cd tasks/astar-island && python train.py
# Check: val_metric should be > 89.47

# If improved: retrain production model
python retrain_gbt.py

# Resubmit with improved model
ASTAR_TOKEN=<token> python run.py

# Restore original if not improved
cp tasks/astar-island/train.py.backup tasks/astar-island/train.py
```

### Step 4: Gauge the round
Check observation stats to understand what kind of round this is:

| obs_settl_rate | Round type | Our strength |
|----------------|-----------|-------------|
| < 0.05 | Extinction | Strong (93-95) |
| 0.05 - 0.15 | Medium | Good (87-93) |
| > 0.15 | High expansion | Weak (72-88) |

High expansion rounds (like R7) are our weakness. If you see obs_settl_rate > 0.15,
the swarm's R7-focused VMs (s1, s2, s3) might have found improvements for this type.

## When a Round Completes

### Step 5: Download ground truth + update model (~3 min)

```bash
ASTAR_TOKEN=<token> python3 -c "
from client import AstarClient
import numpy as np, json
c = AstarClient()
ROUND_NUM = 18  # <-- change this
rounds = c.get_rounds()
round_id = [r for r in rounds if r.round_number==ROUND_NUM][0].id
detail = c.get_round_detail(round_id)

# Save initial states
states = {'initial_states': []}
for seed in range(5):
    s = detail.initial_states[seed]
    states['initial_states'].append({'grid': s.grid, 'settlements': [x.model_dump() for x in s.settlements]})
with open(f'data/round{ROUND_NUM}_initial.json', 'w') as f:
    json.dump(states, f)

# Save GT
for seed in range(5):
    a = c.get_analysis(round_id, seed)
    np.save(f'data/gt_r{ROUND_NUM}_seed{seed}.npy', np.array(a.ground_truth))
    print(f'Seed {seed}: score={a.score:.2f}')
print(f'Round {ROUND_NUM} GT saved.')
"
```

### Step 6: Add to LORO + retrain

1. Add round ID to `ROUNDS` dict in both `train.py` and `retrain_gbt.py`
2. Add to `ROUND_WEIGHTS` in both files: `N: 1.05**M` (where M = round_number - 1)
3. Add to `test_rounds` list in `evaluate_loro()`
4. Only add if the round has observations (check: `ls data/obs_*.jsonl`)
5. Retrain: `python retrain_gbt.py`
6. Run LORO to verify: `python train.py`

### Step 7: Update swarm VMs with new data

```bash
# Deploy new data + code to all VMs
cd tasks/astar-island
for VM in ainm-astar-autoresearch ainm-astar-swarm-a ainm-astar-swarm-b $(seq -f "ainm-astar-s%.0f" 1 10); do
  gcloud compute scp train.py model.py retrain_gbt.py \
    data/round${ROUND_NUM}_initial.json data/gt_r${ROUND_NUM}_seed*.npy \
    data/obs_*.jsonl "$VM:/tmp/astar/data/" \
    --zone=europe-west4-a --project=ai-nm26osl-1823 2>/dev/null &
done
wait

# Restart all swarm agents (they'll pick up new data)
# Use the per-VM launcher scripts in /tmp/astar/launch_*.sh
```

### Step 8: Commit + push

```bash
git add tasks/astar-island/train.py tasks/astar-island/retrain_gbt.py
git add -f tasks/astar-island/data/gbt_models.pkl \
  tasks/astar-island/data/round${ROUND_NUM}_initial.json \
  tasks/astar-island/data/gt_r${ROUND_NUM}_seed*.npy
git commit -m "astar-island: R${ROUND_NUM} scored XX.XX — download GT, add to LORO, retrain"
git push
```

## Monitoring

### Dashboard
```bash
cd tasks/astar-island && python dashboard.py
# Open http://localhost:8050
```

### Quick fleet check
```bash
bash scripts/gcp/collect-astar.sh
```

### Check specific VM
```bash
gcloud compute ssh ainm-astar-s1 --zone=europe-west4-a --project=ai-nm26osl-1823 \
  --ssh-flag="-o StrictHostKeyChecking=no" \
  --command='tail -20 /tmp/astar/swarm.log'
```

### Check VM results
```bash
gcloud compute ssh ainm-astar-s1 --zone=europe-west4-a --project=ai-nm26osl-1823 \
  --ssh-flag="-o StrictHostKeyChecking=no" \
  --command='cat /tmp/astar/results.tsv'
```

## Teardown (after competition)

```bash
# Delete all astar VMs
for VM in ainm-astar-autoresearch ainm-astar-swarm-a ainm-astar-swarm-b $(seq -f "ainm-astar-s%.0f" 1 10) ainm-astar-sweep; do
  gcloud compute instances delete "$VM" --zone=europe-west4-a --project=ai-nm26osl-1823 --quiet &
done
wait
```

## Key Numbers

| Metric | Value |
|--------|-------|
| Baseline WAVG (LORO) | 89.47 |
| Best live score | R17 = 92.99 |
| Best weighted | R17 = 202.92 |
| LORO time (local) | ~360s |
| LORO time (176-core VM) | ~189s |
| Swarm VMs | 13 (3 × c3 + 10 × c2) |
| Total CPUs | 564 |
| Experiments/hour | ~150-210 |
| Features | 77 (31 cell + 23 obs + 23 per-cell) |
| Trained on | 15 rounds (R1-R17 excl R3, R12) |
| PROB_FLOOR | 0.0001 (never change) |

## Leaderboard Score Formula

```
leaderboard_score = MAX(round_score × round_weight) across all rounds
round_weight = 1.05^(round_number - 1)
```

Later rounds are worth more. R18 weight = 1.05^17 = 2.29x.
A score of 90 on R18 = 206 weighted (better than R17's 202.92).
