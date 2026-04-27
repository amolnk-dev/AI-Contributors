# Competition Strategy

**Duration**: 69 hours (March 19 18:00 — March 22 15:00 CET)
**Tasks**: 3 (CV, ML, NLP) — equal weight
**Key insight**: Reliability across all 3 > excellence in 1

## Time Budget

### Phase 1: Rapid Baseline (Hours 0-2)
- Read all 3 challenge descriptions
- Fill task-analysis docs
- Deploy dummy FastAPI endpoints
- Submit naive baselines (even random/constant)
- **Goal**: Non-zero score on ALL 3 tasks

### Phase 2: Strong Baselines (Hours 2-8)
- CV: Pretrained model (ResNet/EfficientNet), basic augmentation
- ML: AutoGluon/XGBoost with default params
- NLP: Pretrained transformer, basic fine-tuning
- Establish cross-validation for each task
- **Goal**: Competitive baselines on all 3

### Phase 3: Overnight Optimization (Night 1, ~Hours 8-20)
- Launch autoresearch loops (protocol: ~/Utvikling/autoresearch-mlx/program.md)
- GCP VMs for CV/large models (PyTorch/CUDA)
- Local Mac for NLP/small models (MLX via autoresearch-mlx)
- **Vertex AI HPO**: Submit parallel hyperparameter tuning jobs via `scripts/gcp/vertex-hpo.sh` — automated search over learning rate, batch size, etc. with Vizier. Runs unattended, reports best trial.
- Each task needs: prepare.py (fixed), train.py (mutable), task.md (metric)
- Agent iterates autonomously: edit train.py → run → keep/discard → repeat
- Monitor GCP via `scripts/gcp/status.sh`

### Phase 4: Iteration (Hours 20-60)
- Review overnight results
- Manual architecture improvements
- Feature engineering (ML), augmentation (CV), prompt tuning (NLP)
- Ensemble construction

### Phase 5: Final Polish (Hours 60-69)
- Ensemble all models per task
- Threshold optimization
- Stress-test APIs (latency, memory)
- Final validation: `scripts/validate.sh`
- Submit final endpoints
- **DO NOT make risky changes in last 3 hours**

## Task Priority

Equal time allocation by default. Shift resources to the most improvable task after baselines are set.

## Pre-Competition Readiness (verified 2026-03-19)

### Local Pipeline
- Setup, tests, validation: ALL PASSING (9/9 tests)
- No prohibited cloud API imports
- All 3 APIs boot and respond to health, metadata, and predict

### GCP Infrastructure
- Project: `ai-nm26osl-1823`, Auth: `devstar18231@gcplab.me`
- GPU quota confirmed in `europe-west4`:
  - A100: 64 (prefer this — use `heavy` tier)
  - L4: 16 (`medium` tier)
  - T4: 4 (`light` tier)
- End-to-end deploy tested: create VM -> deploy code -> API live -> teardown
- Zone fallback: scripts auto-try multiple zones (A100 often stocked out in europe-west4, `us-central1-f` worked)
- SSH + firewall rules verified working

### Monitoring
- `bash scripts/gcp/status.sh` — checks all endpoints with zone auto-detection
- `bash scripts/validate.sh` — full local validation
- `bash scripts/test-task.sh <task>` — live boot + HTTP test

## Sponsors

- NorgesGruppen — likely grocery/retail related CV or ML task
- Tripletex — accounting/ERP, likely NLP task (document understanding?)
- Google Cloud — compute sponsor (use GCP for all training)
