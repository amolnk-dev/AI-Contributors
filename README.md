# Astar Island — Viking Civilisation Prediction

**Weight**: 25% of total score
**Type**: Observation + probabilistic prediction (REST API client)

## Quick Start

```bash
# Set auth token (grab JWT from app.ainm.no cookies)
export ASTAR_TOKEN="your-jwt-token"

# Run for active round
python run.py

# Dry run (no submission)
python run.py --dry-run --visualize

# Resume from checkpoint after crash
python run.py --resume

# Analyze completed round (extracts calibration)
python analyze.py [round_id]
```

## Architecture

Runner-first design — no FastAPI server. We call the competition API, not the other way around.

```
run.py          Main runner: fetch → observe → predict → submit
client.py       HTTP client (rate limit, budget tracking, auth detection)
model.py        Multi-layer prediction engine + ensemble
utils.py        Grid parsing, viewport strategy, normalization, visualization
analyze.py      Post-round ground truth analysis → calibration.json
dtos.py         Pydantic models + terrain mapping constants
```

## Prediction Model

5-layer heuristic:

1. **Static** — Ocean, mountain, deep forest → deterministic
2. **Observed** — Frequency distribution from viewport queries
3. **Unobserved priors** — Based on initial terrain + adjacency (forests, coast, settlements)
4. **Cross-seed transfer** — Pool observations from other seeds with matching initial terrain
5. **Calibration** — Learned priors from past round ground truth

## Query Strategy

50 queries / 5 seeds = 10 per seed. Focus on dynamic regions (settlement neighborhoods).
Greedy set-cover for viewport placement. Repeat same viewport to build frequency estimates.

## Tests

```bash
python -m pytest tests/ -v
```
