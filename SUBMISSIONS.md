# Astar Island — Submission Log

Every submission recorded here: what model, what observations, what score.

## R18 (2026-03-21)

| Field | Value |
|-------|-------|
| Round ID | b0f9d1bf-4b71-4e6e-816c-19c718d29056 |
| Map | 40x40, 5 seeds |
| Status | Submitted (2026-03-21 23:36) |
| Model | 77 features (31 cell + 23 obs + 23 cell_obs), 15 rounds training |
| Features | terrain_diversity_r3 (latest), vectorized extractors |
| GBT | 600 trees, depth 5, lr 0.08, 15 rounds (R1-R17 excl R3,R12) |
| Queries | 50/50 used |
| Submission 1 | 23:36 — immediate with production model |
| Round type | High expansion (57-149 settlements per viewport) |
| Score | TBD |
| Rank | TBD |

### Observations
- Very high settlement counts: 57-149 per viewport (similar to R17)
- Observations saved to `data/obs_b0f9d1bf-4b71-4e6e-816c-19c718d29056.jsonl`

### Resubmissions
- Check swarm for improvements: `bash scripts/gcp/collect-astar.sh`
- If found: resubmit with `ASTAR_TOKEN=... python run.py --resume`

---

## R17 (2026-03-21)

| Field | Value |
|-------|-------|
| Round ID | 3eb0c25d-28fa-48ca-b8e1-fc249e3918e9 |
| Score | **92.99** |
| Rank | 21/283 |
| Weighted | 202.92 |
| Seeds | 93.42 / 93.26 / 92.96 / 93.14 / 92.19 |
| Model | 77 features, terrain_diversity, 14 rounds training |

---

## R16 (2026-03-21)

| Field | Value |
|-------|-------|
| Round ID | 8f664aed-8839-4c85-bed0-77a2cac7c6f5 |
| Score | **88.85** |
| Rank | 4/272 |
| Weighted | 184.69 |
| Seeds | 86.90 / 88.21 / 89.70 / 88.20 / 91.25 |
| Model | 76 features (feature alignment fix), 14 rounds training |
| Notes | First round with aligned features |

---

## R15 (2026-03-21)

| Field | Value |
|-------|-------|
| Score | **92.31** |
| Rank | 28/262 |
| Weighted | 182.62 |

## R14

| Field | Value |
|-------|-------|
| Score | **85.85** |
| Rank | 8/244 |

## R13

| Field | Value |
|-------|-------|
| Score | **93.94** |
| Rank | **1**/186 |
| Notes | Best rank ever! |

## R9

| Field | Value |
|-------|-------|
| Score | **91.77** |
| Rank | 17/221 |

## R4

| Field | Value |
|-------|-------|
| Score | **91.80** |
| Rank | 5/86 |
| Notes | Coastal/inland split deployed |
