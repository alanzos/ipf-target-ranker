# IPF gene ranker

Private implementation for the portfolio note
[Drug target identification for idiopathic pulmonary fibrosis (IPF) via geometric deep learning on a knowledge graph](https://andreslanzos.quarto.pub/portfolio/articles/2026-08-16-ipf-target-ranker/).

Rank 4,283 Open Targets IPF-associated genes so that 139 clinical-stage
mechanisms (phase ≥ 1) sit higher than the rest. Features are 18 public analogue
scores. The graph is STRING v12 (combined score ≥ 700). Models: XGBoost, MLP,
STRING-GCN, STRING-GAT, plus edgeless and permute-y controls.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
python scripts/run_cv.py --smoke          # short check
python scripts/run_cv.py                  # 5x5 CV (the article table)
```

Full CV writes `outputs/cv/metrics.json` and the three article figures under
`outputs/cv/figures/`. Seed 42. Device is `auto` (MPS / CUDA / CPU).

Expected confirmation means (25 test folds, chance AUPRC 0.032):

| Model | AUPRC |
| --- | ---: |
| Open Targets overall | 0.329 ± 0.036 |
| STRING-GCN | 0.322 ± 0.082 |
| XGBoost | 0.305 ± 0.086 |
| STRING-GAT | 0.274 ± 0.084 |
| Edgeless GCN | 0.098 ± 0.033 |
| Permute-y XGBoost | 0.038 ± 0.008 |

STRING-GCN beats edgeless GCN in 25/25 folds.

## Layout

```
data/ipf_matrix.parquet   18 scores + y + OT overall
data/string/              induced STRING subgraph
src/ipf_ranker/           models, CV, metrics
scripts/run_cv.py         one-command experiment
configs/ranker.yaml       frozen hyperparameters
```

GNN training is transductive: test labels are masked, test features still flow
on STRING edges. XGBoost is inductive. Primary metric is AUPRC.

Data licenses: [data/SOURCES.md](data/SOURCES.md). Code is proprietary. All rights reserved.
