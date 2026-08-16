"""AUPRC and Recall@K for gene ranking."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0 or len(y_true) == 0:
        return float("nan")
    k = min(int(k), len(y_true))
    top = np.argsort(-y_score, kind="mergesort")[:k]
    return float(y_true[top].sum() / n_pos)


def ranking_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    ks: tuple[int, ...] = (20, 50, 100),
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        auprc = auroc = float("nan")
    else:
        auprc = float(average_precision_score(y_true, y_score))
        auroc = float(roc_auc_score(y_true, y_score))
    out = {
        "auprc": auprc,
        "auroc": auroc,
        "n": float(len(y_true)),
        "n_pos": float((y_true == 1).sum()),
        "prevalence": float(y_true.mean()) if len(y_true) else float("nan"),
    }
    for k in ks:
        out[f"recall_at_{k}"] = recall_at_k(y_true, y_score, k)
    return out
