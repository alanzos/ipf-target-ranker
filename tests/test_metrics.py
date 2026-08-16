"""Unit tests that do not require PyTorch Geometric."""

from __future__ import annotations

import numpy as np

from ipf_ranker.metrics import ranking_metrics, recall_at_k


def test_recall_at_k_recovers_top_positives() -> None:
    y = np.array([0, 1, 0, 1, 0])
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.0])
    assert recall_at_k(y, score, 2) == 1.0
    assert recall_at_k(y, score, 1) == 0.5


def test_ranking_metrics_includes_prevalence() -> None:
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    m = ranking_metrics(y, score)
    assert m["prevalence"] == 0.5
    assert m["auprc"] > 0.9
    assert "recall_at_20" in m
