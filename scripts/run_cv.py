#!/usr/bin/env python3
"""Reproduce the IPF gene-ranking CV in the portfolio article."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ipf_ranker.cv import (
    fit_degree_gcn,
    fit_gene_gat,
    fit_gene_gcn,
    fit_mlp,
    fit_xgb,
    inner_split,
    pick_device,
    predict_xgb,
)
from ipf_ranker.data import load_ranker_data
from ipf_ranker.metrics import ranking_metrics

BAR_MODELS = [
    "ot_overall",
    "mean_scores",
    "max_scores",
    "xgb",
    "mlp",
    "gcn_string",
    "gcn_edgeless",
    "gat_string",
    "gat_edgeless",
    "gcn_degree",
    "xgb_permute_y",
]
PR_MODELS = [
    "ot_overall",
    "mean_scores",
    "xgb",
    "mlp",
    "gcn_string",
    "gcn_edgeless",
    "gat_string",
    "gat_edgeless",
]
EXAMPLE_GENES = ("TGFB1", "TNF", "PDGFRB", "MUC5B", "COX7B")


def _summarize(fold_rows: list[dict], model: str) -> dict:
    sub = [r for r in fold_rows if r["model"] == model]
    if not sub:
        return {}
    out: dict = {"n_folds": len(sub), "model": model}
    for k in ("auprc", "auroc", "recall_at_20", "recall_at_50", "recall_at_100", "prevalence"):
        vals = np.asarray([r[k] for r in sub], dtype=float)
        out[f"{k}_mean"] = float(np.nanmean(vals))
        out[f"{k}_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def _pr_figure(y: np.ndarray, oof: dict[str, np.ndarray], path: Path, prevalence: float) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for name in PR_MODELS:
        if name not in oof:
            continue
        prec, rec, _ = precision_recall_curve(y, oof[name])
        ax.plot(rec, prec, label=name, linewidth=1.6)
    ax.axhline(prevalence, color="black", linestyle="--", linewidth=1, label=f"chance ({prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("IPF clinical MoA recovery (OOF scores)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _metric_bar(summaries: dict[str, dict], metric: str, path: Path, ylabel: str) -> None:
    names = [n for n in BAR_MODELS if n in summaries]
    means = [summaries[n][f"{metric}_mean"] for n in names]
    sds = [summaries[n][f"{metric}_sd"] for n in names]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=sds, capsize=3, color="#2a6f97")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}, repeated stratified CV (mean +/- sd)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _univariate_bar(X: np.ndarray, y: np.ndarray, columns: list[str], path: Path) -> None:
    rows = []
    for i, col in enumerate(columns):
        score = np.asarray(X[:, i], dtype=float)
        auprc = float(average_precision_score(y, score))
        auprc_neg = float(average_precision_score(y, -score))
        rows.append((col, max(auprc, auprc_neg)))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color="#2a6f97")
    ax.axvline(float(y.mean()), color="black", linestyle="--", linewidth=1, label=f"chance ({y.mean():.3f})")
    ax.set_xlabel("Univariate AUPRC")
    ax.set_title("Per-score recovery of IPF clinical MoA genes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def run(args: argparse.Namespace) -> dict:
    cfg = _load_yaml(args.config)
    if args.smoke:
        n_splits, n_repeats = 2, 1
        cfg["mlp"]["epochs"] = min(int(cfg["mlp"]["epochs"]), 12)
        cfg["gcn"]["epochs"] = min(int(cfg["gcn"]["epochs"]), 15)
        cfg["gat"]["epochs"] = min(int(cfg["gat"]["epochs"]), 15)
        cfg["xgb"]["n_estimators"] = min(int(cfg["xgb"]["n_estimators"]), 40)
    else:
        n_splits = int(cfg["n_splits"])
        n_repeats = int(cfg["n_repeats"])

    np.random.seed(int(cfg["seed"]))
    torch_seed = int(cfg["seed"])
    try:
        import torch

        torch.manual_seed(torch_seed)
    except ImportError:
        pass

    device = pick_device(args.device)
    data = load_ranker_data()
    X, y = data.X, data.y
    deg = data.degree
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng_perm = np.random.default_rng(int(cfg["seed"]))
    y_perm = y.copy()
    rng_perm.shuffle(y_perm)
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=int(cfg["seed"])
    )
    fold_rows: list[dict] = []
    oof_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(data.n_genes, dtype=float))
    oof_n: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(data.n_genes, dtype=float))
    comparators = {
        "ot_overall": data.ot_overall.astype(float),
        "mean_scores": X.mean(axis=1).astype(float),
        "max_scores": X.max(axis=1).astype(float),
        "degree": deg.astype(float),
    }
    print(
        f"IPF ranker CV: n={data.n_genes} pos={data.n_pos} feats={X.shape[1]} "
        f"splits={n_splits}x{n_repeats} device={device}",
        flush=True,
    )
    for fold_i, (tr_all, te) in enumerate(splitter.split(X, y)):
        tr, va = inner_split(tr_all, y, val_frac=float(cfg["val_frac"]), seed=int(cfg["seed"]) + fold_i)
        print(
            f"  fold {fold_i + 1}/{n_splits * n_repeats} train={len(tr)} val={len(va)} test={len(te)}",
            flush=True,
        )
        preds: dict[str, np.ndarray] = dict(comparators)
        xgb_model = fit_xgb(X[tr], y[tr], X[va], y[va], cfg["xgb"], seed=int(cfg["seed"]) + fold_i)
        preds["xgb"] = predict_xgb(xgb_model, X)
        xgb_p = fit_xgb(X[tr], y_perm[tr], X[va], y_perm[va], cfg["xgb"], seed=int(cfg["seed"]) + fold_i)
        preds["xgb_permute_y"] = predict_xgb(xgb_p, X)
        preds["mlp"] = fit_mlp(X, y, tr, va, cfg["mlp"], device)
        preds["gcn_string"] = fit_gene_gcn(
            X, y, tr, va, data.edge_index, data.edge_weight, cfg["gcn"], device, edgeless=False
        )
        preds["gcn_edgeless"] = fit_gene_gcn(
            X, y, tr, va, data.edge_index, data.edge_weight, cfg["gcn"], device, edgeless=True
        )
        preds["gat_string"] = fit_gene_gat(
            X, y, tr, va, data.edge_index, cfg["gat"], device, edgeless=False
        )
        preds["gat_edgeless"] = fit_gene_gat(
            X, y, tr, va, data.edge_index, cfg["gat"], device, edgeless=True
        )
        preds["gcn_degree"] = fit_degree_gcn(
            deg, y, tr, va, data.edge_index, data.edge_weight, cfg["gcn"], device
        )
        for name, score in preds.items():
            metrics = ranking_metrics(y_perm[te] if name == "xgb_permute_y" else y[te], score[te])
            fold_rows.append({"fold": fold_i, "model": name, **metrics})
            oof_sum[name][te] += score[te]
            oof_n[name][te] += 1.0

    oof = {k: np.divide(oof_sum[k], np.maximum(oof_n[k], 1.0)) for k in oof_sum}
    models = sorted({r["model"] for r in fold_rows})
    summaries = {m: _summarize(fold_rows, m) for m in models}
    pd.DataFrame(fold_rows).to_csv(out_dir / "fold_metrics.csv", index=False)
    oof_df = pd.DataFrame(
        {"ensembl_id": data.ensembl_id, "symbol": data.symbols, "y_clinical": y, **oof}
    )
    oof_df.to_parquet(out_dir / "oof_scores.parquet", index=False)
    _pr_figure(y, oof, fig_dir / "pr_curves.png", float(y.mean()))
    _metric_bar(summaries, "auprc", fig_dir / "auprc_cv.png", "AUPRC")
    _univariate_bar(X, y, data.columns, fig_dir / "univariate_auprc.png")

    example_rows = []
    for gene in EXAMPLE_GENES:
        hit = oof_df.loc[oof_df["symbol"] == gene]
        if hit.empty:
            continue
        row = hit.iloc[0]
        example_rows.append(
            {
                "symbol": gene,
                "y_clinical": int(row["y_clinical"]),
                "gcn_string": float(row["gcn_string"]),
                "xgb": float(row["xgb"]),
            }
        )
    payload = {
        "n_genes": data.n_genes,
        "n_pos": data.n_pos,
        "n_features": int(X.shape[1]),
        "columns": data.columns,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "smoke": bool(args.smoke),
        "device": str(device),
        "string_n_edges": int(data.edge_index.shape[1]),
        "summaries": summaries,
        "example_genes": example_rows,
        "chance_auprc": float(y.mean()),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({m: summaries[m].get("auprc_mean") for m in BAR_MODELS if m in summaries}, indent=2), flush=True)
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "ranker.yaml")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "cv")
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
