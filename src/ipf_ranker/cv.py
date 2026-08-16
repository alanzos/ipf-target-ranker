"""Fit helpers for one CV fold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .models import GeneGAT, GeneGCN, TabularMLP
from .train import predict_gene_gnn_proba, predict_proba_mlp, train_gene_gnn_epochs, train_mlp_epochs


@dataclass
class GraphTensors:
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor | None
    edge_index_empty: torch.Tensor
    edge_weight_empty: torch.Tensor


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def inner_split(
    train_idx: np.ndarray,
    y: np.ndarray,
    *,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    yb = y[train_idx]
    if yb.sum() < 2 or (len(yb) - yb.sum()) < 2 or val_frac <= 0:
        return train_idx, train_idx
    tr, va = train_test_split(
        train_idx,
        test_size=val_frac,
        stratify=yb,
        random_state=seed,
    )
    return np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64)


def scale_from_train(X: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    scaler.fit(X[train_idx])
    return scaler.transform(X).astype(np.float32)


def fit_xgb(X_tr, y_tr, X_va, y_va, params: dict, seed: int):
    import xgboost as xgb

    n_pos = max(int(y_tr.sum()), 1)
    n_neg = max(int(len(y_tr) - y_tr.sum()), 1)
    model = xgb.XGBClassifier(
        n_estimators=int(params.get("n_estimators", 300)),
        max_depth=int(params.get("max_depth", 4)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        min_child_weight=int(params.get("min_child_weight", 4)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=n_neg / n_pos,
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
        early_stopping_rounds=int(params.get("early_stopping_rounds", 40)),
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def predict_xgb(model, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def _to_graph_tensors(
    X: np.ndarray,
    y: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
) -> GraphTensors:
    ew = None
    if edge_weight is not None and np.asarray(edge_weight).size:
        ew = torch.tensor(np.asarray(edge_weight), dtype=torch.float32)
    return GraphTensors(
        x=torch.tensor(np.asarray(X), dtype=torch.float32),
        y=torch.tensor(np.asarray(y), dtype=torch.float32),
        edge_index=torch.tensor(np.asarray(edge_index), dtype=torch.long),
        edge_weight=ew,
        edge_index_empty=torch.zeros((2, 0), dtype=torch.long),
        edge_weight_empty=torch.zeros((0,), dtype=torch.float32),
    )


def fit_mlp(X, y, train_idx, val_idx, params: dict, device: torch.device) -> np.ndarray:
    Xs = scale_from_train(X, train_idx)
    model = TabularMLP(
        n_features=Xs.shape[1],
        hidden=list(params.get("hidden", [64, 32])),
        dropout=float(params.get("dropout", 0.3)),
    )
    train_mlp_epochs(
        model,
        Xs[train_idx],
        y[train_idx],
        Xs[val_idx],
        y[val_idx],
        epochs=int(params.get("epochs", 80)),
        batch_size=int(params.get("batch_size", 256)),
        lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
        patience=int(params.get("patience", 12)),
        device=device,
    )
    return predict_proba_mlp(model, Xs, device)


def fit_gene_gcn(
    X,
    y,
    train_idx,
    val_idx,
    edge_index,
    edge_weight,
    params: dict,
    device: torch.device,
    *,
    edgeless: bool = False,
) -> np.ndarray:
    Xs = scale_from_train(X, train_idx)
    g = _to_graph_tensors(Xs, y, edge_index, edge_weight)
    ei = g.edge_index_empty if edgeless else g.edge_index
    ew = g.edge_weight_empty if edgeless else g.edge_weight
    model = GeneGCN(
        in_channels=Xs.shape[1],
        hidden_channels=int(params.get("hidden_channels", 32)),
        num_layers=int(params.get("num_layers", 2)),
        dropout=float(params.get("dropout", 0.3)),
    )
    train_gene_gnn_epochs(
        model,
        g.x,
        g.y,
        ei,
        train_idx,
        val_idx,
        edge_weight=ew,
        epochs=int(params.get("epochs", 100)),
        lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
        patience=int(params.get("patience", 15)),
        device=device,
    )
    return predict_gene_gnn_proba(model, g.x, ei, edge_weight=ew, device=device)


def fit_gene_gat(
    X,
    y,
    train_idx,
    val_idx,
    edge_index,
    params: dict,
    device: torch.device,
    *,
    edgeless: bool = False,
) -> np.ndarray:
    Xs = scale_from_train(X, train_idx)
    g = _to_graph_tensors(Xs, y, edge_index, None)
    ei = g.edge_index_empty if edgeless else g.edge_index
    model = GeneGAT(
        in_channels=Xs.shape[1],
        hidden_channels=int(params.get("hidden_channels", 16)),
        heads=int(params.get("heads", 4)),
        num_layers=int(params.get("num_layers", 2)),
        dropout=float(params.get("dropout", 0.3)),
    )
    train_gene_gnn_epochs(
        model,
        g.x,
        g.y,
        ei,
        train_idx,
        val_idx,
        edge_weight=None,
        epochs=int(params.get("epochs", 100)),
        lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
        patience=int(params.get("patience", 15)),
        device=device,
    )
    return predict_gene_gnn_proba(model, g.x, ei, edge_weight=None, device=device)


def fit_degree_gcn(
    degree,
    y,
    train_idx,
    val_idx,
    edge_index,
    edge_weight,
    params: dict,
    device: torch.device,
) -> np.ndarray:
    X = np.column_stack([degree.astype(np.float32), np.ones(len(degree), dtype=np.float32)])
    return fit_gene_gcn(
        X, y, train_idx, val_idx, edge_index, edge_weight, params, device, edgeless=False
    )
