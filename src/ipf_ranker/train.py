"""Training loops for the MLP and gene GNNs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import ranking_metrics


@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 1e-4
    best: float | None = None
    bad_epochs: int = 0
    should_stop: bool = False

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            return True
        if value > self.best + self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.should_stop = True
        return False


@dataclass
class TrainResult:
    history: list[dict[str, float]] = field(default_factory=list)
    best_state: dict[str, Any] | None = None
    best_val_metric: float | None = None


def _pos_weight(y: np.ndarray) -> torch.Tensor:
    y = np.asarray(y).astype(int)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


def predict_proba_mlp(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(X, dtype=torch.float32, device=device)
        return torch.sigmoid(model(xt)).cpu().numpy()


def train_mlp_epochs(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
    device: str | torch.device = "cpu",
) -> TrainResult:
    device = torch.device(device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(y_train).to(device))
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    stopper = EarlyStopping(patience=patience)
    result = TrainResult()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        val_prob = predict_proba_mlp(model, X_val, device)
        val_auprc = float(ranking_metrics(y_val, val_prob)["auprc"])
        score = val_auprc if not np.isnan(val_auprc) else -1.0
        if stopper.step(score):
            result.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            result.best_val_metric = val_auprc
        if stopper.should_stop:
            break
    if result.best_state is not None:
        model.load_state_dict(result.best_state)
    return result


def train_gene_gnn_epochs(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    edge_weight: torch.Tensor | None = None,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: torch.device | str = "cpu",
) -> TrainResult:
    device = torch.device(device)
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)
    train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    val_np = np.asarray(val_idx, dtype=np.int64)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_np = y.detach().cpu().numpy()
    crit = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(y_np[train_idx]).to(device))
    stopper = EarlyStopping(patience=patience)
    result = TrainResult()
    for _epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x, edge_index, edge_weight)
        loss = crit(logits[train_t], y[train_t])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(x, edge_index, edge_weight)).detach().cpu().numpy()
        val_auprc = float(ranking_metrics(y_np[val_np], val_prob[val_np])["auprc"])
        if stopper.step(val_auprc if not np.isnan(val_auprc) else -1.0):
            result.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            result.best_val_metric = val_auprc
        if stopper.should_stop:
            break
    if result.best_state is not None:
        model.load_state_dict(result.best_state)
    return result


def predict_gene_gnn_proba(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    edge_weight: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    device = torch.device(device)
    model.eval()
    with torch.no_grad():
        logits = model(
            x.to(device),
            edge_index.to(device),
            None if edge_weight is None else edge_weight.to(device),
        )
        return torch.sigmoid(logits).detach().cpu().numpy()
