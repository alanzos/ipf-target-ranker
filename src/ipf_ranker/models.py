"""Gene-level GCN/GAT and a small tabular MLP."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, GCNConv
except ImportError:  # pragma: no cover
    GATConv = None  # type: ignore
    GCNConv = None  # type: ignore


class TabularMLP(nn.Module):
    """Gene features to a ranking logit."""

    def __init__(
        self,
        n_features: int,
        hidden: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        hidden = hidden or [64, 32]
        layers: list[nn.Module] = []
        d = n_features
        for h in hidden:
            layers.extend([nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)])
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GeneGCN(nn.Module):
    """Two-hop GCN: each gene's logit comes from its STRING neighborhood."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if GCNConv is None:
            raise ImportError("torch_geometric is required for GeneGCN")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.head = nn.Linear(hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_weight=edge_weight)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h).squeeze(-1)


class GeneGAT(nn.Module):
    """Two-layer GAT. Layer 1 concatenates 4 heads; layer 2 averages to 16-d."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 16,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if GATConv is None:
            raise ImportError("torch_geometric is required for GeneGAT")
        if num_layers != 2:
            raise ValueError("this distilled GeneGAT is the 2-layer article head")
        self.dropout = dropout
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=heads, dropout=dropout
        )
        self.conv2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_weight
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return self.head(h).squeeze(-1)
