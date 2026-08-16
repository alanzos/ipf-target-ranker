"""Load the frozen gene matrix and STRING subgraph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
FEATURE_COLUMNS = (
    "genetics",
    "expression",
    "expression_geo",
    "expression_ot",
    "pathways",
    "pathways_reactome",
    "network_neighbors",
    "interactome_community",
    "animal_model",
    "causal_inference",
    "rwr_network",
    "disease_submodule",
    "mutated_submodule",
    "knockout",
    "overexpression",
    "tissue_lung",
    "genetics_gwas",
    "knockout_depmap",
)


@dataclass
class RankerData:
    symbols: np.ndarray
    ensembl_id: np.ndarray
    X: np.ndarray
    columns: list[str]
    y: np.ndarray
    ot_overall: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray

    @property
    def n_genes(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_pos(self) -> int:
        return int(self.y.sum())

    @property
    def degree(self) -> np.ndarray:
        deg = np.zeros(self.n_genes, dtype=np.float32)
        if self.edge_index.shape[1]:
            np.add.at(deg, self.edge_index[0], 1.0)
        return deg


def load_ranker_data(root: Path | None = None) -> RankerData:
    root = root or DATA_DIR
    df = pd.read_parquet(root / "ipf_matrix.parquet")
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"matrix missing columns: {missing}")
    genes = [g.strip() for g in (root / "string" / "gene_index.txt").read_text().splitlines() if g.strip()]
    symbols = df["symbol"].astype(str).to_numpy()
    if genes != symbols.tolist():
        raise ValueError("STRING gene_index.txt order must match ipf_matrix.parquet rows")
    edge_index = np.load(root / "string" / "edge_index.npy").astype(np.int64, copy=False)
    edge_weight = np.load(root / "string" / "edge_weight.npy").astype(np.float32, copy=False)
    return RankerData(
        symbols=symbols,
        ensembl_id=df["ensembl_id"].astype(str).to_numpy(),
        X=df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32),
        columns=list(FEATURE_COLUMNS),
        y=df["y_clinical"].to_numpy(dtype=np.int64),
        ot_overall=df["ot_overall"].to_numpy(dtype=np.float32),
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
