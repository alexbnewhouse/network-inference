from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse as sp

from .types import Adjacency


@dataclass(frozen=True)
class CSRAdjacency(Adjacency):
    """CPU adjacency backend using CSR matrix.

    Assumes adjacency is row-normal: rows correspond to target nodes; non-zeros indicate incoming edges from sources.
    For undirected graphs, provide a symmetric matrix.
    """

    mat: sp.csr_matrix

    def __post_init__(self):
        m = self.mat
        if not sp.isspmatrix_csr(m):
            m = m.tocsr()
            object.__setattr__(self, "mat", m)


    def infected_neighbor_counts(self, active: np.ndarray) -> np.ndarray:
        active = active.astype(np.int8, copy=False)
        # counts = A * active
        y = self.mat @ active
        return np.asarray(y).ravel()

    def threshold_met(self, counts: np.ndarray, threshold: float) -> np.ndarray:
        if threshold < 1.0:
            deg = self.degrees().astype(float)
            frac = np.divide(counts, np.maximum(deg, 1.0), dtype=float)
            return frac >= threshold
        else:
            return counts >= int(round(threshold))

    def degrees(self) -> np.ndarray:
        # in-degree = sum across columns (for CSR rows as targets)
        return np.diff(self.mat.indptr)


def to_csr_adjacency(n: int, edges: np.ndarray, directed: bool = False, weights: Optional[np.ndarray] = None) -> CSRAdjacency:
    """Build a CSRAdjacency from edge list.

    edges: array of shape (m,2) with [source, target] indices in [0,n).
    If directed=False, both (u,v) and (v,u) are added.
    Weights default to 1.
    """
    if edges.size == 0:
        mat = sp.csr_matrix((n, n), dtype=np.int8)
        return CSRAdjacency(mat)
    src0 = edges[:, 0].astype(int)
    dst0 = edges[:, 1].astype(int)
    if not directed:
        src = np.concatenate([src0, dst0])
        dst = np.concatenate([dst0, src0])
        data = np.ones_like(src, dtype=np.int8) if weights is None else np.asarray(weights)
        mat = sp.csr_matrix((data, (dst, src)), shape=(n, n))
    else:
        data = np.ones_like(src0, dtype=np.int8) if weights is None else np.asarray(weights)
        mat = sp.csr_matrix((data, (dst0, src0)), shape=(n, n))
    return CSRAdjacency(mat)
