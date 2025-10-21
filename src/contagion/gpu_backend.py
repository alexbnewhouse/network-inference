from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cupy as cp  # type: ignore
    import cupyx.scipy.sparse as cpsp  # type: ignore

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None  # type: ignore
    cpsp = None  # type: ignore

from .types import Adjacency


@dataclass(frozen=True)
class GPUAdjacency(Adjacency):
    """GPU adjacency backend using cupy CSR matrix.

    Requires cupy installation. Falls back gracefully if unavailable.
    """

    mat_cpu: Optional[object] = None  # scipy CSR
    mat_gpu: Optional[object] = None  # cupy CSR

    def __post_init__(self):
        if not CUPY_AVAILABLE:
            raise RuntimeError("cupy is not available; cannot use GPU backend")
        # Transfer to GPU if not already
        if self.mat_gpu is None and self.mat_cpu is not None:
            mat_gpu = cpsp.csr_matrix(self.mat_cpu)  # type: ignore
            object.__setattr__(self, "mat_gpu", mat_gpu)

    def infected_neighbor_counts(self, active: np.ndarray) -> np.ndarray:
        if self.mat_gpu is None:
            raise RuntimeError("GPU matrix not initialized")
        active_gpu = cp.asarray(active, dtype=cp.int8)  # type: ignore
        y_gpu = self.mat_gpu @ active_gpu  # type: ignore
        return cp.asnumpy(y_gpu).ravel()  # type: ignore

    def threshold_met(self, counts: np.ndarray, threshold: float) -> np.ndarray:
        if threshold < 1.0:
            deg = self.degrees().astype(float)
            frac = np.divide(counts, np.maximum(deg, 1.0), dtype=float)
            return frac >= threshold
        else:
            return counts >= int(round(threshold))

    def degrees(self) -> np.ndarray:
        if self.mat_gpu is None:
            raise RuntimeError("GPU matrix not initialized")
        # in-degree = diff of indptr
        indptr = cp.asnumpy(self.mat_gpu.indptr)  # type: ignore
        return np.diff(indptr)


def to_gpu_adjacency(mat_cpu) -> GPUAdjacency:
    """Build GPU adjacency from a scipy CSR matrix."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("cupy is not available")
    return GPUAdjacency(mat_cpu=mat_cpu)
