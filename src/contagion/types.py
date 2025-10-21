from __future__ import annotations

from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class Adjacency(Protocol):
    """Protocol for adjacency backends used by contagion models.

    Implementations should be immutable or treat methods as side-effect free.
    """

    def infected_neighbor_counts(self, active: np.ndarray) -> np.ndarray:  # shape (n,)
        """Return per-node counts of active neighbors that influence the node.

        - For undirected graphs: number of active neighbors.
        - For directed graphs: number of active in-neighbors (predecessors).
        """
        ...

    def threshold_met(self, counts: np.ndarray, threshold: float) -> np.ndarray:
        """Return a boolean mask for nodes that meet a threshold.

        Interpretation:
        - If 0 < threshold < 1: fraction-of-degree threshold
        - If threshold >= 1: integer count threshold
        """
        ...

    def degrees(self) -> np.ndarray:  # shape (n,)
        """Return the relevant degree for thresholding (undirected degree or in-degree)."""
        ...


StateArray = np.ndarray
