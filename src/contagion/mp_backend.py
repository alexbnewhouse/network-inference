from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse as sp

from .types import Adjacency


@dataclass(frozen=True)
class MPAdjacency(Adjacency):
    """Multiprocessing adjacency backend using shared CSR matrix via pickling.

    Workers receive a copy of the adjacency. For large graphs, consider mmap or shared memory.
    """

    mat: sp.csr_matrix

    def __post_init__(self):
        m = self.mat
        if not sp.isspmatrix_csr(m):
            m = m.tocsr()
            object.__setattr__(self, "mat", m)

    def infected_neighbor_counts(self, active: np.ndarray) -> np.ndarray:
        active = active.astype(np.int8, copy=False)
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
        return np.diff(self.mat.indptr)


def _worker_step(args):
    """Worker function: compute next state for a chunk of nodes."""
    model, adj, state, t, node_ids, seed_offset = args
    # Each worker gets a unique RNG stream
    rng = np.random.default_rng(seed_offset)
    model.rng = rng
    next_state, events = model.step(adj, state, t)
    # Return only the chunk of next_state for the assigned nodes
    return next_state[node_ids], events


def run_simulation_mp(
    model,
    adj: Adjacency,
    init_state: np.ndarray,
    timesteps: int,
    seed: Optional[int] = 42,
    early_stop: int = 0,
    workers: int = 0,
):
    """Run simulation with multiprocessing backend.

    Partitions nodes across workers; each worker computes updates for a subset.
    """
    n = len(init_state)
    if workers <= 0:
        workers = mp.cpu_count()

    # Partition nodes
    chunk_size = (n + workers - 1) // workers
    partitions = [
        np.arange(i * chunk_size, min((i + 1) * chunk_size, n), dtype=int) for i in range(workers)
    ]
    partitions = [p for p in partitions if len(p) > 0]

    state = init_state.copy()
    states = [state.copy()]
    events_log = []
    stagnant = 0

    base_rng = np.random.default_rng(seed)
    for t in range(timesteps):
        # Generate per-worker seeds deterministically
        worker_seeds = base_rng.integers(0, 2**31, size=len(partitions))
        tasks = [(model, adj, state, t, part, ws) for part, ws in zip(partitions, worker_seeds)]

        with mp.Pool(processes=len(partitions)) as pool:
            results = pool.map(_worker_step, tasks)

        # Merge results
        next_state = state.copy()
        merged_events: dict = {"t": t}
        for i, (chunk_next, chunk_events) in enumerate(results):
            next_state[partitions[i]] = chunk_next
            # Merge events (simple: collect all)
            for k, v in chunk_events.items():
                if k != "t":
                    if k in merged_events:
                        merged_events[k] = np.concatenate([merged_events[k], v])
                    else:
                        merged_events[k] = v

        events_log.append(merged_events)
        states.append(next_state.copy())

        if np.array_equal(next_state, state):
            stagnant += 1
        else:
            stagnant = 0
        state = next_state
        if early_stop and stagnant >= early_stop:
            break

    from .simulator import RunResult

    return RunResult(states=states, events=events_log)
