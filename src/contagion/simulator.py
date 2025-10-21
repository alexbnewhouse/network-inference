from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import Adjacency


@dataclass
class RunResult:
    states: List[np.ndarray]
    events: List[Dict]


def run_simulation(model, adj: Adjacency, init_state: np.ndarray, timesteps: int, seed: Optional[int] = 42, early_stop: int = 0) -> RunResult:
    rng = np.random.default_rng(seed)
    # attach deterministic rng to model for reproducibility
    model.rng = rng
    state = init_state.copy()
    states = [state.copy()]
    events_log: List[Dict] = []
    stagnant = 0
    for t in range(timesteps):
        next_state, events = model.step(adj, state, t)
        events_log.append(events)
        states.append(next_state.copy())
        if np.array_equal(next_state, state):
            stagnant += 1
        else:
            stagnant = 0
        state = next_state
        if early_stop and stagnant >= early_stop:
            break
    return RunResult(states=states, events=events_log)
