from __future__ import annotations

"""
Contagion models: base interface and concrete models (SI, SIS, SIR, Watts, KReinforcement, MultiStage).
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from .types import Adjacency


@dataclass
class SimulationConfig:
    timesteps: int = 100
    directed: bool = False
    seed: Optional[int] = 42
    backend: str = "cpu"  # cpu|mp|gpu (gpu optional)
    workers: int = 0      # for mp backend; 0=auto
    log_sources: str = "one"  # none|one|all
    early_stop: int = 0   # 0=off; otherwise stop after N steps with no change


class ContagionModel:
    """Base contagion model.

    Contract:
    - state: integer vector of length N with model-specific codes (e.g., 0=S,1=I,2=R)
    - step(): compute next state and emit events
    - run(): iterate step() T times with deterministic RNG
    """

    def __init__(self, n_nodes: int, rng: np.random.Generator):
        self.n = n_nodes
        self.rng = rng

    def step(self, adj: Adjacency, state: np.ndarray, t: int) -> Tuple[np.ndarray, Dict]:
        raise NotImplementedError


class SIModel(ContagionModel):
    def __init__(self, n_nodes: int, rng: np.random.Generator, beta: float = 0.05, threshold: Optional[float] = None):
        super().__init__(n_nodes, rng)
        self.beta = float(beta)
        self.threshold = threshold  # fraction or count depending on adj helper

    def step(self, adj: Adjacency, state: np.ndarray, t: int) -> Tuple[np.ndarray, Dict]:
        S = (state == 0)
        I = (state == 1)
        # exposures: number of infected neighbors
        inf_neighbors = adj.infected_neighbor_counts(I)
        if self.threshold is not None:
            exposed = adj.threshold_met(inf_neighbors, self.threshold)
        else:
            exposed = inf_neighbors > 0
        # infection probability per susceptible node with infected neighbors
        p = 1 - (1 - self.beta) ** inf_neighbors
        infect_now = S & exposed & (self.rng.random(self.n) < p)
        next_state = state.copy()
        next_state[infect_now] = 1
        events = {
            "t": t,
            "infected": np.flatnonzero(infect_now),
        }
        return next_state, events


class SISModel(ContagionModel):
    def __init__(self, n_nodes: int, rng: np.random.Generator, beta: float = 0.05, gamma: float = 0.1, threshold: Optional[float] = None):
        super().__init__(n_nodes, rng)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.threshold = threshold

    def step(self, adj: Adjacency, state: np.ndarray, t: int) -> Tuple[np.ndarray, Dict]:
        S = (state == 0)
        I = (state == 1)
        inf_neighbors = adj.infected_neighbor_counts(I)
        if self.threshold is not None:
            exposed = adj.threshold_met(inf_neighbors, self.threshold)
        else:
            exposed = inf_neighbors > 0
        p_inf = 1 - (1 - self.beta) ** inf_neighbors
        infect_now = S & exposed & (self.rng.random(self.n) < p_inf)
        recover_now = I & (self.rng.random(self.n) < self.gamma)
        next_state = state.copy()
        next_state[infect_now] = 1
        next_state[recover_now] = 0
        events = {"t": t, "infected": np.flatnonzero(infect_now), "recovered": np.flatnonzero(recover_now)}
        return next_state, events


class SIRModel(ContagionModel):
    def __init__(self, n_nodes: int, rng: np.random.Generator, beta: float = 0.05, gamma: float = 0.1, threshold: Optional[float] = None):
        super().__init__(n_nodes, rng)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.threshold = threshold

    def step(self, adj: Adjacency, state: np.ndarray, t: int) -> Tuple[np.ndarray, Dict]:
        S = (state == 0)
        I = (state == 1)
        R = (state == 2)
        inf_neighbors = adj.infected_neighbor_counts(I)
        if self.threshold is not None:
            exposed = adj.threshold_met(inf_neighbors, self.threshold)
        else:
            exposed = inf_neighbors > 0
        p_inf = 1 - (1 - self.beta) ** inf_neighbors
        infect_now = S & exposed & (self.rng.random(self.n) < p_inf)
        recover_now = I & (self.rng.random(self.n) < self.gamma)
        next_state = state.copy()
        next_state[infect_now] = 1
        next_state[recover_now] = 2
        events = {"t": t, "infected": np.flatnonzero(infect_now), "recovered": np.flatnonzero(recover_now)}
        return next_state, events


class WattsThresholdModel(ContagionModel):
    def __init__(self, n_nodes: int, rng: np.random.Generator, phi: float = 0.18):
        super().__init__(n_nodes, rng)
        self.phi = float(phi)

    def step(self, adj: Adjacency, state: np.ndarray, t: int) -> Tuple[np.ndarray, Dict]:
        # 0 = not adopted, 1 = adopted (monotone)
        adopted = (state == 1)
        inf_neighbors = adj.infected_neighbor_counts(adopted)
        deg = adj.degrees()
        frac = np.divide(inf_neighbors, np.maximum(deg, 1), dtype=float)
        adopt_now = (~adopted) & (frac >= self.phi)
        next_state = state.copy()
        next_state[adopt_now] = 1
        events = {"t": t, "adopted": np.flatnonzero(adopt_now)}
        return next_state, events


class KReinforcementModel(ContagionModel):
    def __init__(self, n_nodes: int, rng: np.random.Generator, k: int = 2):
        super().__init__(n_nodes, rng)
        self.k = int(k)

    def step(self, adj: Adjacency, state: np.ndarray, t: int) -> Tuple[np.ndarray, Dict]:
        adopted = (state == 1)
        inf_neighbors = adj.infected_neighbor_counts(adopted)
        adopt_now = (~adopted) & (inf_neighbors >= self.k)
        next_state = state.copy()
        next_state[adopt_now] = 1
        events = {"t": t, "adopted": np.flatnonzero(adopt_now)}
        return next_state, events
