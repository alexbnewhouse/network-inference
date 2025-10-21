from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
import pandas as pd

from .backends import to_csr_adjacency
from .models import SIModel, SISModel, SIRModel
from .simulator import run_simulation
from .analysis import compute_cascade_metrics


@dataclass
class ParameterSpace:
    """Parameter grid for inference."""

    beta_range: tuple[float, float] = (0.01, 0.5)
    gamma_range: Optional[tuple[float, float]] = None
    n_samples: int = 20
    search_mode: str = "grid"  # grid or random


@dataclass
class InferenceResult:
    """Results from parameter inference."""

    best_params: dict
    best_score: float
    all_results: pd.DataFrame


def _run_trial(model_class, adj, init_state, timesteps, seed, params, target_metric: dict) -> float:
    """Run one simulation with given parameters and compute error vs target."""
    model_instance = model_class(len(init_state), np.random.default_rng(0), **params)
    res = run_simulation(model_instance, adj, init_state, timesteps=timesteps, seed=seed)
    metrics = compute_cascade_metrics(res.states, len(init_state))
    # Loss: squared error on final_size
    target_final = target_metric.get("final_size", 0)
    error = (metrics.final_size - target_final) ** 2
    return float(error)


def infer_parameters(
    model_name: str,
    edges_df: pd.DataFrame,
    observed_cascade: dict,
    param_space: ParameterSpace,
    init_state: Optional[np.ndarray] = None,
    timesteps: int = 50,
    seed: int = 42,
    directed: bool = False,
) -> InferenceResult:
    """Infer model parameters via grid or random search.

    Args:
        model_name: "si", "sis", or "sir".
        edges_df: DataFrame with 'source' and 'target' columns.
        observed_cascade: dict with keys like 'final_size', 'initial_seeds'.
        param_space: ParameterSpace defining search ranges.
        init_state: initial state (if None, use initial_seeds from observed).
        timesteps: number of steps to simulate.
        seed: RNG seed.
        directed: whether graph is directed.

    Returns:
        InferenceResult with best params and all trials.
    """
    n = int(max(edges_df["source"].max(), edges_df["target"].max()) + 1)
    edges = edges_df[["source", "target"]].astype(int).to_numpy()
    adj = to_csr_adjacency(n, edges, directed=directed)

    if init_state is None:
        init_state = np.zeros(n, dtype=int)
        seeds = observed_cascade.get("initial_seeds", 1)
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=min(seeds, n), replace=False)
        init_state[idx] = 1

    model_map = {"si": SIModel, "sis": SISModel, "sir": SIRModel}
    model_class = model_map[model_name.lower()]

    # Generate parameter samples
    beta_min, beta_max = param_space.beta_range
    if param_space.search_mode == "grid":
        betas = np.linspace(beta_min, beta_max, param_space.n_samples)
        if param_space.gamma_range:
            gamma_min, gamma_max = param_space.gamma_range
            gammas = np.linspace(gamma_min, gamma_max, param_space.n_samples)
        else:
            gammas = [None]
    else:  # random
        rng = np.random.default_rng(seed + 1)
        betas = rng.uniform(beta_min, beta_max, param_space.n_samples)
        if param_space.gamma_range:
            gamma_min, gamma_max = param_space.gamma_range
            gammas = rng.uniform(gamma_min, gamma_max, param_space.n_samples)
        else:
            gammas = [None] * param_space.n_samples

    trials = []
    for beta in betas:
        for gamma in gammas:
            params = {"beta": beta}
            if gamma is not None:
                params["gamma"] = gamma
            score = _run_trial(model_class, adj, init_state, timesteps, seed, params, observed_cascade)
            trials.append({"beta": beta, "gamma": gamma, "score": score})

    results_df = pd.DataFrame(trials)
    best_idx = results_df["score"].idxmin()
    best_row = results_df.loc[best_idx]
    best_params = {"beta": best_row["beta"]}
    if best_row["gamma"] is not None:
        best_params["gamma"] = best_row["gamma"]
    return InferenceResult(best_params=best_params, best_score=float(best_row["score"]), all_results=results_df)
