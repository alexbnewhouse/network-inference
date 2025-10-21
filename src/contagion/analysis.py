from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CascadeMetrics:
    """Summary metrics for a contagion cascade."""

    final_size: int  # total adopted/infected at end
    peak_time: int  # timestep of maximum new adoptions
    peak_size: int  # max cumulative adopted
    total_steps: int  # number of steps simulated
    initial_seeds: int  # number of initial adopters
    adoption_rate: float  # final_size / n


def compute_cascade_metrics(states: list[np.ndarray], n: int, active_code: int = 1) -> CascadeMetrics:
    """Compute standard cascade metrics from simulation states.

    Args:
        states: List of state arrays from simulation (length T+1).
        n: Total number of nodes.
        active_code: State code for infected/adopted (default 1).
    """
    cumulative = [(s == active_code).sum() for s in states]
    new_per_step = [cumulative[i] - cumulative[i - 1] if i > 0 else cumulative[0] for i in range(len(cumulative))]
    peak_idx = int(np.argmax(new_per_step))
    return CascadeMetrics(
        final_size=int(cumulative[-1]),
        peak_time=peak_idx,
        peak_size=int(max(cumulative)),
        total_steps=len(states) - 1,
        initial_seeds=int(cumulative[0]),
        adoption_rate=float(cumulative[-1]) / n if n > 0 else 0.0,
    )


def compute_rt(states: list[np.ndarray], active_code: int = 1, window: int = 1) -> np.ndarray:
    """Compute effective reproduction number R(t) over time.

    R(t) = (new infections at t) / (infected at t-window).
    Returns array of length T (one per step after initial).
    """
    cumulative = np.array([(s == active_code).sum() for s in states], dtype=float)
    new_infections = np.diff(cumulative)
    infected_prev = cumulative[:-1]
    infected_prev = np.maximum(infected_prev, 1.0)  # avoid division by zero
    rt = new_infections / infected_prev
    return rt


def adoption_curve(states: list[np.ndarray], active_code: int = 1) -> pd.DataFrame:
    """Return a DataFrame with timestep and cumulative adoption count."""
    cumulative = [(s == active_code).sum() for s in states]
    return pd.DataFrame({"timestep": range(len(cumulative)), "adopted": cumulative})


def susceptible_infected_recovered_curves(states: list[np.ndarray]) -> pd.DataFrame:
    """Return SIR compartment counts over time (assumes S=0, I=1, R=2)."""
    timesteps = []
    S_counts = []
    I_counts = []
    R_counts = []
    for t, s in enumerate(states):
        timesteps.append(t)
        S_counts.append(int((s == 0).sum()))
        I_counts.append(int((s == 1).sum()))
        R_counts.append(int((s == 2).sum()))
    return pd.DataFrame({"timestep": timesteps, "S": S_counts, "I": I_counts, "R": R_counts})


def events_to_dataframe(events: list[dict]) -> pd.DataFrame:
    """Convert simulation events log to a flat DataFrame.

    Each row: timestep, event_type, node_id.
    """
    rows = []
    for evt in events:
        t = evt.get("t", -1)
        for key, nodes in evt.items():
            if key == "t":
                continue
            for nid in nodes:
                rows.append({"timestep": t, "event": key, "node": int(nid)})
    return pd.DataFrame(rows)
