from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .backends import to_csr_adjacency
from .models import SIModel, SISModel, SIRModel
from .simulator import run_simulation


@dataclass
class SimpleSimConfig:
    model: str
    beta: float
    gamma: Optional[float]
    timesteps: int
    seed: Optional[int]
    early_stop: int
    directed: bool
    patient_zero: Optional[int]
    initial_frac: Optional[float]
    source_col: str = "source"
    target_col: str = "target"


def _coerce_edges_to_int(df: pd.DataFrame, src_col: str, dst_col: str) -> tuple[np.ndarray, int]:
    # If columns are already ints, cast; otherwise map to categorical codes.
    src = df[src_col]
    dst = df[dst_col]
    if not (pd.api.types.is_integer_dtype(src) and pd.api.types.is_integer_dtype(dst)):
        # Build a joint categorical for consistent mapping
        all_vals = pd.Categorical(pd.concat([src, dst], ignore_index=True))
        codes = all_vals.codes
        src_codes = codes[: len(src)]
        dst_codes = codes[len(src) :]
        edges = np.stack([src_codes, dst_codes], axis=1).astype(int)
        n = int(codes.max() + 1) if codes.size else 0
        return edges, n
    edges = df[[src_col, dst_col]].astype(int).to_numpy()
    n = int(max(df[src_col].max(), df[dst_col].max()) + 1) if len(df) else 0
    return edges, n


def simulate_from_edges_df(df: pd.DataFrame, cfg: SimpleSimConfig):
    # Flexible columns
    for col in (cfg.source_col, cfg.target_col):
        if col not in df.columns:
            raise ValueError(f"edges DataFrame must contain '{cfg.source_col}' and '{cfg.target_col}' columns")
    edges, n = _coerce_edges_to_int(df, cfg.source_col, cfg.target_col)
    adj = to_csr_adjacency(n, edges, directed=cfg.directed)

    rng = np.random.default_rng(cfg.seed)
    if cfg.model.lower() == "si":
        model = SIModel(n, rng, beta=cfg.beta)
    elif cfg.model.lower() == "sis":
        model = SISModel(n, rng, beta=cfg.beta, gamma=float(cfg.gamma or 0.1))
    elif cfg.model.lower() == "sir":
        model = SIRModel(n, rng, beta=cfg.beta, gamma=float(cfg.gamma or 0.1))
    else:
        raise ValueError(f"Unknown model: {cfg.model}")

    init = np.zeros(n, dtype=int)
    if cfg.patient_zero is not None:
        init[int(cfg.patient_zero)] = 1
    elif cfg.initial_frac:
        k = max(1, int(round(n * cfg.initial_frac)))
        idx = rng.choice(n, size=k, replace=False)
        init[idx] = 1
    else:
        # default to one random seed
        if n > 0:
            init[rng.integers(0, n)] = 1

    res = run_simulation(
        model,
        adj,
        init,
        timesteps=cfg.timesteps,
        seed=cfg.seed,
        early_stop=cfg.early_stop,
    )
    return res


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run simple contagion simulations (SI/SIS/SIR)")
    p.add_argument("edges_csv", help="Path to edges CSV")
    p.add_argument("--model", choices=["si", "sis", "sir"], default="si")
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--gamma", type=float)
    p.add_argument("--timesteps", type=int, default=50)
    p.add_argument("--seed", type=int)
    p.add_argument("--early-stop", type=int, default=0)
    p.add_argument("--directed", action="store_true")
    p.add_argument("--patient-zero", type=int)
    p.add_argument("--initial-frac", type=float)
    p.add_argument("--source-col", type=str, default="source")
    p.add_argument("--target-col", type=str, default="target")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    df = pd.read_csv(args.edges_csv)
    cfg = SimpleSimConfig(
        model=args.model,
        beta=args.beta,
        gamma=args.gamma,
        timesteps=args.timesteps,
        seed=args.seed,
        early_stop=args.early_stop,
        directed=bool(args.directed),
        patient_zero=args.patient_zero,
        initial_frac=args.initial_frac,
        source_col=args.source_col,
        target_col=args.target_col,
    )
    res = simulate_from_edges_df(df, cfg)
    infected_counts = [int((s == 1).sum()) for s in res.states]
    print({"timesteps": len(res.states) - 1, "final_infected": infected_counts[-1]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
