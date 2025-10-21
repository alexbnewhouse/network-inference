
"""
Simple Contagion CLI (SI/SIS/SIR)
==================================

Run simple contagion simulations from an edge list CSV.

Example usage:
    python -m src.contagion.cli edges.csv --model sir --beta 0.2 --gamma 0.1 --timesteps 100 --initial-frac 0.05

Arguments:
    edges.csv           Path to edge list CSV (columns: source,target or as specified)
    --model             Contagion model: si, sis, or sir (default: si)
    --beta              Infection rate (default: 0.05)
    --gamma             Recovery rate (SIS/SIR only)
    --timesteps         Number of time steps (default: 50)
    --initial-frac      Fraction of initially infected nodes (default: 1 random node)
    --patient-zero      Index of initial infected node (overrides --initial-frac)
    --source-col        Source column name (default: source)
    --target-col        Target column name (default: target)
    --directed          Treat edges as directed
    --seed              Random seed
    --early-stop        Stop if no new infections for N steps

Outputs:
    Prints summary (timesteps, final infected count) to stdout.
    For more detailed output, modify the script or use the analysis utilities.
"""

import argparse
from dataclasses import dataclass
from typing import Optional
import sys

import numpy as np
import pandas as pd
import os
import json

from .backends import to_csr_adjacency
from .models import SIModel, SISModel, SIRModel
from .simulator import run_simulation
from .config_loader import add_config_argument, merge_config_with_args


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
    """Run contagion simulation from edges DataFrame with validation."""
    # Validate DataFrame
    if df.empty:
        raise ValueError(
            "Empty edge list provided. Please ensure your CSV file contains edge data.\n"
            "Expected format: CSV with 'source' and 'target' columns (or specify with --source-col/--target-col)"
        )
    
    # Check for required columns
    for col in (cfg.source_col, cfg.target_col):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in edges CSV.\n"
                f"Available columns: {list(df.columns)}\n"
                f"Tip: Use --source-col and --target-col to specify column names"
            )
    
    # Validate model-specific parameters
    if cfg.model.lower() in ("sis", "sir") and cfg.gamma is None:
        raise ValueError(
            f"Model '{cfg.model.upper()}' requires --gamma parameter (recovery rate).\n"
            f"Example: --gamma 0.05\n"
            f"Tip: gamma should be between 0.01 and 0.5 for typical spread dynamics"
        )
    
    # Validate parameter ranges
    if not (0 < cfg.beta <= 1.0):
        raise ValueError(
            f"Invalid beta value: {cfg.beta}. Must be between 0 and 1.\n"
            f"Tip: Try values between 0.01 (slow spread) and 0.5 (fast spread)"
        )
    
    if cfg.gamma is not None and not (0 < cfg.gamma <= 1.0):
        raise ValueError(
            f"Invalid gamma value: {cfg.gamma}. Must be between 0 and 1.\n"
            f"Tip: Try values between 0.01 (slow recovery) and 0.3 (fast recovery)"
        )
    
    edges, n = _coerce_edges_to_int(df, cfg.source_col, cfg.target_col)
    
    if n == 0:
        raise ValueError(
            "No valid nodes found in edge list. Please check your data format."
        )
    
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
        if not (0 <= cfg.patient_zero < n):
            raise ValueError(
                f"Invalid patient_zero: {cfg.patient_zero}. Must be between 0 and {n-1} (network has {n} nodes)"
            )
        init[int(cfg.patient_zero)] = 1
    elif cfg.initial_frac:
        if not (0 < cfg.initial_frac <= 1.0):
            raise ValueError(
                f"Invalid initial_frac: {cfg.initial_frac}. Must be between 0 and 1"
            )
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
    p = argparse.ArgumentParser(
        description="Run simple contagion simulations (SI/SIS/SIR)",
        epilog="""
Example:
    python -m src.contagion.cli edges.csv --model sir --beta 0.2 --gamma 0.1 --timesteps 100 --initial-frac 0.05
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("edges_csv", help="Path to edge list CSV (columns: source,target or as specified)")
    p.add_argument("--model", choices=["si", "sis", "sir"], default="si", help="Contagion model (default: si)")
    p.add_argument("--beta", type=float, default=0.05, help="Infection rate (default: 0.05)")
    p.add_argument("--gamma", type=float, help="Recovery rate (SIS/SIR only)")
    p.add_argument("--timesteps", type=int, default=50, help="Number of time steps (default: 50)")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--early-stop", type=int, default=0, help="Stop if no new infections for N steps")
    p.add_argument("--directed", action="store_true", help="Treat edges as directed")
    p.add_argument("--patient-zero", type=int, help="Index of initial infected node (overrides --initial-frac)")
    p.add_argument("--initial-frac", type=float, help="Fraction of initially infected nodes (default: 1 random node)")
    p.add_argument("--source-col", type=str, default="source", help="Source column name (default: source)")
    p.add_argument("--target-col", type=str, default="target", help="Target column name (default: target)")
    p.add_argument("--output-format", choices=["csv", "json", "parquet"], default="csv", help="Format for saving results (default: csv)")
    p.add_argument("--output-path", type=str, help="Path to save results (if not provided, prints summary)")
    add_config_argument(p)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    
    # Merge config file if provided
    args = merge_config_with_args(parser, args, getattr(args, 'config', None))
    
    try:
        df = pd.read_csv(args.edges_csv)
    except FileNotFoundError:
        print(f"Error: Edge file not found: {args.edges_csv}", file=sys.stderr)
        print(f"Please ensure the file exists and the path is correct.", file=sys.stderr)
        return 1
    except pd.errors.EmptyDataError:
        print(f"Error: Edge file is empty: {args.edges_csv}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error reading edge file: {e}", file=sys.stderr)
        return 1
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
    
    try:
        print("Running simulation...")
        res = simulate_from_edges_df(df, cfg)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error during simulation: {e}", file=sys.stderr)
        return 1
    
    infected_counts = [int((s == 1).sum()) for s in res.states]
    summary = {"timesteps": len(res.states) - 1, "final_infected": infected_counts[-1]}
    if args.output_path:
        df_out = pd.DataFrame({"timestep": list(range(len(infected_counts))), "infected": infected_counts})
        ext = args.output_format
        out_path = args.output_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
        if ext == "csv":
            df_out.to_csv(out_path, index=False)
        elif ext == "json":
            df_out.to_json(out_path, orient="records", indent=2)
        elif ext == "parquet":
            df_out.to_parquet(out_path, index=False)
        print(f"Results saved to {out_path} ({ext.upper()})")
    else:
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
