from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .backends import to_csr_adjacency
from .models import WattsThresholdModel, KReinforcementModel
from .simulator import run_simulation
from .analysis import compute_cascade_metrics, events_to_dataframe


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run complex contagion simulations (Watts/KReinforcement)")
    p.add_argument("edges_csv", help="Path to edges CSV")
    p.add_argument("--model", choices=["watts", "k"], required=True)
    p.add_argument("--phi", type=float, help="Threshold for Watts model (fraction of neighbors)")
    p.add_argument("--k", type=int, help="Count threshold for K-reinforcement model")
    p.add_argument("--timesteps", type=int, default=50)
    p.add_argument("--seed", type=int)
    p.add_argument("--early-stop", type=int, default=0)
    p.add_argument("--directed", action="store_true")
    p.add_argument("--patient-zero", type=int)
    p.add_argument("--initial-frac", type=float)
    p.add_argument("--source-col", type=str, default="source")
    p.add_argument("--target-col", type=str, default="target")
    p.add_argument("--output-dir", type=str, help="Directory to save outputs (events, summary)")
    return p


def _coerce_edges_to_int(df: pd.DataFrame, src_col: str, dst_col: str) -> tuple[np.ndarray, int]:
    src = df[src_col]
    dst = df[dst_col]
    if not (pd.api.types.is_integer_dtype(src) and pd.api.types.is_integer_dtype(dst)):
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    df = pd.read_csv(args.edges_csv)

    edges, n = _coerce_edges_to_int(df, args.source_col, args.target_col)
    adj = to_csr_adjacency(n, edges, directed=args.directed)

    rng = np.random.default_rng(args.seed)
    if args.model == "watts":
        phi = args.phi if args.phi is not None else 0.18
        model = WattsThresholdModel(n, rng, phi=phi)
    elif args.model == "k":
        k_val = args.k if args.k is not None else 2
        model = KReinforcementModel(n, rng, k=k_val)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    init = np.zeros(n, dtype=int)
    if args.patient_zero is not None:
        init[int(args.patient_zero)] = 1
    elif args.initial_frac:
        k_init = max(1, int(round(n * args.initial_frac)))
        idx = rng.choice(n, size=k_init, replace=False)
        init[idx] = 1
    else:
        if n > 0:
            init[rng.integers(0, n)] = 1

    res = run_simulation(model, adj, init, timesteps=args.timesteps, seed=args.seed, early_stop=args.early_stop)
    metrics = compute_cascade_metrics(res.states, n)

    summary = {
        "model": args.model,
        "phi": args.phi,
        "k": args.k,
        "timesteps": len(res.states) - 1,
        "final_size": metrics.final_size,
        "peak_time": metrics.peak_time,
        "adoption_rate": metrics.adoption_rate,
    }
    print(json.dumps(summary, indent=2))

    if args.output_dir:
        out_path = Path(args.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        events_df = events_to_dataframe(res.events)
        events_df.to_csv(out_path / "events.csv", index=False)
        with open(out_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Outputs saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
