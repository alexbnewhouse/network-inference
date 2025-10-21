from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .inference import infer_parameters, ParameterSpace


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Infer contagion model parameters from observed cascades")
    p.add_argument("edges_csv", help="Path to edges CSV")
    p.add_argument("--model", choices=["si", "sis", "sir"], required=True)
    p.add_argument("--observed-final-size", type=int, required=True, help="Observed final cascade size")
    p.add_argument("--observed-initial-seeds", type=int, default=1)
    p.add_argument("--beta-min", type=float, default=0.01)
    p.add_argument("--beta-max", type=float, default=0.5)
    p.add_argument("--gamma-min", type=float)
    p.add_argument("--gamma-max", type=float)
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--search-mode", choices=["grid", "random"], default="grid")
    p.add_argument("--timesteps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--directed", action="store_true")
    p.add_argument("--source-col", type=str, default="source")
    p.add_argument("--target-col", type=str, default="target")
    p.add_argument("--output-dir", type=str, help="Directory to save results")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Read edges
    df = pd.read_csv(args.edges_csv)
    # Rename columns to standard names if needed
    if args.source_col != "source" or args.target_col != "target":
        df = df.rename(columns={args.source_col: "source", args.target_col: "target"})

    observed = {"final_size": args.observed_final_size, "initial_seeds": args.observed_initial_seeds}
    gamma_range = None
    if args.gamma_min is not None and args.gamma_max is not None:
        gamma_range = (args.gamma_min, args.gamma_max)

    param_space = ParameterSpace(
        beta_range=(args.beta_min, args.beta_max),
        gamma_range=gamma_range,
        n_samples=args.n_samples,
        search_mode=args.search_mode,
    )

    result = infer_parameters(
        model_name=args.model,
        edges_df=df,
        observed_cascade=observed,
        param_space=param_space,
        timesteps=args.timesteps,
        seed=args.seed,
        directed=args.directed,
    )

    output = {
        "model": args.model,
        "best_params": result.best_params,
        "best_score": result.best_score,
        "n_trials": len(result.all_results),
    }
    print(json.dumps(output, indent=2))

    if args.output_dir:
        out_path = Path(args.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        result.all_results.to_csv(out_path / "trials.csv", index=False)
        with open(out_path / "best_params.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
