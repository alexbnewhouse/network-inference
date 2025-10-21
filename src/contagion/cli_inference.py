
"""
Contagion Parameter Inference CLI
=================================

Infer SI/SIS/SIR model parameters from observed cascade data.

Example usage:
    python -m src.contagion.cli_inference edges.csv --model sir --observed-final-size 100 --beta-min 0.01 --beta-max 0.5 --gamma-min 0.01 --gamma-max 0.5 --n-samples 30 --search-mode random

Arguments:
    edges.csv               Path to edge list CSV (columns: source,target or as specified)
    --model                 Model type: si, sis, or sir (required)
    --observed-final-size   Observed final cascade size (required)
    --observed-initial-seeds Number of initial seeds in observed cascade (default: 1)
    --beta-min, --beta-max  Range for beta (default: 0.01, 0.5)
    --gamma-min, --gamma-max Range for gamma (SIS/SIR only)
    --n-samples             Number of parameter samples (default: 20)
    --search-mode           Parameter search mode: grid or random (default: grid)
    --timesteps             Number of time steps (default: 50)
    --seed                  Random seed (default: 42)
    --directed              Treat edges as directed
    --source-col            Source column name (default: source)
    --target-col            Target column name (default: target)
    --output-dir            Directory to save results

Outputs:
    Prints best-fit parameters and summary to stdout.
    For more detailed output, use --output-dir.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .inference import infer_parameters, ParameterSpace
from .config_loader import add_config_argument, merge_config_with_args


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Infer contagion model parameters from observed cascades",
        epilog="""
Example:
    python -m src.contagion.cli_inference edges.csv --model sir --observed-final-size 100 --beta-min 0.01 --beta-max 0.5 --gamma-min 0.01 --gamma-max 0.5 --n-samples 30 --search-mode random
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("edges_csv", help="Path to edge list CSV (columns: source,target or as specified)")
    p.add_argument("--model", choices=["si", "sis", "sir"], required=True, help="Model type (required)")
    p.add_argument("--observed-final-size", type=int, required=True, help="Observed final cascade size (required)")
    p.add_argument("--observed-initial-seeds", type=int, default=1, help="Number of initial seeds in observed cascade (default: 1)")
    p.add_argument("--beta-min", type=float, default=0.01, help="Minimum beta value (default: 0.01)")
    p.add_argument("--beta-max", type=float, default=0.5, help="Maximum beta value (default: 0.5)")
    p.add_argument("--gamma-min", type=float, help="Minimum gamma value (SIS/SIR only)")
    p.add_argument("--gamma-max", type=float, help="Maximum gamma value (SIS/SIR only)")
    p.add_argument("--n-samples", type=int, default=20, help="Number of parameter samples (default: 20)")
    p.add_argument("--search-mode", choices=["grid", "random"], default="grid", help="Parameter search mode (default: grid)")
    p.add_argument("--timesteps", type=int, default=50, help="Number of time steps (default: 50)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--directed", action="store_true", help="Treat edges as directed")
    p.add_argument("--source-col", type=str, default="source", help="Source column name (default: source)")
    p.add_argument("--target-col", type=str, default="target", help="Target column name (default: target)")
    p.add_argument("--output-dir", type=str, help="Directory to save results")
    p.add_argument("--output-format", choices=["csv", "json", "parquet"], default="csv", help="Format for saving results (default: csv)")
    p.add_argument("--output-path", type=str, help="Path to save results (if not provided, prints summary)")
    add_config_argument(p)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    
    # Merge config file if provided
    args = merge_config_with_args(parser, args, getattr(args, 'config', None))

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
    print("Parameter inference complete.")
    if getattr(args, "output_path", None):
        ext = getattr(args, "output_format", "csv")
        out_path = args.output_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
        if ext == "csv":
            result.all_results.to_csv(out_path + "_trials.csv", index=False)
            pd.DataFrame([output]).to_csv(out_path + "_best_params.csv", index=False)
        elif ext == "json":
            result.all_results.to_json(out_path + "_trials.json", orient="records", indent=2)
            with open(out_path + "_best_params.json", "w") as f:
                json.dump(output, f, indent=2)
        elif ext == "parquet":
            result.all_results.to_parquet(out_path + "_trials.parquet", index=False)
            pd.DataFrame([output]).to_parquet(out_path + "_best_params.parquet", index=False)
        print(f"Results saved to {out_path}_trials.{ext} and {out_path}_best_params.{ext}")
    elif getattr(args, "output_dir", None):
        out_path = Path(args.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        result.all_results.to_csv(out_path / "trials.csv", index=False)
        with open(out_path / "best_params.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {out_path}")
    else:
        print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
