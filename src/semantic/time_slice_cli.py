"""
Time-Sliced Semantic Network CLI
===============================

Build semantic networks for each time slice (e.g., month, week) in your data.

Example usage:
    python -m src.semantic.time_slice_cli --input data.csv --outdir output/ --slice-col timestamp --freq M

Arguments:
    --input      Input CSV file with timestamp column
    --outdir     Output directory
    --slice-col  Column to slice on (default: timestamp)
    --freq       Slice frequency: M=month, W=week (default: M)
    --max-rows   Limit number of rows to load (optional)

Outputs:
    For each time slice, semantic network CSVs are written to the output directory.
"""
import argparse
import pandas as pd
import os
from .time_slice import TimeSlicedSemanticPipeline
from .build_semantic_network import build_semantic_from_df

def main():
    ap = argparse.ArgumentParser(
        description="Run time-sliced semantic network pipeline",
        epilog="""
Example:
    python -m src.semantic.time_slice_cli --input data.csv --outdir output/ --slice-col timestamp --freq M
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Input CSV file with timestamp column")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--slice-col", type=str, default="timestamp", help="Column to slice on (default: timestamp)")
    ap.add_argument("--freq", type=str, default="M", help="Slice frequency: M=month, W=week (default: M)")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to load (optional)")
    args = ap.parse_args()

    import os
    import pandas as pd
    print("Loading input data...")
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input, nrows=args.max_rows)
    pipeline = TimeSlicedSemanticPipeline(slice_col=args.slice_col, freq=args.freq)
    ext = getattr(args, "output_format", "csv") if hasattr(args, "output_format") else "csv"
    def build_fn(group, outdir):
        # Save each slice in the selected format
        base = os.path.join(outdir, f"slice_{group[args.slice_col].iloc[0]}")
        build_semantic_from_df(
            group,
            outdir,
            min_df=5,
            max_vocab=None,
            window=10,
            topk=20,
            cds=0.75,
            use_gpu=False,
            use_igraph=False,
            spacy_gpu=False,
        )
        # Optionally convert CSVs to other formats
        nodes_csv = os.path.join(outdir, "nodes.csv")
        edges_csv = os.path.join(outdir, "edges.csv")
        if ext != "csv":
            if os.path.exists(nodes_csv):
                df_nodes = pd.read_csv(nodes_csv)
                if ext == "json":
                    df_nodes.to_json(base + "_nodes.json", orient="records", indent=2)
                elif ext == "parquet":
                    df_nodes.to_parquet(base + "_nodes.parquet", index=False)
            if os.path.exists(edges_csv):
                df_edges = pd.read_csv(edges_csv)
                if ext == "json":
                    df_edges.to_json(base + "_edges.json", orient="records", indent=2)
                elif ext == "parquet":
                    df_edges.to_parquet(base + "_edges.parquet", index=False)
    print("Running time-sliced pipeline...")
    pipeline.run(df, build_fn, args.outdir)
    print("Time-sliced outputs saved in", args.outdir)

if __name__ == "__main__":
    main()
