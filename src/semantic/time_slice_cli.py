"""
CLI for running time-sliced semantic network pipeline.
"""
import argparse
import pandas as pd
import os
from .time_slice import TimeSlicedSemanticPipeline
from .build_semantic_network import build_semantic_from_df

def main():
    ap = argparse.ArgumentParser(description="Run time-sliced semantic network pipeline")
    ap.add_argument("--input", required=True, help="Input CSV file with timestamp column")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--slice-col", type=str, default="timestamp", help="Column to slice on (default: timestamp)")
    ap.add_argument("--freq", type=str, default="M", help="Slice frequency: M=month, W=week")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to load")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input, nrows=args.max_rows)
    pipeline = TimeSlicedSemanticPipeline(slice_col=args.slice_col, freq=args.freq)
    # For each slice, build semantic network and write CSV outputs
    def build_fn(group, outdir):
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
    pipeline.run(df, build_fn, args.outdir)

if __name__ == "__main__":
    main()
