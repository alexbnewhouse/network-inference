"""
CLI for running the knowledge graph extraction pipeline.
"""
import argparse
import pandas as pd
import os
from .kg_pipeline import KnowledgeGraphPipeline

def main():
    ap = argparse.ArgumentParser(description="Run knowledge graph extraction pipeline")
    ap.add_argument("--input", required=True, help="Input CSV file containing text data")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to load")
    ap.add_argument("--model", default="en_core_web_sm", help="spaCy model name or path (default: en_core_web_sm)")
    ap.add_argument("--text-col", default="text", help="Column name for text content (default: text)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df_raw = pd.read_csv(args.input, nrows=args.max_rows)
    if args.text_col not in df_raw.columns:
        raise ValueError(f"Text column '{args.text_col}' not found in input CSV. Available: {list(df_raw.columns)}")
    df = pd.DataFrame({"text": df_raw[args.text_col].fillna("").astype(str)})
    kg = KnowledgeGraphPipeline(ner_model=args.model)
    kg.run(df, args.outdir)

if __name__ == "__main__":
    main()
