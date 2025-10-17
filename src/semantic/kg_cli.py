"""
CLI for running the knowledge graph extraction pipeline.
"""
import argparse
import pandas as pd
import os
from .kg_pipeline import KnowledgeGraphPipeline

def main():
    ap = argparse.ArgumentParser(description="Run knowledge graph extraction pipeline")
    ap.add_argument("--input", required=True, help="Input CSV file with 'text' column")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to load")
    ap.add_argument("--model", default="en_core_web_sm", help="spaCy model name or path (default: en_core_web_sm)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input, nrows=args.max_rows)
    kg = KnowledgeGraphPipeline(ner_model=args.model)
    kg.run(df, args.outdir)

if __name__ == "__main__":
    main()
