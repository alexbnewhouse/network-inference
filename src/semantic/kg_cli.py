"""
Knowledge Graph Extraction CLI
=============================

Extract a knowledge graph from text data using spaCy NER.

Example usage:
    python -m src.semantic.kg_cli --input data.csv --outdir output/ --model en_core_web_sm

Arguments:
    --input      Input CSV file containing text data
    --outdir     Output directory
    --max-rows   Limit number of rows to load (optional)
    --model      spaCy model name or path (default: en_core_web_sm)
    --text-col   Column name for text content (default: text)

Outputs:
    Knowledge graph files in the output directory.
"""
import argparse
import pandas as pd
import os
from .kg_pipeline import KnowledgeGraphPipeline

def main():
    ap = argparse.ArgumentParser(
        description="Run knowledge graph extraction pipeline",
        epilog="""
Example:
    python -m src.semantic.kg_cli --input data.csv --outdir output/ --model en_core_web_sm
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Input CSV file containing text data")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to load")
    ap.add_argument("--model", default="en_core_web_sm", help="spaCy model name or path (default: en_core_web_sm)")
    ap.add_argument("--text-col", default="text", help="Column name for text content (default: text)")
    args = ap.parse_args()

    import os
    import pandas as pd
    print("Loading input data...")
    os.makedirs(args.outdir, exist_ok=True)
    df_raw = pd.read_csv(args.input, nrows=args.max_rows)
    if args.text_col not in df_raw.columns:
        raise ValueError(f"Text column '{args.text_col}' not found in input CSV. Available: {list(df_raw.columns)}")
    df = pd.DataFrame({"text": df_raw[args.text_col].fillna("").astype(str)})
    kg = KnowledgeGraphPipeline(ner_model=args.model)
    print("Extracting knowledge graph...")
    kg.run(df, args.outdir)
    # Optionally convert CSVs to other formats
    ext = getattr(args, "output_format", "csv") if hasattr(args, "output_format") else "csv"
    base = os.path.join(args.outdir, "kg")
    nodes_csv = os.path.join(args.outdir, "nodes.csv")
    edges_csv = os.path.join(args.outdir, "edges.csv")
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
    print(f"Knowledge graph outputs saved in {args.outdir}")

if __name__ == "__main__":
    main()
