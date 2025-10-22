"""
Knowledge Graph Extraction CLI
=============================

Extract a knowledge graph from text data using enhanced spaCy NER with:
- Entity normalization and filtering
- Entity co-occurrence detection
- Dependency-based relation extraction

Example usage:
    # Basic usage with small model
    python -m src.semantic.kg_cli --input data.csv --outdir output/ --model en_core_web_sm
    
    # Better quality with medium model
    python -m src.semantic.kg_cli --input data.csv --outdir output/ --model en_core_web_md --min-freq 3
    
    # Sentence-level co-occurrence
    python -m src.semantic.kg_cli --input data.csv --outdir output/ --window 0

Arguments:
    --input           Input CSV file containing text data
    --outdir          Output directory
    --max-rows        Limit number of rows to load (optional)
    --model           spaCy model (recommend en_core_web_md or en_core_web_lg for better quality)
    --text-col        Column name for text content (default: text)
    --min-freq        Minimum entity frequency to include (default: 2)
    --window          Character window for co-occurrence (default: 100, use 0 for sentence-level)
    --no-dependencies Disable dependency parsing (faster but fewer relations)

Outputs:
    kg_nodes.csv - Entity nodes with type and frequency
    kg_edges.csv - Entity relationships with weights
"""
import argparse
import pandas as pd
import os
from .kg_pipeline import KnowledgeGraphPipeline

def main():
    ap = argparse.ArgumentParser(
        description="Run knowledge graph extraction pipeline with enhanced NER",
        epilog="""
Example:
    python -m src.semantic.kg_cli --input data.csv --outdir output/ --model en_core_web_md --min-freq 3
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Input CSV file containing text data")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to load")
    ap.add_argument("--model", default="en_core_web_sm", 
                    help="spaCy model name (recommend en_core_web_md or en_core_web_lg for better quality)")
    ap.add_argument("--text-col", default="text", help="Column name for text content (default: text)")
    ap.add_argument("--min-freq", type=int, default=2, 
                    help="Minimum entity frequency to include in graph (default: 2)")
    ap.add_argument("--window", type=int, default=100, 
                    help="Character window for entity co-occurrence (default: 100, use 0 for sentence-level)")
    ap.add_argument("--no-dependencies", action="store_true", 
                    help="Disable dependency-based relation extraction (faster but less relations)")
    ap.add_argument("--time-col", default=None,
                    help="Column name for timestamps (enables temporal grouping)")
    ap.add_argument("--group-by-time", default=None, choices=['hour', 'daily', 'weekly', 'monthly'],
                    help="Group by time period (requires --time-col)")
    ap.add_argument("--add-sentiment", action="store_true",
                    help="Add sentiment analysis to entities and edges (requires vaderSentiment)")
    args = ap.parse_args()

    import os
    import pandas as pd
    print("Loading input data...")
    os.makedirs(args.outdir, exist_ok=True)
    df_raw = pd.read_csv(args.input, nrows=args.max_rows)
    if args.text_col not in df_raw.columns:
        raise ValueError(f"Text column '{args.text_col}' not found in input CSV. Available: {list(df_raw.columns)}")
    
    # Check for temporal grouping
    if args.group_by_time and not args.time_col:
        raise ValueError("--time-col is required when using --group-by-time")
    
    if args.time_col and args.time_col not in df_raw.columns:
        raise ValueError(f"Time column '{args.time_col}' not found in input CSV. Available: {list(df_raw.columns)}")
    
    # Initialize pipeline with enhanced parameters
    window = None if args.window == 0 else args.window
    kg = KnowledgeGraphPipeline(
        ner_model=args.model,
        min_entity_freq=args.min_freq,
        cooccurrence_window=window,
        use_dependencies=not args.no_dependencies
    )
    
    # Handle temporal grouping
    if args.group_by_time:
        print(f"Grouping by {args.group_by_time}...")
        df_raw['timestamp'] = pd.to_datetime(df_raw[args.time_col], errors='coerce')
        df_raw = df_raw.dropna(subset=['timestamp'])
        
        # Create time groups
        if args.group_by_time == 'hour':
            df_raw['time_group'] = df_raw['timestamp'].dt.strftime('%Y-%m-%d_%H')
        elif args.group_by_time == 'daily':
            df_raw['time_group'] = df_raw['timestamp'].dt.strftime('%Y-%m-%d')
        elif args.group_by_time == 'weekly':
            df_raw['time_group'] = df_raw['timestamp'].dt.to_period('W').astype(str)
        elif args.group_by_time == 'monthly':
            df_raw['time_group'] = df_raw['timestamp'].dt.strftime('%Y-%m')
        
        # Process each time group
        for time_group, group_df in df_raw.groupby('time_group'):
            print(f"\nProcessing {time_group} ({len(group_df)} documents)...")
            df = pd.DataFrame({"text": group_df[args.text_col].fillna("").astype(str)})
            
            # Create subdirectory for this time period
            time_outdir = os.path.join(args.outdir, str(time_group))
            os.makedirs(time_outdir, exist_ok=True)
            
            kg.run(df, time_outdir, show_progress=False)
        
        print(f"\n✓ Temporal KG extraction complete. Outputs in {args.outdir}/[time_period]/")
    else:
        # Standard single KG extraction
        df = pd.DataFrame({"text": df_raw[args.text_col].fillna("").astype(str)})
        print("Extracting knowledge graph...")
        nodes_df, edges_df, ents_per_doc = kg.run(df, args.outdir)
        
        # Add sentiment analysis if requested
        if args.add_sentiment:
            try:
                from .kg_sentiment import add_sentiment_to_kg
                add_sentiment_to_kg(df, nodes_df, edges_df, ents_per_doc, args.outdir)
            except ImportError:
                print("⚠️  Warning: vaderSentiment not installed. Run: pip install vaderSentiment")
            except Exception as e:
                print(f"⚠️  Warning: Could not add sentiment analysis: {e}")
    
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
