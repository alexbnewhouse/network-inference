"""
Community Detection CLI
======================

Detect communities in a semantic network using Louvain or Leiden algorithms.

Example usage:
    python -m src.semantic.community_cli --nodes nodes.csv --edges edges.csv --outdir output/ --method louvain

Arguments:
    --nodes      Nodes CSV file (must have 'id' and 'token' columns)
    --edges      Edges CSV file (must have 'src', 'dst', 'weight' columns)
    --outdir     Output directory
    --method     Community detection method: louvain (default) or leiden

Outputs:
    communities.csv in the output directory, listing detected communities and their summary stats.
"""
import argparse
import pandas as pd
import networkx as nx
import os
from .community import CommunityDetector

def main():
    ap = argparse.ArgumentParser(
        description="Run community detection on semantic graph",
        epilog="""
Example:
    python -m src.semantic.community_cli --nodes nodes.csv --edges edges.csv --outdir output/ --method louvain
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--nodes", required=True, help="Nodes CSV file (must have 'id' and 'token' columns)")
    ap.add_argument("--edges", required=True, help="Edges CSV file (must have 'src', 'dst', 'weight' columns)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--method", type=str, default="louvain", help="Community detection method: louvain (default) or leiden")
    args = ap.parse_args()

    import os
    import pandas as pd
    print("Loading nodes and edges...")
    os.makedirs(args.outdir, exist_ok=True)
    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)
    G = nx.from_pandas_edgelist(edges, 'src', 'dst', edge_attr='weight')
    id2tok = dict(zip(nodes['id'], nodes['token']))
    cd = CommunityDetector(method=args.method)
    print("Detecting communities...")
    partition = cd.detect(G)
    summaries = cd.summarize(G, partition, id2tok)
    df_out = pd.DataFrame(summaries)
    out_base = os.path.join(args.outdir, "communities")
    ext = getattr(args, "output_format", "csv") if hasattr(args, "output_format") else "csv"
    if ext == "csv":
        df_out.to_csv(out_base + ".csv", index=False)
    elif ext == "json":
        df_out.to_json(out_base + ".json", orient="records", indent=2)
    elif ext == "parquet":
        df_out.to_parquet(out_base + ".parquet", index=False)
    print(f"Detected {len(summaries)} communities. Output: {out_base}.{ext}")

if __name__ == "__main__":
    main()
