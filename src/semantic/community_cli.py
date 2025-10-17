"""
CLI for community detection and reporting.
"""
import argparse
import pandas as pd
import networkx as nx
import os
from .community import CommunityDetector

def main():
    ap = argparse.ArgumentParser(description="Run community detection on semantic graph")
    ap.add_argument("--nodes", required=True, help="Nodes CSV file")
    ap.add_argument("--edges", required=True, help="Edges CSV file")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--method", type=str, default="louvain", help="Community detection method: louvain or leiden")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)
    G = nx.from_pandas_edgelist(edges, 'src', 'dst', edge_attr='weight')
    id2tok = dict(zip(nodes['id'], nodes['token']))
    cd = CommunityDetector(method=args.method)
    partition = cd.detect(G)
    summaries = cd.summarize(G, partition, id2tok)
    outp = os.path.join(args.outdir, "communities.csv")
    pd.DataFrame(summaries).to_csv(outp, index=False)
    print(f"Detected {len(summaries)} communities. Output: {outp}")

if __name__ == "__main__":
    main()
