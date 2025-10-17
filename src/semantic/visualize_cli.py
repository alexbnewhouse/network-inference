from __future__ import annotations

import argparse
import os

from .visualize import load_graph, plot_top_terms, plot_top_edges, get_ego_subgraph, export_pyvis


def main():
    ap = argparse.ArgumentParser(description="Visualize semantic network outputs")
    ap.add_argument("--outdir", required=True, help="Directory containing nodes/edges parquet or csv")
    ap.add_argument("--topn", type=int, default=25, help="Top-N terms and edges to plot")
    ap.add_argument("--ego-token", type=str, default=None, help="Token for ego-network visualization")
    ap.add_argument("--ego-k", type=int, default=20, help="Number of neighbors for ego-network")
    args = ap.parse_args()

    nodes, edges, G = load_graph(args.outdir)
    os.makedirs(os.path.join(args.outdir, "viz"), exist_ok=True)

    plot_top_terms(nodes, os.path.join(args.outdir, "viz", "top_terms.png"), topn=args.topn)
    plot_top_edges(edges, nodes, os.path.join(args.outdir, "viz", "top_edges.png"), topn=args.topn)

    H, tok, _center = get_ego_subgraph(G, nodes, args.ego_token, args.ego_k)
    if H is not None:
        # Lazy import to avoid global dependency
        from .visualize import draw_ego_network

        draw_ego_network(G, nodes, tok, os.path.join(args.outdir, "viz", f"ego_{tok or 'top'}.png"), k=args.ego_k)
        export_pyvis(H, nodes, os.path.join(args.outdir, "viz", f"ego_{tok or 'top'}.html"))


if __name__ == "__main__":
    main()
