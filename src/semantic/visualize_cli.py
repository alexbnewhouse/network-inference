
"""
Semantic Network Visualization CLI
=================================

Visualize top terms, edges, and ego-networks from semantic network outputs.

Example usage:
    python -m src.semantic.visualize_cli --outdir output/ --topn 25 --ego-token climate --ego-k 20

Arguments:
    --outdir     Directory containing nodes/edges CSV
    --topn       Top-N terms and edges to plot (default: 25)
    --ego-token  Token for ego-network visualization (optional)
    --ego-k      Number of neighbors for ego-network (default: 20)

Outputs:
    PNG and HTML visualizations in the output directory's 'viz' subfolder.
"""

from __future__ import annotations

import argparse
import os

from .visualize import load_graph, plot_top_terms, plot_top_edges, get_ego_subgraph, export_pyvis

def main():
    ap = argparse.ArgumentParser(
        description="Visualize semantic network outputs",
        epilog="""
Example:
    python -m src.semantic.visualize_cli --outdir output/ --topn 25 --ego-token climate --ego-k 20
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--outdir", required=True, help="Directory containing nodes/edges CSV")
    ap.add_argument("--topn", type=int, default=25, help="Top-N terms and edges to plot (default: 25)")
    ap.add_argument("--ego-token", type=str, default=None, help="Token for ego-network visualization (optional)")
    ap.add_argument("--ego-k", type=int, default=20, help="Number of neighbors for ego-network (default: 20)")
    args = ap.parse_args()

    import os
    print("Loading graph and preparing visualizations...")
    nodes, edges, G = load_graph(args.outdir)
    viz_dir = os.path.join(args.outdir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    plot_top_terms(nodes, os.path.join(viz_dir, "top_terms.png"), topn=args.topn)
    plot_top_edges(edges, nodes, os.path.join(viz_dir, "top_edges.png"), topn=args.topn)
    H, tok, _center = get_ego_subgraph(G, nodes, args.ego_token, args.ego_k)
    if H is not None:
        from .visualize import draw_ego_network
        draw_ego_network(G, nodes, tok, os.path.join(viz_dir, f"ego_{tok or 'top'}.png"), k=args.ego_k)
        export_pyvis(H, nodes, os.path.join(viz_dir, f"ego_{tok or 'top'}.html"))
    print(f"Visualizations saved in {viz_dir}")

if __name__ == "__main__":
    main()
