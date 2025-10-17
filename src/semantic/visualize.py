import os
from typing import Optional, Tuple

import pandas as pd
import networkx as nx


def load_graph(outdir: str) -> Tuple[pd.DataFrame, pd.DataFrame, nx.Graph]:
    nodes_p = os.path.join(outdir, "nodes.csv")
    edges_p = os.path.join(outdir, "edges.csv")
    if not (os.path.exists(nodes_p) and os.path.exists(edges_p)):
        raise FileNotFoundError(f"Expected nodes.csv and edges.csv in {outdir}")
    nodes = pd.read_csv(nodes_p)
    edges = pd.read_csv(edges_p)
    # Build graph
    G = nx.Graph()
    for r in nodes.itertuples(index=False):
        nid = int(getattr(r, "id"))
        G.add_node(
            nid,
            token=str(getattr(r, "token")),
            doc_freq=int(getattr(r, "doc_freq")),
            term_freq=int(getattr(r, "term_freq")),
        )
    for r in edges.itertuples(index=False):
        G.add_edge(int(getattr(r, "src")), int(getattr(r, "dst")), weight=float(getattr(r, "weight")))
    return nodes, edges, G


def plot_top_terms(nodes: pd.DataFrame, outpath: str, topn: int = 25) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    top = nodes.nlargest(topn, "term_freq").copy()
    plt.figure(figsize=(8, max(4, topn * 0.3)))
    sns.barplot(data=top, y="token", x="term_freq", color="#4C78A8")
    plt.title(f"Top {topn} tokens by term frequency")
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_top_edges(edges: pd.DataFrame, nodes: pd.DataFrame, outpath: str, topn: int = 25) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    id2tok = nodes.set_index("id")["token"].to_dict()
    top = edges.nlargest(topn, "weight").copy()
    top["pair"] = top.apply(lambda r: f"{id2tok.get(int(r['src']), r['src'])} — {id2tok.get(int(r['dst']), r['dst'])}", axis=1)
    plt.figure(figsize=(10, max(4, topn * 0.35)))
    sns.barplot(data=top, y="pair", x="weight", color="#F58518")
    plt.title(f"Top {topn} PPMI edges")
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def get_ego_subgraph(G, nodes_df: pd.DataFrame, token: Optional[str], k: int = 20) -> Tuple[Optional[nx.Graph], Optional[str], Optional[int]]:
    # Resolve token to node id; if not provided, pick the highest-degree node
    id2tok = nodes_df.set_index("id")["token"].to_dict()
    tok2id = {v: k for k, v in id2tok.items()}
    if token and token in tok2id:
        center = tok2id[token]
    else:
        center = max(G.nodes(), key=lambda n: G.degree(n)) if G.number_of_nodes() > 0 else None
        token = id2tok.get(center, None) if center is not None else None
    if center is None:
        return None, None, None

    # Get top-k neighbors by weight
    nbrs = sorted(G[center].items(), key=lambda x: -x[1].get("weight", 0.0))[:k]
    nodes = [center] + [n for n, _ in nbrs]
    H = G.subgraph(nodes).copy()
    return H, token, center


def draw_ego_network(G: nx.Graph, nodes_df: pd.DataFrame, token: Optional[str], outpath: str, k: int = 20) -> Optional[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, token, _center = get_ego_subgraph(G, nodes_df, token, k)
    if H is None:
        return None

    pos = nx.spring_layout(H, seed=42, k=0.6)
    plt.figure(figsize=(8, 6))
    # Node sizes scaled by degree within H
    deg = {n: H.degree(n) for n in H.nodes()}
    sizes = [100 + 30 * deg[n] for n in H.nodes()]
    id2tok = nodes_df.set_index("id")["token"].to_dict()
    labels = {n: id2tok.get(n, str(n)) for n in H.nodes()}
    nx.draw_networkx_edges(H, pos, alpha=0.4)
    nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color="#72B7B2")
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=8)
    import matplotlib.pyplot as plt
    plt.title(f"Ego network around: {token}")
    plt.axis("off")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    return token


def export_pyvis(H: nx.Graph, nodes_df: pd.DataFrame, outpath: str) -> bool:
    try:
        from pyvis.network import Network
    except Exception:
        return False
    id2tok = nodes_df.set_index("id")["token"].to_dict()
    net = Network(height="750px", width="100%", notebook=False, cdn_resources="in_line")
    for n in H.nodes():
        net.add_node(int(n), label=id2tok.get(n, str(n)))
    for u, v, d in H.edges(data=True):
        net.add_edge(int(u), int(v), value=float(d.get("weight", 1.0)))
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    net.write_html(outpath)
    return True
