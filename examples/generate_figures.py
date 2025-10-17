#!/usr/bin/env python3
"""Generate static figures for README/examples.

Outputs under examples/figures/:
- similarity_heatmap.png
- transformer_network.png
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.semantic.transformers_enhanced import TransformerEmbeddings, TransformerSemanticNetwork

FIG_DIR = Path(__file__).resolve().parent / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def similarity_heatmap():
    sentences = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "Machine learning uses neural networks.",
        "Climate change affects global temperatures.",
    ]
    embedder = TransformerEmbeddings()
    embs = embedder.encode(sentences, show_progress=False)
    sim = embedder.compute_similarity_matrix(embs)

    plt.figure(figsize=(6, 5))
    sns.heatmap(sim, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
                xticklabels=[f"S{i+1}" for i in range(len(sentences))],
                yticklabels=[f"S{i+1}" for i in range(len(sentences))])
    plt.title('Sentence Similarity (Cosine)')
    plt.tight_layout()
    out = FIG_DIR / 'similarity_heatmap.png'
    plt.savefig(out, dpi=160)
    print(f"Saved {out}")


def transformer_network():
    docs = [
        "AI models are improving rapidly.",
        "Deep learning uses neural networks with many layers.",
        "Solar energy is a renewable resource.",
        "Wind turbines generate electricity from wind.",
        "Battery technology enables electric vehicles.",
        "Neural networks are used in computer vision.",
    ]
    builder = TransformerSemanticNetwork()
    edges = builder.build_document_network(docs, similarity_threshold=0.25, top_k=4)

    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(r['src'], r['dst'], weight=r['weight'])

    plt.figure(figsize=(7, 5))
    pos = nx.spring_layout(G, seed=42)
    w = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=600)
    # Draw edges with uniform width and alpha scaled separately
    # NetworkX draw function expects a float width, so we draw with default width
    nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.4, edge_color='gray')
    nx.draw_networkx_labels(G, pos)
    plt.title('Transformer Document Network')
    plt.axis('off')
    plt.tight_layout()
    out = FIG_DIR / 'transformer_network.png'
    plt.savefig(out, dpi=160)
    print(f"Saved {out}")


if __name__ == '__main__':
    similarity_heatmap()
    transformer_network()
