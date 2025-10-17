#!/usr/bin/env python3
"""
Benchmark co-occurrence vs transformer networks.

Outputs:
- CSV summary of timings and graph stats
- Optional plots (if --plot)

Usage:
  python3 benchmarks/benchmark_methods.py --docs 100 500 1000 --repeats 1 --out benchmarks/results.csv
"""
from __future__ import annotations
import argparse
import time
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
import sys

import numpy as np
import pandas as pd

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.semantic.transformers_enhanced import TransformerSemanticNetwork

try:
    from src.semantic.build_semantic_network import build_semantic_from_df
except Exception:
    build_semantic_from_df = None


@dataclass
class BenchmarkResult:
    method: str
    docs: int
    vocab: int | None
    build_time_s: float
    nodes: int
    edges: int
    density: float


def generate_docs(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate sample documents using the examples generator if available."""
    rng = np.random.default_rng(seed)
    # Prefer the examples generator
    try:
        examples_dir = Path(__file__).resolve().parents[1] / 'examples'
        sys.path.insert(0, str(examples_dir))
        from sample_data import generate_news_dataset  # type: ignore
        df = generate_news_dataset(n_docs=n, seed=seed)
    except Exception:
        texts = [f"Document {i} about AI, climate, and markets {rng.integers(0, 1000)}" for i in range(n)]
        df = pd.DataFrame({'text': texts})
    return df


def summarize_edges_df(edges: pd.DataFrame, src_col: str, dst_col: str) -> tuple[int, int, float]:
    import networkx as nx
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(r[src_col], r[dst_col])
    return G.number_of_nodes(), G.number_of_edges(), nx.density(G)


def benchmark_transformers(df: pd.DataFrame, similarity_threshold: float, top_k: int) -> BenchmarkResult:
    builder = TransformerSemanticNetwork()
    start = time.time()
    edges = builder.build_document_network(df['text'].tolist(), similarity_threshold=similarity_threshold, top_k=top_k)
    elapsed = time.time() - start
    nodes, edges_n, density = summarize_edges_df(edges, 'src', 'dst')
    return BenchmarkResult('transformer', len(df), None, elapsed, nodes, edges_n, density)


def benchmark_cooccurrence(df: pd.DataFrame, min_df: int, topk: int) -> BenchmarkResult | None:
    if build_semantic_from_df is None:
        return None
    import tempfile
    tmpdir = Path(tempfile.mkdtemp())
    start = time.time()
    build_semantic_from_df(df, str(tmpdir), min_df=min_df, topk=topk)
    elapsed = time.time() - start
    edges_path = tmpdir / 'edges.csv'
    nodes_path = tmpdir / 'nodes.csv'
    if not edges_path.exists() or not nodes_path.exists():
        return None
    edges = pd.read_csv(edges_path)
    nodes = pd.read_csv(nodes_path)
    # Best-effort column detection
    src_col = 'source' if 'source' in edges.columns else edges.columns[0]
    dst_col = 'target' if 'target' in edges.columns else edges.columns[1]
    n_nodes, n_edges, density = summarize_edges_df(edges, src_col, dst_col)
    return BenchmarkResult('cooccurrence', len(df), len(nodes) if 'term' in nodes.columns else None, elapsed, n_nodes, n_edges, density)


def main():
    p = argparse.ArgumentParser(description='Benchmark methods')
    p.add_argument('--docs', type=int, nargs='+', default=[200, 500], help='Document counts to test')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--out', type=Path, default=Path('benchmarks/results.csv'))
    p.add_argument('--plot', action='store_true')
    p.add_argument('--transformer-threshold', type=float, default=0.3)
    p.add_argument('--transformer-topk', type=int, default=10)
    p.add_argument('--cooccur-min-df', type=int, default=5)
    p.add_argument('--cooccur-topk', type=int, default=20)
    args = p.parse_args()

    results: list[BenchmarkResult] = []

    for n in args.docs:
        for r in range(args.repeats):
            df = generate_docs(n, seed=42 + r)
            # Transformer
            tr = benchmark_transformers(df, args.transformer_threshold, args.transformer_topk)
            results.append(tr)
            # Co-occurrence
            cr = benchmark_cooccurrence(df, args.cooccur_min_df, args.cooccur_topk)
            if cr is not None:
                results.append(cr)

    # Write CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"Wrote results to {args.out}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            df_res = pd.DataFrame([asdict(r) for r in results])
            plt.figure(figsize=(10,6))
            sns.barplot(data=df_res, x='docs', y='build_time_s', hue='method')
            plt.title('Build Time by Method')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")


if __name__ == '__main__':
    main()
