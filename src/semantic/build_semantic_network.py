from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd
import polars as pl
from tqdm import tqdm
import multiprocessing as mp
from scipy import sparse

# Support running as module or script
try:
    from .preprocess import tokenize
    from .cooccur import build_vocab, cooccurrence, compute_ppmi
    from .graph_build import sparsify_topk, to_networkx
except ImportError:  # pragma: no cover
    from preprocess import tokenize
    from cooccur import build_vocab, cooccurrence, compute_ppmi
    from graph_build import sparsify_topk, to_networkx


def read_dataset(path: str, max_rows: int | None = None) -> pd.DataFrame:
    # Use Polars for fast CSV reading if available
    try:
        df = pl.read_csv(path, n_rows=max_rows, null_values=["NA", "NaN", "None"])
        if "text" not in df.columns:
            raise ValueError("Input CSV must contain a 'text' column")
        if "subject" not in df.columns:
            df = df.with_columns(pl.lit("").alias("subject"))
        df = df.with_columns([
            pl.col("text").fill_null("").cast(pl.Utf8),
            pl.col("subject").fill_null("").cast(pl.Utf8),
        ])
        return df.to_pandas()
    except Exception:
        # Fallback to pandas
        df = pd.read_csv(path, nrows=max_rows, na_values=["NA", "NaN", "None"], keep_default_na=True)
        assert "text" in df.columns, "Input CSV must contain a 'text' column"
        if "subject" not in df.columns:
            df["subject"] = None
        df["text"] = df["text"].fillna("")
        df["subject"] = df["subject"].fillna("")
        return df


def build_docs(df: pd.DataFrame) -> List[List[str]]:
    # Optionally use spaCy with GPU for tokenization
    import os
    spacy_gpu = os.environ.get("SPACY_GPU", "0") == "1"
    if spacy_gpu:
        import spacy
        try:
            nlp = spacy.load("en_core_web_trf")
        except Exception:
            spacy.cli.download("en_core_web_trf")
            nlp = spacy.load("en_core_web_trf")
        nlp.max_length = 2_000_000
        nlp.enable_pipe("transformer")
        nlp.to_gpu()
        def spacy_tokenize(text):
            doc = nlp(text)
            return [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha and len(t) > 1]
        rows = list(df.itertuples(index=False))
        texts = [(str(getattr(row, "subject", "") or "") + "\n" + str(getattr(row, "text", "") or "")).strip() for row in rows]
        docs = list(tqdm(nlp.pipe(texts, batch_size=128), total=len(texts), desc="spaCy GPU tokenizing"))
        docs = [[t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha and len(t) > 1] for doc in docs]
        return docs
    else:
        # Multiprocessing tokenization
        def _row_to_text(row):
            subj = str(getattr(row, "subject", "") or "")
            txt = str(getattr(row, "text", "") or "")
            return (subj + "\n" + txt).strip()
        rows = list(df.itertuples(index=False))
        with mp.Pool(mp.cpu_count()) as pool:
            docs = list(tqdm(pool.imap(tokenize, map(_row_to_text, rows)), total=len(rows), desc="Tokenizing (mp)"))
        return docs


def recompute_df(docs: List[List[str]], vocab: dict[str, int]) -> pd.DataFrame:
    from collections import Counter

    df_counts = Counter()
    for toks in docs:
        df_counts.update(set(t for t in toks if t in vocab))
    # Token frequency (tf)
    tf_counts = Counter()
    for toks in docs:
        for t in toks:
            if t in vocab:
                tf_counts[t] += 1
    items = []
    for t, idx in vocab.items():
        items.append({"id": idx, "token": t, "doc_freq": int(df_counts.get(t, 0)), "term_freq": int(tf_counts.get(t, 0))})
    return pd.DataFrame(items).sort_values("doc_freq", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Build a PPMI-weighted semantic co-occurrence network from CSV (scalable, GPU-ready)")
    ap.add_argument("--input", required=True, help="Path to CSV file with 'text' and optional 'subject'")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--min-df", type=int, default=5, help="Minimum document frequency for vocabulary")
    ap.add_argument("--max-vocab", type=int, default=None, help="Maximum vocabulary size")
    ap.add_argument("--window", type=int, default=10, help="Context window size for co-occurrence")
    ap.add_argument("--topk", type=int, default=20, help="Top-k neighbors to keep per node (0 for all)")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to load for testing")
    ap.add_argument("--cds", type=float, default=0.75, help="Context distribution smoothing exponent for PPMI")
    ap.add_argument("--polars", action="store_true", help="Use Polars for CSV reading (default: auto)")
    ap.add_argument("--igraph", action="store_true", help="Use igraph for large graph construction")
    ap.add_argument("--spacy-gpu", action="store_true", help="Use spaCy with GPU for tokenization (requires CUDA)")
    ap.add_argument("--gpu", action="store_true", help="Use CuPy for matrix ops (requires CUDA)")
    args = ap.parse_args()

    # Set env var for spaCy GPU
    if args.spacy_gpu:
        import os
        os.environ["SPACY_GPU"] = "1"

    os.makedirs(args.outdir, exist_ok=True)

    # Chunked reading for very large files (if max_rows not set)
    if args.max_rows is None:
        # Use Polars scan_csv for streaming if available
        try:
            scan = pl.scan_csv(args.input, null_values=["NA", "NaN", "None"])
            # Only select needed columns
            scan = scan.select([pl.col("text"), pl.col("subject")])
            # Collect in batches
            batch_size = 100_000
            docs = []
            nrows = 0
            for df in scan.iter_batches(batch_size=batch_size):
                pdf = df.collect().to_pandas()
                docs.extend(build_docs(pdf))
                nrows += len(pdf)
                print(f"Processed {nrows} rows...")
        except Exception:
            # Fallback to pandas full read
            df = read_dataset(args.input, max_rows=args.max_rows)
            docs = build_docs(df)
    else:
        df = read_dataset(args.input, max_rows=args.max_rows)
        docs = build_docs(df)

    vocab = build_vocab(docs, min_df=args.min_df, max_vocab=args.max_vocab)
    id2tok: List[str] = [""] * len(vocab)
    for t, i in vocab.items():
        id2tok[i] = t

    from .graph_build import sparsify_topk_sparse, to_networkx_sparse, to_igraph_sparse
    # Use CuPy for matrix ops if requested and available
    use_gpu = args.gpu
    try:
        if use_gpu:
            import cupy as cp
            import cupyx.scipy.sparse as cupy_sparse
            print("Using CuPy for matrix ops.")
            # TODO: implement GPU cooccurrence/PPMI if needed
            # For now, fallback to CPU for cooccurrence, but use CuPy for PPMI
            coo, token_counts, total_tokens = cooccurrence(docs, vocab, window=args.window)
            # Convert to CuPy sparse
            coo_gpu = cupy_sparse.coo_matrix((cp.array(coo.data), (cp.array(coo.row), cp.array(coo.col))), shape=coo.shape)
            token_counts_gpu = cp.array(token_counts)
            from .cooccur import compute_ppmi_gpu
            ppmi = compute_ppmi_gpu(coo_gpu, token_counts_gpu, total_tokens, cds=args.cds)
            # Move back to CPU for output
            ppmi = ppmi.get()
        else:
            coo, token_counts, total_tokens = cooccurrence(docs, vocab, window=args.window)
            ppmi = compute_ppmi(coo, token_counts, total_tokens, cds=args.cds)
    except ImportError:
        coo, token_counts, total_tokens = cooccurrence(docs, vocab, window=args.window)
        ppmi = compute_ppmi(coo, token_counts, total_tokens, cds=args.cds)
    if args.topk and args.topk > 0:
        ppmi = sparsify_topk_sparse(ppmi, topk=args.topk)

    # Save nodes and edges as Parquet (sparse)
    nodes_df = recompute_df(docs, vocab)
    coo_ppmi = ppmi.tocoo()
    edges_df = pd.DataFrame({
        "src": coo_ppmi.row,
        "dst": coo_ppmi.col,
        "weight": coo_ppmi.data
    })
    edges_df = edges_df[edges_df["weight"] > 0]
    if not edges_df.empty:
        edges_df = edges_df.sort_values("weight", ascending=False).reset_index(drop=True)

    nodes_path = os.path.join(args.outdir, "nodes.parquet")
    edges_path = os.path.join(args.outdir, "edges.parquet")
    try:
        nodes_df.to_parquet(nodes_path, index=False)
        edges_df.to_parquet(edges_path, index=False)
    except Exception as e:
        nodes_path = os.path.join(args.outdir, "nodes.csv")
        edges_path = os.path.join(args.outdir, "edges.csv")
        nodes_df.to_csv(nodes_path, index=False)
        edges_df.to_csv(edges_path, index=False)

    # Save graph
    try:
        if args.igraph:
            G = to_igraph_sparse(ppmi, id2tok)
            G.write_graphml(os.path.join(args.outdir, "graph_igraph.graphml"))
        else:
            import networkx as nx
            G = to_networkx_sparse(ppmi, id2tok)
            nx.write_graphml(G, os.path.join(args.outdir, "graph.graphml"))
    except Exception:
        pass

    print(f"Saved nodes to {nodes_path}")
    print(f"Saved edges to {edges_path}")


if __name__ == "__main__":
    main()
