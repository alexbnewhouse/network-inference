from __future__ import annotations

import numpy as np
from collections import Counter
from typing import Dict, Iterable, List, Tuple
from scipy import sparse
import multiprocessing as mp


def _doc_pairs_worker(args):
    doc, vocab, window = args
    ids = [vocab[t] for t in doc if t in vocab]
    n = len(ids)
    pairs = []
    for i, wi in enumerate(ids):
        left = max(0, i - window)
        right = min(n, i + window + 1)
        for j in range(left, right):
            if j == i:
                continue
            wj = ids[j]
            if wj == wi:
                continue
            a, b = (wi, wj) if wi < wj else (wj, wi)
            pairs.append((a, b))
    return pairs, ids

def compute_ppmi_gpu(coo, token_counts, total_tokens, cds=0.75, eps=1e-12):
    import cupy as cp
    sm_counts = cp.power(token_counts, cds)
    Z = sm_counts.sum() + eps
    total_co = coo.data.sum() + eps
    pi = sm_counts / Z
    data = []
    for idx in range(len(coo.data)):
        i, j = coo.row[idx], coo.col[idx]
        cij = coo.data[idx]
        pij = cij / total_co
        denom = pi[i] * pi[j] + eps
        val = cp.log2(cp.maximum(pij / denom, eps))
        data.append(val if val > 0 else 0.0)
    data = cp.array(data, dtype=cp.float32)
    data[data < 0] = 0.0
    return type(coo)((data, (coo.row, coo.col)), shape=coo.shape)

def build_vocab(tokenized_docs: Iterable[List[str]], min_df: int = 5, max_vocab: int | None = None) -> Dict[str, int]:
    df = Counter()
    for toks in tokenized_docs:
        df.update(set(toks))
    items = [(t, c) for t, c in df.items() if c >= min_df]
    items.sort(key=lambda x: (-x[1], x[0]))
    if max_vocab is not None:
        items = items[:max_vocab]
    return {t: i for i, (t, _) in enumerate(items)}


def cooccurrence(
    tokenized_docs: Iterable[List[str]],
    vocab: Dict[str, int],
    window: int = 10,
) -> Tuple[sparse.coo_matrix, np.ndarray, int]:
    """
    Compute symmetric co-occurrence counts within a sliding window for each document.
    Returns: (coo_matrix, token_counts, total_tokens)
    """
    vocab_size = len(vocab)
    # Multiprocessing for large docs using top-level worker to avoid pickling issues
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(_doc_pairs_worker, ((doc, vocab, window) for doc in tokenized_docs))

    pair_counts = Counter()
    token_counts = np.zeros(vocab_size, dtype=np.int64)
    total_tokens = 0
    for pairs, ids in results:
        pair_counts.update(pairs)
        for idx in ids:
            token_counts[idx] += 1
        total_tokens += len(ids)

    # Build sparse co-occurrence matrix
    rows, cols, data = [], [], []
    for (i, j), v in pair_counts.items():
        rows.append(i)
        cols.append(j)
        data.append(v)
    coo = sparse.coo_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size), dtype=np.float32)
    return coo, token_counts, total_tokens


def compute_ppmi(
    coo: sparse.coo_matrix,
    token_counts: np.ndarray,
    total_tokens: int,
    cds: float = 0.75,
    eps: float = 1e-12,
) -> sparse.coo_matrix:
    """Compute Positive PMI with context distribution smoothing (cds) using sparse matrices."""
    sm_counts = np.power(token_counts, cds)
    Z = sm_counts.sum() + eps
    total_co = coo.data.sum() + eps
    pi = sm_counts / Z
    # For each nonzero entry
    data = []
    for idx in range(len(coo.data)):
        i, j = coo.row[idx], coo.col[idx]
        cij = coo.data[idx]
        pij = cij / total_co
        denom = pi[i] * pi[j] + eps
        val = np.log2(max(pij / denom, eps))
        data.append(val if val > 0 else 0.0)
    data = np.array(data, dtype=np.float32)
    data[data < 0] = 0.0
    return sparse.coo_matrix((data, (coo.row, coo.col)), shape=coo.shape)
