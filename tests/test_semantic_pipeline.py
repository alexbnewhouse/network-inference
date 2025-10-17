import os

import pandas as pd

from src.semantic.build_semantic_network import build_docs
from src.semantic.cooccur import build_vocab, cooccurrence, compute_ppmi


def test_pipeline_tiny(tmp_path):
    data = pd.DataFrame({
        "subject": ["hello", "world"],
        "text": ["cats and dogs", ">>123 cats vs dogs"],
    })
    docs = build_docs(data)
    vocab = build_vocab(docs, min_df=1)
    pairs, counts, total = cooccurrence(docs, vocab, window=2)
    ppmi = compute_ppmi(pairs, counts, total)
    assert len(vocab) >= 2
    assert any(weight > 0 for weight in ppmi.values())
