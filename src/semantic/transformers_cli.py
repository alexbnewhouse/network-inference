"""
CLI for transformer-enhanced semantic network building.
"""
import argparse
import pandas as pd
import os
from .transformers_enhanced import TransformerSemanticNetwork


def main():
    ap = argparse.ArgumentParser(description="Build transformer-based semantic network")
    ap.add_argument("--input", required=True, help="Input CSV file containing text data")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", 
                    help="Sentence transformer model name")
    ap.add_argument("--similarity-threshold", type=float, default=0.5,
                    help="Minimum similarity for edge creation")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Keep top-k most similar documents per document")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit number of rows")
    ap.add_argument("--device", default="cpu", help="Device: cpu, cuda, or mps")
    ap.add_argument("--mode", default="document", choices=["document", "term"],
                    help="Build document or term network")
    ap.add_argument("--text-col", default="text", help="Column name for text content (default: text)")
    ap.add_argument("--subject-col", default=None, help="Optional column for subject/thread when mode=term (default: subject if exists)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    df_raw = pd.read_csv(args.input, nrows=args.max_rows)
    if args.text_col not in df_raw.columns:
        raise ValueError(f"Text column '{args.text_col}' not found in input CSV. Available: {list(df_raw.columns)}")
    df = pd.DataFrame()
    df["text"] = df_raw[args.text_col].fillna("").astype(str)
    texts = df["text"].tolist()
    
    builder = TransformerSemanticNetwork(model_name=args.model, device=args.device)
    
    if args.mode == "document":
        edges_df = builder.build_document_network(
            texts,
            similarity_threshold=args.similarity_threshold,
            top_k=args.top_k
        )
    else:
        # For term mode, need to extract vocabulary first
        from .preprocess import tokenize
        from .cooccur import build_vocab
        # Optionally include subject/thread in tokenization context
        subj_col = args.subject_col if args.subject_col else ("subject" if "subject" in df_raw.columns else None)
        if subj_col and subj_col in df_raw.columns:
            combo_texts = (df_raw[subj_col].fillna("").astype(str) + "\n" + df["text"]).tolist()
        else:
            combo_texts = texts
        docs = [tokenize(t) for t in combo_texts]
        vocab = build_vocab(docs, min_df=5)
        terms = list(vocab.keys())
        edges_df = builder.build_term_network(
            terms,
            similarity_threshold=args.similarity_threshold,
            top_k=args.top_k
        )
    
    edges_path = os.path.join(args.outdir, "transformer_edges.csv")
    edges_df.to_csv(edges_path, index=False)
    print(f"Saved {len(edges_df)} edges to {edges_path}")


if __name__ == "__main__":
    main()
