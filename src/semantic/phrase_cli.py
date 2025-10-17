"""
CLI for phrase/bigram/MWE promotion.
"""
import argparse
import pandas as pd
import os
from .phrase_mwe import PhraseDetector

def main():
    ap = argparse.ArgumentParser(description="Detect and promote high-PMI bigrams as phrases.")
    ap.add_argument("--input", required=True, help="Input CSV file with 'text' column")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--min-count", type=int, default=10, help="Minimum bigram count")
    ap.add_argument("--min-pmi", type=float, default=5.0, help="Minimum PMI for phrase promotion")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to load")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input, nrows=args.max_rows)
    docs = [str(x).split() for x in df["text"]]
    detector = PhraseDetector(min_count=args.min_count, min_pmi=args.min_pmi)
    pmi = detector.fit(docs)
    new_docs = detector.promote_phrases(docs, pmi)
    df["text_phrased"] = [" ".join(doc) for doc in new_docs]
    outp = os.path.join(args.outdir, "phrased.csv")
    df.to_csv(outp, index=False)
    print(f"Promoted {len(pmi)} phrases. Output: {outp}")

if __name__ == "__main__":
    main()
