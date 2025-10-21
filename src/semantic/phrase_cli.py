"""
Phrase Promotion CLI
===================

Detect and promote high-PMI bigrams as phrases in text data.

Example usage:
    python -m src.semantic.phrase_cli --input data.csv --outdir output/ --min-count 10 --min-pmi 5.0

Arguments:
    --input      Input CSV file with 'text' column
    --outdir     Output directory
    --min-count  Minimum bigram count (default: 10)
    --min-pmi    Minimum PMI for phrase promotion (default: 5.0)
    --max-rows   Limit number of rows to load (optional)

Outputs:
    phrased.csv in the output directory, with a new 'text_phrased' column.
"""
import argparse
import pandas as pd
import os
from .phrase_mwe import PhraseDetector

def main():
    ap = argparse.ArgumentParser(
        description="Detect and promote high-PMI bigrams as phrases.",
        epilog="""
Example:
    python -m src.semantic.phrase_cli --input data.csv --outdir output/ --min-count 10 --min-pmi 5.0
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Input CSV file with 'text' column")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--min-count", type=int, default=10, help="Minimum bigram count")
    ap.add_argument("--min-pmi", type=float, default=5.0, help="Minimum PMI for phrase promotion")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to load")
    args = ap.parse_args()

    import os
    import pandas as pd
    print("Loading input data...")
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input, nrows=args.max_rows)
    docs = [str(x).split() for x in df["text"]]
    detector = PhraseDetector(min_count=args.min_count, min_pmi=args.min_pmi)
    print("Detecting phrases...")
    pmi = detector.fit(docs)
    new_docs = detector.promote_phrases(docs, pmi)
    df["text_phrased"] = [" ".join(doc) for doc in new_docs]
    out_base = os.path.join(args.outdir, "phrased")
    ext = getattr(args, "output_format", "csv") if hasattr(args, "output_format") else "csv"
    if ext == "csv":
        df.to_csv(out_base + ".csv", index=False)
    elif ext == "json":
        df.to_json(out_base + ".json", orient="records", indent=2)
    elif ext == "parquet":
        df.to_parquet(out_base + ".parquet", index=False)
    print(f"Promoted {len(pmi)} phrases. Output: {out_base}.{ext}")

if __name__ == "__main__":
    main()
