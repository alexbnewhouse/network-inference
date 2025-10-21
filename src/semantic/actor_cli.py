"""
Actor/Reply Network Extraction CLI
==================================

Extract within-thread actor/reply networks from forum or social media data.

Example usage:
    python -m src.semantic.actor_cli --input posts.csv --outdir output/ --thread-col thread_id --post-col post_id --text-col text

Arguments:
    --input      Input CSV file with thread/post/author columns
    --outdir     Output directory
    --max-rows   Limit number of rows to load (optional)
    --thread-col Column name for thread IDs (default: thread_id; fallback to 'subject' if present)
    --post-col   Column name for post IDs or 'index' to use row index (default: post_id)
    --text-col   Column name for post text (default: text)
    --author-tripcode-col Column for author tripcode (optional)
    --author-capcode-col  Column for author capcode (optional)
    --author-id-col       Column for per-thread poster ID (optional)

Outputs:
    Actor/reply network files in the output directory.
"""
import argparse
import pandas as pd
import os
from .actor_network import ActorNetworkPipeline


def normalize_actor_df(df, thread_col, post_col, text_col,
                       author_trip_col, author_cap_col, author_id_col):
    # Ensure required columns exist with standard names expected by pipeline
    out = pd.DataFrame(index=df.index)
    # Thread column
    if thread_col and thread_col in df.columns:
        out["thread_id"] = df[thread_col]
    elif "subject" in df.columns:
        out["thread_id"] = df["subject"]
    else:
        out["thread_id"] = 1  # single thread fallback
    # Post column
    if post_col == "index":
        out["post_id"] = (df.reset_index().index + 1).values
    elif post_col and post_col in df.columns:
        out["post_id"] = df[post_col]
    elif "post_id" in df.columns:
        out["post_id"] = df["post_id"]
    else:
        out["post_id"] = (df.reset_index().index + 1).values
    # Text column
    if text_col in df.columns:
        out["text"] = df[text_col]
    else:
        raise ValueError(f"Text column '{text_col}' not found in input CSV")
    # Author columns (optional)
    out["author_tripcode"] = df[author_trip_col] if author_trip_col in df.columns else None
    out["author_capcode"] = df[author_cap_col] if author_cap_col in df.columns else None
    out["author_poster_id_thread"] = df[author_id_col] if author_id_col in df.columns else None
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Run actor/reply network extraction pipeline",
        epilog="""
Example:
    python -m src.semantic.actor_cli --input posts.csv --outdir output/ --thread-col thread_id --post-col post_id --text-col text
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Input CSV file with thread/post/author columns")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to load")
    ap.add_argument("--thread-col", default="thread_id", help="Column name for thread IDs (default: thread_id; fallback to 'subject' if present)")
    ap.add_argument("--post-col", default="post_id", help="Column name for post IDs or 'index' to use row index (default: post_id)")
    ap.add_argument("--text-col", default="text", help="Column name for post text (default: text)")
    ap.add_argument("--author-tripcode-col", default="author_tripcode", help="Column name for author tripcode (optional)")
    ap.add_argument("--author-capcode-col", default="author_capcode", help="Column name for author capcode (optional)")
    ap.add_argument("--author-id-col", default="author_poster_id_thread", help="Column name for per-thread poster ID (optional)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input, nrows=args.max_rows)
    norm_df = normalize_actor_df(
        df,
        thread_col=args.thread_col,
        post_col=args.post_col,
        text_col=args.text_col,
        author_trip_col=args.author_tripcode_col,
        author_cap_col=args.author_capcode_col,
        author_id_col=args.author_id_col,
    )
    actor = ActorNetworkPipeline()
    actor.run(norm_df, args.outdir)


if __name__ == "__main__":
    main()
