"""
Within-Thread Actor/Reply Network Extraction
- Parse threads/posts
- Extract reply/quote edges
- Handle tripcode/capcode/ID
- Compute network metrics
- Output per-thread actor networks (edges, metrics)
"""
import os
import pandas as pd
from collections import defaultdict, Counter

class ActorNetworkPipeline:
    def __init__(self):
        pass

    def extract_edges(self, df):
        """Extract reply/quote edges within threads."""
        # Assumes df has columns: thread_id, post_id, author_tripcode, author_capcode, author_poster_id_thread, text
        edges = []
        for thread_id, group in df.groupby("thread_id"):
            post_auth = {}
            for _, row in group.iterrows():
                post_auth[row["post_id"]] = row.get("author_tripcode") or row.get("author_capcode") or row.get("author_poster_id_thread") or f"anon_{row['post_id']}"
            for _, row in group.iterrows():
                src = post_auth.get(row["post_id"])
                # Find quoted post_ids in text (>>12345)
                import re
                quotes = re.findall(r">>(\d+)", str(row["text"]))
                for qid in quotes:
                    qid = int(qid)
                    dst = post_auth.get(qid)
                    if dst:
                        edges.append({"thread_id": thread_id, "src": src, "dst": dst, "post_id": row["post_id"], "quoted_post_id": qid})
        # Ensure DataFrame has expected columns even if empty
        return pd.DataFrame(edges, columns=["thread_id", "src", "dst", "post_id", "quoted_post_id"])

    def compute_metrics(self, edges_df):
        """Compute basic network metrics per thread."""
        if edges_df.empty:
            return pd.DataFrame(columns=["thread_id", "n_edges", "n_actors", "max_degree"])
        metrics = []
        for thread_id, group in edges_df.groupby("thread_id"):
            actors = set(group["src"]).union(set(group["dst"]))
            n_edges = len(group)
            n_actors = len(actors)
            degree = Counter(group["src"]).most_common(1)[0][1] if n_edges else 0
            metrics.append({"thread_id": thread_id, "n_edges": n_edges, "n_actors": n_actors, "max_degree": degree})
        return pd.DataFrame(metrics)

    def run(self, df, outdir):
        edges_df = self.extract_edges(df)
        metrics_df = self.compute_metrics(edges_df)
        os.makedirs(outdir, exist_ok=True)
        edges_df.to_csv(os.path.join(outdir, "actor_edges.csv"), index=False)
        metrics_df.to_csv(os.path.join(outdir, "actor_metrics.csv"), index=False)
        print(f"Actor network: {len(edges_df)} edges, {len(metrics_df)} threads")
