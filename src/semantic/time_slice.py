"""
Time-Sliced Semantic Network Pipeline
- Partition data by month/week
- Build semantic networks per slice
 - Output per-slice CSV graph files and summary stats
"""
import os
import pandas as pd
from dateutil import parser
from collections import defaultdict

class TimeSlicedSemanticPipeline:
    def __init__(self, slice_col="timestamp", freq="M"):
        self.slice_col = slice_col
        self.freq = freq  # 'M' for month, 'W' for week

    def run(self, df, build_semantic_fn, outdir):
        # Assumes df has a datetime column (self.slice_col)
        df[self.slice_col] = pd.to_datetime(df[self.slice_col], errors="coerce")
        df = df.dropna(subset=[self.slice_col])
        df["slice"] = df[self.slice_col].dt.to_period(self.freq)
        for slice_val, group in df.groupby("slice"):
            print(f"Processing slice {slice_val} ({len(group)})")
            slice_dir = os.path.join(outdir, f"slice_{slice_val}")
            os.makedirs(slice_dir, exist_ok=True)
            build_semantic_fn(group, slice_dir)
