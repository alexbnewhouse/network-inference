# Quick Start: Processing Your 1M /pol/ Posts

## Step 1: No Installation Needed!

The batch processing method works out of the box - no FAISS required!

```bash
cd /Users/alexnewhouse/network_inference
source .venv/bin/activate
```

**Note**: FAISS is optional and has compatibility issues with Python 3.12+. The default batch processing is reliable and production-ready.

## Step 2: Run the Transformer CLI

```bash
python -m src.semantic.transformers_cli \
  --input ../Dropbox/accdb_etl_pipeline/data/pol_archive_0.csv \
  --outdir output/pol_transformer_network \
  --text-col body \
  --device cuda \
  --top-k 20 \
  --similarity-threshold 0.5
```

## What Will Happen

### Phase 1: Encoding (estimate: 12-15 minutes)
```
Encoding documents...
Batches: 100%|████████| 31250/31250 [12:34<00:00, 41.43it/s]
```

### Phase 2: Batch Processing (estimate: 100-120 minutes)
```
Using memory-efficient batch processing (1,000,000 documents)...
Processing batch 1/100...
Processing batch 2/100...
...
Processing batch 100/100...
Found 12,456,789 edges
```

### Phase 3: Saving Results
```
Saved to: output/pol_transformer_network/edges.csv
```

## Expected Results

**Output File**: `output/pol_transformer_network/edges.csv`

**Format**:
```csv
source,target,similarity
0,1,0.876543
0,5,0.823456
1,2,0.912345
...
```

**Size**: ~500-800 MB for 10-20M edges

**Time**: ~2 hours total on RTX 5090

**Memory**: 
- RAM: ~20-25 GB peak
- VRAM: ~8 GB

**Stable**: No FAISS crashes, production-ready!

## Troubleshooting

### If you get "CUDA out of memory"

The encoding phase uses GPU memory. If you run out:
```bash
# Use CPU for encoding (slower but works)
python -m src.semantic.transformers_cli \
  --input ../Dropbox/accdb_etl_pipeline/data/pol_archive_0.csv \
  --outdir output/pol_transformer_network \
  --text-col body \
  --device cpu \
  --top-k 20
```

### If you get "Out of memory" during batch processing

Reduce the batch size:
```bash
python -m src.semantic.transformers_cli \
  --input ../Dropbox/accdb_etl_pipeline/data/pol_archive_0.csv \
  --outdir output/pol_transformer_network \
  --text-col body \
  --device cuda \
  --top-k 20 \
  --batch-size 5000  # Smaller batches use less RAM
```

### If processing is too slow

```bash
# Reduce number of edges (faster)
python -m src.semantic.transformers_cli \
  --input ../Dropbox/accdb_etl_pipeline/data/pol_archive_0.csv \
  --outdir output/pol_transformer_network \
  --text-col body \
  --device cuda \
  --top-k 10 \
  --similarity-threshold 0.7
```

### If still having memory issues

Process a sample first:
```bash
python -m src.semantic.transformers_cli \
  --input ../Dropbox/accdb_etl_pipeline/data/pol_archive_0.csv \
  --outdir output/pol_sample_network \
  --text-col body \
  --device cuda \
  --top-k 20 \
  --similarity-threshold 0.5 \
  --max-rows 100000  # Process only 100K posts first
```

## Parameters Explained

| Parameter | Value | Why |
|-----------|-------|-----|
| `--input` | pol_archive_0.csv | Your data file |
| `--outdir` | output/pol_transformer_network | Where to save results |
| `--text-col` | body | Column with post text |
| `--device` | cuda | Use GPU (change to 'cpu' on Mac) |
| `--top-k` | 20 | Keep 20 most similar posts per post |
| `--similarity-threshold` | 0.5 | Only keep edges with sim ≥ 0.5 |

## Advanced Options

### Fewer edges (faster, less memory)
```bash
--top-k 10 \
--similarity-threshold 0.7
```

### More edges (more complete network)
```bash
--top-k 50 \
--similarity-threshold 0.3
```

### Different embedding model
```bash
--model sentence-transformers/all-mpnet-base-v2  # Better quality, slower
```

## What to Do with Results

### Load in Python
```python
import pandas as pd
import networkx as nx

# Load edges
edges = pd.read_csv('output/pol_transformer_network/edges.csv')

# Create network
G = nx.from_pandas_edgelist(
    edges,
    source='source',
    target='target',
    edge_attr='similarity'
)

# Analyze
print(f"Nodes: {G.number_of_nodes():,}")
print(f"Edges: {G.number_of_edges():,}")
print(f"Density: {nx.density(G):.6f}")

# Find communities
import community
communities = community.best_partition(G.to_undirected())
```

### Visualize in Gephi
1. Save as GraphML:
```python
nx.write_graphml(G, 'pol_network.graphml')
```

2. Open in [Gephi](https://gephi.org/)
3. Apply ForceAtlas2 layout
4. Color by communities

### Export for analysis
```python
# Get most central posts
centrality = nx.degree_centrality(G)
top_posts = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:100]

# Load original text
posts = pd.read_csv('../Dropbox/accdb_etl_pipeline/data/pol_archive_0.csv')
for post_id, score in top_posts[:10]:
    print(f"\nPost {post_id} (centrality: {score:.4f}):")
    print(posts.iloc[post_id]['body'][:200] + "...")
```

## Estimated Timeline

- ✅ Activate environment: 10 seconds
- ✅ Run encoding: 12-15 minutes
- ✅ Run batch processing: 100-120 minutes
- ✅ Load and analyze: 5-10 minutes

**Total**: ~2 hours from start to finish

**Reliable**: No FAISS installation, no crashes, production-ready!

## Questions?

- Read [SCALING_GUIDE.md](SCALING_GUIDE.md) for detailed explanation
- Check [TRANSFORMER_SCALING_UPDATE.md](TRANSFORMER_SCALING_UPDATE.md) for technical details
- See [GPU_SENTIMENT_GUIDE.md](GPU_SENTIMENT_GUIDE.md) for GPU optimization tips

Ready to go! 🚀
