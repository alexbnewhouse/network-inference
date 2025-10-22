# Scaling Transformer Networks to Large Datasets

> **TL;DR**: The system **automatically optimizes** for your dataset size. Just run your command - no configuration needed! For datasets >10K, optionally install FAISS for better performance: `pip install faiss-cpu`

## Problem: Memory Limitations

When processing large datasets (>10K documents), computing full similarity matrices becomes memory-prohibitive:

- **10K documents**: 100M matrix entries = 381 MB (manageable)
- **100K documents**: 10B matrix entries = 37 GB (borderline)
- **1M documents**: 1T matrix entries = 3.6 TB (impossible)

## Solution: Automatic Method Selection

The `build_document_network()` function **automatically selects** the best approach based on dataset size:

### Auto-Detection Logic

1. **<10K documents**: Uses full similarity matrix (fastest for small datasets)
2. **>10K documents + FAISS available**: Uses FAISS for efficient approximate nearest neighbors
3. **>10K documents + no FAISS**: Falls back to memory-efficient batch processing

**You don't need to do anything** - the system automatically picks the best method!

### How FAISS Works

Instead of computing a full N×N similarity matrix, FAISS:
1. Builds an efficient index of document embeddings
2. For each document, searches for only the top-k most similar
3. Reduces memory from O(N²) to O(N×k)

**Memory Savings Example (1M documents, top-k=20)**:
- Full matrix: 3.6 TB
- FAISS approach: ~1.5 GB (2400x reduction)

## Installation

### FAISS CPU (Mac/Linux/Windows)
```bash
pip install faiss-cpu
```

### FAISS GPU (NVIDIA only - much faster)
```bash
pip install faiss-gpu
```

**Note**: FAISS GPU requires CUDA. Use CPU version on Mac.

## Usage

### Automatic Mode (Default - Recommended)

The system **automatically selects** the best method - you don't need to do anything!

```bash
python -m src.semantic.transformers_cli \
  --input large_dataset.csv \
  --outdir output/ \
  --device cuda \
  --text-col body \
  --top-k 20
```

**What Happens Automatically**:
- **<10K docs**: Uses full similarity matrix (fastest for small datasets)
- **>10K docs with FAISS**: Automatically switches to FAISS for efficient search
- **>10K docs without FAISS**: Automatically falls back to memory-efficient batch processing

**No configuration needed!** The code detects your dataset size and available libraries.

### Manual Override (Optional)

Most users won't need this, but you can override the automatic selection:

```bash
# Disable FAISS (force batch processing even if FAISS available)
python -m src.semantic.transformers_cli \
  --input data.csv \
  --outdir output/ \
  --no-faiss \
  --batch-size 5000

# Adjust batch size for memory-efficient processing
python -m src.semantic.transformers_cli \
  --input data.csv \
  --outdir output/ \
  --batch-size 20000  # Larger batches = faster but more memory
```

## Performance Comparison

| Dataset Size | Method | Memory | Time (CUDA) |
|-------------|--------|--------|-------------|
| 10K docs | Full matrix | 381 MB | ~30 sec |
| 100K docs | FAISS | ~1.5 GB | ~5 min |
| 1M docs | FAISS | ~15 GB | ~50 min |
| 1M docs | Full matrix | **3.6 TB** | ❌ Impossible |

**Hardware**: RTX 4090, 24GB VRAM, 64GB RAM

## Recommendations by Dataset Size

### Small Datasets (<10K documents)
- **Just use default settings** - no configuration needed!
- Full similarity matrix used automatically (fastest method)
- No need for FAISS installation
- Works great on CPU

### Medium Datasets (10K-100K documents)
- **Recommended**: Install FAISS for best performance
  ```bash
  pip install faiss-cpu  # or faiss-gpu for NVIDIA
  ```
- **No FAISS?** No problem - auto-falls back to batch processing
- Use default settings (automatically picks best method)
- Set `--top-k 20` to limit edges per document

### Large Datasets (>100K documents)
- **Install FAISS GPU** for acceptable performance:
  ```bash
  pip install faiss-gpu  # NVIDIA only
  ```
- Use GPU device: `--device cuda`
- Default settings work well (auto-optimized)
- Optional: Increase batch size: `--batch-size 20000`

### Very Large Datasets (>1M documents)
- **Required**: FAISS GPU installation
- Use CUDA device: `--device cuda`
- System automatically optimizes for scale
- Optional tuning:
  - Reduce edges: `--top-k 10`
  - Higher threshold: `--similarity-threshold 0.7`
  - Sample first: `--max-rows 500000`

## Troubleshooting

### Memory Error with FAISS Installed

If you still get memory errors with FAISS:

1. **Reduce top-k**: Use `--top-k 10` instead of 20
2. **Increase threshold**: Use `--similarity-threshold 0.7` to keep only strongest edges
3. **Process in chunks**: Split input CSV and combine results
4. **Use sampling**: Add `--max-rows 100000` to process subset

### FAISS Not Available

If FAISS import fails, the system falls back to batch processing:

```
⚠️  FAISS not available. Install with: pip install faiss-cpu or faiss-gpu
Falling back to batch processing (slower)...
```

Install FAISS or use `--batch-size` to control memory:

```bash
python -m src.semantic.transformers_cli \
  --input data.csv \
  --outdir output/ \
  --no-faiss \
  --batch-size 5000  # Process 5K docs at a time
```

### Slow Performance

For large datasets without GPU:

1. **Install FAISS GPU**: `pip install faiss-gpu` (NVIDIA only)
2. **Use CUDA**: `--device cuda`
3. **Reduce top-k**: `--top-k 10` (fewer edges to compute)
4. **Filter text**: Pre-process to include only relevant documents

## Example: Processing 1M /pol/ Posts

```bash
# Install FAISS GPU (one-time setup)
pip install faiss-gpu

# Process with automatic FAISS
python -m src.semantic.transformers_cli \
  --input pol_archive_0.csv \
  --outdir output/pol_network \
  --text-col body \
  --device cuda \
  --top-k 20 \
  --similarity-threshold 0.5

# Output:
# Encoding documents... 100%|████████| 1000000/1000000 [12:34<00:00, 1325.23it/s]
# Using FAISS for efficient similarity search (1,000,000 documents)...
# Building edge list... 100%|████████| 1000000/1000000 [35:42<00:00, 467.12it/s]
# Found 8,234,567 edges
# Saved to: output/pol_network/edges.csv
```

**Estimated Time**:
- Encoding (CUDA): ~13 minutes
- FAISS search: ~36 minutes
- **Total**: ~50 minutes for 1M documents

**Memory Usage**:
- Peak RAM: ~15 GB
- VRAM: ~8 GB

## API Usage

```python
from src.semantic.transformers_enhanced import TransformerSemanticNetwork

# Initialize
builder = TransformerSemanticNetwork(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"
)

# Build network (FAISS automatically used for large datasets)
edges_df = builder.build_document_network(
    texts=documents,
    similarity_threshold=0.5,
    top_k=20,
    use_faiss=True,  # Auto-enabled for >10K docs
    batch_size=10000  # Only used if FAISS unavailable
)
```

## Technical Details

### FAISS Index Types

The implementation uses `IndexFlatIP` (Inner Product):
- Exact search (not approximate)
- Cosine similarity via normalized embeddings
- Best balance of accuracy and simplicity

For even larger datasets (>10M), consider approximate indexes:
- `IndexIVFFlat`: Faster search, slight accuracy loss
- `IndexHNSWFlat`: Graph-based, excellent recall

### Normalization

Embeddings are L2-normalized before FAISS indexing:
```python
faiss.normalize_L2(embeddings)
```

This converts dot product to cosine similarity:
- cosine(a, b) = dot(a, b) / (||a|| × ||b||)
- After normalization: cosine(a, b) = dot(a, b)

### Edge Deduplication

Document networks are undirected, so edges are naturally deduplicated:
- If doc A → doc B (sim=0.8), doc B → doc A (sim=0.8)
- Only one edge stored per pair

## References

- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [Sentence Transformers](https://www.sbert.net/)
- [GPU Sentiment Guide](GPU_SENTIMENT_GUIDE.md)
