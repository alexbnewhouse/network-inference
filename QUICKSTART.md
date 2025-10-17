# Quick Reference Guide

## Common Workflows

### 1. Basic Semantic Network (Fast, Small Dataset)

```bash
# For quick analysis of <10K documents
python -m src.semantic.build_semantic_network \
  --input data.csv \
  --outdir output/semantic \
  --min-df 3 \
  --topk 15 \
  --max-rows 5000
```

**Expected time**: 1-2 minutes  
**Output files**: nodes.csv, edges.csv, graph.graphml  
**Memory usage**: <500 MB

### 2. Large-Scale Semantic Network (Production)

```bash
# For complete dataset with optimization
python -m src.semantic.build_semantic_network \
  --input data.csv \
  --outdir output/semantic \
  --min-df 10 \
  --max-vocab 50000 \
  --window 10 \
  --topk 20 \
  --cds 0.75
```

**Expected time**: 10-30 minutes for 100K documents  
**Output files**: nodes.csv, edges.csv, graph.graphml  
**Memory usage**: 2-8 GB

### 3. GPU-Accelerated Processing (Large Dataset)

```bash
# Requires CUDA GPU
python -m src.semantic.build_semantic_network \
  --input data.csv \
  --outdir output/semantic \
  --min-df 10 \
  --gpu \
  --spacy-gpu
```

**Expected speedup**: 5-10x faster  
**Requirements**: NVIDIA GPU, cupy-cuda12x  
**Memory usage**: GPU VRAM + 2-4 GB RAM

### 4. Knowledge Graph Extraction (Transformer)

```bash
# High-quality entity extraction
python -m src.semantic.kg_cli \
  --input data.csv \
  --outdir output/kg \
  --model en_core_web_trf \
  --max-rows 10000
```

**Expected time**: 5-10 minutes for 10K documents  
**Output files**: kg_nodes.csv, kg_edges.csv  
**Accuracy**: ~90% F1 for entity recognition

### 5. Transformer-Based Semantic Network

```bash
# Embedding-based similarity network
python -m src.semantic.transformers_cli \
  --input data.csv \
  --outdir output/transformer \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --similarity-threshold 0.5 \
  --top-k 20 \
  --device cpu
```

**Expected time**: 2-5 minutes for 1K documents  
**Output files**: transformer_edges.csv  
**Memory usage**: 2-4 GB

### 6. Community Detection

```bash
# Find communities in existing network
python -m src.semantic.community_cli \
  --nodes output/semantic/nodes.csv \
  --edges output/semantic/edges.csv \
  --outdir output/communities
```

**Expected time**: <1 minute for 50K nodes  
**Output files**: communities.csv  
**Algorithm**: Louvain (modularity optimization)

### 7. Time-Sliced Analysis

```bash
# Track network evolution monthly
python -m src.semantic.time_slice_cli \
  --input data.csv \
  --outdir output/timeslices \
  --slice-col timestamp \
  --freq M \
  --max-rows 50000
```

**Expected time**: 10-30 minutes (builds multiple networks)  
**Output**: Separate network per time slice  
**Use case**: Trend analysis, topic evolution

### 8. Actor/Reply Network (Forum/Social)

```bash
# Extract reply networks from threaded discussions
python -m src.semantic.actor_cli \
  --input data.csv \
  --outdir output/actors \
  --thread-col subject \
  --post-col index \
  --text-col text
```

**Expected time**: 1-5 minutes  
**Output files**: actor_edges.csv, actor_metrics.csv  
**Pattern**: Looks for >>123 style replies

## Command Line Options Quick Reference

### build_semantic_network

| Option | Default | Description |
|--------|---------|-------------|
| --input | Required | Input CSV file |
| --outdir | Required | Output directory |
| --min-df | 5 | Min document frequency |
| --max-vocab | None | Max vocabulary size |
| --window | 10 | Co-occurrence window |
| --topk | 20 | Top-k edges per node |
| --cds | 0.75 | Context dist smoothing |
| --max-rows | None | Limit input rows |
| --gpu | False | Use GPU acceleration |
| --spacy-gpu | False | GPU for tokenization |
| --igraph | False | Use igraph library |

### kg_cli

| Option | Default | Description |
|--------|---------|-------------|
| --input | Required | Input CSV file |
| --outdir | Required | Output directory |
| --model | en_core_web_sm | spaCy model name |
| --max-rows | None | Limit input rows |

### transformers_cli

| Option | Default | Description |
|--------|---------|-------------|
| --input | Required | Input CSV file |
| --outdir | Required | Output directory |
| --model | all-MiniLM-L6-v2 | Sentence transformer |
| --similarity-threshold | 0.5 | Min similarity for edge |
| --top-k | 20 | Top-k per document |
| --device | cpu | cpu, cuda, or mps |
| --mode | document | document or term |

### community_cli

| Option | Default | Description |
|--------|---------|-------------|
| --nodes | Required | nodes.csv file |
| --edges | Required | edges.csv file |
| --outdir | Required | Output directory |

### actor_cli

| Option | Default | Description |
|--------|---------|-------------|
| --input | Required | Input CSV file |
| --outdir | Required | Output directory |
| --thread-col | thread_id | Thread ID column |
| --post-col | post_id | Post ID column |
| --text-col | text | Text column |

### time_slice_cli

| Option | Default | Description |
|--------|---------|-------------|
| --input | Required | Input CSV file |
| --outdir | Required | Output directory |
| --slice-col | timestamp | Datetime column |
| --freq | M | M=monthly, W=weekly |

## Python API Quick Examples

### Basic Semantic Network

```python
from src.semantic.build_semantic_network import build_semantic_from_df
import pandas as pd

df = pd.read_csv("data.csv")
build_semantic_from_df(df, "output/", min_df=5, topk=20)
```

### Knowledge Graph

```python
from src.semantic.kg_pipeline import KnowledgeGraphPipeline
import pandas as pd

df = pd.read_csv("data.csv")
kg = KnowledgeGraphPipeline(ner_model="en_core_web_sm")
kg.run(df, "output/kg")
```

### Transformer Network

```python
from src.semantic.transformers_enhanced import TransformerSemanticNetwork
import pandas as pd

df = pd.read_csv("data.csv")
builder = TransformerSemanticNetwork()
edges = builder.build_document_network(
    df["text"].tolist(),
    similarity_threshold=0.5,
    top_k=20
)
```

### Load and Analyze Network

```python
from src.semantic.visualize import load_graph
import networkx as nx

G = load_graph("output/semantic")
print(f"Nodes: {len(G.nodes())}")
print(f"Edges: {len(G.edges())}")
print(f"Density: {nx.density(G):.4f}")

# Get most central nodes
centrality = nx.degree_centrality(G)
top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
```

## Troubleshooting Checklist

### Installation Issues

- [ ] Using Python 3.12 (not 3.14)?
- [ ] Virtual environment activated?
- [ ] All requirements installed: `pip install -r requirements.txt`
- [ ] spaCy model downloaded: `python -m spacy download en_core_web_sm`

### Memory Issues

- [ ] Reduce vocabulary: `--max-vocab 20000`
- [ ] Increase min-df: `--min-df 10`
- [ ] Reduce top-k: `--topk 10`
- [ ] Test with subset: `--max-rows 1000`

### Speed Issues

- [ ] Enable multiprocessing (automatic)
- [ ] Use GPU if available: `--gpu --spacy-gpu`
- [ ] Use smaller spaCy model: `en_core_web_sm` instead of `trf`
- [ ] Reduce max-rows for testing

### Output Issues

- [ ] Check output directory exists and is writable
- [ ] Verify CSV has required columns (text, subject, etc.)
- [ ] Check for empty or malformed input data
- [ ] Look for error messages in terminal output

## Performance Benchmarks

### Semantic Network (co-occurrence + PPMI)

| Documents | Vocab | Time (CPU) | Time (GPU) | Memory |
|-----------|-------|------------|------------|--------|
| 1,000 | 5K | 10s | 5s | 200MB |
| 10,000 | 20K | 2min | 30s | 1GB |
| 100,000 | 50K | 20min | 4min | 4GB |
| 1,000,000 | 100K | 3hr | 30min | 16GB |

### Knowledge Graph (NER)

| Documents | Model | Time | Entities | Memory |
|-----------|-------|------|----------|--------|
| 1,000 | sm | 30s | ~3K | 500MB |
| 1,000 | trf | 2min | ~3.5K | 2GB |
| 10,000 | sm | 5min | ~30K | 1GB |
| 10,000 | trf | 20min | ~35K | 4GB |

### Transformer Embeddings

| Documents | Model | Device | Time | Memory |
|-----------|-------|--------|------|--------|
| 1,000 | MiniLM | CPU | 1min | 2GB |
| 1,000 | MiniLM | GPU | 10s | 2GB+1GB VRAM |
| 10,000 | MiniLM | CPU | 10min | 4GB |
| 10,000 | MiniLM | GPU | 1.5min | 4GB+2GB VRAM |

*Benchmarks on: MacBook Air M2 (CPU), NVIDIA V100 (GPU)*

## File Size Estimates

### Input Data

- 1K documents (~1KB each): 1 MB CSV
- 10K documents: 10 MB CSV
- 100K documents: 100 MB CSV
- 1M documents: 1 GB CSV

### Output Data (Semantic Network)

| Vocab Size | Top-K | Nodes CSV | Edges CSV | GraphML |
|------------|-------|-----------|-----------|---------|
| 5,000 | 20 | 500 KB | 2 MB | 5 MB |
| 20,000 | 20 | 2 MB | 8 MB | 20 MB |
| 50,000 | 20 | 5 MB | 20 MB | 50 MB |
| 100,000 | 20 | 10 MB | 40 MB | 100 MB |

## Best Practices

### For Small Datasets (<10K documents)

```bash
python -m src.semantic.build_semantic_network \
  --input data.csv \
  --outdir output/ \
  --min-df 2 \
  --topk 20
```

### For Medium Datasets (10K-100K documents)

```bash
python -m src.semantic.build_semantic_network \
  --input data.csv \
  --outdir output/ \
  --min-df 5 \
  --max-vocab 50000 \
  --topk 20
```

### For Large Datasets (>100K documents)

```bash
python -m src.semantic.build_semantic_network \
  --input data.csv \
  --outdir output/ \
  --min-df 10 \
  --max-vocab 50000 \
  --topk 15 \
  --gpu
```

### For High-Quality Analysis

```bash
# Use transformer models
python -m src.semantic.kg_cli \
  --model en_core_web_trf

python -m src.semantic.transformers_cli \
  --model sentence-transformers/all-mpnet-base-v2
```

## Environment Variables

```bash
# Disable GPU visibility
export CUDA_VISIBLE_DEVICES=""

# Set specific GPU
export CUDA_VISIBLE_DEVICES=0

# Enable spaCy GPU tokenization
export SPACY_GPU=1

# Set multiprocessing method
export TOKENIZERS_PARALLELISM=false  # Avoid warnings
```

## Common Errors and Solutions

### Error: `ModuleNotFoundError: No module named 'spacy'`
**Solution**: `pip install spacy && python -m spacy download en_core_web_sm`

### Error: `OSError: [E050] Can't find model 'en_core_web_trf'`
**Solution**: `python -m spacy download en_core_web_trf`

### Error: `CUDA out of memory`
**Solution**: Reduce batch size or use CPU: `--device cpu`

### Error: `KeyError: 'text'`
**Solution**: Ensure CSV has 'text' column or specify: `--text-col your_column`

### Error: `ValueError: empty vocabulary`
**Solution**: Lower min-df: `--min-df 1` or increase dataset size

## Next Steps

After running pipelines:

1. **Visualize**: Load graphs in Gephi/Cytoscape using GraphML files
2. **Analyze**: Use Jupyter notebooks for interactive exploration
3. **Compare**: Run community detection and time-slice analysis
4. **Iterate**: Adjust parameters based on network quality metrics
