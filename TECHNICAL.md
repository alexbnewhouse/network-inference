# Technical Documentation

## Architecture Overview

### Pipeline Architecture

```
Input Data (CSV)
    ↓
Preprocessing
    ├── Tokenization
    ├── Normalization  
    └── Stopword Removal
    ↓
Feature Extraction
    ├── Co-occurrence Counting
    ├── Transformer Embeddings
    └── Entity Recognition
    ↓
Network Construction
    ├── PPMI Weighting
    ├── Similarity Matrices
    └── Entity Linking
    ↓
Post-processing
    ├── Sparsification (Top-K)
    ├── Community Detection
    └── Temporal Slicing
    ↓
Output (CSV, GraphML)
```

## Core Algorithms

### PPMI Calculation

The Positive Pointwise Mutual Information algorithm measures word association strength:

```python
def compute_ppmi(co_occurrence_matrix, word_frequencies, total_words, cds=0.75):
    """
    Compute PPMI with context distribution smoothing.
    
    Mathematical formulation:
    1. Smooth word frequencies: f_smooth(w) = f(w)^cds
    2. Normalize: P(w) = f_smooth(w) / Σf_smooth(w')
    3. For each word pair (w1, w2):
       PMI(w1,w2) = log2(P(w1,w2) / (P(w1) * P(w2)))
    4. PPMI(w1,w2) = max(0, PMI(w1,w2))
    
    Complexity: O(V^2) where V = vocabulary size
    Memory: O(E) where E = number of edges (sparse storage)
    """
```

**Why CDS = 0.75?**
- Empirically optimal value from Levy & Goldberg (2014)
- Balances frequent vs rare word contributions
- Too low (<0.5): underweights common words
- Too high (>0.9): similar to no smoothing

### Co-occurrence Counting

Sliding window approach with symmetric counting:

```python
def count_cooccurrences(documents, vocab, window=10):
    """
    Count word co-occurrences within sliding window.
    
    Algorithm:
    1. For each document:
       a. Convert to word IDs using vocabulary
       b. For each position i:
          - Look at window [i-w, i+w]
          - Count all unique pairs
    2. Aggregate counts across documents
    3. Build sparse matrix
    
    Complexity: O(N * L * W^2) where:
        N = number of documents
        L = average document length
        W = window size
        
    Optimization: Multiprocessing across documents
    """
```

**Window Size Selection**:
- Small (2-5): Captures syntactic relationships
- Medium (10-15): Captures semantic associations
- Large (20+): Captures topical similarity

### Top-K Sparsification

Reduces network density while preserving structure:

```python
def sparsify_topk(matrix, k=20):
    """
    Keep only top-k edges per node.
    
    Algorithm:
    1. For each row (node):
       a. Get all edge weights
       b. Sort descending
       c. Keep top k
    2. Rebuild sparse matrix
    
    Complexity: O(V * E/V * log(E/V))
    Memory reduction: From O(V^2) to O(V*k)
    """
```

**Effect on Network Properties**:
- Degree distribution: Max degree = k
- Connectivity: May create disconnected components
- Communities: Generally preserved
- Centrality: Local structure maintained

### Community Detection (Louvain)

Greedy modularity optimization:

```python
def louvain_communities(graph):
    """
    Louvain algorithm for community detection.
    
    Algorithm:
    1. Initialize: each node in own community
    2. First phase (local optimization):
       For each node:
           Try moving to neighbor communities
           Keep move that maximizes modularity gain
       Repeat until no improvement
    3. Second phase (aggregation):
       Build new graph where nodes = communities
       Edges weighted by inter-community connections
    4. Repeat phases 1-2 on new graph
    5. Stop when modularity cannot be increased
    
    Complexity: O((E + V) * log(V))
    Quality: Near-optimal modularity
    """
```

**Modularity Definition**:
```
Q = (1/2m) * Σ[A_ij - (k_i * k_j)/(2m)] * δ(c_i, c_j)

where:
m = total edge weight
A_ij = edge weight between i and j
k_i = sum of edge weights for node i
c_i = community of node i
δ(x,y) = 1 if x==y, else 0
```

## Transformer Models

### Sentence Embeddings

Using sentence-transformers for semantic similarity:

```python
class TransformerEmbeddings:
    """
    Generate contextualized embeddings.
    
    Process:
    1. Load pre-trained model (e.g., all-MiniLM-L6-v2)
    2. Tokenize text into subword units
    3. Pass through transformer layers
    4. Pool token embeddings (mean, CLS, etc.)
    5. L2-normalize output vectors
    
    Model sizes:
    - all-MiniLM-L6-v2: 384 dimensions, 23M parameters
    - all-mpnet-base-v2: 768 dimensions, 110M parameters
    - paraphrase-multilingual: 768 dim, supports 50+ languages
    
    Complexity: O(N * L^2 * D) where:
        N = batch size
        L = sequence length
        D = model dimension
    """
```

### Similarity Computation

Cosine similarity between embeddings:

```python
def cosine_similarity(embeddings):
    """
    Compute pairwise cosine similarity.
    
    Formula:
    sim(a,b) = (a · b) / (||a|| * ||b||)
    
    For normalized vectors: sim(a,b) = a · b
    
    Implementation:
    1. L2-normalize embeddings (if not already)
    2. Matrix multiply: S = E * E^T
    3. Result: n×n similarity matrix
    
    Complexity: O(n^2 * d) where:
        n = number of embeddings
        d = embedding dimension
        
    Memory: O(n^2) - can be large for many documents
    
    Optimization: Compute in blocks for large n
    """
```

### Named Entity Recognition

Transformer-based NER with spaCy:

```python
def extract_entities(texts, model="en_core_web_trf"):
    """
    Extract named entities using transformer model.
    
    Architecture:
    1. Subword tokenization (WordPiece/SentencePiece)
    2. Transformer encoder (e.g., RoBERTa)
    3. Token classification head
    4. CRF layer for sequence consistency
    5. Entity span merging
    
    Entity Types (OntoNotes):
    - PERSON: People, including fictional
    - ORG: Companies, agencies, institutions
    - GPE: Geopolitical entities (countries, cities)
    - LOC: Non-GPE locations (mountains, bodies of water)
    - DATE: Absolute or relative dates/periods
    - TIME: Times smaller than a day
    - MONEY: Monetary values
    - QUANTITY: Measurements
    - ORDINAL: First, second, etc.
    - CARDINAL: Numerals not covered by other types
    
    Performance (en_core_web_trf on OntoNotes):
    - Precision: ~90%
    - Recall: ~89%
    - F1: ~89.5%
    """
```

## Performance Optimization

### Memory Management

**Sparse Matrix Storage**:
```python
# Dense matrix: 50,000 × 50,000 = 2.5 billion floats = 10 GB
# Sparse matrix: ~1M non-zero = 4 MB (2500x reduction)

from scipy import sparse
coo = sparse.coo_matrix((data, (rows, cols)), shape=(V, V))
```

**Vocabulary Limiting**:
```python
# Before: 200,000 terms
# After filtering (min_df=5, max_vocab=50000): 50,000 terms
# Memory reduction: 16x less
# Processing speed: 4x faster
```

### Multiprocessing

**Document-level Parallelism**:
```python
with mp.Pool(mp.cpu_count()) as pool:
    results = pool.map(process_document, documents)
    
# Speedup: Near-linear with CPU count
# Overhead: Pickling/unpickling data
# Best for: Large numbers of small/medium documents
```

**GPU Acceleration**:
```python
# CuPy for matrix operations
import cupy as cp
gpu_matrix = cp.asarray(cpu_matrix)
result = cp.matmul(gpu_matrix, gpu_matrix.T)

# Speedup: 10-100x for large matrices
# Memory limit: GPU VRAM (typically 8-40 GB)
```

### Batch Processing

**Transformer Encoding**:
```python
# Process in batches to utilize GPU efficiently
batch_size = 32  # Adjust based on sequence length and GPU memory

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    embeddings = model.encode(batch)
    
# GPU utilization: ~90%+
# Speed: ~1000 sentences/second on V100
```

## Output Formats

### CSV Format

**Nodes File** (`nodes.csv`):
```csv
id,token,doc_freq,term_freq
0,climate,1234,5678
1,change,2345,8901
```

**Edges File** (`edges.csv`):
```csv
src,dst,weight
0,1,8.234
1,5,6.123
```

### GraphML Format

XML-based format for Gephi, Cytoscape:
```xml
<graphml>
  <graph edgedefault="undirected">
    <node id="0">
      <data key="token">climate</data>
    </node>
    <edge source="0" target="1">
      <data key="weight">8.234</data>
    </edge>
  </graph>
</graphml>
```

**Advantages**:
- Preserves node/edge attributes
- Standard format for visualization tools
- Human-readable

**Disadvantages**:
- Larger file size than binary formats
- Slower to parse for very large networks

## Testing

### Unit Tests

```python
# Test PPMI computation
def test_ppmi():
    # Given known co-occurrence counts
    # Expected PPMI values should match analytical result
    assert ppmi_value == pytest.approx(expected, rel=1e-3)

# Test sparsification
def test_topk():
    # After top-k, each row should have at most k non-zero entries
    assert (sparse_matrix.getnnz(axis=1) <= k).all()
```

### Integration Tests

```python
# Test full pipeline
def test_pipeline_end_to_end():
    df = load_test_data()
    build_semantic_from_df(df, "test_output/")
    assert os.path.exists("test_output/nodes.csv")
    assert os.path.exists("test_output/edges.csv")
```

### Performance Benchmarks

```python
import time

def benchmark_cooccurrence():
    start = time.time()
    result = cooccurrence(docs, vocab, window=10)
    elapsed = time.time() - start
    
    docs_per_sec = len(docs) / elapsed
    print(f"Speed: {docs_per_sec:.0f} docs/sec")
```

## Validation

### Network Quality Metrics

```python
def validate_network(G):
    """
    Check network quality.
    
    Metrics:
    - Giant component size (should be >50% of nodes)
    - Average degree (should be >2 for connected network)
    - Clustering coefficient (indicates community structure)
    - Modularity (>0.3 indicates good communities)
    """
    gcc = max(nx.connected_components(G), key=len)
    gcc_pct = len(gcc) / len(G) * 100
    
    avg_degree = sum(dict(G.degree()).values()) / len(G)
    clustering = nx.average_clustering(G)
    
    print(f"Giant component: {gcc_pct:.1f}%")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Clustering: {clustering:.3f}")
```

### Entity Recognition Quality

```python
def evaluate_ner(predictions, gold_standard):
    """
    Evaluate NER quality.
    
    Metrics:
    - Precision: % of predicted entities that are correct
    - Recall: % of true entities that were found
    - F1: Harmonic mean of precision and recall
    
    Entity-level vs Token-level:
    - Entity-level: Stricter, exact span match required
    - Token-level: Looser, partial credit for overlap
    """
    tp = true_positives(predictions, gold_standard)
    fp = false_positives(predictions, gold_standard)
    fn = false_negatives(predictions, gold_standard)
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1
```

## Common Pitfalls

### 1. Vocabulary Too Large

**Problem**: Running out of memory with 500K+ vocabulary.

**Solution**:
```python
# Filter by document frequency
--min-df 10  # Ignore terms in <10 documents

# Limit vocabulary size
--max-vocab 50000  # Keep only top 50K terms

# More aggressive filtering
--min-df 20 --max-vocab 20000
```

### 2. Dense Network

**Problem**: Network too dense for visualization (millions of edges).

**Solution**:
```python
# Aggressive top-k filtering
--topk 5  # Only 5 strongest edges per node

# Higher similarity threshold for transformers
--similarity-threshold 0.7  # Only strong similarities

# Filter by edge weight
edges_df = edges_df[edges_df['weight'] > threshold]
```

### 3. GPU Out of Memory

**Problem**: CUDA out of memory during batch processing.

**Solution**:
```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Process in chunks
for chunk in chunks(texts, 10000):
    process_chunk(chunk)

# Use gradient checkpointing (for training)
model.gradient_checkpointing_enable()
```

### 4. Slow Processing

**Problem**: Taking hours on medium datasets.

**Solution**:
```python
# Enable multiprocessing
# (Automatic in build_docs)

# Use GPU if available
--gpu --spacy-gpu

# Reduce max-rows for testing
--max-rows 10000

# Use faster models
--model en_core_web_sm  # Instead of trf
```

## References

### Academic Papers

1. **PPMI**: Levy & Goldberg (2014) "Neural Word Embedding as Implicit Matrix Factorization"
2. **Context Distribution Smoothing**: Mikolov et al. (2013) "Distributed Representations of Words and Phrases"
3. **Louvain Algorithm**: Blondel et al. (2008) "Fast unfolding of communities in large networks"
4. **Sentence Transformers**: Reimers & Gurevych (2019) "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
5. **Transformer NER**: Devlin et al. (2019) "BERT: Pre-training of Deep Bidirectional Transformers"

### Software Documentation

- spaCy: https://spacy.io/
- NetworkX: https://networkx.org/documentation/stable/
- Sentence Transformers: https://www.sbert.net/
- Hugging Face: https://huggingface.co/docs

### Tutorials

- Graph Analysis: https://networkx.org/documentation/stable/tutorial.html
- NLP with spaCy: https://course.spacy.io/
- Transformers: https://huggingface.co/course/
