# Concepts and Terminology

Deep dive into the methods, statistics, and theory behind network inference.

## Table of Contents

- [Core Statistical Methods](#core-statistical-methods)
- [Network Analysis Concepts](#network-analysis-concepts)
- [NLP and Transformer Concepts](#nlp-and-transformer-concepts)
- [Data Processing](#data-processing)
- [When to Use What](#when-to-use-what)

## Core Statistical Methods

### PPMI (Positive Pointwise Mutual Information)

**What it is**: A statistical measure that quantifies how much more often two words appear together than would be expected by chance.

**Why it matters**: PPMI weights in semantic networks indicate genuine associations between words, filtering out coincidental co-occurrences. Without PPMI, common words would dominate the network simply because they appear frequently, not because they're meaningfully related.

**Mathematical Formula**:
```
PMI(w1, w2) = log2(P(w1, w2) / (P(w1) × P(w2)))
PPMI(w1, w2) = max(0, PMI(w1, w2))
```

Where:
- `P(w1, w2)` = probability of w1 and w2 co-occurring
- `P(w1)` = probability of w1 appearing
- `P(w2)` = probability of w2 appearing

**Interpretation**:
- **PPMI = 0**: Words co-occur at chance level (negative PMI truncated to 0)
- **PPMI = 5**: Words co-occur 2^5 = 32 times more than chance
- **PPMI = 10**: Words co-occur 2^10 = 1024 times more than chance

**Example**: 
If "climate" appears in 5% of documents, "change" appears in 8%, but they co-occur in 3% (not 0.4% = 5% × 8%), their PMI is:

```
PMI = log2(0.03 / (0.05 × 0.08)) = log2(7.5) ≈ 2.9
```

This indicates "climate" and "change" are 7.5× more likely to co-occur than random chance.

### Co-occurrence

**What it is**: The phenomenon of words appearing near each other in text, defined by a sliding window (typically 5-10 words in each direction).

**Why it matters**: Co-occurrence patterns reveal semantic relationships. Words that co-occur frequently tend to be semantically related, even if they don't appear in the exact same syntactic constructions.

**Window size considerations**:

| Window | Captures | Use Case |
|--------|----------|----------|
| 2-3 | Syntax, collocations | Phrase detection |
| 5-10 | Semantic relatedness | General semantic networks |
| 15-20 | Broader topic associations | Topic modeling |
| 50+ | Document-level co-occurrence | Very loose associations |

**Example with window=2**:

Text: "The quick brown fox jumped over the lazy dog"

Co-occurrences for "fox":
- Distance 1: "brown", "jumped"
- Distance 2: "quick", "over"

Within window=2, "fox" co-occurs with: {quick, brown, jumped, over}

**Directional vs Symmetric**:
- **Symmetric** (default): "fox-jumped" = "jumped-fox" (same edge)
- **Directional**: "fox→jumped" ≠ "jumped→fox" (separate edges, captures word order)

### Context Distribution Smoothing (CDS)

**What it is**: A statistical technique that reduces the dominance of very frequent words by raising their counts to a fractional power (typically α=0.75).

**Why it matters**: Without smoothing, the most common words (like "said", "people", "time") would dominate co-occurrence patterns, even though they're often semantically uninformative. CDS balances the contributions of frequent and rare words.

**Formula**: 
```
smoothed_count(w) = count(w)^α
```

Where α is typically 0.75 (can range from 0.5 to 1.0)

**Effect on word counts**:

| Original Count | α=0.75 | α=0.5 | Explanation |
|----------------|--------|-------|-------------|
| 10,000 | 1,778 | 100 | Very common word heavily downweighted |
| 1,000 | 316 | 32 | Common word moderately downweighted |
| 100 | 31 | 10 | Mid-frequency word slightly downweighted |
| 10 | 5.6 | 3.2 | Rare word minimally affected |

**Ratio compression example**:
- Original: 10,000 / 100 = 100:1 ratio
- With α=0.75: 1,778 / 31 = 57:1 ratio
- With α=0.5: 100 / 10 = 10:1 ratio

**When to adjust**:
- **α=1.0** (no smoothing): When all words are equally important
- **α=0.75** (default): Balanced, works well for most text
- **α=0.5** (aggressive): Strong smoothing for highly skewed distributions

**Our implementation**:
```bash
python -m src.semantic.build_semantic_network \
  --input data.csv \
  --cds 0.75  # default value
```

## Network Analysis Concepts

### Nodes and Edges

**Nodes** (vertices): Represent entities in the network
- **Semantic networks**: Words or terms
- **Knowledge graphs**: Named entities (people, places, organizations)
- **Actor networks**: Users, authors, or accounts
- **Document networks**: Individual documents

**Edges** (links): Represent relationships between nodes
- **Undirected**: No inherent direction (mutual co-occurrence)
- **Directed**: Has direction (A→B, like replies or citations)
- **Weighted**: Strength of connection (PPMI, similarity score)
- **Unweighted**: Binary (connected or not)

**Network density**:
```
Density = (actual edges) / (possible edges)
        = E / (N × (N-1) / 2)   # for undirected networks
```

Where E = edge count, N = node count

**Dense vs Sparse**:
- Dense (>10%): Highly interconnected, may be hard to interpret
- Sparse (<1%): Easy to visualize, focuses on strong relationships
- Very sparse (<0.1%): May miss important patterns

### Graph Sparsification (Top-K)

**What it is**: Keeping only the K strongest edges per node to reduce network size and complexity.

**Why it matters**:
1. **Computational**: Full networks scale as O(N²) edges
2. **Visual**: Dense networks are unreadable
3. **Analytical**: Weak edges add noise, not signal
4. **Storage**: Sparse networks are much smaller

**Top-K strategies**:

| K Value | Result | Use Case |
|---------|--------|----------|
| 5-10 | Very sparse | Quick visualization, core relationships only |
| 15-20 | Balanced | General analysis, standard setting |
| 30-50 | Detailed | Comprehensive analysis, research |
| >100 | Nearly complete | Mathematical analysis, algorithms |

**Example**: With 10,000 vocabulary words:
- **No sparsification**: Up to ~50M edges (10k × 10k / 2)
- **Top-K=20**: Exactly 200K edges (10k × 20)
- **Reduction**: 250× smaller!

**Implementation**:
For each node, we:
1. Rank all potential edges by weight (PPMI or similarity)
2. Keep only the top K highest-weighted edges
3. Discard the rest

**Trade-offs**:
- **Low K**: Cleaner, faster, but may miss important weak connections
- **High K**: More complete, but noisier and slower

### Community Detection

**What it is**: Algorithms that identify groups of nodes that are more densely connected to each other than to the rest of the network.

**Why it matters**: Communities often correspond to:
- **Topics**: Clusters of semantically related words
- **Themes**: Groups of co-occurring concepts
- **Social groups**: Cliques in social networks
- **Functional modules**: Related entities in knowledge graphs

**Louvain Algorithm**:

The Louvain algorithm is a hierarchical method that:
1. Starts with each node in its own community
2. Iteratively moves nodes to maximize modularity
3. Aggregates communities into super-nodes
4. Repeats until no improvement

**Modularity**: Measures how good a community division is:
```
Q = (1/2m) Σ[Aij - (ki×kj)/(2m)] δ(ci, cj)
```

Where:
- Aij = edge weight between i and j
- ki, kj = degree of nodes i and j
- m = total edge weight
- δ(ci, cj) = 1 if i and j in same community, else 0

**Interpretation**:
- Q > 0.3: Significant community structure
- Q > 0.5: Strong community structure
- Q < 0.1: Weak or no communities

**Usage**:
```bash
python -m src.semantic.community_cli \
  --edges edges.csv \
  --outdir output/
```

**Output**: CSV with node→community mappings, allowing you to:
- Color nodes by community in visualizations
- Analyze community characteristics
- Track community evolution over time

### Network Metrics

**Node-level metrics**:

| Metric | Measures | Interpretation |
|--------|----------|----------------|
| **Degree** | Number of connections | How connected a word/entity is |
| **Strength** | Sum of edge weights | Total association strength |
| **Betweenness** | How often node is on shortest paths | Bridge between topics |
| **Eigenvector** | Connections to important nodes | Influence/prestige |
| **Clustering** | How interconnected neighbors are | Local cohesion |

**Network-level metrics**:

| Metric | Measures | Interpretation |
|--------|----------|----------------|
| **Average degree** | Mean connections per node | Overall connectivity |
| **Diameter** | Longest shortest path | Network span |
| **Average path length** | Mean shortest path | Navigability |
| **Clustering coefficient** | Tendency to form triangles | Local structure |
| **Modularity** | Community structure strength | Topic separation |

## NLP and Transformer Concepts

### Named Entity Recognition (NER)

**What it is**: Automatically identifying and classifying named entities in text into predefined categories.

**Entity types**:

| Type | Description | Examples |
|------|-------------|----------|
| **PERSON** | People, including fictional | "Barack Obama", "Sherlock Holmes" |
| **ORG** | Organizations | "Microsoft", "UN", "Harvard" |
| **GPE** | Geopolitical entities | "Paris", "California", "EU" |
| **LOC** | Non-GPE locations | "Mount Everest", "Pacific Ocean" |
| **DATE** | Absolute or relative dates | "June 1st", "yesterday", "2024" |
| **TIME** | Times smaller than a day | "3pm", "midnight" |
| **MONEY** | Monetary values | "$100", "€50", "ten dollars" |
| **PERCENT** | Percentage | "20%", "fifty percent" |
| **PRODUCT** | Objects, vehicles, foods | "iPhone", "Toyota Camry" |
| **EVENT** | Named events | "World War II", "Olympics" |
| **WORK_OF_ART** | Titles of creative works | "Hamlet", "Mona Lisa" |

**Accuracy considerations**:

| Model | F1 Score | Speed | Use Case |
|-------|----------|-------|----------|
| `en_core_web_sm` | ~85% | Fast | Quick prototyping, large datasets |
| `en_core_web_md` | ~88% | Medium | Balanced accuracy/speed |
| `en_core_web_lg` | ~90% | Slow | Better accuracy |
| `en_core_web_trf` | ~92% | Very slow | Best accuracy, small datasets |

### Transformers

**What they are**: Neural network architectures (BERT, RoBERTa, GPT) that process text bidirectionally using self-attention mechanisms.

**Key innovation**: Unlike older models (Word2Vec, GloVe) that give each word a single vector, transformers generate **contextualized embeddings** where the same word has different representations depending on context.

**Example**:
```
"I went to the bank to deposit money"     → bank₁ (financial)
"I sat on the river bank and fished"      → bank₂ (geographic)
```

Traditional embeddings: bank = same vector in both
Transformers: bank₁ ≠ bank₂ (different contexts)

**Attention mechanism**: For each word, the model looks at all other words to determine importance weights.

Example sentence: "The cat sat on the mat"

When encoding "sat", attention might focus heavily on:
- "cat" (subject)
- "on" (preposition indicating relation)
- "mat" (object)

And less on:
- "The" (determiner, less semantic content)

**Models we use**:

| Model | Size | Dimensions | Speed | Use Case |
|-------|------|------------|-------|----------|
| `all-MiniLM-L6-v2` | 23M params | 384 | Fast | Default, balanced |
| `all-mpnet-base-v2` | 110M params | 768 | Medium | Better quality |
| `paraphrase-multilingual` | 118M params | 384 | Medium | Non-English text |

### Embeddings

**What they are**: Dense vector representations of words or documents in continuous space (typically 100-1000 dimensions).

**Key property**: Semantic similarity ≈ geometric proximity

**Example in 2D** (actual embeddings are 384+ dimensions):
```
     king •─────────────• queen
      |                    |
      |                    |
      |                    |
     man •─────────────• woman
```

The relationship "king→queen" is similar to "man→woman" in embedding space.

**Cosine similarity**:
```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Range: [-1, 1]
- 1.0: Identical vectors
- 0.0: Orthogonal (unrelated)
- -1.0: Opposite directions

**Our usage**:
```python
from src.semantic.transformers_enhanced import TransformerEmbeddings

embedder = TransformerEmbeddings()
embeddings = embedder.encode(texts)  # texts → 384-dim vectors
similarity_matrix = embedder.compute_similarity_matrix(embeddings)
```

### Sentence Transformers

**What they are**: Transformer models specifically fine-tuned to generate semantically meaningful sentence-level embeddings.

**How they work**:
1. Process full sentence through BERT-like model
2. Apply pooling (usually mean pooling) over all token embeddings
3. Optionally add dense layer for final embedding
4. Fine-tune with siamese/triplet network on sentence pairs

**Training data**: Millions of sentence pairs with similarity labels:
- Similar: paraphrases, translations, entailment
- Dissimilar: random pairs, contradictions

**Advantages over averaging word vectors**:
- Captures sentence structure
- Better semantic similarity
- Consistent dimensionality
- Pre-trained on similarity task

## Data Processing

### Tokenization

**What it is**: Breaking text into discrete units (tokens) for processing.

**Our pipeline**:
```
Raw text → Lowercase → Remove punctuation → Split on whitespace
         → Remove stopwords → Filter by length → Output tokens
```

**Example**:
```
Input:  "The quick brown fox jumped over the lazy dog!"
After lowercase: "the quick brown fox jumped over the lazy dog!"
After punctuation removal: "the quick brown fox jumped over the lazy dog"
After tokenization: ["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]
After stopword removal: ["quick", "brown", "fox", "jumped", "lazy", "dog"]
Final tokens: ["quick", "brown", "fox", "jumped", "lazy", "dog"]
```

**Stopwords**: Common words removed because they carry little semantic content:
```
a, an, the, is, are, was, were, be, been, being, have, has, had, 
do, does, did, will, would, could, should, may, might, must,
of, for, in, on, at, to, from, with, by, about, as, into
```

**Length filtering**: Remove very short tokens (usually <2 characters)
- Keeps: "go", "AI", "US"
- Removes: "a", "I", "x" (unless specifically needed)

### Document Frequency (DF)

**What it is**: The number of documents containing a term (not the total occurrences).

**Example**:
```
Doc 1: "the cat sat on the mat"  → cat appears once
Doc 2: "the cat and the dog"     → cat appears once
Doc 3: "birds fly high"           → cat doesn't appear

cat: DF = 2 (appears in 2 documents)
cat: TF = 2 (total occurrences = 2)
```

**Min-DF filtering**: Remove terms with DF < threshold

| min-df | Effect | Use Case |
|--------|--------|----------|
| 1 | Keep all terms | Small, clean datasets |
| 2-3 | Remove hapax legomena (once-only) | Reduce typos |
| 5-10 | Remove rare terms | Standard cleaning |
| 20-50 | Remove very rare terms | Large, noisy datasets |
| 100+ | Keep only common terms | Focus on major topics |

**Max-DF filtering**: Remove terms appearing in too many documents
```bash
--max-df 0.8  # Remove terms in >80% of documents
```

Removes ubiquitous terms that don't distinguish documents.

### Term Frequency (TF)

**What it is**: Total number of times a term appears across all documents.

**DF vs TF**:
```
Doc 1: "cat cat cat dog"
Doc 2: "dog dog bird"
Doc 3: "bird bird bird bird"

       TF    DF
cat:    3     1
dog:    3     2
bird:   5     2
```

**TF-IDF**: Combines TF and DF to identify important terms:
```
TF-IDF(term, doc) = TF(term, doc) × log(N / DF(term))
```

Where N = total documents

High TF-IDF = frequent in this document, rare overall = distinctive term

## When to Use What

### Method Comparison

| Method | Speed | Accuracy | Dataset Size | Semantic Depth | Use Case |
|--------|-------|----------|--------------|----------------|----------|
| **Co-occurrence + PPMI** | ⚡⚡⚡ Fast | ⭐⭐ Good | Any | Surface-level | Quick analysis, large data |
| **Transformers (document)** | 🐌 Slow | ⭐⭐⭐ Best | <10K docs | Deep semantic | High-quality, smaller data |
| **Transformers (term)** | 🚶 Medium | ⭐⭐⭐ Best | <50K terms | Deep semantic | Vocabulary analysis |
| **Knowledge Graph** | ⚡⚡ Fast | ⭐⭐ Good | Any | Entity-focused | Entity relationships |
| **Actor Network** | ⚡⚡⚡ Very fast | ⭐⭐⭐ Exact | Any | Social structure | Social networks |

### Decision Tree

```
Do you have entity data (people, orgs, places)?
├─ Yes → Knowledge Graph (kg_cli)
└─ No → Do you need deep semantic understanding?
    ├─ Yes → Dataset < 10K documents?
    │   ├─ Yes → Transformer Network (transformers_cli)
    │   └─ No → Co-occurrence + sample/chunk
    └─ No → How large is your dataset?
        ├─ < 100K docs → Co-occurrence (build_semantic_network)
        └─ > 100K docs → Co-occurrence + GPU + sparsification
```

### Performance Expectations

**10,000 documents, 10,000 vocab, window=10, top-k=20**:

| Method | Hardware | Time | Memory |
|--------|----------|------|--------|
| Co-occurrence | CPU (8 cores) | 2 min | 500 MB |
| Co-occurrence | GPU (RTX 3090) | 30 sec | 2 GB |
| Transformers (doc) | CPU | 15 min | 1 GB |
| Transformers (doc) | GPU | 3 min | 4 GB |
| Transformers (term) | CPU | 5 min | 800 MB |
| Knowledge Graph | CPU | 2 min | 400 MB |

**Scaling behavior**:

| Dataset Size | Co-occurrence | Transformers | Knowledge Graph |
|--------------|---------------|--------------|-----------------|
| 1K docs | 10 sec | 1 min | 10 sec |
| 10K docs | 2 min | 15 min | 2 min |
| 100K docs | 20 min | Hours (sample!) | 20 min |
| 1M docs | 3 hours (GPU) | Not recommended | 3 hours |

## Further Reading

- **PPMI Theory**: Church & Hanks (1990), "Word Association Norms, Mutual Information, and Lexicography"
- **Transformers**: Vaswani et al. (2017), "Attention Is All You Need"
- **Sentence Transformers**: Reimers & Gurevych (2019), "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- **Community Detection**: Blondel et al. (2008), "Fast unfolding of communities in large networks"
- **Network Analysis**: Newman (2018), "Networks: An Introduction"

## Glossary

- **PPMI**: Positive Pointwise Mutual Information - statistical association measure
- **Co-occurrence**: Words appearing near each other in text
- **CDS**: Context Distribution Smoothing - reducing frequency bias
- **Top-K**: Keeping only K strongest edges per node
- **Embedding**: Dense vector representation of text
- **Cosine similarity**: Angle-based similarity measure for vectors
- **NER**: Named Entity Recognition - identifying people, places, etc.
- **Transformer**: Attention-based neural network architecture
- **Community**: Densely connected group of nodes in network
- **Modularity**: Quality measure for community detection
- **DF**: Document Frequency - documents containing a term
- **TF**: Term Frequency - total occurrences of a term
- **Sparsification**: Reducing network density by removing weak edges
