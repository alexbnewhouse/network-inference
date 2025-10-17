# Network Inference: Semantic & Knowledge Graph Analysis

A comprehensive toolkit for extracting, analyzing, and visualizing semantic networks, knowledge graphs, and actor networks from text data. Built with scalability in mind, supporting GPU acceleration and transformer-based models.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Overview](#overview)
- [Key Concepts & Terminology](#key-concepts--terminology)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Advanced Features](#advanced-features)
- [API Reference](#api-reference)

## Overview

This toolkit provides multiple approaches to extract and analyze relationships from text:

1. **Semantic Networks**: Co-occurrence-based networks using PPMI weighting
2. **Transformer Networks**: Embedding-based semantic similarity networks  
3. **Knowledge Graphs**: Entity-relation extraction with NER
4. **Actor Networks**: Reply/mention networks from threaded discussions
5. **Community Detection**: Identify clusters and communities in networks
6. **Temporal Analysis**: Time-sliced network evolution

## Key Concepts & Terminology

### Core Statistical Methods

#### PPMI (Positive Pointwise Mutual Information)

**What it is**: A statistical measure that quantifies how much more often two words appear together than would be expected by chance.

**Why it matters**: PPMI weights in semantic networks indicate genuine associations between words, filtering out coincidental co-occurrences.

**Formula**: 
```
PMI(w1, w2) = log2(P(w1, w2) / (P(w1) × P(w2)))
PPMI(w1, w2) = max(0, PMI(w1, w2))
```

Where:
- `P(w1, w2)` = probability of w1 and w2 appearing together
- `P(w1)` = probability of w1 appearing
- `P(w2)` = probability of w2 appearing

**Example**: If "climate" and "change" appear together much more often than their individual frequencies would predict, they have high PPMI.

#### Co-occurrence

**What it is**: The phenomenon of words appearing near each other in text within a defined window (e.g., 10 words).

**Why it matters**: Co-occurrence patterns reveal semantic relationships - words that co-occur frequently are often semantically related.

**Window size**: The number of words to look before and after a target word. A window of 10 means we check 10 words in each direction.

**Example**: In "The quick brown fox jumped", with a window of 2:
- "quick" co-occurs with: "The", "brown", "fox"
- "brown" co-occurs with: "The", "quick", "fox", "jumped"

#### Context Distribution Smoothing (CDS)

**What it is**: A technique that reduces the influence of very frequent words in co-occurrence calculations by raising word counts to a power (typically 0.75).

**Why it matters**: Without smoothing, very common words would dominate the network. CDS balances the contribution of frequent and rare words.

**Formula**: `smoothed_count(w) = count(w)^0.75`

**Example**: A word appearing 10,000 times gets smoothed to ~1778, while a word appearing 100 times smooths to ~31 - reducing the ratio from 100:1 to 57:1.

### Network Analysis Concepts

#### Nodes and Edges

**Nodes**: In semantic networks, nodes represent words/terms. In knowledge graphs, nodes represent entities. In actor networks, nodes represent users/authors.

**Edges**: Connections between nodes. In semantic networks, edge weights represent PPMI scores. In knowledge graphs, edges represent relationships.

#### Graph Sparsification (Top-K)

**What it is**: Keeping only the top K strongest edges for each node to reduce network density.

**Why it matters**: Large networks can be overwhelming and computationally expensive. Top-K filtering keeps the most important relationships while making visualization and analysis tractable.

**Example**: With top-K=20, each word in the semantic network connects to only its 20 most strongly associated words.

#### Community Detection

**What it is**: Algorithms that identify clusters of densely connected nodes in a network.

**Why it matters**: Communities often represent topics, themes, or social groups. They help understand the structure of large networks.

**Methods used**: 
- Louvain algorithm: Fast, hierarchical community detection
- Modularity optimization: Finds divisions that maximize within-community connections

### NLP & Transformer Concepts

#### Named Entity Recognition (NER)

**What it is**: Identifying and classifying named entities (people, organizations, locations, dates, etc.) in text.

**Why it matters**: Entities are the building blocks of knowledge graphs and help identify key actors and concepts.

**Entity types**: PERSON, ORG (organization), GPE (geopolitical entity), DATE, MONEY, etc.

#### Transformers

**What it is**: Neural network architectures (like BERT, RoBERTa) that understand text context bidirectionally using attention mechanisms.

**Why it matters**: Transformers capture semantic meaning better than co-occurrence methods, understanding that "bank" in "river bank" differs from "bank account".

**Models available**:
- `all-MiniLM-L6-v2`: Fast, lightweight sentence embeddings
- `en_core_web_trf`: spaCy transformer for NER and parsing
- `bert-base-uncased`: Original BERT for various tasks

#### Embeddings

**What it is**: Dense vector representations of words or documents in continuous space where semantic similarity corresponds to geometric proximity.

**Why it matters**: Embeddings enable semantic similarity calculations and neural network processing of text.

**Example**: "king" - "man" + "woman" ≈ "queen" in embedding space.

### Data Processing Terms

#### Tokenization

**What it is**: Breaking text into individual words or subword units (tokens).

**Why it matters**: The first step in all text analysis - how you tokenize affects everything downstream.

**Our approach**: 
- Lowercase normalization
- Stopword removal (common words like "the", "is", "and")
- Minimum token length (>1 character)
- Alphanumeric filtering

#### Document Frequency (DF)

**What it is**: The number of documents in which a term appears.

**Why it matters**: Very rare terms (low DF) may be typos or noise. Very common terms (high DF) may be uninformative. 

**Min-DF filtering**: We exclude terms appearing in fewer than min_df documents (default: 5) to reduce vocabulary size and noise.

#### Term Frequency (TF)

**What it is**: The total number of times a term appears across all documents.

**Why it matters**: Indicates term importance but must be balanced with DF to avoid overweighting common words.

## Features

### 🔤 Semantic Co-occurrence Networks

- **PPMI-weighted edges**: Statistical significance testing for word associations
- **Context window**: Configurable sliding window (default: 10 words)
- **GPU acceleration**: CuPy support for matrix operations
- **Vocabulary filtering**: Min document frequency and max vocabulary size
- **Top-K sparsification**: Keep only strongest relationships per node
- **Multiple output formats**: CSV (nodes/edges), GraphML

### 🤖 Transformer-Enhanced Analysis

- **Sentence embeddings**: Pre-trained models via sentence-transformers
- **Semantic similarity networks**: Document and term networks based on embeddings
- **Enhanced NER**: Transformer-based entity recognition (en_core_web_trf)
- **Topic modeling**: BERTopic integration for automatic topic discovery
- **GPU support**: CUDA and Apple Silicon (MPS) acceleration

### 🕸️ Knowledge Graph Extraction

- **Named Entity Recognition**: spaCy-based entity extraction
- **Entity linking**: Wikidata and custom gazetteer support
- **Relation extraction**: Pattern-based and dependency-based methods
- **Property graphs**: Node/edge CSV and GraphML export

### 👥 Actor Network Analysis

- **Reply network extraction**: Parse >>quote patterns in threaded discussions
- **Author identification**: Handle tripcodes, capcodes, and per-thread IDs
- **Network metrics**: Degree centrality, thread statistics
- **Flexible column mapping**: Adapt to various data schemas

### 🕐 Temporal Analysis

- **Time-sliced networks**: Build separate networks for time periods
- **Evolution tracking**: Compare network structure over time
- **Configurable frequencies**: Monthly, weekly, or custom slicing

### 📊 Community Detection

- **Louvain algorithm**: Fast, multi-scale community detection
- **Modularity optimization**: Find natural divisions in networks
- **Community statistics**: Size, density, internal/external edges

### 🎨 Visualization

- **NetworkX integration**: Standard graph visualization
- **pyvis support**: Interactive HTML network visualizations
- **Matplotlib/Seaborn**: Statistical plots and distributions
- **GraphML export**: Compatible with Gephi, Cytoscape, etc.

## Installation

### Requirements

- **Python 3.12+** (3.14 has some package compatibility issues)
- **Operating Systems**: macOS, Linux, Windows
- **Optional**: CUDA for GPU acceleration (Linux/Windows)

### Basic Installation

```bash
# Clone repository
git clone https://github.com/alexbnewhouse/network-inference.git
cd network-inference

# Create virtual environment (Python 3.12 recommended)
python3.12 -m venv .venv312
source .venv312/bin/activate  # On Windows: .venv312\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Enhanced Installation (with transformers)

```bash
# Install with transformer support
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_sm      # Small model (fast)
python -m spacy download en_core_web_trf     # Transformer model (accurate)

# Install optional packages
pip install sentence-transformers            # Sentence embeddings
pip install bertopic                         # Topic modeling
pip install scikit-learn                     # ML utilities
```

### GPU Support (Linux/Windows)

```bash
# For NVIDIA CUDA
pip install cupy-cuda12x

# Verify GPU availability
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

## Quick Start

### 1. Build a Basic Semantic Network

```bash
# Using co-occurrence with PPMI weighting
python -m src.semantic.build_semantic_network \
  --input your_data.csv \
  --outdir output/semantic \
  --min-df 5 \
  --window 10 \
  --topk 20
```

**Outputs**:
- `output/semantic/nodes.csv`: Vocabulary with frequencies
- `output/semantic/edges.csv`: PPMI-weighted edges
- `output/semantic/graph.graphml`: Network for visualization

### 2. Extract a Knowledge Graph

```bash
# Using transformer-based NER
python -m src.semantic.kg_cli \
  --input your_data.csv \
  --outdir output/kg \
  --model en_core_web_trf \
  --max-rows 10000
```

**Outputs**:
- `output/kg/kg_nodes.csv`: Entities with types
- `output/kg/kg_edges.csv`: Entity co-occurrences

### 3. Build Transformer-Based Network

```bash
# Using sentence embeddings
python -m src.semantic.transformers_cli \
  --input your_data.csv \
  --outdir output/transformer \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --similarity-threshold 0.5 \
  --top-k 20 \
  --device cpu
```

### 4. Detect Communities

```bash
# Run Louvain community detection
python -m src.semantic.community_cli \
  --nodes output/semantic/nodes.csv \
  --edges output/semantic/edges.csv \
  --outdir output/communities
```

## Usage Guide

### Data Format

Your input CSV should have at minimum:

- `text` (required): The text content to analyze
- `subject` (optional): Thread subject or document title
- `timestamp` (optional): For temporal analysis

**Example CSV**:
```csv
text,subject,timestamp
"This is the first document","Topic A",2024-01-01
"Here is another document","Topic B",2024-01-02
```

### Semantic Network Pipeline

The semantic network pipeline processes text through several stages:

```
Text → Tokenization → Vocabulary Building → Co-occurrence → PPMI → Sparsification → Graph
```

**Command line options**:

```bash
python -m src.semantic.build_semantic_network \
  --input data.csv \              # Input CSV file
  --outdir output/ \              # Output directory
  --min-df 5 \                    # Min document frequency
  --max-vocab 50000 \             # Max vocabulary size
  --window 10 \                   # Co-occurrence window
  --topk 20 \                     # Top-K edges per node
  --cds 0.75 \                    # Context distribution smoothing
  --max-rows 100000 \             # Limit rows (for testing)
  --gpu \                         # Use GPU acceleration
  --spacy-gpu \                   # Use spaCy GPU tokenization
  --igraph                        # Use igraph instead of networkx
```

### Knowledge Graph Extraction

Extract entities and their relationships:

```bash
python -m src.semantic.kg_cli \
  --input data.csv \
  --outdir output/kg \
  --model en_core_web_sm \        # or en_core_web_trf for better accuracy
  --max-rows 5000
```

**Entity types extracted**: PERSON, ORG, GPE, DATE, MONEY, PRODUCT, EVENT, etc.

### Actor Network Analysis

For discussion forums or social media with reply chains:

```bash
python -m src.semantic.actor_cli \
  --input data.csv \
  --outdir output/actors \
  --thread-col subject \          # Column for thread IDs
  --post-col index \              # Column for post IDs (or 'index')
  --text-col text \               # Column with text content
  --author-tripcode-col tripcode  # Optional author identifier columns
```

**The pipeline looks for reply patterns like**: `>>12345` to build reply networks.

### Time-Sliced Analysis

Track network evolution over time:

```bash
python -m src.semantic.time_slice_cli \
  --input data.csv \
  --outdir output/timeslices \
  --slice-col timestamp \         # Datetime column
  --freq M                        # M=monthly, W=weekly
```

**Outputs**: Separate network for each time slice in `output/timeslices/slice_YYYY-MM/`

### Community Detection

Identify clusters in existing networks:

```bash
python -m src.semantic.community_cli \
  --nodes output/semantic/nodes.csv \
  --edges output/semantic/edges.csv \
  --outdir output/communities
```

**Output**: `communities.csv` with node-to-community assignments

## Advanced Features

### GPU Acceleration

For large datasets, GPU acceleration can provide significant speedups:

```bash
# Semantic network with GPU
python -m src.semantic.build_semantic_network \
  --input large_data.csv \
  --outdir output/ \
  --gpu \                         # Use CuPy for matrix operations
  --spacy-gpu                     # Use GPU for tokenization
```

**Requirements**: NVIDIA GPU, CUDA toolkit, cupy-cuda12x

### Transformer Models

Use state-of-the-art NLP models:

```bash
# Knowledge graph with transformer
python -m src.semantic.kg_cli \
  --input data.csv \
  --outdir output/ \
  --model en_core_web_trf         # Transformer-based spaCy model

# Embedding-based network
python -m src.semantic.transformers_cli \
  --input data.csv \
  --outdir output/ \
  --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --device cuda                   # Use GPU for encoding
```

### Phrase Detection

Detect and promote multi-word expressions:

```bash
python -m src.semantic.phrase_cli \
  --input data.csv \
  --outdir output/phrases \
  --min-count 10 \                # Minimum phrase frequency
  --min-pmi 5.0                   # Minimum PMI for phrase
```

**Example**: Detects "climate_change", "machine_learning", "new_york" as single tokens.

## API Reference

### Python API

Use the toolkit programmatically:

```python
from src.semantic.build_semantic_network import build_semantic_from_df
from src.semantic.kg_pipeline import KnowledgeGraphPipeline
from src.semantic.transformers_enhanced import TransformerSemanticNetwork
import pandas as pd

# Load data
df = pd.read_csv("data.csv")

# Build semantic network
build_semantic_from_df(
    df, 
    outdir="output/semantic",
    min_df=5,
    window=10,
    topk=20
)

# Extract knowledge graph
kg = KnowledgeGraphPipeline(ner_model="en_core_web_sm")
kg.run(df, "output/kg")

# Build transformer network
trans_net = TransformerSemanticNetwork(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
edges = trans_net.build_document_network(
    df["text"].tolist(),
    similarity_threshold=0.5,
    top_k=20
)
```

### Jupyter Notebooks

Explore interactively:

```python
# In Jupyter notebook
%load_ext autoreload
%autoreload 2

import pandas as pd
from src.semantic.visualize import load_graph, plot_degree_distribution

# Load network
G = load_graph("output/semantic")

# Analyze
plot_degree_distribution(G)
```

See `notebooks/explore_networks.ipynb` for full examples.

## Project Structure

```
network_inference/
├── src/
│   └── semantic/
│       ├── build_semantic_network.py   # Main semantic network pipeline
│       ├── transformers_enhanced.py    # Transformer models
│       ├── kg_pipeline.py              # Knowledge graph extraction
│       ├── actor_network.py            # Actor/reply networks
│       ├── community.py                # Community detection
│       ├── cooccur.py                  # Co-occurrence & PPMI
│       ├── graph_build.py              # Graph construction
│       ├── preprocess.py               # Tokenization & cleaning
│       ├── visualize.py                # Visualization utilities
│       └── *_cli.py                    # Command-line interfaces
├── notebooks/
│   └── explore_networks.ipynb          # Interactive analysis
├── tests/
│   ├── test_semantic_pipeline.py
│   └── test_kg_actor.py
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## Performance Tips

1. **Start small**: Test with `--max-rows 1000` before processing full datasets
2. **Use GPU**: For datasets >100K documents, GPU acceleration is recommended
3. **Adjust vocabulary**: Use `--max-vocab` to limit memory usage
4. **Top-K filtering**: Lower `--topk` values reduce output size and improve clarity
5. **Batch processing**: For very large datasets, process in chunks with time slicing

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'spacy'`
```bash
# Solution: Install spaCy and download model
pip install spacy
python -m spacy download en_core_web_sm
```

**Issue**: NumPy build errors on Python 3.14
```bash
# Solution: Use Python 3.12
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

**Issue**: Out of memory with large datasets
```bash
# Solution: Use sparsification and vocabulary limits
python -m src.semantic.build_semantic_network \
  --input data.csv \
  --max-vocab 50000 \
  --topk 10
```

**Issue**: Slow processing
```bash
# Solution: Enable GPU or use multiprocessing
export CUDA_VISIBLE_DEVICES=0
python -m src.semantic.build_semantic_network --gpu
```

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{network_inference_2024,
  title = {Network Inference: Semantic and Knowledge Graph Analysis Toolkit},
  author = {Newhouse, Alex},
  year = {2024},
  url = {https://github.com/alexbnewhouse/network-inference}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Built with:
- [spaCy](https://spacy.io/) - NLP pipeline
- [NetworkX](https://networkx.org/) - Graph analysis
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [scikit-learn](https://scikit-learn.org/) - ML utilities
- [Polars](https://www.pola.rs/) - Fast data processing

## Contact

- **GitHub**: [@alexbnewhouse](https://github.com/alexbnewhouse)
- **Repository**: [network-inference](https://github.com/alexbnewhouse/network-inference)

For bugs and feature requests, please open an issue on GitHub.
