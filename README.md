# Network Inference Toolkit

Fast, scalable semantic networks, knowledge graphs, and actor networks from text data. GPU-accelerated with transformer support.

[![Python 3.12-3.13](https://img.shields.io/badge/python-3.12--3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Recent Updates (Oct 2025):** 
> - **NEW: Automatic Scaling for Large Datasets** 🚀
>   - Transformer networks now process 1M+ documents automatically
>   - Memory-efficient batch processing (no FAISS dependency issues!)
>   - Works reliably on Python 3.12+ 
>   - No configuration needed - just run your command!
>   - See [SCALING_GUIDE.md](SCALING_GUIDE.md) and [TRANSFORMER_SCALING_UPDATE.md](TRANSFORMER_SCALING_UPDATE.md)
> - **NEW: GPU-Accelerated Transformer Sentiment Analysis** 🤖
>   - Contextual sentiment understanding (vs lexicon-based VADER)
>   - Optimized for RTX 5090 and high-end GPUs
>   - 85-90% accuracy on social media text (vs 75-80% VADER)
>   - Perfect for nuanced sentiment in extremist discourse
>   - See [GPU_SENTIMENT_GUIDE.md](GPU_SENTIMENT_GUIDE.md) for comprehensive guide
> - **NEW: Complete Onboarding Training Package** 🎓
>   - 60-90 minute hands-on tutorial with 10,000 synthetic posts
>   - Learn both anonymized and actor-based analysis
>   - Ethical decision-making and privacy techniques
>   - Real-world Iron March case study included
>   - See [ONBOARDING_TRAINING_GUIDE.md](ONBOARDING_TRAINING_GUIDE.md)
> - **NEW: Complete Knowledge Graph Platform for Social Scientists**
>   - Temporal analysis with event detection & trajectory classification
>   - Enhanced sentiment: stance detection, framing analysis, temporal trends
>   - User-entity networks with community detection
>   - Auto-generated quality reports
>   - See [KG_FOR_SOCIAL_SCIENTISTS.md](KG_FOR_SOCIAL_SCIENTISTS.md) for comprehensive guide
> - All CLIs now support config files (JSON/YAML) via `--config`
> - Multiple output formats: CSV, JSON, Parquet via `--output-format`
> - See [CHANGELOG_USABILITY.md](CHANGELOG_USABILITY.md) for full details

---

## New to This? Start Here!

**If you're a social scientist new to computational text analysis**, follow this path:

### 🎓 Complete Training Track (Recommended)
1. **[ONBOARDING_TRAINING_GUIDE.md](ONBOARDING_TRAINING_GUIDE.md)** - **NEW!** 60-90 min hands-on tutorial with 10K synthetic dataset
   - Part 1: Anonymized analysis (semantic networks, knowledge graphs, temporal)
   - Part 2: Actor-based analysis (user networks, communities, influence)
   - Part 3: Comparing approaches with decision matrix
   - Part 4: Ethical best practices and anonymization techniques
   - Includes real Iron March case study scenario

### 📖 Documentation Path (Self-Paced)
2. **[GETTING_STARTED.md](GETTING_STARTED.md)** - Step-by-step beginner guide (zero to first network in 15 min)
3. **[tutorials/01_hello_world.md](tutorials/01_hello_world.md)** - 5-minute hands-on tutorial with sample data
4. **[KG_FOR_SOCIAL_SCIENTISTS.md](KG_FOR_SOCIAL_SCIENTISTS.md)** - Complete guide to knowledge graphs for research
5. **[DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md)** - How to format your CSV files

**Already familiar with Python?** Jump to [Quick Start](#quick-start) below.

---

## Ethics & Privacy Warning

**IMPORTANT**: Before analyzing social media data, review **[ETHICS.md](ETHICS.md)** for:
- IRB requirements & exemptions for social media research
- Privacy & anonymization best practices
- Responsible reporting of sensitive data
- Platform-specific ethical considerations (4chan, Reddit, Twitter)

This toolkit is designed for **research purposes only**. Use responsibly and follow your institution's ethics guidelines.

---

## Quick Start

**First time?** Try this 2-minute demo with sample data:

```bash
# 1. Clone and enter the directory
git clone https://github.com/alexbnewhouse/network-inference.git
cd network-inference

# 2. Install dependencies (use Python 3.12 or 3.13 - NOT 3.14)
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Generate sample data and build your first network
python examples/sample_data.py
python -m src.semantic.build_semantic_network \
  --input examples/sample_news.csv \
  --outdir output/my_first_network \
  --min-df 2 --topk 15
```

**Output**: You'll get `nodes.csv`, `edges.csv`, and `graph.graphml` files ready to:
- Analyze in Excel/Python/R (CSV files)
- Visualize in [Gephi](https://gephi.org/) (graphml file)
- Import into your statistical software

**Next steps**: See [GETTING_STARTED.md](GETTING_STARTED.md) to analyze your own data

> **Python Version Note**: Use Python 3.12 or 3.13. Python 3.14 is too new and some dependencies don't compile yet.

## What You Can Build

**Choose your analysis type based on your research question:**

| Network Type | Best For | Example Research Questions | Speed |
|-------------|----------|---------------------------|-------|
| **Semantic Network** | Topic/concept relationships | "What ideas co-occur in climate change discourse?" | Fast (1-10 min) |
| **Knowledge Graph** | Entity relationships | "Which politicians are discussed together?" | Fast (1-5 min) |
| **Temporal KG** ✨ | Discourse evolution | "How did discussion of Russia change over time?" | Medium (10-60 min) |
| **Sentiment Analysis** ✨ | Stance & framing | "Are comments pro- or anti-immigration?" | Fast (5-15 min) |
| **User-Entity Networks** ✨ | Community detection | "Which users discuss similar topics?" | Fast (5-20 min) |
| **Actor Network** | Social relationships | "Who replies to whom in this forum?" | Fast (<1 min) |
| **Transformer Network** | Deep semantic similarity | "Which documents are semantically similar?" | Medium (5-30 min) |
| **Time-Sliced** | Network evolution | "How did the network structure change monthly?" | Medium (10-60 min) |

✨ = **New features in Oct 2025** - See [KG_FOR_SOCIAL_SCIENTISTS.md](KG_FOR_SOCIAL_SCIENTISTS.md) for detailed guides

**Not sure which to use?** Start with [GETTING_STARTED.md](GETTING_STARTED.md) - it walks through choosing the right tool.

## Installation

**Recommended**: Python 3.12 or 3.13 (Python 3.14 not yet supported)

### Basic Installation (CPU only - good for most use cases)

```bash
git clone https://github.com/alexbnewhouse/network-inference.git
cd network-inference
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Troubleshooting**: If you get errors, make sure you're using Python 3.12 or 3.13:
```bash
python --version  # Should show Python 3.12.x or 3.13.x
```

### With Transformers (Better accuracy)

```bash
# After basic installation
python -m spacy download en_core_web_trf
pip install sentence-transformers bertopic
```

### With GPU (Much faster for large datasets)

```bash
# NVIDIA GPU required
pip install cupy-cuda12x
```

## Common Workflows

### 1. Basic Semantic Network

**Best for**: Quick analysis, exploratory work, <100K documents

```bash
python -m src.semantic.build_semantic_network \
  --input data.csv \
  --outdir output/semantic \
  --min-df 5 \
  --topk 20
```

<details>
<summary>What this does</summary>

- Tokenizes text and builds vocabulary
- Computes word co-occurrences within a sliding window
- Weights edges with PPMI (statistical significance)
- Keeps top 20 strongest connections per word
- Outputs network in multiple formats

**Parameters**:
- `--min-df 5`: Ignore words appearing in <5 documents
- `--topk 20`: Keep only 20 strongest edges per node
- `--window 10`: Look ±10 words for co-occurrences (default)

</details>

### 2. Knowledge Graph Extraction

**Best for**: Finding entities (people, orgs, places) and their relationships

```bash
python -m src.semantic.kg_cli \
  --input data.csv \
  --outdir output/kg \
  --model en_core_web_sm \
  --max-rows 10000
```

<details>
<summary>What this does</summary>

- Runs Named Entity Recognition on text
- Extracts PERSON, ORG, GPE, DATE entities
- Creates co-occurrence network of entities
- Outputs entity nodes and relationship edges

**Use `en_core_web_trf` for better accuracy (slower)**

</details>

### 2b. Knowledge Graph with GPU-Accelerated Sentiment 🚀 NEW!

**Best for**: Nuanced sentiment analysis with contextual understanding

```bash
# Fast VADER sentiment (CPU, lexicon-based)
python -m src.semantic.kg_cli \
  --input data.csv \
  --outdir output/kg_sentiment \
  --add-sentiment \
  --sentiment-model vader

# Accurate transformer sentiment (GPU, contextual)
python -m src.semantic.kg_cli \
  --input data.csv \
  --outdir output/kg_sentiment \
  --add-sentiment \
  --sentiment-model transformer \
  --sentiment-device cuda \
  --sentiment-batch-size 128
```

<details>
<summary>What this does</summary>

**VADER (Default)**:
- Fast lexicon-based sentiment (~1000 docs/sec)
- Good for obvious sentiment patterns
- CPU-only, no special hardware needed

**Transformer (NEW)**:
- Contextual sentiment with RoBERTa trained on 58M tweets
- Detects sarcasm, irony, coded language
- 85-90% accuracy vs 75-80% for VADER
- GPU-accelerated (RTX 5090, A100, etc.)
- Entity-aware: analyzes sentiment in context windows around mentions

**Outputs**:
- `kg_nodes_with_sentiment.csv`: Entities with sentiment scores
- `entity_sentiment.csv`: Detailed sentiment statistics
- Controversy detection (high variance = contested entities)

**Models available**:
- `cardiffnlp/twitter-roberta-base-sentiment` (default, social media)
- `nlptown/bert-base-multilingual-uncased-sentiment` (5-class)
- `distilbert-base-uncased-finetuned-sst-2-english` (lightweight)

**See [GPU_SENTIMENT_GUIDE.md](GPU_SENTIMENT_GUIDE.md) for detailed guide and benchmarks**

</details>

### 3. Transformer Semantic Network

**Best for**: High-quality semantic similarity, smaller datasets

```bash
python -m src.semantic.transformers_cli \
  --input data.csv \
  --outdir output/transformer \
  --mode document \
  --similarity-threshold 0.5 \
  --top-k 20 \
  --max-rows 5000
```

<details>
<summary>What this does</summary>

- Encodes documents using sentence transformers
- Computes cosine similarity between all pairs
- Keeps similarities above threshold
- Creates network where edges = semantic similarity

**Modes**:
- `document`: Connect similar documents
- `term`: Connect similar vocabulary terms

</details>

### 4. Actor/Reply Network

**Best for**: Forums, social media, threaded discussions

```bash
python -m src.semantic.actor_cli \
  --input data.csv \
  --outdir output/actors \
  --text-col text \
  --thread-col thread_id \
  --post-col post_id
```

<details>
<summary>What this does</summary>

- Parses reply patterns (e.g., `>>12345`)
- Builds directed network of who replies to whom
- Computes centrality and participation metrics
- Works with tripcodes, IDs, or any author identifier

</details>

### 5. Time-Sliced Analysis

**Best for**: Tracking topic evolution, temporal trends

```bash
python -m src.semantic.time_slice_cli \
  --input data.csv \
  --outdir output/timeslices \
  --time-col timestamp \
  --freq M
```

<details>
<summary>What this does</summary>

- Splits data into time periods (M=monthly, W=weekly, D=daily)
- Builds separate semantic network for each period
- Enables comparison of network structure over time
- Tracks emergence/decline of topics

</details>

### 6. Community Detection

**Best for**: Finding topic clusters, identifying themes

```bash
python -m src.semantic.community_cli \
  --edges output/semantic/edges.csv \
  --outdir output/communities
```

<details>
<summary>What this does</summary>

- Runs Louvain algorithm to find communities
- Groups densely connected nodes
- Outputs community assignments
- Computes community statistics

</details>

### 7. Network Contagion & Diffusion

**NEW**: Simulate information spread, adoption cascades, and epidemic dynamics on your networks.

```bash
# Simple contagion (SI/SIS/SIR models)
python -m src.contagion.cli output/semantic/edges.csv \
  --model sir --beta 0.1 --gamma 0.05 --timesteps 100

# Complex contagion (Watts threshold model)
python -m src.contagion.cli_complex output/semantic/edges.csv \
  --model watts --phi 0.18 --timesteps 50 --output-dir results/

# Infer parameters from observed cascades
python -m src.contagion.cli_inference output/semantic/edges.csv \
  --model si --observed-final-size 150 --n-samples 20
```

<details>
<summary>What this does</summary>

- Simulates disease/information spread on any network
- Models: SI, SIS, SIR (simple), Watts/K-reinforcement (complex)
- Supports CPU, multiprocessing, and GPU acceleration
- Infers transmission parameters from observed data
- Outputs: cascade timeseries, adoption curves, events log

**See [CONTAGION.md](CONTAGION.md) for full guide**

</details>

### 8. Advanced Knowledge Graph Analysis (NEW)

**For social scientists**: Temporal analysis, sentiment, and user networks

<details>
<summary><b>Temporal KG: Track discourse evolution</b></summary>

```bash
# Generate weekly knowledge graphs with sentiment
python -m src.semantic.kg_cli \
  --input data.csv \
  --outdir output/kg_temporal \
  --text-col text \
  --time-col created_at \
  --group-by-time weekly \
  --add-sentiment

# Analyze entity timeline
python -m src.semantic.kg_temporal_cli \
  --kg-dir output/kg_temporal \
  --entity "Russia" \
  --report russia_timeline.md
```

**Features**:
- Timeline tracking (first/last seen, persistence, peak periods)
- Event detection (automatic spike identification)
- Trajectory classification (emerging/declining/stable/episodic)
- Period comparison (new/lost/growing entities)
</details>

<details>
<summary><b>Enhanced Sentiment: Stance & framing analysis</b></summary>

```bash
# Detect stance (pro/anti/neutral)
python -m src.semantic.kg_sentiment_enhanced_cli \
  --input data.csv \
  --text-col text \
  --entity "Russia" \
  --stance --framing --group-by board

# Track sentiment over time
python -m src.semantic.kg_sentiment_enhanced_cli \
  --temporal output/kg_temporal \
  --entity "Russia" \
  --trends --shifts
```

**Features**:
- Stance detection with pattern matching
- Framing analysis (how entities are described)
- Temporal sentiment tracking
- Group comparison (boards, communities)
- Shift detection (significant changes)
</details>

<details>
<summary><b>User-Entity Networks: Community detection</b></summary>

```bash
# Build user-entity bipartite network
python -m src.semantic.kg_user_entity_network_cli \
  --kg-dir output/kg \
  --data data.csv \
  --user-col user_id \
  --text-col text \
  --stats --communities --export-all output/networks
```

**Features**:
- Bipartite graphs (users ↔ entities)
- User-user projection (by shared interests)
- Entity-entity projection (by shared audiences)
- Community detection (identify user clusters)
- Similarity metrics (Jaccard, cosine, overlap)
</details>

**📖 Complete Guide**: [KG_FOR_SOCIAL_SCIENTISTS.md](KG_FOR_SOCIAL_SCIENTISTS.md) - 10,000+ word guide with examples, case studies, and research design advice

## Working with Your Own Data

**See [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md) for complete formatting guide**

### Minimum Requirements

Your data needs to be a **CSV file** with at least one text column:

```csv
text
"This is my first social media post about politics"
"Another post discussing healthcare policy"
"A third post mentioning both politics and healthcare"
```

That's it! Save this as `my_data.csv` and you're ready to analyze.

### Recommended Format (enables all features)

Include these optional columns to unlock advanced features:

```csv
text,timestamp,user_id,thread_id,post_id
"Post content","2024-01-01 12:00:00","user123","thread_1","post_1"
"Reply content","2024-01-01 12:05:00","user456","thread_1","post_2"
```

**What each column enables**:
- `text` (required) - The content to analyze
- `timestamp` - Time-sliced analysis, temporal trends
- `user_id` - User-entity networks, community detection
- `thread_id` - Actor networks, reply patterns
- `post_id` - Reply network construction

### Using Different Column Names

Your CSV has different column names? No problem! Just tell the tool:

```bash
# If your text column is called "message" instead of "text"
python -m src.semantic.build_semantic_network \
  --input data.csv \
  --text-col message \
  --outdir output
```

**Common variations**: `body`, `content`, `message`, `post_text`, `comment` all work fine.

### Real-World Examples

See **[REAL_DATA_USAGE.md](REAL_DATA_USAGE.md)** for:
- 4chan data formatting
- Reddit data formatting  
- Twitter/X data formatting
- Survey data formatting
- News article formatting

**📖 4chan-Specific Guide**: See **[CHAN_QUICKSTART.md](CHAN_QUICKSTART.md)** for a comprehensive guide to analyzing 4chan data, including:
- Complete workflow examples
- Board-level comparisons
- Entity evolution tracking
- Temporal analysis for ephemeral data
- Ethics and privacy best practices

## Output Files

All pipelines generate standard output files:

### Semantic Network Output

```
output/semantic/
├── nodes.csv          # Word, frequency, document_frequency
├── edges.csv          # source, target, similarity (PPMI weight)
└── graph.graphml      # Complete network for Gephi/Cytoscape
```

### Knowledge Graph Output

```
output/kg/
├── kg_nodes.csv       # Entity, type (PERSON/ORG/GPE/etc)
└── kg_edges.csv       # source, target, co-occurrence count
```

### Transformer Network Output

```
output/transformer/
└── transformer_edges.csv   # source, target, similarity (cosine)
```

**Note**: Transformer networks now **automatically scale** to large datasets:
- **<10K docs**: Uses full similarity matrix (fastest)
- **>10K docs**: Automatically uses memory-efficient batch processing (stable, production-ready)
- **Optional FAISS**: Available but has compatibility issues with Python 3.12+

No special installation needed! See [SCALING_GUIDE.md](SCALING_GUIDE.md) for processing 100K-1M+ documents.

## Performance Guide

### Dataset Size Recommendations

| Documents | Method | Time | Memory | GPU? | Notes |
|-----------|--------|------|--------|------|-------|
| <1K | Any | Seconds | <100MB | No | All methods work great |
| 1K-10K | Semantic/KG/Transformer | 1-5 min | <500MB | No | Transformers use full matrix |
| 10K-100K | Semantic/Transformer | 5-30 min | 1-4GB | Recommended | Auto-batch processing |
| 100K-1M | Transformer+GPU | 30-120 min | 4-20GB | Yes | Batch processing scales well |
| >1M | Transformer+GPU | 1-3 hours | 20GB+ | Yes | See SCALING_GUIDE.md |

**Transformers**: Now scale to millions of documents with automatic batch processing. No special installation needed!

### Speed Optimization

```bash
# 1. Limit vocabulary
--max-vocab 50000

# 2. Reduce network density
--topk 10

# 3. Use GPU
--gpu --spacy-gpu

# 4. Test on subset first
--max-rows 1000
```

### Memory Optimization

```bash
# 1. Increase min document frequency
--min-df 10

# 2. Reduce top-k
--topk 10

# 3. Use smaller vocabulary
--max-vocab 20000
```

## Examples

See **[examples/](examples/)** directory for:
- `1_transformer_networks.ipynb` - Complete transformer walkthrough
- `2_topic_modeling.ipynb` - BERTopic integration
- `3_comparison.ipynb` - Co-occurrence vs transformer comparison
- `sample_data.py` - Generate test datasets

Run benchmarks:
```bash
python benchmarks/benchmark_methods.py --docs 100 --repeats 3
```

## Python API

Use programmatically:

```python
import pandas as pd
from src.semantic.transformers_enhanced import TransformerSemanticNetwork

# Load data
df = pd.read_csv("data.csv")
texts = df["text"].tolist()

# Build network
builder = TransformerSemanticNetwork()
edges = builder.build_document_network(
    texts,
    similarity_threshold=0.5,
    top_k=20
)

# edges is a pandas DataFrame with columns: source, target, similarity
edges.to_csv("output/edges.csv", index=False)
```

See **[API.md](API.md)** for complete API documentation.

## Visualization

### Quick Visualization

```bash
python -m src.semantic.visualize_cli \
  --nodes output/semantic/nodes.csv \
  --edges output/semantic/edges.csv \
  --outdir output/viz
```

### Use with Gephi

1. Open Gephi
2. File → Open → Select `graph.graphml`
3. Apply Force Atlas 2 layout
4. Color by modularity class (communities)

### Interactive HTML

```python
from src.semantic.visualize import create_interactive_network

create_interactive_network(
    nodes_file="output/semantic/nodes.csv",
    edges_file="output/semantic/edges.csv",
    output_file="network.html"
)
```

## Troubleshooting

### Common Issues

<details>
<summary><b>ModuleNotFoundError: No module named 'spacy'</b></summary>

```bash
pip install spacy
python -m spacy download en_core_web_sm
```
</details>

<details>
<summary><b>Out of memory errors</b></summary>

```bash
# Reduce vocabulary and edges
python -m src.semantic.build_semantic_network \
  --input data.csv \
  --max-vocab 20000 \
  --topk 10 \
  --min-df 10
```
</details>

<details>
<summary><b>Processing is slow</b></summary>

```bash
# Enable GPU or reduce dataset
--gpu                    # Use CUDA acceleration
--max-rows 10000        # Process subset
--max-vocab 30000       # Limit vocabulary
```
</details>

<details>
<summary><b>Column not found errors</b></summary>

```bash
# Check your columns
python -c "import pandas as pd; print(pd.read_csv('data.csv').columns.tolist())"

# Use --text-col to specify
python -m src.semantic.build_semantic_network \
  --input data.csv \
  --text-col body  # or whatever your column is named
```
</details>

<details>
<summary><b>Python 3.14 compatibility issues</b></summary>

```bash
# Use Python 3.12
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```
</details>

## Advanced Topics

For in-depth explanations of methods and theory:

- **[CONCEPTS.md](CONCEPTS.md)** - Detailed explanation of PPMI, co-occurrence, transformers, embeddings
- **[TECHNICAL.md](TECHNICAL.md)** - Implementation details, algorithms, optimization
- **[QUICKSTART.md](QUICKSTART.md)** - Additional workflow examples

## Testing

Run tests:

```bash
# All tests
python -m unittest discover tests -v

# Specific test
python tests/test_transformers.py
```

All tests passing: ✅ (35/35)

## Best Practices

### Data Preparation

**Text Preprocessing**
- Clean your data before building networks (remove duplicates, handle missing values)
- For small datasets (<1K docs), use min_df=2-3; for large datasets (>10K), use min_df=10+
- Ensure text column has meaningful content (avoid very short documents)

**Performance Optimization**
- Use `--max-rows` to test on a subset before processing full datasets
- For large networks (>100K edges), use Parquet format: `--output-format parquet`
- Enable multiprocessing where available to speed up processing

**Network Quality**
- Start with restrictive parameters (higher min_df, lower topk) to reduce noise
- Use `--min-pmi` to filter weak co-occurrences (5.0+ recommended)
- For transformers, set `--similarity-threshold 0.6+` to keep only strong connections

### Workflow Recommendations

**1. Iterative Refinement**
```bash
# Start small
python -m src.semantic.build_semantic_network --input data.csv --outdir test/ --max-rows 1000

# Inspect results, then scale up
python -m src.semantic.build_semantic_network --input data.csv --outdir output/ --min-df 10
```

**2. Using Config Files**
```bash
# Create config.json with all parameters
python -m src.contagion.cli --config examples/contagion_config.json

# Override specific params from CLI
python -m src.contagion.cli --config config.json --beta 0.2
```

**3. Combining Multiple Networks**
- Build semantic network first (fast baseline)
- Add transformer edges for high-value terms
- Use knowledge graph to identify entities
- Combine networks using NetworkX merge operations

### Common Pitfalls

❌ **Don't**
- Process millions of documents without testing on a sample first
- Use default parameters without understanding your data
- Ignore edge weights (they contain important information)
- Skip data cleaning (garbage in = garbage out)

✅ **Do**
- Validate your network visually (use `visualize_cli`)
- Check node/edge counts before downstream analysis
- Save intermediate results to avoid reprocessing
- Use appropriate output formats (CSV for inspection, Parquet for performance)

## Troubleshooting

### Installation Issues

**Problem: spaCy model not found**
```bash
# Solution: Download the model explicitly
python -m spacy download en_core_web_sm
```

**Problem: CUDA/GPU errors**
```bash
# Solution: Fallback to CPU or install correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Or specify CPU explicitly
python -m src.semantic.transformers_cli --device cpu
```

**Problem: Import errors for optional dependencies**
```bash
# Solution: Install missing packages
pip install sentence-transformers  # For transformer models
pip install pyyaml                 # For YAML config files
pip install cupy-cuda12x           # For GPU acceleration (optional)
```

### Performance Issues

**Problem: Slow network building**
- **Solution**: Use `--max-rows` to limit input size, or increase `--min-df` to reduce vocabulary
- **Check**: Are you processing too many low-frequency terms? Increase min_df
- **Check**: Using transformers on CPU? Consider switching to semantic networks or use GPU

**Problem: Out of memory**
- **Solution**: Process in batches using `--max-rows`, or use time-sliced pipeline
- **Solution**: Increase min_df to reduce vocabulary size
- **Solution**: Lower topk parameter to reduce edge count

**Problem: Network too sparse/dense**
- **Too sparse**: Lower min_df, increase topk, decrease similarity thresholds
- **Too dense**: Increase min_df, decrease topk, increase similarity thresholds

### Output Issues

**Problem: Empty or tiny network**
- **Check**: Is min_df too high for your dataset size?
- **Check**: Does your data have a 'text' column? Use `--text-col` to specify
- **Check**: Are documents too short? Try lowering min_df or min_count

**Problem: Cannot open output files**
- **Solution**: Check file permissions and ensure output directory exists
- **Solution**: Try different output format: `--output-format json`

**Problem: Results don't match expectations**
- **Check**: Review the vocabulary in nodes.csv - are stopwords being included?
- **Check**: Inspect edge weights - are they meaningful?
- **Solution**: Visualize a sample: `python -m src.semantic.visualize_cli --outdir output/`

### Contagion Simulation Issues

**Problem: No spread / everyone infected immediately**
- **Solution**: Adjust beta (infection rate) - try values between 0.01-0.5
- **Check**: Network connectivity - disconnected components won't spread
- **Solution**: For complex contagion, adjust phi (threshold) parameter

**Problem: Inference not converging**
- **Solution**: Increase n_samples for grid search
- **Solution**: Widen parameter ranges (beta_min, beta_max)
- **Check**: Is observed_final_size realistic for your network size?

### Data Format Issues

**Problem: Column not found errors**
- **Solution**: Specify column names: `--source-col src --target-col dst`
- **Solution**: Check CSV headers match expected names
- **Solution**: For text data, use `--text-col column_name`

**Problem: Non-integer node IDs**
- **Don't worry**: The toolkit automatically converts string IDs to integers
- **Note**: Original IDs are preserved in the mapping

### Getting Help

If you encounter issues not covered here:

1. **Check error messages carefully** - they often contain the solution
2. **Run with `--max-rows 100`** to test on a small sample
3. **Review examples/** - working configurations for common use cases
4. **Open an issue** on GitHub with:
   - Command you ran
   - Error message (full traceback)
   - Sample of your data format (first few rows)
   - Python version and OS

## Project Structure

```
network_inference/
├── src/semantic/          # Core pipeline modules
│   ├── build_semantic_network.py
│   ├── transformers_enhanced.py
│   ├── kg_pipeline.py
│   └── *_cli.py          # Command-line interfaces
├── examples/              # Jupyter notebooks
├── tests/                 # Unit tests
├── benchmarks/            # Performance benchmarks
└── docs/                  # Additional documentation
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{network_inference_2024,
  title = {Network Inference: Semantic Network Analysis Toolkit},
  author = {Newhouse, Alex},
  year = {2024},
  url = {https://github.com/alexbnewhouse/network-inference}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/alexbnewhouse/network-inference/issues)
- **Docs**: [Full Documentation](https://github.com/alexbnewhouse/network-inference/wiki)
- **Email**: See GitHub profile

Built with [spaCy](https://spacy.io/), [NetworkX](https://networkx.org/), [Sentence Transformers](https://www.sbert.net/), and [scikit-learn](https://scikit-learn.org/).
