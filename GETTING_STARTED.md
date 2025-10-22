# Getting Started with Network Inference

**A step-by-step guide for absolute beginners**

This guide will get you from zero to your first network analysis in **15 minutes**.

---

## Who This Is For

- **Social scientists** who want to analyze text data at scale
- **Researchers** studying online communities, social movements, or discourse
- **Anyone** with text data (social media posts, surveys, news articles) who wants to find patterns

**No prior programming experience required** - just follow the steps!

---

## What You'll Learn

1. Install the toolkit
2. Run your first analysis
3. Understand the output
4. Choose the right tool for your research

---

## Prerequisites

### Required
- **Computer**: Mac, Windows, or Linux
- **Python 3.12 or 3.13**: [Download here](https://www.python.org/downloads/)
  - **Important**: Use Python 3.12 or 3.13, NOT 3.14 (too new, dependencies don't compile yet)
- **15 minutes** of your time

### Optional (but recommended)
- **Text editor**: [VS Code](https://code.visualstudio.com/) for viewing files
- **Gephi**: [Download here](https://gephi.org/) for visualizing networks

---

## Step 1: Installation (5 minutes)

### 1.1 Check Your Python Version

First, make sure you have Python 3.12 or 3.13:

```bash
python3 --version
# Should show: Python 3.12.x or Python 3.13.x
```

**Don't have the right version?** Download Python 3.12 from [python.org](https://www.python.org/downloads/)

### 1.2 Download the Toolkit

Open Terminal (Mac/Linux) or Command Prompt (Windows) and run:

```bash
# Download from GitHub
git clone https://github.com/alexbnewhouse/network-inference.git
cd network-inference
```

### 1.2 Download the Toolkit

Open Terminal (Mac/Linux) or Command Prompt (Windows) and run:

```bash
# Download from GitHub
git clone https://github.com/alexbnewhouse/network-inference.git
cd network-inference
```

**Don't have git?** Download the [ZIP file](https://github.com/alexbnewhouse/network-inference/archive/refs/heads/main.zip) and unzip it.

### 1.3 Create Virtual Environment

```bash
# Create isolated Python environment (use python3.12 or python3.13)
python3.12 -m venv .venv

# Activate it
source .venv/bin/activate  # Mac/Linux
# OR
.venv\Scripts\activate  # Windows
```

**You'll see `(.venv)` in your terminal** - this means it worked!

### 1.4 Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Download language model (required for text processing)
python -m spacy download en_core_web_sm
```

**This takes 2-3 minutes.** Get coffee! ☕

**If you get compilation errors**: Make sure you're using Python 3.12 or 3.13, not 3.14.

### 1.5 Verify Installation

```bash
# Check that everything works
python -c "import spacy, pandas, networkx; print('✓ Installation successful!')"
```

**If you see "✓ Installation successful!" - you're ready!**

**Problems?** See [Troubleshooting](#troubleshooting) below.

---

## Step 2: Your First Network (5 minutes)

We'll analyze 100 sample news headlines to create a semantic network showing which topics are connected.

### 2.1 Generate Sample Data

```bash
python examples/sample_data.py
```

This creates `examples/sample_news.csv` with 100 headlines about AI, climate, finance, and health.

### 2.2 Run Analysis

```bash
python -m src.semantic.build_semantic_network \
  --input examples/sample_news.csv \
  --outdir output/my_first_network \
  --min-df 2 \
  --topk 15
```

**What's happening:**
- Reading `sample_news.csv` (100 headlines)
- Finding words that co-occur frequently
- Building a network of related concepts
- Saving results to `output/my_first_network/`

**Takes ~10 seconds!** ⚡

### 2.3 Check Your Results

```bash
ls output/my_first_network/
```

You should see 3 files:
- **`nodes.csv`** - All words/concepts in the network
- **`edges.csv`** - Connections between words (with strength scores)
- **`graph.graphml`** - Network file for visualization

---

## Step 3: Understanding Your Results (3 minutes)

### 3.1 Look at the Nodes

```bash
head -20 output/my_first_network/nodes.csv
```

Example output:
```
term,frequency
climate,45
technology,38
research,32
...
```

**Interpretation**:
- Each row is a word/concept
- `frequency` = how often it appears
- Higher frequency = more central to the discourse

### 3.2 Look at the Edges

```bash
head -20 output/my_first_network/edges.csv
```

Example output:
```
source,target,weight
climate,change,8.342
technology,artificial,7.891
research,study,6.523
...
```

**Interpretation**:
- Each row connects two words
- `weight` = strength of association (higher = stronger)
- High weights mean words frequently appear together

### 3.3 Visualize (Optional)

**Option 1: Open in Gephi** (best for exploration)
1. Download [Gephi](https://gephi.org/)
2. Open Gephi → File → Open → Select `graph.graphml`
3. Click "Run" on ForceAtlas 2 layout
4. You'll see clusters of related concepts!

**Option 2: Quick Python visualization**

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_graphml("output/my_first_network/graph.graphml")
nx.draw(G, with_labels=True, node_size=50, font_size=8)
plt.savefig("my_network.png", dpi=300, bbox_inches='tight')
print("Saved to my_network.png!")
```

---

## Step 4: Analyze Your Own Data (2 minutes setup)

### 4.1 Prepare Your CSV

Your CSV needs **one text column**. That's it!

```csv
text
"Your first post or document"
"Your second post or document"
"Your third post or document"
```

**Save as**: `my_data.csv`

**See [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md) for details on optional columns (timestamps, user IDs, etc.)**

### 4.2 Run Analysis

```bash
python -m src.semantic.build_semantic_network \
  --input my_data.csv \
  --outdir output/my_analysis \
  --min-df 5 \
  --topk 20
```

**Parameters explained:**
- `--min-df 5` = ignore words appearing in <5 documents (reduces noise)
- `--topk 20` = keep 20 strongest connections per word (keeps network manageable)

**Adjust for your data size:**
- **<1K documents**: Use `--min-df 2 --topk 15`
- **1K-10K documents**: Use `--min-df 5 --topk 20` (default)
- **>10K documents**: Use `--min-df 10 --topk 20`

---

## Next Steps: Choosing Your Analysis

Now that you know the basics, pick the analysis that fits your research:

### For Social Scientists Studying Discourse

**→ Read [KG_FOR_SOCIAL_SCIENTISTS.md](KG_FOR_SOCIAL_SCIENTISTS.md)**

This 10,000-word guide covers:
- Knowledge graphs (entities, relationships, temporal analysis)
- Sentiment & stance detection
- User-entity networks
- Real research examples
- Ethics & IRB compliance

**Best for**: Tracking narratives, detecting events, measuring polarization

### For Fast Topic Analysis

**→ See [QUICKSTART.md](QUICKSTART.md)**

Quick reference with commands for:
- Semantic networks (concepts)
- Knowledge graphs (entities)
- Actor networks (users)
- Community detection
- Time-sliced analysis

**Best for**: Exploratory analysis, quick insights

### For Advanced Users

**→ See [API.md](API.md)** and **[TECHNICAL.md](TECHNICAL.md)**

Programming interface and technical details:
- Python API for custom pipelines
- Performance optimization
- Algorithm details (PPMI, embeddings, NER)
- Integration with other tools

**Best for**: Building custom analysis pipelines, scaling to millions of documents

---

## Common Research Workflows

### Workflow 1: "What topics are discussed in this dataset?"

```bash
# Build semantic network
python -m src.semantic.build_semantic_network \
  --input data.csv --outdir output/topics

# Detect communities (topic clusters)
python -m src.semantic.community_cli \
  --nodes output/topics/nodes.csv \
  --edges output/topics/edges.csv \
  --outdir output/communities
```

### Workflow 2: "Who/what are the key actors/entities?"

```bash
# Extract entities (people, places, organizations)
python -m src.semantic.kg_cli \
  --input data.csv \
  --outdir output/kg \
  --model en_core_web_sm \
  --add-sentiment
```

### Workflow 3: "How has discourse evolved over time?"

```bash
# Requires 'created_at' column in your CSV
python -m src.semantic.kg_cli \
  --input data.csv \
  --outdir output/temporal \
  --time-col created_at \
  --group-by-time monthly \
  --add-sentiment

# Analyze timeline
python -m src.semantic.kg_temporal_cli \
  --input output/temporal \
  --entity "Russia" \
  --report russia_timeline.md
```

### Workflow 4: "What are user communities/clusters?"

```bash
# Requires 'user_id' column
# First build knowledge graph
python -m src.semantic.kg_cli \
  --input data.csv \
  --outdir output/kg

# Then analyze user-entity patterns
python -m src.semantic.kg_user_entity_network_cli \
  --kg-dir output/kg \
  --data data.csv \
  --user-col user_id \
  --text-col text \
  --communities \
  --export-all output/networks
```

---

## Troubleshooting

### "Command not found: python3.12"
**Solution**: You don't have Python 3.12 installed.
- Download from [python.org](https://www.python.org/downloads/)
- Or use `python3.11` or `python3.10` (toolkit works with 3.10+)

### "pip: command not found"
**Solution**: Try `python -m pip` instead of `pip`:
```bash
python -m pip install -r requirements.txt
```

### "ModuleNotFoundError: No module named 'spacy'"
**Solution**: Activate your virtual environment first:
```bash
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

### "Memory Error" or computer freezes
**Solution**: Your dataset is too large. Try a subset:
```bash
--max-rows 1000  # Process first 1000 rows
```

### Analysis is very slow
**Solutions**:
1. **Use GPU** (if you have NVIDIA GPU):
   ```bash
   pip install cupy-cuda12x
   # Then add --gpu flag to commands
   ```

2. **Reduce vocabulary**:
   ```bash
   --max-vocab 50000  # Limit to top 50K words
   ```

3. **Use smaller model**:
   ```bash
   --model en_core_web_sm  # Instead of _trf
   ```

### "Column 'text' not found"
**Solution**: Specify your text column name:
```bash
--text-col body  # If your column is named 'body'
```

See [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md) for details.

---

## Ethics & Privacy ⚠️

**Before analyzing data with user information:**

1. **Check if your data is public** (4chan, public Twitter, public Reddit)
   - If yes: You can analyze, but aggregate in publications
   - If no: You need IRB approval

2. **Anonymize user IDs** if reporting results:
   ```python
   import hashlib
   df['user_id'] = df['user_id'].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16])
   ```

3. **Don't report individual behaviors** - aggregate only
   - ✅ "30% of users mention Entity X"
   - ❌ "User user_12345 posts extremist content"

**See [ETHICS.md](ETHICS.md) for comprehensive guidelines on privacy, IRB requirements, and best practices.**

---

## Getting Help

### Documentation
- **This guide**: General getting started
- **[DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md)**: CSV format details
- **[QUICKSTART.md](QUICKSTART.md)**: Command reference
- **[KG_FOR_SOCIAL_SCIENTISTS.md](KG_FOR_SOCIAL_SCIENTISTS.md)**: In-depth research guide
- **[ETHICS.md](ETHICS.md)**: Privacy and IRB guidelines

### Examples
- **`examples/`**: Sample notebooks with visualizations
- **`tutorials/`**: Step-by-step tutorials

### Community
- **GitHub Issues**: Report bugs or request features
- **Email**: [Your email for questions]

---

## What's Next?

**You now know how to:**
- ✅ Install the toolkit
- ✅ Run basic analysis
- ✅ Understand network outputs
- ✅ Analyze your own data

**Continue learning:**
1. Try the [sample notebooks](examples/) for interactive exploration
2. Read the [social science guide](KG_FOR_SOCIAL_SCIENTISTS.md) for research applications
3. Explore advanced features in [QUICKSTART.md](QUICKSTART.md)

**Happy analyzing!** 🎉

---

## Quick Command Cheatsheet

```bash
# Semantic network (topics/concepts)
python -m src.semantic.build_semantic_network --input data.csv --outdir output/semantic

# Knowledge graph (entities/people/places)
python -m src.semantic.kg_cli --input data.csv --outdir output/kg

# With sentiment analysis
python -m src.semantic.kg_cli --input data.csv --outdir output/kg --add-sentiment

# Temporal analysis (requires 'created_at' column)
python -m src.semantic.kg_cli --input data.csv --outdir output/temporal --group-by-time weekly

# Actor network (requires 'user_id' column)
python -m src.semantic.actor_cli --input data.csv --outdir output/actors

# Community detection
python -m src.semantic.community_cli --nodes output/semantic/nodes.csv --edges output/semantic/edges.csv --outdir output/communities
```

Save this for reference! 📋
