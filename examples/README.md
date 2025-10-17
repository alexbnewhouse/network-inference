# Examples Directory

This directory contains Jupyter notebooks and sample data demonstrating the various features of the network inference toolkit.

## 📓 Notebooks

### 1. Transformer Networks (`1_transformer_networks.ipynb`)
**Complete walkthrough of transformer-based semantic networks**

Topics covered:
- Understanding sentence embeddings
- Building document similarity networks
- Creating term/concept networks
- Model comparison (MiniLM vs MPNet)
- Network analysis and visualization
- Community detection

**Prerequisites**: `sentence-transformers`, `scikit-learn`

**Run time**: ~10-15 minutes

### 2. Topic Modeling (`2_topic_modeling.ipynb`)
**Automated topic discovery with BERTopic**

Topics covered:
- BERTopic fundamentals
- Training topic models
- Visualizing topic hierarchies
- Comparing with LDA
- Topic evolution over time

**Prerequisites**: `bertopic` (optional)

**Run time**: ~5-10 minutes

### 3. Method Comparison (`3_comparison.ipynb`) [Coming Soon]
**Side-by-side comparison of different approaches**

Topics covered:
- Co-occurrence vs Transformer networks
- Speed benchmarks
- Quality metrics
- When to use which method

## 📊 Sample Data

### Generated Datasets

Run `python3 sample_data.py` to generate:

1. **`sample_news.csv`** - 100 news headlines
   - Categories: AI/Tech, Climate, Finance, Health
   - Time range: Last 365 days
   - Use for: Topic modeling, trend analysis

2. **`sample_forum.csv`** - 200 forum posts
   - 10 discussion threads
   - Reply structure included
   - Use for: Actor networks, community detection

3. **`sample_research.csv`** - 50 research abstracts
   - AI/ML research topics
   - Years: 2022-2025
   - Use for: Semantic networks, citation analysis

### Data Format

All datasets follow this structure:

```python
# Minimum required columns
df = pd.DataFrame({
    'text': ['document 1', 'document 2', ...],  # Required
})

# Optional columns for advanced features
df = pd.DataFrame({
    'text': ['...'],           # Required: main text content
    'id': [0, 1, 2, ...],      # Optional: document IDs
    'timestamp': [...],        # Optional: for time-slice analysis
    'category': [...],         # Optional: ground truth labels
    'subject': [...],          # Optional: for thread/topic grouping
})
```

## 🚀 Quick Start

### Option 1: Run All Notebooks

```bash
cd examples
jupyter lab
# Open and run notebooks in order
```

### Option 2: Try Individual Features

```python
import pandas as pd
import sys
sys.path.append('..')

from src.semantic.transformers_enhanced import TransformerSemanticNetwork

# Load sample data
df = pd.read_csv('sample_news.csv')

# Build network
builder = TransformerSemanticNetwork()
edges = builder.build_document_network(
    documents=df['text'].tolist(),
    similarity_threshold=0.3,
    top_k=5
)

print(f"Created network with {len(edges)} edges!")
```

## 📋 Requirements

### Basic Examples
```bash
pip install pandas numpy matplotlib seaborn networkx
```

### Transformer Examples
```bash
pip install sentence-transformers scikit-learn
```

### Topic Modeling Examples
```bash
pip install bertopic
```

### All Features
```bash
pip install -r ../requirements.txt
```

## 💡 Usage Tips

### For Learning
1. Start with `1_transformer_networks.ipynb` to understand basics
2. Move to `2_topic_modeling.ipynb` for advanced clustering
3. Generate your own data with `sample_data.py`

### For Development
1. Use sample datasets to test new features
2. Modify data generation scripts for custom scenarios
3. Run notebooks to validate changes

### For Demonstrations
1. Sample datasets provide reproducible results
2. Notebooks include visualizations suitable for presentations
3. Each notebook runs in <15 minutes

## 🔧 Customization

### Generate Custom Data

Edit `sample_data.py`:

```python
# Add new topics
topics = {
    'Your Topic': [
        "Template sentence 1",
        "Template sentence 2",
    ]
}

# Adjust size
df = generate_news_dataset(n_docs=500)  # More documents
```

### Modify Notebooks

```python
# Try different models
embedder = TransformerEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Adjust network parameters
edges = builder.build_document_network(
    similarity_threshold=0.5,  # Stricter similarity
    top_k=10  # More connections
)
```

## 📈 Expected Outputs

### Network Files
- `*_nodes.csv` - Node attributes (degree, centrality, etc.)
- `*_edges.csv` - Edge list with weights
- `*.graphml` - Graph file for Gephi/Cytoscape

### Visualizations
- Similarity heatmaps
- Network graphs with NetworkX
- Topic visualizations with BERTopic
- Degree distributions
- Community structures

## 🐛 Troubleshooting

### "Module not found" errors
```bash
# Make sure you're in the examples directory
cd /path/to/network_inference/examples

# And that parent directory is in path
import sys
sys.path.insert(0, '..')
```

### "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### "CUDA out of memory"
```python
# Use CPU instead
embedder = TransformerEmbeddings(device='cpu')
```

### Notebook kernel issues
```bash
# Install kernel
python3 -m ipykernel install --user --name=network_inference

# Select kernel in Jupyter: Kernel -> Change Kernel -> network_inference
```

## 📚 Additional Resources

- [Main README](../README.md) - Full documentation
- [TECHNICAL.md](../TECHNICAL.md) - Algorithm details
- [QUICKSTART.md](../QUICKSTART.md) - Command-line examples

## 🤝 Contributing

Found a bug in an example? Have an idea for a new notebook?
1. Check [CONTRIBUTING.md](../CONTRIBUTING.md)
2. Open an issue with the `examples` label
3. Submit a PR with improvements!

## 📄 License

All example code and sample data are released under the same MIT License as the main project.

---

**Happy Learning!** 🎉

If you find these examples helpful, please consider starring the repository!
