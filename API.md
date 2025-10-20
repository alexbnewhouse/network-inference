# Python API Reference

Use the toolkit programmatically in your Python scripts.

## Quick Examples

### Semantic Network

```python
import pandas as pd
from src.semantic.build_semantic_network import build_semantic_from_df

# Load data
df = pd.read_csv("data.csv")

# Build network
build_semantic_from_df(
    df,
    outdir="output/semantic",
    min_df=5,
    window=10,
    topk=20,
    text_col="text"  # or your column name
)

# Output files: nodes.csv, edges.csv, graph.graphml
```

### Transformer Networks

```python
from src.semantic.transformers_enhanced import TransformerSemanticNetwork
import pandas as pd

# Initialize builder
builder = TransformerSemanticNetwork(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"  # or "cuda" for GPU
)

# Load texts
df = pd.read_csv("data.csv")
texts = df["text"].tolist()

# Build document network
doc_edges = builder.build_document_network(
    texts,
    similarity_threshold=0.5,
    top_k=20
)

# doc_edges is a DataFrame with columns: source, target, similarity
doc_edges.to_csv("output/doc_network.csv", index=False)

# Build term network
from src.semantic.preprocess import tokenize
from src.semantic.cooccur import build_vocab

docs = [tokenize(t) for t in texts]
vocab = build_vocab(docs, min_df=5)
terms = list(vocab.keys())

term_edges = builder.build_term_network(
    terms,
    similarity_threshold=0.5,
    top_k=20
)

term_edges.to_csv("output/term_network.csv", index=False)
```

### Knowledge Graph

```python
from src.semantic.kg_pipeline import KnowledgeGraphPipeline
import pandas as pd

# Initialize pipeline
kg = KnowledgeGraphPipeline(
    ner_model="en_core_web_sm"  # or "en_core_web_trf"
)

# Load data
df = pd.read_csv("data.csv")

# Extract knowledge graph
kg.run(df, "output/kg")

# Output files: kg_nodes.csv, kg_edges.csv
```

### Transformer Embeddings

```python
from src.semantic.transformers_enhanced import TransformerEmbeddings
import numpy as np

# Initialize embedder
embedder = TransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# Encode texts
texts = ["First document", "Second document", "Third document"]
embeddings = embedder.encode(texts, show_progress=True)

# embeddings shape: (3, 384)
print(embeddings.shape)

# Compute similarity matrix
sim_matrix = embedder.compute_similarity_matrix(embeddings)

# sim_matrix shape: (3, 3)
# sim_matrix[i,j] = cosine similarity between texts i and j
print(sim_matrix)
```

### Named Entity Recognition

```python
from src.semantic.transformers_enhanced import TransformerNER
import pandas as pd

# Initialize NER
ner = TransformerNER(model_name="en_core_web_sm")

# Extract entities from single text
text = "Apple Inc. is headquartered in Cupertino, California."
entities = ner.extract_entities(text)

# entities is a DataFrame with columns: text, label, start, end
print(entities)
#           text label start  end
# 0   Apple Inc.   ORG     0    11
# 1    Cupertino   GPE    33    42
# 2   California   GPE    44    54
```

### Community Detection

```python
import pandas as pd
from src.semantic.community import detect_communities_louvain

# Load edges
edges = pd.read_csv("output/semantic/edges.csv")

# Detect communities
communities = detect_communities_louvain(edges)

# communities is a dict: {node_id: community_id}
print(communities)

# Save to file
community_df = pd.DataFrame([
    {"node": node, "community": comm}
    for node, comm in communities.items()
])
community_df.to_csv("output/communities.csv", index=False)
```

## Core Classes

### TransformerEmbeddings

Generate embeddings using pre-trained transformer models.

```python
class TransformerEmbeddings:
    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize embeddings model.
        
        Args:
            model_name: HuggingFace model name
            device: "cpu", "cuda", or "mps"
        """
        
    def encode(self, 
               texts: List[str], 
               batch_size: int = 32,
               show_progress: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Array of shape (n_texts, embedding_dim)
            
        Raises:
            ValueError: If texts is empty
        """
        
    def compute_similarity_matrix(self, 
                                  embeddings_or_texts) -> np.ndarray:
        """
        Compute cosine similarity matrix.
        
        Args:
            embeddings_or_texts: Either embeddings array or list of texts
            
        Returns:
            Similarity matrix of shape (n, n)
        """
```

**Example**:
```python
embedder = TransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedder.encode(["Hello", "World"])
sim_matrix = embedder.compute_similarity_matrix(embeddings)
```

### TransformerSemanticNetwork

Build semantic networks using transformer embeddings.

```python
class TransformerSemanticNetwork:
    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize network builder.
        
        Args:
            model_name: HuggingFace model name
            device: "cpu", "cuda", or "mps"
        """
        
    def build_document_network(self,
                               texts: List[str],
                               similarity_threshold: float = 0.5,
                               top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Build document similarity network.
        
        Args:
            texts: List of documents
            similarity_threshold: Minimum similarity for edge
            top_k: Keep top-k most similar per document
            
        Returns:
            DataFrame with columns: source, target, similarity
        """
        
    def build_term_network(self,
                           terms: List[str],
                           similarity_threshold: float = 0.5,
                           top_k: Optional[int] = 20) -> pd.DataFrame:
        """
        Build term similarity network.
        
        Args:
            terms: List of terms/tokens
            similarity_threshold: Minimum similarity for edge
            top_k: Keep top-k most similar per term
            
        Returns:
            DataFrame with columns: source, target, similarity
        """
```

**Example**:
```python
builder = TransformerSemanticNetwork()
texts = ["Document one", "Document two", "Document three"]
edges = builder.build_document_network(
    texts,
    similarity_threshold=0.3,
    top_k=5
)
```

### TransformerNER

Named entity recognition using transformer models.

```python
class TransformerNER:
    def __init__(self, model_name: str = "en_core_web_trf"):
        """
        Initialize NER pipeline.
        
        Args:
            model_name: spaCy model name
        """
        
    def extract_entities(self, text: str) -> pd.DataFrame:
        """
        Extract named entities from text.
        
        Args:
            text: Input text string
            
        Returns:
            DataFrame with columns: text, label, start, end
            Returns empty DataFrame if no entities found
        """
```

**Example**:
```python
ner = TransformerNER("en_core_web_sm")
text = "Elon Musk founded SpaceX in 2002."
entities = ner.extract_entities(text)
# Returns: text="Elon Musk", label="PERSON"
#          text="SpaceX", label="ORG"
#          text="2002", label="DATE"
```

### KnowledgeGraphPipeline

Extract entities and build knowledge graphs.

```python
class KnowledgeGraphPipeline:
    def __init__(self, 
                 ner_model: str = "en_core_web_sm",
                 linker=None,
                 rel_patterns=None):
        """
        Initialize KG pipeline.
        
        Args:
            ner_model: spaCy model for NER
            linker: Optional entity linker
            rel_patterns: Optional relation patterns
        """
        
    def run(self, df: pd.DataFrame, outdir: str):
        """
        Extract KG and save to disk.
        
        Args:
            df: DataFrame with text data
            outdir: Output directory
        """
```

**Example**:
```python
kg = KnowledgeGraphPipeline("en_core_web_sm")
df = pd.read_csv("data.csv")
kg.run(df, "output/kg")
# Creates: kg_nodes.csv, kg_edges.csv
```

## Utility Functions

### Tokenization

```python
from src.semantic.preprocess import tokenize

text = "The quick brown fox jumped!"
tokens = tokenize(text)
# Returns: ["quick", "brown", "fox", "jumped"]
```

### Vocabulary Building

```python
from src.semantic.cooccur import build_vocab

docs = [
    ["word1", "word2", "word3"],
    ["word1", "word3", "word4"],
    ["word2", "word4", "word5"]
]

vocab = build_vocab(docs, min_df=2)
# Returns: {"word1": freq, "word2": freq, "word3": freq, "word4": freq}
# word5 excluded (appears in only 1 doc)
```

### Co-occurrence Counting

```python
from src.semantic.cooccur import cooccurrence

docs = [["cat", "sat", "mat"], ["dog", "sat", "rug"]]
vocab = {"cat": 0, "sat": 1, "mat": 2, "dog": 3, "rug": 4}

pairs, counts, total = cooccurrence(docs, vocab, window=2)
# Returns co-occurrence counts for word pairs within window
```

### PPMI Computation

```python
from src.semantic.cooccur import compute_ppmi

ppmi_matrix = compute_ppmi(pairs, counts, total, cds_alpha=0.75)
# Returns PPMI-weighted co-occurrence matrix
```

### Graph Visualization

```python
from src.semantic.visualize import load_graph, plot_degree_distribution
import matplotlib.pyplot as plt

# Load graph from GraphML
G = load_graph("output/semantic/graph.graphml")

# Plot degree distribution
plot_degree_distribution(G)
plt.savefig("degree_dist.png")

# Or create interactive HTML
from src.semantic.visualize import create_interactive_network

create_interactive_network(
    nodes_file="output/semantic/nodes.csv",
    edges_file="output/semantic/edges.csv",
    output_file="network.html"
)
```

## Working with Output Files

### Reading Network Data

```python
import pandas as pd
import networkx as nx

# Read nodes and edges
nodes = pd.read_csv("output/semantic/nodes.csv")
edges = pd.read_csv("output/semantic/edges.csv")

# Build NetworkX graph
G = nx.Graph()
for _, row in nodes.iterrows():
    G.add_node(row['word'], frequency=row['frequency'])
for _, row in edges.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['similarity'])

# Analyze
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G):.4f}")

# Find most central nodes
degree_cent = nx.degree_centrality(G)
top_nodes = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 most connected nodes:", top_nodes)
```

### Reading GraphML

```python
import networkx as nx

# Load graph
G = nx.read_graphml("output/semantic/graph.graphml")

# NetworkX graph ready for analysis
communities = nx.community.louvain_communities(G, weight='weight')
print(f"Found {len(communities)} communities")
```

## Error Handling

### Empty Input Handling

```python
from src.semantic.transformers_enhanced import TransformerEmbeddings

embedder = TransformerEmbeddings()

try:
    embeddings = embedder.encode([])
except ValueError as e:
    print(f"Error: {e}")  # "Input list is empty"
```

### Missing Columns

```python
import pandas as pd
from src.semantic.build_semantic_network import build_semantic_from_df

df = pd.read_csv("data.csv")

try:
    build_semantic_from_df(df, "output", text_col="nonexistent_col")
except ValueError as e:
    print(f"Error: {e}")  # Column not found
```

## Integration Examples

### With Pandas

```python
import pandas as pd
from src.semantic.transformers_enhanced import TransformerSemanticNetwork

# Read data
df = pd.read_csv("data.csv")

# Filter and clean
df = df[df['text'].str.len() > 50]  # Only longer texts
df = df.dropna(subset=['text'])

# Build network
builder = TransformerSemanticNetwork()
edges = builder.build_document_network(
    df['text'].tolist(),
    similarity_threshold=0.5
)

# Add document metadata to edges
edges['source_title'] = edges['source'].map(lambda i: df.iloc[i]['title'])
edges['target_title'] = edges['target'].map(lambda i: df.iloc[i]['title'])

edges.to_csv("output/enriched_edges.csv", index=False)
```

### With NetworkX

```python
import networkx as nx
from src.semantic.transformers_enhanced import TransformerSemanticNetwork

builder = TransformerSemanticNetwork()
edges_df = builder.build_document_network(texts)

# Convert to NetworkX
G = nx.Graph()
for _, row in edges_df.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['similarity'])

# Compute metrics
pagerank = nx.pagerank(G, weight='weight')
betweenness = nx.betweenness_centrality(G, weight='weight')

# Find communities
communities = nx.community.louvain_communities(G, weight='weight')
```

### With Visualization

```python
import matplotlib.pyplot as plt
import networkx as nx
from src.semantic.visualize import load_graph

# Load graph
G = load_graph("output/semantic")

# Draw with matplotlib
pos = nx.spring_layout(G, k=0.5, iterations=50)
nx.draw_networkx_nodes(G, pos, node_size=100, alpha=0.6)
nx.draw_networkx_edges(G, pos, alpha=0.2)
nx.draw_networkx_labels(G, pos, font_size=8)

plt.axis('off')
plt.tight_layout()
plt.savefig("network.png", dpi=300)
```

## Best Practices

1. **Always check for empty inputs**:
   ```python
   if not texts or len(texts) == 0:
       raise ValueError("No texts provided")
   ```

2. **Handle missing data**:
   ```python
   df['text'] = df['text'].fillna("")
   texts = [t for t in df['text'].tolist() if len(t.strip()) > 0]
   ```

3. **Use appropriate thresholds**:
   ```python
   # For noisy data, higher threshold
   edges = builder.build_document_network(texts, similarity_threshold=0.7)
   
   # For clean data, lower threshold
   edges = builder.build_document_network(texts, similarity_threshold=0.3)
   ```

4. **Save intermediate results**:
   ```python
   # Save embeddings to avoid recomputing
   embeddings = embedder.encode(texts)
   np.save("embeddings.npy", embeddings)
   
   # Load later
   embeddings = np.load("embeddings.npy")
   ```

5. **Use progress bars for long operations**:
   ```python
   from tqdm import tqdm
   
   results = []
   for text in tqdm(texts, desc="Processing"):
       result = process(text)
       results.append(result)
   ```

## See Also

- [CONCEPTS.md](CONCEPTS.md) - Detailed explanations of methods
- [REAL_DATA_USAGE.md](REAL_DATA_USAGE.md) - Real dataset examples
- [examples/](examples/) - Jupyter notebooks with walkthroughs
