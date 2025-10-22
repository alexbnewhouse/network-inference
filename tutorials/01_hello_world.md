# Tutorial 1: Hello World - Your First Network in 5 Minutes

**Goal**: Build your first semantic network from sample data and understand what it shows.

**Time**: 5 minutes  
**Prerequisites**: Toolkit installed (see [GETTING_STARTED.md](../GETTING_STARTED.md))

---

## What We'll Build

A network showing which topics are related in 100 news headlines.

**Input**: 100 headlines about AI, climate, finance, health  
**Output**: Network showing "AI" connects to "technology", "climate" connects to "change", etc.

---

## Step 1: Generate Sample Data (30 seconds)

```bash
cd network-inference
python examples/sample_data.py
```

**Output**:
```
✓ Created examples/sample_news.csv (100 rows)
✓ Created examples/sample_forum.csv (200 rows)
✓ Created examples/sample_research.csv (50 rows)
```

---

## Step 2: Build Network (10 seconds)

```bash
python -m src.semantic.build_semantic_network \
  --input examples/sample_news.csv \
  --outdir output/hello_world \
  --min-df 2 \
  --topk 15
```

**What's happening:**
1. Loading 100 headlines
2. Finding words that appear together frequently
3. Calculating association strengths (PPMI weights)
4. Keeping top 15 connections per word
5. Saving network files

**Expected output**:
```
Loading data...
Building vocabulary...
Computing co-occurrences...
Building network...
✓ Saved 3 files to output/hello_world/
```

---

## Step 3: Explore Results (2 minutes)

### Look at the Network Files

```bash
ls output/hello_world/
```

You should see:
- `nodes.csv` - Words in the network
- `edges.csv` - Connections between words
- `graph.graphml` - Network file for Gephi

### View Top Words

```bash
head -15 output/hello_world/nodes.csv
```

Example output:
```
term,frequency
climate,45
technology,38
research,32
change,30
study,28
system,25
energy,24
data,22
model,20
...
```

**Interpretation**:
- "climate" appears in 45 headlines
- "technology" appears in 38 headlines
- These are the most central concepts

### View Strongest Connections

```bash
head -15 output/hello_world/edges.csv
```

Example output:
```
source,target,weight
climate,change,8.342
artificial,intelligence,7.891
machine,learning,7.654
renewable,energy,6.523
financial,market,6.234
...
```

**Interpretation**:
- "climate" and "change" are strongly connected (weight=8.34)
- They often appear together → they form a topic
- High weights mean strong associations

---

## Step 4: Understand What You Built

### What is a Semantic Network?

A **semantic network** maps concepts and their relationships:
- **Nodes** (words) = concepts discussed in the text
- **Edges** (connections) = how often concepts appear together
- **Weights** = strength of association

### What Can You Learn?

From this simple 100-headline network, you can see:

1. **Main Topics**: 
   - Climate & environment (climate, energy, renewable)
   - AI & technology (artificial, intelligence, machine, learning)
   - Finance (financial, market, economic)
   - Health (health, medical, treatment)

2. **Topic Structure**:
   - Which concepts cluster together
   - Which topics are isolated vs. connected

3. **Discourse Patterns**:
   - How journalists/authors frame issues
   - Which concepts are linked in public discourse

---

## Step 5: Visualize (Optional, 2 minutes)

### Option 1: Gephi (Best)

If you have [Gephi](https://gephi.org/) installed:

1. Open Gephi
2. File → Open → Select `output/hello_world/graph.graphml`
3. Click "OK" on import settings
4. Go to "Layout" panel → Select "ForceAtlas 2"
5. Click "Run" 
6. Stop after 30 seconds

**You'll see**: Clusters of related words forming topic groups!

### Option 2: Python (Quick)

Create a file `visualize.py`:

```python
import networkx as nx
import matplotlib.pyplot as plt

# Load network
G = nx.read_graphml("output/hello_world/graph.graphml")

# Get top 30 nodes by degree (most connected)
top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:30]
subgraph = G.subgraph([n for n, d in top_nodes])

# Draw
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(subgraph, k=2, iterations=50)
nx.draw_networkx(subgraph, pos, 
                 node_size=100, 
                 font_size=10,
                 with_labels=True,
                 node_color='lightblue',
                 edge_color='gray',
                 alpha=0.7)
plt.axis('off')
plt.tight_layout()
plt.savefig("network.png", dpi=300, bbox_inches='tight')
print("✓ Saved to network.png")
```

Run it:
```bash
python visualize.py
```

---

## What's Next?

### Try Your Own Data

Replace `sample_news.csv` with your own data:

```csv
text
"Your first post"
"Your second post"
"Your third post"
```

Run the same command:
```bash
python -m src.semantic.build_semantic_network \
  --input my_data.csv \
  --outdir output/my_network \
  --min-df 2 \
  --topk 15
```

### Try Different Network Types

**Knowledge Graph** (extract entities like people, places):
```bash
python -m src.semantic.kg_cli \
  --input examples/sample_news.csv \
  --outdir output/kg_hello \
  --model en_core_web_sm
```

**With Sentiment Analysis**:
```bash
python -m src.semantic.kg_cli \
  --input examples/sample_news.csv \
  --outdir output/kg_sentiment \
  --add-sentiment
```

### Learn More

- **[GETTING_STARTED.md](../GETTING_STARTED.md)** - Full beginner guide
- **[DATA_REQUIREMENTS.md](../DATA_REQUIREMENTS.md)** - CSV format details
- **[QUICKSTART.md](../QUICKSTART.md)** - Command reference
- **[KG_FOR_SOCIAL_SCIENTISTS.md](../KG_FOR_SOCIAL_SCIENTISTS.md)** - Research applications

---

## Troubleshooting

### "Command not found"
Make sure you're in the `network-inference` directory and your virtual environment is activated:
```bash
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

### "File not found: examples/sample_news.csv"
Run the sample data generator first:
```bash
python examples/sample_data.py
```

### No output files created
Check for error messages. Common causes:
- Empty or invalid CSV
- Missing required columns
- Insufficient memory (try smaller dataset)

---

## Summary

**You just:**
✅ Generated sample data  
✅ Built a semantic network  
✅ Explored the output files  
✅ Learned how to interpret results  

**Next**: Try with your own data or explore advanced features!

**Questions?** See [GETTING_STARTED.md](../GETTING_STARTED.md) troubleshooting section.
