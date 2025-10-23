# 4chan Data Analysis: Complete Guide

**A comprehensive guide to extracting and analyzing semantic/entity relationships over time in 4chan data**

---

## Table of Contents

1. [Overview](#overview)
2. [Data Format & Structure](#data-format--structure)
3. [Quick Start](#quick-start)
4. [Analysis Workflows](#analysis-workflows)
5. [Board-Level Comparisons](#board-level-comparisons)
6. [Entity Evolution Tracking](#entity-evolution-tracking)
7. [Ethics & Privacy](#ethics--privacy)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### Why This Toolkit Works Well for 4chan

**4chan presents unique challenges for social media analysis:**
- **Ephemeral content**: Posts deleted after ~24 hours (boards auto-prune)
- **Anonymous by default**: Most users post without persistent identifiers
- **Optional tripcodes**: Cryptographic user identifiers for those who want them
- **Board-specific cultures**: /pol/, /b/, /sci/ have very different discourse patterns
- **High volume**: Popular boards generate thousands of posts per hour
- **Offensive content**: Requires careful ethical handling

**This toolkit is specifically designed to handle:**
- ✅ **Anonymous data**: No persistent user IDs required
- ✅ **Temporal analysis**: Track discourse evolution despite ephemeral nature
- ✅ **Semantic networks**: Understand topic relationships without user tracking
- ✅ **Entity extraction**: Identify people, places, organizations discussed
- ✅ **Board comparisons**: Compare discourse patterns across communities
- ✅ **Tripcode support**: Optional user tracking when available
- ✅ **Scale**: Process millions of posts efficiently

---

## Data Format & Structure

### Minimum Required Format

Your 4chan CSV needs **at least** a text column:

```csv
body
"Anonymous post about current events"
"Another post discussing technology"
"Reply referencing >>12345678"
```

### Recommended Format (Full Features)

Include these columns to unlock all features:

```csv
no,thread_id,board,body,time,tripcode
12345678,12340000,pol,"Post text here",1704115200,
12345679,12340000,pol,"Reply to >>12345678",1704115260,!Ep8pui8Vw2
12345680,12341111,int,"Different thread",1704115320,
```

**Column descriptions:**

| Column | Type | Description | Required? | Example |
|--------|------|-------------|-----------|---------|
| `body` | string | Post content | **Required** | "Discussing politics today" |
| `no` | integer | Post number (unique ID) | Recommended | 12345678 |
| `thread_id` | integer | Thread number (OP post no) | Recommended | 12340000 |
| `board` | string | Board name | Recommended | "pol", "int", "sci" |
| `time` | integer/string | Unix timestamp or datetime | Recommended | 1704115200 or "2024-01-01 12:00:00" |
| `tripcode` | string | Optional user identifier | Optional | "!Ep8pui8Vw2" |
| `name` | string | Name field (usually "Anonymous") | Optional | "Anonymous" |
| `subject` | string | Thread subject/title | Optional | "General Discussion" |

### Timestamp Format

4chan uses Unix timestamps. Convert to readable format:

```python
import pandas as pd

df = pd.read_csv("4chan_data.csv")

# Convert Unix timestamp to datetime
df['created_at'] = pd.to_datetime(df['time'], unit='s')

# Or if already in datetime format
df['created_at'] = pd.to_datetime(df['time'])

df.to_csv("4chan_data_formatted.csv", index=False)
```

### Handling Anonymous Data

**Most 4chan posts are anonymous.** The toolkit handles this gracefully:

```python
# Option 1: Leave tripcode column empty for anonymous posts
# Option 2: Use post number (no) as unique identifier
# Option 3: Omit user columns entirely for fully anonymous analysis

# For anonymous analysis (recommended):
df = df[['body', 'created_at', 'board', 'thread_id']]
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/alexbnewhouse/network-inference.git
cd network-inference

# Install dependencies (Python 3.12 or 3.13)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 5-Minute Test

```bash
# Generate sample 4chan-style data
python examples/sample_4chan_data.py

# Build semantic network (topic relationships)
python -m src.semantic.build_semantic_network \
  --input examples/sample_4chan.csv \
  --text-col body \
  --outdir output/chan_semantic \
  --min-df 3 --topk 20

# Extract entities (people, places, organizations)
python -m src.semantic.kg_cli \
  --input examples/sample_4chan.csv \
  --text-col body \
  --outdir output/chan_kg \
  --max-rows 1000
```

**Output**: `nodes.csv`, `edges.csv`, and `graph.graphml` ready for analysis!

---

## Analysis Workflows

### 1. Semantic Network Analysis

**Goal**: Understand what topics co-occur in discussion

**Best for**: Exploratory analysis, topic discovery, discourse mapping

```bash
python -m src.semantic.build_semantic_network \
  --input 4chan_data.csv \
  --text-col body \
  --outdir output/semantic \
  --min-df 10 \
  --topk 20 \
  --window 10
```

**What you get:**
- `nodes.csv`: Vocabulary terms with frequencies
- `edges.csv`: Co-occurrence relationships (PPMI-weighted)
- `graph.graphml`: Network for Gephi visualization

**Interpretation example:**
```
source,target,similarity
trump,biden,8.32
election,fraud,7.89
media,fake,6.54
```
→ "trump" and "biden" frequently co-occur, suggesting linked discussion

### 2. Knowledge Graph Extraction

**Goal**: Extract named entities and their relationships

**Best for**: Tracking people, organizations, places discussed over time

```bash
python -m src.semantic.kg_cli \
  --input 4chan_data.csv \
  --text-col body \
  --outdir output/kg \
  --model en_core_web_sm \
  --max-rows 50000
```

**What you get:**
- `kg_nodes.csv`: Entities (PERSON, ORG, GPE, DATE)
- `kg_edges.csv`: Entity co-occurrences

**Example output:**
```
entity,entity_type,frequency
Trump,PERSON,1523
Russia,GPE,892
FBI,ORG,456
```

**Use the larger model for better accuracy:**
```bash
# Install larger model (more accurate)
python -m spacy download en_core_web_trf

python -m src.semantic.kg_cli \
  --input 4chan_data.csv \
  --text-col body \
  --model en_core_web_trf \
  --outdir output/kg_accurate
```

### 3. Temporal Knowledge Graphs

**Goal**: Track how entities and relationships evolve over time

**Best for**: Understanding discourse shifts, tracking controversies, event detection

```bash
# Generate weekly knowledge graphs
python -m src.semantic.kg_cli \
  --input 4chan_data.csv \
  --text-col body \
  --time-col created_at \
  --group-by-time weekly \
  --outdir output/kg_temporal \
  --add-sentiment

# Analyze entity timeline
python -m src.semantic.kg_temporal_cli \
  --kg-dir output/kg_temporal \
  --entity "Russia" \
  --report entity_timeline.md
```

**What you get:**
- Weekly/daily/monthly knowledge graphs
- Entity timeline tracking (first seen, peak periods, trajectory)
- Event detection (automatic spike identification)
- Sentiment over time (if `--add-sentiment` used)

**Timeline analysis output:**
```markdown
# Entity Timeline: Russia

## Overview
- **First seen**: 2024-01-01 (Week 1)
- **Last seen**: 2024-03-15 (Week 11)
- **Peak period**: Week 5 (523 mentions)
- **Trajectory**: Episodic (sporadic spikes)
- **Total mentions**: 2,847

## Events Detected
1. **Week 5 spike** (+450% from baseline)
   - Coincides with [geopolitical event]
   - Co-mentioned with: Ukraine, NATO, sanctions
```

### 4. Sentiment Analysis

**Goal**: Understand emotional tone and stance toward entities

**Best for**: Controversy detection, stance analysis, framing studies

```bash
# Fast VADER sentiment (lexicon-based)
python -m src.semantic.kg_cli \
  --input 4chan_data.csv \
  --text-col body \
  --outdir output/kg_sentiment \
  --add-sentiment \
  --sentiment-model vader

# Accurate transformer sentiment (contextual, GPU-accelerated)
python -m src.semantic.kg_cli \
  --input 4chan_data.csv \
  --text-col body \
  --outdir output/kg_sentiment \
  --add-sentiment \
  --sentiment-model transformer \
  --sentiment-device cuda \
  --sentiment-batch-size 128
```

**What you get:**
- Entity-level sentiment scores
- Controversy detection (high variance = contested entity)
- Sentiment distribution analysis

**Example output:**
```
entity,mean_sentiment,std_sentiment,controversy_score
Trump,-0.32,0.78,high
Biden,-0.28,0.75,high
Europe,0.12,0.34,low
```
→ Trump and Biden are contested (high variance), Europe is neutral

### 5. Board-Level Comparison

**Goal**: Compare discourse patterns across different boards

**Best for**: Understanding community differences, cross-board analysis

```bash
# Build knowledge graphs for each board separately
for board in pol int sci his; do
  python -m src.semantic.kg_cli \
    --input 4chan_data.csv \
    --text-col body \
    --board-filter $board \
    --outdir output/kg_${board} \
    --add-sentiment
done

# Compare boards (manual analysis of outputs)
# Or use Python API for programmatic comparison
```

**Python API for board comparison:**
```python
import pandas as pd

# Load entity frequencies per board
pol_nodes = pd.read_csv("output/kg_pol/kg_nodes.csv")
int_nodes = pd.read_csv("output/kg_int/kg_nodes.csv")

# Find board-specific entities
pol_only = set(pol_nodes['entity']) - set(int_nodes['entity'])
int_only = set(int_nodes['entity']) - set(pol_nodes['entity'])

print(f"/pol/ specific entities: {pol_only}")
print(f"/int/ specific entities: {int_only}")

# Compare sentiment
pol_sentiment = pol_nodes.groupby('entity')['sentiment'].mean()
int_sentiment = int_nodes.groupby('entity')['sentiment'].mean()

# Entities with largest sentiment difference
diff = (pol_sentiment - int_sentiment).abs().sort_values(ascending=False)
print(f"Most polarized entities across boards:\n{diff.head(10)}")
```

---

## Board-Level Comparisons

### Filtering by Board

Most CLIs support board filtering:

```bash
# Analyze /pol/ only
python -m src.semantic.kg_cli \
  --input 4chan_data.csv \
  --text-col body \
  --board-col board \
  --board-filter pol \
  --outdir output/kg_pol
```

### Automated Multi-Board Analysis

Create a shell script for batch processing:

```bash
#!/bin/bash
# analyze_boards.sh

BOARDS=("pol" "int" "sci" "his" "biz")
INPUT="4chan_data.csv"

for BOARD in "${BOARDS[@]}"; do
  echo "Analyzing /$BOARD/..."
  
  # Semantic network
  python -m src.semantic.build_semantic_network \
    --input $INPUT \
    --text-col body \
    --board-filter $BOARD \
    --outdir output/${BOARD}_semantic \
    --min-df 5 --topk 15
  
  # Knowledge graph
  python -m src.semantic.kg_cli \
    --input $INPUT \
    --text-col body \
    --board-filter $BOARD \
    --outdir output/${BOARD}_kg \
    --add-sentiment
done

echo "Board analysis complete!"
```

Run it:
```bash
chmod +x analyze_boards.sh
./analyze_boards.sh
```

### Comparative Analysis Script

```python
# compare_boards.py
import pandas as pd
import matplotlib.pyplot as plt

boards = ['pol', 'int', 'sci', 'his', 'biz']
board_stats = []

for board in boards:
    # Load knowledge graph nodes
    nodes = pd.read_csv(f"output/{board}_kg/kg_nodes.csv")
    
    board_stats.append({
        'board': board,
        'unique_entities': len(nodes),
        'person_mentions': len(nodes[nodes['entity_type'] == 'PERSON']),
        'org_mentions': len(nodes[nodes['entity_type'] == 'ORG']),
        'gpe_mentions': len(nodes[nodes['entity_type'] == 'GPE']),
        'avg_sentiment': nodes['sentiment'].mean() if 'sentiment' in nodes.columns else None
    })

df_stats = pd.DataFrame(board_stats)
print(df_stats)

# Visualize
df_stats.plot(x='board', y='unique_entities', kind='bar')
plt.title('Unique Entities by Board')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('board_comparison.png')
```

---

## Entity Evolution Tracking

### Research Question Example

**"How did discussion of 'Russia' evolve on /pol/ during the Ukraine conflict?"**

### Step 1: Generate Temporal Knowledge Graphs

```bash
# Weekly knowledge graphs with sentiment
python -m src.semantic.kg_cli \
  --input 4chan_pol_2022.csv \
  --text-col body \
  --time-col created_at \
  --group-by-time weekly \
  --add-sentiment \
  --outdir output/pol_kg_temporal
```

### Step 2: Analyze Entity Timeline

```bash
python -m src.semantic.kg_temporal_cli \
  --kg-dir output/pol_kg_temporal \
  --entity "Russia" \
  --report russia_analysis.md \
  --plot-trends
```

### Step 3: Compare Related Entities

```python
# analyze_entity_evolution.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

kg_dir = Path("output/pol_kg_temporal")
entities = ["Russia", "Ukraine", "NATO", "Putin", "Zelensky"]

# Collect timeline data
timelines = {}
for entity in entities:
    entity_data = []
    
    for period_dir in sorted(kg_dir.glob("period_*")):
        period = period_dir.name.replace("period_", "")
        nodes = pd.read_csv(period_dir / "kg_nodes.csv")
        
        if entity in nodes['entity'].values:
            row = nodes[nodes['entity'] == entity].iloc[0]
            entity_data.append({
                'period': period,
                'frequency': row['frequency'],
                'sentiment': row.get('sentiment', 0)
            })
    
    timelines[entity] = pd.DataFrame(entity_data)

# Plot frequency over time
plt.figure(figsize=(12, 6))
for entity, df in timelines.items():
    plt.plot(df['period'], df['frequency'], marker='o', label=entity)

plt.xlabel('Time Period')
plt.ylabel('Mention Frequency')
plt.title('Entity Evolution on /pol/ During Ukraine Conflict')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('entity_evolution.png')
```

### Event Detection

Temporal analysis automatically detects spikes:

```bash
python -m src.semantic.kg_temporal_cli \
  --kg-dir output/pol_kg_temporal \
  --detect-events \
  --event-threshold 2.5  # 2.5x baseline = significant spike
```

**Output**: Automatic event detection report
```
Events Detected:

Week 8: Russia spike (+450%)
  - Co-mentioned with: Ukraine (+380%), NATO (+290%)
  - Sentiment shift: -0.15 → -0.42 (more negative)
  - Context: Likely related to military escalation

Week 12: NATO spike (+320%)
  - Co-mentioned with: Finland (+520%), Sweden (+410%)
  - Sentiment shift: -0.28 → -0.08 (less negative)
  - Context: Likely related to expansion discussions
```

---

## Ethics & Privacy

### 4chan-Specific Considerations

**Good news**: 4chan is one of the most privacy-friendly platforms to analyze:

✅ **Advantages:**
- Fully public, intended for anyone to view
- Most users anonymous (no persistent identity to protect)
- Expectation of ephemerality (posts auto-delete)
- No user profiles to de-anonymize

⚠️ **Still be responsible:**
- Aggregate in publications (don't quote individual posts verbatim)
- Paraphrase offensive content
- Avoid amplifying calls for violence
- Consider content warnings for sensitive topics
- Report patterns, not individuals

### IRB Guidance

**For most 4chan research, IRB exemption is likely** (US institutions):
- Public data, no interaction with users
- Anonymous (no persistent identifiers to track)
- Focus on discourse patterns, not individuals
- Secondary analysis of existing data

**However, always check with your institution!** Policies vary.

### Anonymization Best Practices

Even though 4chan is anonymous, follow these practices:

```python
import pandas as pd
import hashlib

df = pd.read_csv("4chan_data.csv")

# 1. Hash tripcodes (if present)
if 'tripcode' in df.columns:
    df['tripcode'] = df['tripcode'].apply(
        lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16] 
        if pd.notna(x) else None
    )

# 2. Remove post numbers (prevents reverse lookup)
if 'no' in df.columns:
    df = df.drop(columns=['no'])

# 3. Aggregate timestamps to hour/day
df['created_at'] = pd.to_datetime(df['time'], unit='s').dt.floor('H')

# 4. Remove very rare terms (potential identifiers)
# Handle during analysis with --min-df parameter

df.to_csv("4chan_data_anonymized.csv", index=False)
```

### Reporting Guidelines

**In publications:**

❌ **Don't:**
- Quote unique phrases verbatim (Google-searchable)
- Link to specific threads or posts
- Identify individual users (even by tripcodes)
- Cherry-pick offensive quotes without context

✅ **Do:**
- Paraphrase representative quotes
- Report aggregate statistics
- Provide content warnings for offensive material
- Contextualize findings ("On /pol/, a politically incorrect board...")
- Include methods section with data handling details

**Example methods section:**
> "We analyzed 500,000 posts from 4chan's /pol/ board collected via archive.org during January-June 2022. Posts are publicly accessible and anonymous (97% posted without tripcodes). We extracted entities using spaCy 3.8 and grouped posts by week for temporal analysis. Exact quotes were paraphrased to prevent post re-identification. This research was determined exempt by [University] IRB under category 4 (public data, secondary analysis)."

**See [ETHICS.md](ETHICS.md) for comprehensive guidance.**

---

## Advanced Features

### 1. Reply Network Analysis

**Goal**: Understand conversation structure and reply patterns

4chan uses `>>12345678` format for replies. The toolkit parses this:

```bash
python -m src.semantic.actor_cli \
  --input 4chan_data.csv \
  --text-col body \
  --thread-col thread_id \
  --post-col no \
  --outdir output/reply_network
```

**What you get:**
- Who replies to whom (even anonymously)
- Thread structure and conversation flow
- Centrality metrics (who gets most replies)

### 2. Tripcode Networks

**For posts with tripcodes** (optional user tracking):

```bash
python -m src.semantic.kg_user_entity_network_cli \
  --kg-dir output/kg \
  --data 4chan_data.csv \
  --user-col tripcode \
  --text-col body \
  --stats --communities
```

**What you get:**
- User-entity bipartite networks
- Which tripcodes discuss which entities
- Community detection (groups with shared interests)

### 3. Transformer Semantic Networks

**For higher-quality semantic similarity** (slower):

```bash
python -m src.semantic.transformers_cli \
  --input 4chan_data.csv \
  --text-col body \
  --mode document \
  --similarity-threshold 0.5 \
  --top-k 20 \
  --max-rows 5000 \
  --outdir output/transformer
```

**Best for**: Document clustering, finding semantically similar posts

### 4. Community Detection

**Find topic clusters in your network:**

```bash
# Build network first
python -m src.semantic.build_semantic_network \
  --input 4chan_data.csv \
  --text-col body \
  --outdir output/semantic

# Detect communities
python -m src.semantic.community_cli \
  --edges output/semantic/edges.csv \
  --outdir output/communities
```

**What you get:**
- Topic clusters (groups of related terms)
- Community assignments
- Modularity scores (how well-separated communities are)

### 5. Time-Sliced Analysis

**Alternative to temporal KG** - track network structure evolution:

```bash
python -m src.semantic.time_slice_cli \
  --input 4chan_data.csv \
  --text-col body \
  --time-col created_at \
  --freq W \
  --outdir output/timeslices
```

**What you get:**
- Separate semantic networks per time period
- Compare network structure over time
- Track emergence/decline of topics

---

## Troubleshooting

### Common Issues

#### "Column 'text' not found"

4chan data uses `body` not `text`:
```bash
--text-col body
```

#### "Could not parse timestamp"

Convert Unix timestamps:
```python
df['created_at'] = pd.to_datetime(df['time'], unit='s')
```

#### "Out of memory"

Process in chunks:
```bash
--max-rows 50000  # Process 50K rows at a time
```

Or use time-slicing:
```bash
--group-by-time monthly  # One month at a time
```

#### "Network too sparse"

Lower minimum document frequency:
```bash
--min-df 3  # Instead of default 10
```

#### "Processing too slow"

Use GPU acceleration:
```bash
--gpu  # For spaCy
--sentiment-device cuda  # For transformer sentiment
```

Or sample data:
```bash
--max-rows 10000  # Test on subset first
```

### Performance Tips

**For large datasets (>100K posts):**

1. **Test on subset first:**
   ```bash
   --max-rows 10000
   ```

2. **Use time-slicing:**
   ```bash
   --group-by-time weekly
   ```

3. **Limit vocabulary:**
   ```bash
   --min-df 20 --max-vocab 50000
   ```

4. **Enable GPU:**
   ```bash
   --gpu --sentiment-device cuda
   ```

5. **Use parallel processing:**
   ```bash
   # Multiple boards in parallel
   parallel -j 4 python -m src.semantic.kg_cli \
     --input 4chan_data.csv \
     --board-filter {} \
     --outdir output/kg_{} \
     ::: pol int sci his
   ```

---

## Complete Example Workflow

### Research Question
**"How did anti-immigrant discourse evolve on /pol/ during the 2024 election?"**

### Step-by-Step Analysis

```bash
# 1. Prepare data (convert timestamps, filter board)
python -c "
import pandas as pd
df = pd.read_csv('4chan_full_archive.csv')
df = df[df['board'] == 'pol']  # /pol/ only
df['created_at'] = pd.to_datetime(df['time'], unit='s')
df = df[df['created_at'].between('2024-01-01', '2024-11-30')]
df.to_csv('pol_2024.csv', index=False)
print(f'Filtered to {len(df)} posts')
"

# 2. Build temporal knowledge graphs (monthly)
python -m src.semantic.kg_cli \
  --input pol_2024.csv \
  --text-col body \
  --time-col created_at \
  --group-by-time monthly \
  --add-sentiment \
  --outdir output/pol_2024_kg_temporal

# 3. Analyze key entities
python -m src.semantic.kg_temporal_cli \
  --kg-dir output/pol_2024_kg_temporal \
  --entity "immigration" \
  --report immigration_timeline.md

python -m src.semantic.kg_temporal_cli \
  --kg-dir output/pol_2024_kg_temporal \
  --entity "Trump" \
  --report trump_timeline.md

# 4. Detect events/spikes
python -m src.semantic.kg_temporal_cli \
  --kg-dir output/pol_2024_kg_temporal \
  --detect-events \
  --event-threshold 2.0

# 5. Analyze entity relationships
python -m src.semantic.build_semantic_network \
  --input pol_2024.csv \
  --text-col body \
  --outdir output/pol_2024_semantic \
  --min-df 50 --topk 20

# 6. Visualize (open in Gephi)
# Use output/pol_2024_semantic/graph.graphml

# 7. Export for further analysis
python -c "
import pandas as pd
from pathlib import Path

# Aggregate entity mentions over time
kg_dir = Path('output/pol_2024_kg_temporal')
results = []

for period_dir in sorted(kg_dir.glob('period_*')):
    period = period_dir.name
    nodes = pd.read_csv(period_dir / 'kg_nodes.csv')
    
    results.append({
        'period': period,
        'total_entities': len(nodes),
        'immigration_mentions': nodes[nodes['entity'].str.contains('immigr', case=False)]['frequency'].sum(),
        'avg_sentiment': nodes['sentiment'].mean() if 'sentiment' in nodes.columns else None
    })

df = pd.DataFrame(results)
df.to_csv('pol_2024_summary.csv', index=False)
print(df)
"
```

### Expected Output

```
period,total_entities,immigration_mentions,avg_sentiment
period_2024-01,487,127,-0.32
period_2024-02,523,145,-0.38
period_2024-03,501,198,-0.41
...
period_2024-11,612,289,-0.52
```

**Interpretation:**
- Immigration mentions increased from 127 (Jan) to 289 (Nov)
- Sentiment became more negative (-0.32 → -0.52)
- Total entities increased (broader discourse scope)

---

## Next Steps

**After this guide:**
1. **[KG_FOR_SOCIAL_SCIENTISTS.md](KG_FOR_SOCIAL_SCIENTISTS.md)** - Deep dive into knowledge graphs
2. **[ETHICS.md](ETHICS.md)** - Comprehensive ethics guidelines
3. **[SCALING_GUIDE.md](SCALING_GUIDE.md)** - Processing millions of posts
4. **[GPU_SENTIMENT_GUIDE.md](GPU_SENTIMENT_GUIDE.md)** - Advanced sentiment analysis

**Example analyses:**
- See `examples/` for Jupyter notebooks
- Check `tutorials/` for hands-on walkthroughs

**Questions?** Open an issue on GitHub!

---

**Ready to analyze 4chan data responsibly and effectively!** 🚀

This toolkit is designed specifically for the challenges of anonymous, ephemeral social media. Use it to understand discourse dynamics, track entity relationships, and detect emerging narratives—all while respecting privacy and ethical research practices.
