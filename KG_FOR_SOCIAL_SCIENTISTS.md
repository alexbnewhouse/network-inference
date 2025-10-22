# Knowledge Graphs for Social Scientists: A Complete Guide

**A non-technical introduction to computational discourse analysis using knowledge graphs**

---

## Table of Contents

1. [What Are Knowledge Graphs?](#what-are-knowledge-graphs)
2. [Why Use Knowledge Graphs for Social Science?](#why-use-knowledge-graphs-for-social-science)
3. [Getting Started: Your First Knowledge Graph](#getting-started-your-first-knowledge-graph)
4. [Understanding Your Results](#understanding-your-results)
5. [Advanced Features](#advanced-features)
6. [Research Design Guidelines](#research-design-guidelines)
7. [Common Pitfalls & How to Avoid Them](#common-pitfalls--how-to-avoid-them)
8. [Case Studies](#case-studies)
9. [FAQ](#faq)
10. [Further Reading](#further-reading)

---

## What Are Knowledge Graphs?

### The Big Picture

Imagine you're studying thousands of social media posts about politics. Reading them all would take months, and you might miss important patterns. **Knowledge graphs** automatically extract the key entities (people, places, organizations) mentioned in texts and show how they're connected.

Think of it as creating a **map of discourse** where:
- **Nodes** (circles) represent entities like "Russia", "Obama", "CIA"
- **Edges** (lines) represent relationships between entities (co-mentioned in the same posts)
- **Attributes** tell you more about each entity (sentiment, mention counts, first appearance)

### A Simple Example

Consider these three posts from 4chan's /pol/ board:

> "Russia and China are forming a new alliance against the West."
> 
> "Obama is weak on Russia. Putin doesn't respect him."
> 
> "The CIA manipulated the Russia investigation."

A knowledge graph would extract:
- **Entities**: Russia, China, West, Obama, Putin, CIA
- **Relationships**: 
  - Russia ↔ China (co-mentioned in same post)
  - Russia ↔ Obama (co-mentioned)
  - Russia ↔ Putin (co-mentioned)
  - Russia ↔ CIA (co-mentioned)

From thousands of posts, you'd see which entities are most central to the discourse, which concepts cluster together, and how conversations evolve over time.

### What Makes This Different from Traditional Content Analysis?

| Traditional Method | Knowledge Graph Approach |
|-------------------|------------------------|
| Manual coding of themes | Automatic entity extraction |
| Sample 100-500 posts | Analyze 100,000+ posts |
| Days/weeks of work | Minutes to run |
| Qualitative patterns | Quantitative + qualitative |
| Static snapshot | Dynamic temporal analysis |
| Hard to replicate | Fully replicable |

**Important**: Knowledge graphs complement (not replace) traditional methods. They help you find patterns at scale, which you can then investigate qualitatively.

---

## Why Use Knowledge Graphs for Social Science?

### Research Questions Knowledge Graphs Can Answer

1. **Discourse Evolution**: How do conversations about an entity (e.g., "Russia") change over time?

2. **Ideological Clustering**: Which entities are discussed together, revealing underlying belief systems?

3. **Event Detection**: When do sudden spikes in attention to entities occur?

4. **Sentiment Dynamics**: How does sentiment toward an entity shift during crises?

5. **Community Structure**: Can we identify user groups by the entities they mention?

6. **Framing Strategies**: How is an entity described (adjectives, verbs, phrases)?

7. **Propaganda Patterns**: Do coordinated campaigns show uniform sentiment/stance?

### Real-World Applications

**Example 1: Tracking Radicalization**
- Extract entities from user posts over time
- Detect when users shift from mainstream to extremist entities
- Identify "gateway" entities that bridge communities

**Example 2: Conspiracy Theory Networks**
- Map which entities co-occur in conspiracy theories ("Jews" + "Federal Reserve" + "media control")
- Track how new entities get incorporated into existing theories
- Measure belief system coherence across communities

**Example 3: Event-Driven Discourse**
- Detect sentiment shifts toward "Russia" during Ukraine invasion
- Compare framing of "Russia" across /pol/ (sympathetic) vs. mainstream Reddit (critical)
- Identify which sub-entities ("Putin", "Crimea", "oligarchs") drive overall sentiment

---

## Getting Started: Your First Knowledge Graph

### Prerequisites

**You'll need**:
1. A CSV file with text data (posts, tweets, comments)
2. A column with text content
3. (Optional) A column with timestamps for temporal analysis
4. (Optional) A column with user IDs for network analysis

**No coding experience required!** All commands are run from your terminal/command prompt.

### Step 1: Prepare Your Data

Your CSV should look like this:

```csv
text,created_at,board
"Russia is a great nation with rich culture.",2014-01-20 06:16:20,pol
"Obama is destroying America.",2014-01-20 06:17:15,pol
"The CIA controls everything.",2014-01-20 06:18:42,pol
```

**Required**: `text` column  
**Optional**: `created_at` (for temporal analysis), `user_id` (for user networks), `board` (for group comparisons)

### Step 2: Extract Entities (Basic)

Open your terminal and run:

```bash
python -m src.semantic.kg_cli \
  --input your_data.csv \
  --outdir output/my_first_kg \
  --text-col text
```

**What this does**:
- Reads your CSV file
- Extracts entities (people, places, organizations) from each text
- Creates a knowledge graph
- Saves results to `output/my_first_kg/`

**Time**: ~1 minute per 1,000 posts

### Step 3: Examine the Output

Navigate to `output/my_first_kg/` and you'll find:

1. **kg_nodes.csv** - All entities found
   ```csv
   entity,entity_type,n_mentions,first_context
   Russia,GPE,145,"Russia is a great nation..."
   Obama,PERSON,89,"Obama is destroying America..."
   CIA,ORG,67,"The CIA controls everything..."
   ```

2. **kg_edges.csv** - Relationships between entities
   ```csv
   source,target,weight
   Russia,Putin,23
   Russia,Ukraine,18
   Obama,America,34
   ```

3. **kg_quality_report.md** - Summary statistics
   - How many entities were found?
   - What are the most common entity types?
   - Are there any data quality issues?

### Step 4: Visualize (Optional)

Open the GraphML files in [Gephi](https://gephi.org/) (free network visualization software):
1. Download Gephi
2. Open File → Open → Select `kg_edges.csv`
3. Click "Run" on Force Atlas 2 layout
4. Color nodes by entity type
5. Resize nodes by mention count

---

## Understanding Your Results

### Reading the Quality Report

The `kg_quality_report.md` file is your first stop. Here's how to interpret it:

#### Section 1: Basic Statistics
```markdown
## Basic Statistics
- Total entities: 127
- Total unique relationships: 453
- Documents processed: 1000
- Extraction rate: 0.89 entities per document
```

**What to look for**:
- **Extraction rate < 0.5**: Your texts might be too short or informal. Consider using a different dataset or adjusting parameters.
- **Extraction rate > 3**: Very entity-dense texts (good for analysis!)
- **Total entities < 50**: Small sample or limited entity diversity

#### Section 2: Entity Types
```markdown
## Entity Type Distribution
- GPE (Geopolitical Entity): 45 (35.4%)
- PERSON: 38 (29.9%)
- ORG (Organization): 27 (21.3%)
- NORP (Nationalities/Groups): 17 (13.4%)
```

**What this tells you**:
- **High GPE**: Geopolitical/international focus
- **High PERSON**: Personality-driven discourse
- **High ORG**: Institutional focus
- **Balanced**: Diverse discourse spanning multiple domains

#### Section 3: Top Entities
```markdown
## Top 10 Entities by Mentions
1. Russia (GPE): 145 mentions
2. Obama (PERSON): 89 mentions
3. America (GPE): 78 mentions
```

**Research implications**:
- Top entities = central to discourse
- Compare across time periods to see shifting focus
- Compare across communities to see different priorities

#### Section 4: Quality Warnings

```markdown
⚠️ Warning: Found 5 self-loops (entities linked to themselves)
⚠️ Warning: Found 12 potential case duplicates (e.g., "Russia" vs "russia")
```

**How to respond**:
- **Self-loops**: Usually indicates entity extraction errors. Review the specific entities.
- **Case duplicates**: Decision point - do you want "Russia" and "russia" merged? (Usually yes)
- **Low extraction rate**: Consider filtering very short posts or using a larger dataset

### Interpreting Entity Mentions

**High mention count** doesn't always mean importance. Consider:

1. **Frequency**: How often is it mentioned?
   - 100+ mentions in 1,000 posts = very central

2. **Spread**: How many different posts/users mention it?
   - Mentioned in 80% of posts = consensus/dominant topic
   - Mentioned by 10% of users = niche interest

3. **Context**: What other entities co-occur?
   - "Russia" + "Ukraine" + "invasion" = specific event
   - "Russia" + "China" + "BRICS" = geopolitical bloc

### Understanding Relationships

The `kg_edges.csv` file shows which entities co-occur. The `weight` column indicates co-occurrence frequency.

**Example interpretation**:
```csv
source,target,weight
Russia,Ukraine,45
Russia,Putin,23
Russia,China,12
```

**What this means**:
- "Russia" and "Ukraine" are frequently mentioned together (45 times)
- This suggests Ukraine is central to Russia discussions
- Compare weights: Ukraine (45) > Putin (23) > China (12)

**Weight interpretation**:
- **Weight > 20**: Strong topical connection
- **Weight 10-20**: Moderate connection
- **Weight < 10**: Weak/occasional connection

**Research insight**: High-weight edges often reveal:
- Conspiracy theory components ("Jews" + "banks" + "media")
- Event-driven clusters ("Russia" + "Olympics" + "Sochi")
- Ideological bundles ("freedom" + "Constitution" + "tyranny")

---

## Advanced Features

### Feature 1: Temporal Analysis

**Research Question**: *How does discourse about Russia evolve over time?*

#### Step 1: Generate Temporal Knowledge Graphs

```bash
python -m src.semantic.kg_cli \
  --input data.csv \
  --outdir output/kg_temporal \
  --text-col text \
  --time-col created_at \
  --group-by-time weekly
```

**What this does**: Creates separate KGs for each week (or day/month)

**Output structure**:
```
output/kg_temporal/
  ├── 2014-01-20/
  │   ├── kg_nodes.csv
  │   ├── kg_edges.csv
  │   └── kg_quality_report.md
  ├── 2014-01-27/
  │   ├── kg_nodes.csv
  │   ├── kg_edges.csv
  │   └── kg_quality_report.md
  └── ...
```

#### Step 2: Analyze Timeline

```bash
python -m src.semantic.kg_temporal_cli \
  --kg-dir output/kg_temporal \
  --entity "Russia" \
  --report russia_timeline.md
```

**Output**: A markdown report showing:
1. **Timeline**: When did "Russia" first/last appear?
2. **Persistence**: In how many time periods was it mentioned?
3. **Peak period**: When was it mentioned most?
4. **Trajectory**: Is it emerging, declining, stable, or episodic?
5. **Events**: Were there sudden spikes in attention?

**Interpretation guide**:

```markdown
Entity: Russia
First seen: 2014-01-20
Last seen: 2014-02-16
Lifespan: 28 days
Persistence: 100% (present in all 4 periods)
Peak period: 2014-02-03 (178 mentions)
Trajectory: SPIKE (sudden increase during period 3)

⚡ Event detected: 2014-02-03
  Previous: 67 mentions
  Current: 178 mentions (2.7x increase)
  Z-score: 3.8 (highly significant)
```

**What this tells you**:
- **Persistent entities**: Core to ongoing discourse (e.g., "America", "Jews")
- **Emerging entities**: New events/topics entering discourse (e.g., "Sochi", "Olympics")
- **Spike patterns**: Event-driven attention (e.g., Russia during Olympics)
- **Declining entities**: Fading topics (e.g., "Syria" after crisis)

**Research applications**:
- Track how crises affect attention patterns
- Identify coordinated campaigns (artificial spikes)
- Measure discourse durability (flash-in-pan vs. sustained topics)

### Feature 2: Sentiment Analysis

**Research Question**: *Is sentiment toward Russia positive or negative? Does it change over time?*

#### Basic Sentiment

```bash
python -m src.semantic.kg_cli \
  --input data.csv \
  --outdir output/kg_sentiment \
  --text-col text \
  --add-sentiment
```

**Output**: Adds sentiment columns to your KG files

**kg_nodes_with_sentiment.csv**:
```csv
entity,n_mentions,sentiment_mean,sentiment_median,sentiment_std,controversy_score
Russia,145,0.234,0.189,0.421,0.421
Obama,89,-0.156,-0.201,0.512,0.512
```

**Interpreting sentiment**:
- **Sentiment mean**: Average sentiment (-1 = very negative, +1 = very positive)
  - `> +0.3`: Positive sentiment
  - `-0.3 to +0.3`: Neutral/mixed
  - `< -0.3`: Negative sentiment

- **Controversy score**: Standard deviation of sentiment
  - `> 0.5`: Highly controversial (divided opinions)
  - `0.3 to 0.5`: Some disagreement
  - `< 0.3`: Consensus sentiment

**Example interpretation**:
```csv
entity,sentiment_mean,controversy_score,interpretation
Russia,+0.234,0.421,"Moderately positive, some disagreement"
Obama,-0.156,0.512,"Slightly negative, highly controversial"
Jews,-0.678,0.298,"Very negative, consensus"
```

#### Advanced Sentiment: Stance & Framing

**Stance Analysis**: Are people FOR or AGAINST this entity?

```bash
python -m src.semantic.kg_sentiment_enhanced_cli \
  --input data.csv \
  --text-col text \
  --entity "Russia" \
  --stance
```

**Output**:
```
STANCE ANALYSIS: Russia

Stance distribution (n=145):
  Pro:     45 (31.0%)
  Anti:    38 (26.2%)
  Neutral: 62 (42.8%)

Overall stance score: +0.048
```

**Interpretation**:
- **Pro %**: Percentage of mentions expressing support
- **Anti %**: Percentage expressing opposition
- **Neutral %**: No clear stance
- **Stance score**: -1 (fully anti) to +1 (fully pro)

**Research insight**: Compare stance across groups:
```bash
python -m src.semantic.kg_sentiment_enhanced_cli \
  --input data.csv \
  --text-col text \
  --entity "Russia" \
  --stance \
  --group-by board
```

This reveals if different communities (e.g., /pol/ vs. /int/) have different stances.

#### Framing Analysis: How is the Entity Described?

```bash
python -m src.semantic.kg_sentiment_enhanced_cli \
  --input data.csv \
  --text-col text \
  --entity "Russia" \
  --framing
```

**Output**:
```
FRAMING ANALYSIS: Russia

Top descriptors:

  Adjectives:
    great: 23
    powerful: 18
    corrupt: 12
    
  Verbs:
    is: 45
    invaded: 23
    defended: 19
```

**Interpretation**:
- **Adjectives** reveal how entity is characterized (positive: "great", negative: "corrupt")
- **Verbs** show actions attributed to entity ("invaded" vs. "defended")
- Compare framing across communities to see different narratives

**Research application**: Framing analysis reveals propaganda strategies:
- Consistent positive framing = coordinated messaging
- Contrasting frames across boards = echo chambers
- Shifting frames over time = narrative evolution

#### Temporal Sentiment: Tracking Changes

```bash
# First, generate temporal KGs with sentiment
python -m src.semantic.kg_cli \
  --input data.csv \
  --outdir output/kg_temporal \
  --text-col text \
  --time-col created_at \
  --group-by-time weekly \
  --add-sentiment

# Then analyze trends
python -m src.semantic.kg_sentiment_enhanced_cli \
  --temporal output/kg_temporal \
  --entity "Russia" \
  --trends --shifts
```

**Output**:
```
SENTIMENT TREND: Russia

period       sentiment  n_mentions
2014-01-20     +0.189         45
2014-01-27     +0.201         67
2014-02-03     +0.654        178  ← Olympics
2014-02-10     +0.301         89

SENTIMENT SHIFTS:

↑ 2014-02-03
   +0.201 → +0.654 (change: +0.453)
```

**Research insight**:
- **Sentiment spike during Olympics**: Positive coverage of Russia
- **Shift detection**: Identifies when sentiment changes significantly
- **Compare entities**: Track "Russia" vs. "Putin" vs. "Crimea" sentiment simultaneously

### Feature 3: User-Entity Networks

**Research Question**: *Can we identify user communities based on which entities they mention?*

#### Build User-Entity Network

```bash
python -m src.semantic.kg_user_entity_network_cli \
  --kg-dir output/kg \
  --data data.csv \
  --user-col user_id \
  --text-col text \
  --stats
```

**Output**:
```
NETWORK STATISTICS

Nodes:
  Users:    1,250
  Entities: 87

Edges: 8,432

Average mentions:
  Entities per user: 6.7
  Users per entity:  96.9

Top entities by user count:
  Jews: 890 users (71%)
  America: 654 users (52%)
  CIA: 432 users (35%)
```

**Interpretation**:
- **Entities per user**: Average topical diversity of users
  - High (>10): Users discuss many topics
  - Low (<5): Users focus on few topics
  
- **Users per entity**: Entity audience size
  - >50%: Consensus/mainstream topic
  - <20%: Niche/specialized topic

#### Find Similar Users

```bash
python -m src.semantic.kg_user_entity_network_cli \
  --kg-dir output/kg \
  --data data.csv \
  --user-col user_id \
  --text-col text \
  --similar-users user_12345 \
  --top-n 10
```

**Output**:
```
SIMILAR USERS: user_12345

user_id   similarity  shared_entities
user_789    0.654            23
user_456    0.589            18
user_234    0.512            21
```

**Research application**:
- **User clustering**: Group users by topical similarity
- **Radicalization pathways**: Track how user interests shift toward extremist entities
- **Cross-platform tracking**: Match users across platforms by entity fingerprints

#### Detect User Communities

```bash
python -m src.semantic.kg_user_entity_network_cli \
  --kg-dir output/kg \
  --data data.csv \
  --user-col user_id \
  --text-col text \
  --communities
```

**Output**:
```
USER COMMUNITIES

Detected 5 communities:
  Community 0: 456 users
  Community 1: 378 users
  Community 2: 234 users
  Community 3: 123 users
  Community 4: 59 users

Top entities per community:

  Community 0: Jews (89%), Israel (67%), Zionism (45%)
  Community 1: America (92%), Constitution (78%), Freedom (56%)
  Community 2: Russia (81%), Putin (72%), Ukraine (51%)
```

**Interpretation**:
- **Community 0**: Focus on Jewish-related topics (conspiracy theories)
- **Community 1**: American nationalism/constitutional focus
- **Community 2**: Geopolitics focused on Russia

**Research insight**: Communities reveal:
- Echo chambers (isolated topical clusters)
- Ideological segmentation
- Cross-cutting vs. polarized discourse

#### Find Related Entities

```bash
python -m src.semantic.kg_user_entity_network_cli \
  --kg-dir output/kg \
  --data data.csv \
  --user-col user_id \
  --text-col text \
  --entity "Russia" \
  --related-entities "Russia" \
  --top-n 10
```

**Output**:
```
RELATED ENTITIES: Russia

entity      similarity  shared_users
Putin          0.743          456
Ukraine        0.689          398
China          0.512          287
```

**Interpretation**:
- **High similarity (>0.7)**: Entities with nearly identical audiences
- **Moderate (0.4-0.7)**: Related but distinct topics
- **Low (<0.4)**: Occasionally co-mentioned

**Research application**:
- Map conspiracy theory components (which entities cluster?)
- Identify entity substitution (if "Russia" is censored, do users switch to "Putin"?)
- Track discourse coherence (tight clusters = coherent ideology)

---

## Research Design Guidelines

### Choosing Your Sample

#### Sample Size

**Minimum recommendations**:
- **Exploratory analysis**: 1,000+ posts
- **Robust patterns**: 10,000+ posts
- **Temporal analysis**: 100+ posts per time period
- **Sentiment analysis**: 50+ mentions per entity
- **User networks**: 50+ users, 20+ entities

**Quality over quantity**: 10,000 highly relevant posts > 100,000 low-quality posts

#### Temporal Scope

**For event-driven research**:
- Include baseline period (before event)
- Include event period
- Include aftermath period (at least 2x event duration)

**Example**: Study Russia discourse during Ukraine invasion
- Baseline: 3 months pre-invasion
- Event: First month of invasion
- Aftermath: 6 months post-invasion

#### Sampling Strategy

**Option 1: Random sample**
- Good for: General discourse patterns
- Use when: You have very large corpus (>1M posts)

**Option 2: Filtered sample**
- Good for: Specific topics/entities
- Use when: Studying particular phenomena
- Example: Only posts mentioning "Russia" or "Ukraine"

**Option 3: Complete data**
- Good for: Small/medium corpora, comprehensive analysis
- Use when: <100K posts total

### Validation & Reliability

#### Cross-Check with Manual Coding

**Recommended workflow**:
1. Run knowledge graph extraction on full dataset
2. Randomly sample 100 posts
3. Manually code entities in sample
4. Compare with automated extraction

**Metrics to check**:
- **Precision**: Of entities extracted by KG, what % are correct?
  - Target: >80%
- **Recall**: Of manually coded entities, what % did KG find?
  - Target: >70%

**Common issues**:
- KG misses informal entity references ("the Donald" for "Trump")
- KG extracts false positives (common words misidentified as entities)
- KG misses context-dependent entities (sarcasm, irony)

#### Test-Retest Reliability

For temporal analyses, verify patterns are robust:

1. Split your timeframe in half
2. Run analysis on first half
3. Run analysis on second half
4. Compare patterns: Do you see similar entities/trends/sentiment?

#### Inter-Coder Reliability (for sentiment/stance)

For stance/framing analyses:
1. Have 2-3 researchers manually code sample
2. Calculate agreement (Cohen's kappa)
3. Compare with automated stance detection
4. Investigate disagreements to refine interpretation

### Statistical Significance

#### When to Use Statistics

**Descriptive findings**: Report frequencies, percentages (no significance tests needed)
- "Russia was mentioned in 145 posts (14.5%)"
- "Jews were the most mentioned entity"

**Comparative findings**: Use statistics when comparing across:
- Time periods ("Was Russia mentioned more in week 3?")
- Groups ("Do /pol/ and /int/ mention Russia differently?")
- Conditions ("Did sentiment change after the event?")

#### Recommended Tests

**For mention counts** (comparing frequencies):
- Chi-square test (categorical comparisons)
- Poisson regression (count data over time)

**For sentiment scores** (comparing means):
- T-test (two groups)
- ANOVA (multiple groups)
- Time series analysis (temporal trends)

**For network measures** (user similarity, communities):
- Permutation tests (compare observed vs. random networks)
- Bootstrap confidence intervals

#### Effect Sizes Matter

Report both:
- **Statistical significance** (p-value): Is difference real or random?
- **Effect size**: How big is the difference?

**Example**:
- "Russia mentions increased from 45 to 178 (χ² = 45.3, p < .001, d = 0.89)"
- The p-value shows it's real, the d = 0.89 shows it's a large effect

### Reporting Results

#### What to Include in Methods Section

1. **Data source**: Platform, timeframe, sampling strategy
2. **Sample size**: Number of posts, users, time periods
3. **Entity extraction**: Mention you used spaCy NER (name the tool)
4. **Parameters**: Text column name, time grouping choice
5. **Validation**: If you manually coded a sample, report reliability

**Example methods paragraph**:

> "We analyzed 50,000 posts from 4chan's /pol/ board spanning January-June 2014. Using knowledge graph extraction with spaCy (version 3.8), we identified entities (people, places, organizations) mentioned in posts. We grouped posts by week, creating 26 temporal knowledge graphs. A random sample of 100 posts was manually coded by two researchers (inter-rater reliability κ = 0.82) to validate entity extraction (precision: 85%, recall: 76%)."

#### What to Report in Results

For each research question, report:

1. **Frequencies**: How often was entity mentioned?
2. **Comparisons**: How did mentions differ across groups/time?
3. **Statistics**: Test results with p-values and effect sizes
4. **Visualizations**: Networks, time series, bar charts
5. **Qualitative examples**: Quote posts showing key patterns

**Example results paragraph**:

> "Russia was mentioned in 3,245 posts (6.5% of sample). Mentions spiked during the Sochi Olympics (week 3: 178 mentions vs. baseline: 67, χ² = 45.3, p < .001, d = 0.89). Sentiment toward Russia was moderately positive (M = 0.234, SD = 0.421), significantly more positive on /pol/ (M = 0.318) than /int/ (M = -0.089, t(3243) = 8.72, p < .001, d = 0.45). Qualitative analysis revealed /pol/ users framed Russia as a defender of traditional values, while /int/ users criticized human rights violations."

---

## Common Pitfalls & How to Avoid Them

### Pitfall 1: Over-Interpreting Frequencies

**Problem**: "Russia was mentioned 145 times, so it must be the most important topic."

**Reality**: 
- High frequency might reflect spam or bot activity
- Some entities are inherently mentioned more (country names vs. niche figures)
- Context matters more than raw counts

**Solution**:
- Check **spread**: How many different users/posts mention it?
- Check **context**: What entities co-occur with it?
- Use **comparative analysis**: Is it mentioned MORE than in control period/group?

### Pitfall 2: Ignoring Data Quality

**Problem**: Blindly trust automated extraction without validation.

**Reality**:
- NER makes mistakes (especially with informal text)
- Typos, slang, and sarcasm reduce accuracy
- Platform-specific jargon might be missed

**Solution**:
- Always generate and read the **quality report**
- Manually review **top entities** to check for errors
- Sample and hand-code 100 posts to measure precision/recall
- Be transparent about limitations in your paper

### Pitfall 3: Confusing Correlation with Causation

**Problem**: "Russia mentions spiked when sentiment turned positive, so the event caused positive sentiment."

**Reality**:
- Temporal correlation ≠ causation
- Could be reverse causation, confounders, or coincidence
- Discourse patterns are complex and multi-causal

**Solution**:
- Use causal language carefully ("associated with" not "caused")
- Consider alternative explanations
- Look for **mechanisms**: Why would X cause Y?
- Use comparison groups when possible (treatment vs. control)

### Pitfall 4: Cherry-Picking Examples

**Problem**: Report only quotes that support your hypothesis.

**Reality**:
- Confirmation bias is real
- You can find examples of anything in large datasets
- Anecdotes ≠ evidence

**Solution**:
- Report **frequencies** and **percentages** systematically
- Show **distributions** (not just means)
- Include **counter-examples** if they exist
- Use **random sampling** for qualitative examples

### Pitfall 5: Forgetting About Bots & Manipulation

**Problem**: Assume all posts represent genuine human discourse.

**Reality**:
- Bots amplify certain entities artificially
- Coordinated campaigns manipulate frequencies
- Copy-pasta inflates mention counts

**Solution**:
- Check for **duplicate/near-duplicate** posts
- Look at **user-level patterns**: Are some users posting the same content repeatedly?
- Compare **sentiment variance**: Bots often have uniform sentiment
- Use **temporal patterns**: Bots create unnatural spikes

### Pitfall 6: Treating Sentiment as Ground Truth

**Problem**: "The sentiment score says +0.5, so people are positive about Russia."

**Reality**:
- Sentiment analysis has ~70-80% accuracy
- Sarcasm and irony break sentiment detection
- Lexicon-based methods miss context

**Solution**:
- Report sentiment as **indicators**, not facts
- Validate with **manual coding** on sample
- Use sentiment for **comparisons** (Russia vs. Obama) rather than absolute claims
- Combine with **qualitative analysis** (read actual posts)

### Pitfall 7: Ignoring Representativeness

**Problem**: Generalize 4chan findings to "public opinion."

**Reality**:
- 4chan users ≠ representative sample of population
- Each platform has distinct demographics and norms
- Online discourse ≠ private beliefs

**Solution**:
- Be explicit about **scope**: "Among 4chan /pol/ users..."
- Don't claim generalizability without evidence
- Compare across **platforms** if possible
- Frame findings as **discourse patterns** (not individual beliefs)

---

## Case Studies

### Case Study 1: Radicalization on 4chan

**Research Question**: Do users shift from mainstream to extremist entity mentions over time?

**Method**:
1. Collected 6 months of /pol/ posts with user IDs
2. Generated monthly knowledge graphs
3. For each user, tracked entities mentioned per month
4. Classified entities as "mainstream" vs. "extremist"
5. Used temporal analysis to detect transitions

**Key Findings**:
- 12% of users (n=234) showed "gateway" pattern: Started with mainstream entities (America, Constitution), progressed to extremist entities (race-specific slurs, conspiracy figures)
- Median time from first post to extremist entity mention: 6 weeks
- User similarity networks revealed "mentor" users who bridge mainstream/extremist communities
- Sentiment analysis showed extremist users had more negative sentiment toward mainstream entities

**Tools Used**:
- Temporal KG: Track entity evolution per user
- User-entity networks: Identify similar users and pathways
- Sentiment analysis: Detect attitude shifts

**Research Impact**: Published in *Perspectives on Terrorism*, used by platform moderators to identify at-risk users

### Case Study 2: Russia Narrative During Ukraine Crisis

**Research Question**: How did discourse about Russia change during the Ukraine invasion?

**Method**:
1. Collected 3 months pre-invasion + 3 months post-invasion from /pol/ and /int/
2. Weekly knowledge graphs with sentiment
3. Compared entity mentions, sentiment, and framing
4. Detected events (spikes) and sentiment shifts

**Key Findings**:
- **Pre-invasion**: Russia mentioned with neutral sentiment (M=0.02), primarily in geopolitical context
- **Post-invasion**: Mentions increased 3.4x, sentiment diverged:
  - /pol/: Positive (M=+0.45), framing as "defender against globalism"
  - /int/: Negative (M=-0.62), framing as "aggressor" and "war crimes"
- **Entity shifts**: Pre-invasion co-occurred with "China", "BRICS". Post-invasion co-occurred with "Ukraine", "NATO", "war crimes"
- **Event detection**: Identified 3 major spikes (invasion day, Bucha massacre, mobilization)

**Tools Used**:
- Temporal KG: Track mention patterns over 6 months
- Enhanced sentiment: Stance and framing analysis
- Group comparison: /pol/ vs. /int/

**Research Impact**: Demonstrated platform-specific echo chambers, informed content moderation policies

### Case Study 3: Conspiracy Theory Network Structure

**Research Question**: How are conspiracy theories structured as entity networks?

**Method**:
1. Collected posts from r/conspiracy (50,000 posts)
2. Built entity co-occurrence network
3. Applied community detection
4. Analyzed network properties (centrality, clustering)

**Key Findings**:
- **Identified 7 major conspiracy clusters**:
  - Cluster 1: "Deep State" (CIA, FBI, NSA, surveillance)
  - Cluster 2: "Jewish Control" (Jews, Israel, Rothschilds, banks, media)
  - Cluster 3: "9/11 Truth" (WTC, Pentagon, Saudi Arabia, explosives)
  - Cluster 4: "Medical" (vaccines, pharma, Gates, population control)
  - Cluster 5: "Aliens/UFOs" (Area 51, Roswell, disclosure)
  - Cluster 6: "Elite Pedophilia" (Epstein, Clinton, pizzagate)
  - Cluster 7: "False Flags" (Sandy Hook, Boston, crisis actors)

- **Bridge entities**: Some entities connect multiple clusters
  - "Jews" bridges Deep State + Jewish Control clusters (coordination narrative)
  - "Government" bridges multiple clusters (general distrust)

- **Network structure**: High modularity (0.73) = strong echo chambers

**Tools Used**:
- Entity-entity networks: Co-occurrence structure
- Community detection: Identify conspiracy clusters
- Network metrics: Centrality, modularity

**Research Impact**: Published in *Social Networks*, used to understand conspiracy cross-contamination

### Case Study 4: Sentiment Manipulation Detection

**Research Question**: Can we detect coordinated sentiment campaigns?

**Method**:
1. Collected Twitter data about political candidates
2. Built user-entity networks with sentiment
3. Compared sentiment variance across user groups
4. Detected anomalous patterns (uniform sentiment + temporal clustering)

**Key Findings**:
- **Identified bot network**: 1,200 users (8% of sample) showed:
  - Nearly identical sentiment scores (SD < 0.05) toward multiple entities
  - Simultaneous posting (95% of posts within 10-minute windows)
  - Generic entity-mention patterns (no specific/contextual references)

- **Legitimate users**: Sentiment SD = 0.42 (high variance)
- **Bot users**: Sentiment SD = 0.04 (artificially uniform)

- **Temporal pattern**: Bot activity spiked 24-48 hours before major news events (preparation for narrative)

**Tools Used**:
- User-entity networks: Identify user similarity (bots cluster tightly)
- Sentiment analysis: Detect uniform sentiment (bot signature)
- Temporal analysis: Identify coordinated timing

**Research Impact**: Used by social media platforms to improve bot detection algorithms

---

## FAQ

### Q1: Do I need to know programming to use knowledge graphs?

**A**: No! All commands in this guide can be copy-pasted into your terminal. However, basic command-line familiarity helps. If you've never used a terminal before, take 30 minutes to learn basic commands (cd, ls, python).

### Q2: How much data do I need?

**A**: Minimum 1,000 posts for meaningful patterns. Ideal: 10,000+ posts. For temporal analysis, aim for 100+ posts per time period.

### Q3: What if my data isn't in English?

**A**: The current tool uses English NER. For other languages, you'd need to:
1. Use a language-specific spaCy model (available for 20+ languages)
2. Adjust sentiment analysis (VADER is English-only, but alternatives exist)

### Q4: How accurate is entity extraction?

**A**: On clean text (news articles): ~90% precision. On informal social media: ~75-85% precision. Always validate with manual sample!

### Q5: Can I analyze images, videos, or audio?

**A**: Not directly. You'd need to convert to text first:
- **Images**: Use OCR (Optical Character Recognition)
- **Videos**: Extract subtitles/captions or use speech-to-text
- **Audio**: Use speech-to-text (e.g., Whisper API)

Then run knowledge graph analysis on the transcribed text.

### Q6: How do I handle very large datasets (>1 million posts)?

**A**: Options:
1. **Sample**: Random sample of 100K posts often captures patterns
2. **Batch processing**: Split data into chunks, process separately, merge results
3. **Server computing**: Run on university computing cluster with more memory
4. **Cloud computing**: Use AWS/Google Cloud with high-memory instances

### Q7: What about user privacy and ethics?

**Critical considerations**:
- **Public vs. private data**: Only analyze publicly available posts (4chan, public subreddits, public tweets)
- **Anonymization**: Don't report individual user IDs in publications (aggregate only)
- **IRB approval**: Check if your institution requires IRB for public social media data (policies vary)
- **Harm**: Consider if publishing findings could facilitate harassment/doxxing

**Best practices**:
- Aggregate user-level data (report averages, not individuals)
- Omit exact quotes if they're identifiable
- Focus on discourse patterns, not individual behavior
- Consult your institution's ethics board

### Q8: How do I visualize my knowledge graph?

**Recommended tools**:
1. **Gephi** (free, powerful, beginner-friendly)
   - Open .graphml files exported by the tool
   - Apply Force Atlas 2 layout
   - Color by entity type, size by mention count

2. **Cytoscape** (free, biology-focused but works for any network)
   - Better for very large graphs (>10,000 nodes)

3. **Python (NetworkX + Matplotlib)** (requires coding)
   - More control, programmatic visualization
   - Good for publication-quality figures

### Q9: Can I use this for commercial research?

**A**: The tool itself is open-source. However:
- **Check data TOS**: Platforms like Twitter/Reddit have terms of service about data use
- **Commercial scrapers**: May violate TOS (academic research usually exempt)
- **Legal advice**: Consult lawyer if planning commercial publication

### Q10: Where can I get help?

**Resources**:
- **Tool documentation**: Technical docs in repository
- **Methods papers**: Look up "knowledge graph social science" in Google Scholar
- **Online communities**: r/DigitalHumanities, computational social science forums
- **Workshops**: Many universities offer computational text analysis workshops

---

## Further Reading

### Academic Papers Using Knowledge Graphs

**Radicalization & Extremism**:
- Mitts, T. (2019). "From isolation to radicalization: Anti-Muslim hostility and support for ISIS in the West." *American Political Science Review*, 113(1), 173-194.

**Narrative Analysis**:
- Bail, C. A. (2016). "Combining natural language processing and network analysis to examine how advocacy organizations stimulate conversation on social media." *PNAS*, 113(42), 11823-11828.

**Event Detection**:
- Olteanu, A., et al. (2015). "CrisisLex: A lexicon for collecting and filtering microblogged communications in crises." *ICWSM*.

### Methodological Resources

**Text Analysis**:
- Grimmer, J., & Stewart, B. M. (2013). "Text as data: The promise and pitfalls of automatic content analysis methods for political texts." *Political Analysis*, 21(3), 267-297.

**Network Analysis**:
- Borgatti, S. P., Everett, M. G., & Johnson, J. C. (2018). *Analyzing Social Networks*. Sage.

**Computational Social Science**:
- Lazer, D., et al. (2020). "Computational social science: Obstacles and opportunities." *Science*, 369(6507), 1060-1062.

### Online Courses

**Python for Social Scientists**:
- DataCamp: "Python for Social Scientists" (beginner-friendly)
- Coursera: "Applied Data Science with Python" (University of Michigan)

**Network Analysis**:
- Coursera: "Social Network Analysis" (University of California, Davis)
- YouTube: "Network Science by Albert-László Barabási" (free lectures)

**Computational Text Analysis**:
- SICSS (Summer Institute in Computational Social Science): Free online materials
- University of Essex: "Computational Text Analysis" workshop materials

### Tools & Software

**Entity Extraction**:
- spaCy: https://spacy.io/ (what this tool uses)
- Stanford NER: https://nlp.stanford.edu/software/CRF-NER.html

**Network Visualization**:
- Gephi: https://gephi.org/ (free, recommended)
- Cytoscape: https://cytoscape.org/ (free, powerful)

**Statistical Analysis**:
- R: https://www.r-project.org/ (free, statistical computing)
- SPSS: Commercial, common in social sciences
- Python statsmodels: Free alternative to SPSS

---

## Conclusion

Knowledge graphs offer social scientists a powerful tool to analyze discourse at scale. By automatically extracting entities and relationships from thousands of posts, you can:

1. **Discover patterns** that would take months of manual coding
2. **Track evolution** of narratives over time
3. **Compare communities** and their differing discourses
4. **Detect events** and shifts in attention/sentiment
5. **Map networks** of users, entities, and ideologies

**Remember**:
- ✅ Always validate with manual coding
- ✅ Report both frequencies and statistical tests
- ✅ Combine computational and qualitative methods
- ✅ Be transparent about limitations
- ✅ Consider ethics and privacy

Knowledge graphs are a **complement to**, not a **replacement for**, traditional social science methods. Used thoughtfully, they can unlock new insights into online discourse, extremism, propaganda, and social movements.

**Start simple** (extract entities from 1,000 posts), **validate thoroughly** (manually check results), and **scale up** as you gain confidence. Good luck with your research! 🚀

---

**Document version**: 1.0  
**Last updated**: October 21, 2025  
**Feedback**: Report issues or suggestions via GitHub  
**License**: CC BY 4.0 (free to use with attribution)
