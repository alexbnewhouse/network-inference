# Network Analysis Onboarding Training Guide

**A hands-on tutorial using real data to learn both anonymized and actor-based analysis**

---

## 📚 What You'll Learn

By the end of this training, you'll be able to:

1. ✅ Build **semantic networks** to map discourse topics
2. ✅ Extract **knowledge graphs** to find entities and relationships
3. ✅ Conduct **actor network analysis** to understand user behavior
4. ✅ Perform **temporal analysis** to track changes over time
5. ✅ Apply **sentiment analysis** to measure attitudes
6. ✅ Decide when to use **anonymized vs. identified** approaches
7. ✅ Understand **ethical considerations** for each method

**Time**: 60-90 minutes  
**Prerequisites**: Toolkit installed (see [GETTING_STARTED.md](GETTING_STARTED.md))

---

## 📊 Training Dataset

We'll use a realistic dataset of **10,000 social media posts** covering:
- **3 months** of activity (Jan-Mar 2024)
- **250 users** with realistic posting patterns
- **3,469 discussion threads**
- **5 topics**: geopolitics, tech, politics, culture, economics
- **Entities**: countries, politicians, companies, events

**Generate the data**:
```bash
python3 examples/generate_training_data.py
```

**Output files**:
- `examples/training_data.csv` - Full dataset (10,000 posts)
- `examples/training_data_users.csv` - User metadata

---

## Part 1: Anonymized Analysis (30 minutes)

**When to use**: Public data research, exploratory analysis, IRB-exempt studies

**Key principle**: Focus on **patterns and discourse**, not individual users

### Exercise 1.1: Semantic Network Analysis (10 min)

**Goal**: Identify main topics and how they're connected

```bash
# Build semantic network
python3 -m src.semantic.build_semantic_network \
  --input examples/training_data.csv \
  --text-col text \
  --outdir output/training_semantic \
  --min-df 10 \
  --topk 20
```

**What happens**:
1. Extracts frequent words (appearing in 10+ posts)
2. Calculates co-occurrence patterns
3. Builds network of related concepts
4. Keeps top 20 connections per word

**Explore the results**:
```bash
# View top concepts
head -20 output/training_semantic/nodes.csv

# View strongest connections
head -20 output/training_semantic/edges.csv
```

**Expected findings**:
- Clusters around "Russia", "China", "Ukraine"
- Tech cluster: "AI", "cryptocurrency", "cybersecurity"
- Political cluster: "Biden", "Trump", "election"

**Research questions you can answer**:
- What are the dominant topics?
- Which issues are discussed together?
- Are there distinct topic communities?

**Ethical note**: ✅ No user data needed, focuses on discourse patterns

---

### Exercise 1.2: Knowledge Graph - Entity Extraction (10 min)

**Goal**: Extract people, places, organizations mentioned

```bash
# Extract entities
python3 -m src.semantic.kg_cli \
  --input examples/training_data.csv \
  --text-col text \
  --outdir output/training_kg \
  --model en_core_web_sm \
  --max-rows 5000
```

**Note**: Using 5,000 posts for speed (full 10K takes ~5 min)

**Check the quality report**:
```bash
cat output/training_kg/kg_quality_report.md
```

**Expected entities**:
- **PERSON**: Biden, Trump, Putin, Musk
- **GPE** (countries): Russia, China, Ukraine, US
- **ORG**: NATO, UN, Google, Tesla

**View top entities**:
```bash
head -30 output/training_kg/kg_nodes.csv
```

**Research questions you can answer**:
- Which political figures are most discussed?
- Which countries are mentioned together?
- What organizations are central to discourse?

**Ethical note**: ✅ Public figure entities, no personal user data

---

### Exercise 1.3: Temporal Analysis (10 min)

**Goal**: Track how topics evolve over 3 months

```bash
# Generate weekly knowledge graphs
python3 -m src.semantic.kg_cli \
  --input examples/training_data.csv \
  --text-col text \
  --time-col created_at \
  --group-by-time weekly \
  --outdir output/training_temporal \
  --max-rows 5000
```

**This creates 13 weekly snapshots** (Jan-Mar 2024)

**Analyze timeline**:
```bash
python3 -m src.semantic.kg_temporal_cli \
  --kg-dir output/training_temporal \
  --timeline \
  --top-n 15
```

**Expected patterns**:
- Election-related entities spike in February
- Tech entities show steady growth
- Geopolitical entities cluster around specific weeks

**View specific entity**:
```bash
python3 -m src.semantic.kg_temporal_cli \
  --kg-dir output/training_temporal \
  --entity "Ukraine"
```

**Research questions you can answer**:
- When did specific topics emerge or decline?
- Were there sudden spikes (events)?
- How did discourse shift over time?

**Ethical note**: ✅ Aggregate patterns, no user tracking

---

## Part 2: Actor-Based Analysis (30 minutes)

**When to use**: User behavior studies, influence research, community mapping

**Key principle**: Study **network structures** while protecting privacy

### Exercise 2.1: Actor Network - Who Talks to Whom? (10 min)

**Goal**: Map user interaction patterns through thread participation

```bash
# Build actor network from thread co-participation
python3 -m src.semantic.actor_cli \
  --input examples/training_data.csv \
  --text-col text \
  --thread-col thread_id \
  --author-col user_id_hashed \
  --outdir output/training_actors
```

**Note**: Using `user_id_hashed` (anonymized IDs)

**Check the network**:
```bash
# View network statistics
head -20 output/training_actors/actor_metrics.csv
```

**Metrics explained**:
- `degree`: How many other users this user interacts with
- `betweenness`: How often user bridges different groups
- `eigenvector`: Influence score based on connections

**Top actors**:
```bash
sort -t',' -k2 -nr output/training_actors/actor_metrics.csv | head -10
```

**Research questions you can answer**:
- Who are the most connected users?
- Who bridges different communities?
- Are there isolated vs. highly connected users?

**Ethical consideration**: ⚠️ Uses hashed IDs to prevent re-identification

---

### Exercise 2.2: User-Entity Networks (10 min)

**Goal**: Discover user communities based on shared interests

```bash
# Build user-entity bipartite network
python3 -m src.semantic.kg_user_entity_network_cli \
  --kg-dir output/training_kg \
  --data examples/training_data.csv \
  --user-col user_id_hashed \
  --text-col text \
  --stats \
  --communities \
  --export-all output/training_networks
```

**View statistics**:
```
NETWORK STATISTICS
Nodes:
  Users: ~200-250
  Entities: ~40-60

Edges: ~800-1200
Average entities per user: ~4-6
```

**Community detection results**:
```
USER COMMUNITIES
Detected 3-5 communities:
  Community 0: Geopolitics-focused users
  Community 1: Tech-focused users
  Community 2: US politics-focused users
```

**Explore user profiles**:
```bash
# Pick a random hashed user ID from output
python3 -m src.semantic.kg_user_entity_network_cli \
  --kg-dir output/training_kg \
  --data examples/training_data.csv \
  --user-col user_id_hashed \
  --text-col text \
  --user <HASHED_USER_ID>
```

**Research questions you can answer**:
- Do users cluster by topic interest?
- Which entities define communities?
- Can we identify user "types" (geopolitics vs. tech focused)?

**Ethical consideration**: ⚠️ Use hashed IDs, report aggregate patterns only

---

### Exercise 2.3: Influence and Sentiment (10 min)

**Goal**: Combine network position with sentiment

```bash
# Add sentiment to knowledge graph
python3 -m src.semantic.kg_cli \
  --input examples/training_data.csv \
  --text-col text \
  --outdir output/training_kg_sentiment \
  --max-rows 5000 \
  --add-sentiment
```

**Check sentiment results**:
```bash
cat output/training_kg_sentiment/sentiment_summary.txt
```

**Expected findings**:
- Most positive entities: Tech companies, some politicians
- Most negative entities: Adversarial countries, controversial figures
- Controversial entities: Political figures with mixed sentiment

**Compare sentiment across users**:
```bash
# Group sentiment analysis
python3 -m src.semantic.kg_sentiment_enhanced_cli \
  --input examples/training_data.csv \
  --text-col text \
  --entity "Biden" \
  --stance
```

**Research questions you can answer**:
- Do central users have different sentiment patterns?
- Are influential users more positive/negative?
- Do communities have distinct sentiment profiles?

**Ethical consideration**: ⚠️ Sentiment at aggregate level, not individual profiling

---

## Part 3: Comparing Approaches (15 minutes)

### When to Use Anonymized Analysis

✅ **Use when**:
- Studying discourse patterns, not individuals
- Data is publicly posted (4chan, public Reddit/Twitter)
- Research question focuses on topics, not users
- IRB exemption possible (no human subjects)
- Exploratory phase of research

**Examples**:
- "How is climate change framed in online discourse?"
- "What conspiracy theories co-occur?"
- "How did discussion of AI evolve over time?"

**Outputs**:
- Semantic networks
- Knowledge graphs
- Temporal topic trends
- Entity co-occurrence

**Protections**:
- No user IDs needed
- Results report aggregate patterns
- Publications quote paraphrased text
- Lower re-identification risk

---

### When to Use Actor-Based Analysis

✅ **Use when**:
- Studying social influence, not just discourse
- Research questions involve user behavior
- Need to track individuals over time
- IRB approval obtained
- Data has clear user identifiers

**Examples**:
- "Who are the influential users spreading misinformation?"
- "Do users radicalize through network connections?"
- "Which users bridge echo chambers?"

**Outputs**:
- Actor networks (user-user)
- User-entity networks
- Influence metrics
- Community structure

**Protections**:
- ⚠️ Use hashed IDs (not raw usernames)
- ⚠️ IRB approval required
- ⚠️ Report aggregates, not individuals
- ⚠️ Secure data storage
- ⚠️ K-anonymity filtering (group size >k users)

---

### Decision Matrix

| Research Question | Anonymized | Actor-Based | Why |
|-------------------|------------|-------------|-----|
| What topics are discussed? | ✅ | ❌ | No users needed |
| Which entities co-occur? | ✅ | ❌ | Pattern-focused |
| How did discourse evolve? | ✅ | ❌ | Temporal trends |
| Who are influential users? | ❌ | ✅ | Need user metrics |
| Do users form communities? | ❌ | ✅ | Need user networks |
| Who spreads specific narratives? | ❌ | ✅ | Need user tracking |
| What's the sentiment toward X? | ✅ | Either | Can be aggregate |
| Do similar users share sentiment? | ❌ | ✅ | Need user comparison |

---

## Part 4: Ethical Best Practices (15 minutes)

### Privacy Protection Checklist

**Before Analysis**:
- [ ] Check if data is truly public (4chan yes, private forums no)
- [ ] Determine if IRB approval needed (see [ETHICS.md](ETHICS.md))
- [ ] Decide if user tracking necessary for research question
- [ ] Plan anonymization strategy if using user IDs

**During Analysis**:
- [ ] Use hashed IDs, not raw usernames/emails
- [ ] Store data on encrypted drives
- [ ] Limit team access to raw data
- [ ] Never share user-post mappings publicly

**When Reporting**:
- [ ] Report aggregate statistics only
- [ ] Paraphrase quotes to prevent searchability
- [ ] Use k-anonymity (groups of k+ users)
- [ ] Avoid naming specific users (unless public figures)
- [ ] Include ethical statement in publications

---

### Anonymization Techniques

**1. Hash User IDs**:
```python
import hashlib

def hash_user_id(user_id, salt="your_secret_salt"):
    """One-way hash to prevent re-identification."""
    return hashlib.sha256(f"{user_id}{salt}".encode()).hexdigest()[:16]

# Apply to dataframe
df['user_id_hashed'] = df['user_id'].apply(hash_user_id)
```

**2. Temporal Rounding**:
```python
# Round timestamps to week (prevents timing attacks)
df['created_at_week'] = pd.to_datetime(df['created_at']).dt.to_period('W')
```

**3. K-Anonymity Filtering**:
```python
# Only report groups with 5+ users
community_sizes = df.groupby('community').size()
large_communities = community_sizes[community_sizes >= 5].index
df_filtered = df[df['community'].isin(large_communities)]
```

**4. Paraphrase Quotes**:
```
❌ DON'T: "Russia is a terrorist state that must be stopped"
✅ DO: "One user expressed strong opposition to Russian actions"
```

---

## Part 5: Real-World Application (Bonus)

### Scenario: Analyzing Iron March Data

You've been asked to study the Iron March neo-Nazi forum dataset. How should you approach it?

**Step 1: Assess Data Type**
- ❌ Not currently public forum (defunct)
- ⚠️ Data is "leaked" (ethical gray area)
- ⚠️ Contains extremist content
- ✅ Public interest justification (understanding extremism)

**Step 2: Choose Approach**

**Option A: Anonymized (Recommended for initial analysis)**
```bash
# Export from R (see IRON_MARCH_GUIDE.md)
# Then analyze discourse patterns
python3 -m src.semantic.kg_cli \
  --input iron_march_posts.csv \
  --text-col text \
  --time-col created_at \
  --group-by-time monthly \
  --outdir output/iron_march_discourse
```

**Research questions**:
- What ideological concepts co-occur?
- How did discourse evolve 2011-2017?
- Which historical figures are central?

**Option B: Actor-Based (Requires IRB approval)**
```bash
# User network analysis
python3 -m src.semantic.actor_cli \
  --input iron_march_posts.csv \
  --text-col text \
  --thread-col topic_id \
  --author-col user_id_hashed \  # ⚠️ Hash the IDs!
  --outdir output/iron_march_actors
```

**Research questions**:
- Who are influential users?
- Do users form distinct communities?
- Are there recruitment patterns?

**Step 3: Ethical Safeguards**
- ✅ IRB approval (even though data is leaked)
- ✅ Hash all user IDs
- ✅ Never publish user-post mappings
- ✅ Aggregate results only
- ✅ Paraphrase all quotes
- ✅ Focus on systemic patterns, not individuals
- ✅ Secure data storage (encrypted, access-controlled)

**See full guide**: [IRON_MARCH_GUIDE.md](IRON_MARCH_GUIDE.md)

---

## Summary & Next Steps

### What You've Learned

✅ **Anonymized analysis**: Semantic networks, knowledge graphs, temporal trends  
✅ **Actor analysis**: User networks, communities, influence metrics  
✅ **Ethical decision-making**: When to anonymize, how to protect privacy  
✅ **Tool mastery**: 8+ CLI tools for different analysis types

### Recommended Learning Path

**Week 1: Master Anonymized Analysis**
- Practice with public datasets (Reddit, Twitter, news)
- Build semantic networks for different topics
- Extract knowledge graphs
- Track temporal trends

**Week 2: Actor Network Analysis**
- Read [ETHICS.md](ETHICS.md) thoroughly
- Get IRB approval if needed
- Practice with hashed IDs
- Learn community detection

**Week 3: Advanced Techniques**
- Sentiment analysis
- Stance detection
- User-entity networks
- Temporal knowledge graphs

**Week 4: Real Research Project**
- Apply to your own research question
- Write up results
- Prepare publication
- Document ethical procedures

---

## Quick Reference Commands

### Anonymized Analysis
```bash
# Semantic network
python3 -m src.semantic.build_semantic_network \
  --input data.csv --text-col text --outdir output/semantic

# Knowledge graph
python3 -m src.semantic.kg_cli \
  --input data.csv --text-col text --outdir output/kg

# Temporal KG
python3 -m src.semantic.kg_cli \
  --input data.csv --time-col created_at --group-by-time weekly \
  --outdir output/temporal

# Sentiment
python3 -m src.semantic.kg_cli \
  --input data.csv --add-sentiment --outdir output/kg_sentiment
```

### Actor-Based Analysis
```bash
# Actor network (⚠️ use hashed IDs!)
python3 -m src.semantic.actor_cli \
  --input data.csv --thread-col thread_id \
  --author-col user_id_hashed --outdir output/actors

# User-entity networks
python3 -m src.semantic.kg_user_entity_network_cli \
  --kg-dir output/kg --data data.csv \
  --user-col user_id_hashed --communities

# Community detection
python3 -m src.semantic.community_cli \
  --edges output/actors/edges.csv --outdir output/communities
```

---

## Getting Help

- **General guide**: [GETTING_STARTED.md](GETTING_STARTED.md)
- **Ethics**: [ETHICS.md](ETHICS.md)
- **Data format**: [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md)
- **Research applications**: [KG_FOR_SOCIAL_SCIENTISTS.md](KG_FOR_SOCIAL_SCIENTISTS.md)
- **Iron March specific**: [IRON_MARCH_GUIDE.md](IRON_MARCH_GUIDE.md)

**Questions?** Open a GitHub issue or consult your IRB for ethical guidance.

---

**Training complete! You're ready to conduct responsible network analysis.** 🎓🔬
