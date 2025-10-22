# Using Network Inference with Iron March Data

**A complete guide for analyzing the Iron March neo-Nazi forum dataset**

---

## Overview

The Iron March dataset (from the R package `github.com/knapply/ironmarch`) contains data from a defunct neo-Nazi message board. This guide shows how to export the data from R and analyze it using our network inference toolkit.

**⚠️ IMPORTANT**: Before working with this data, review [ETHICS.md](ETHICS.md) for:
- IRB requirements for extremism research
- Privacy and safety considerations
- Responsible reporting practices
- De-identification techniques

---

## Table of Contents

1. [Data Available](#data-available)
2. [Exporting from R](#exporting-from-r)
3. [Network Analysis Workflows](#network-analysis-workflows)
4. [Research Questions](#research-questions)
5. [Code Examples](#code-examples)
6. [Safety Considerations](#safety-considerations)

---

## Data Available

The Iron March R package provides three main data types relevant for network analysis:

### 1. Direct Messages (`build_messages()`)
- **22,309 private messages** between users
- **Fields**: msg_id, msg_topic_id, msg_date, msg_post, msg_author_id, msg_ip_address
- **Network Type**: User-to-user communication network
- **Use Cases**: Private communication patterns, direct influence networks

### 2. Forum Posts (`im_forums_dfs$forums_posts`)
- **195,128 public forum posts**
- **Fields**: pid, author_id, author_name, post_date, post (content), topic_id, ip_address
- **Network Types**: User-entity networks, semantic networks, temporal analysis
- **Use Cases**: Public discourse analysis, topic evolution, user communities

### 3. Forum Threads (`im_forums_dfs$forums_topics`)
- **7,168 discussion threads**
- **Fields**: tid, title, posts, starter_id, start_date, last_post, forum_id, views
- **Network Types**: Thread participation networks, temporal networks
- **Use Cases**: Topic tracking, conversation flow, engagement patterns

### 4. Member Information (`build_members()`)
- **1,542 registered users**
- **Fields**: member_id, name, email, joined, ip_address, posts, member_group_id
- **Use Cases**: User attribution, demographic analysis, activity patterns

---

## Exporting from R

### Step 1: Install and Load Iron March Package

```r
# Install from GitHub
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
remotes::install_github("knapply/ironmarch")

library(ironmarch)
library(dplyr)
library(readr)
```

### Step 2: Export Direct Messages to CSV

```r
# Build messages dataframe
messages <- ironmarch::build_messages()

# Export for network analysis
write_csv(messages, "iron_march_messages.csv")

# Preview structure
glimpse(messages)
# Columns: msg_id, msg_topic_id, msg_date, msg_post, msg_author_id, msg_ip_address
```

### Step 3: Export Forum Posts to CSV

```r
# Get forum posts
posts <- ironmarch::im_forums_dfs$forums_posts

# Clean and prepare for export
posts_clean <- posts %>%
  select(
    post_id = pid,
    user_id = author_id,
    user_name = author_name,
    created_at = post_date,
    text = post,
    thread_id = topic_id,
    ip_address
  ) %>%
  # Remove HTML tags from post content
  mutate(
    text = gsub("<[^>]*>", "", text),
    text = gsub("&nbsp;", " ", text),
    text = gsub("&quot;", "\"", text),
    text = gsub("&amp;", "&", text)
  )

write_csv(posts_clean, "iron_march_posts.csv")
```

### Step 4: Export Thread Metadata

```r
# Get thread information
threads <- ironmarch::im_forums_dfs$forums_topics

threads_clean <- threads %>%
  select(
    thread_id = tid,
    title,
    thread_posts = posts,
    user_id = starter_id,
    created_at = start_date,
    last_post = last_post,
    forum_id,
    views
  )

write_csv(threads_clean, "iron_march_threads.csv")
```

### Step 5: Export Member Information

```r
# Get member data
members <- ironmarch::build_members()

members_clean <- members %>%
  select(
    user_id = member_id,
    username = name,
    joined,
    total_posts = posts,
    ip_address,
    group_id = member_group_id
  )

write_csv(members_clean, "iron_march_members.csv")
```

---

## Network Analysis Workflows

### Workflow 1: User Communication Network (Direct Messages)

**Research Question**: Who are the central actors in private communications?

```bash
# Build user-to-user network from DMs
python -m src.semantic.actor_cli \
  --input iron_march_messages.csv \
  --text-col msg_post \
  --thread-col msg_topic_id \
  --post-col msg_id \
  --author-col msg_author_id \
  --outdir output/dm_network

# Analyze centrality
python -m src.semantic.community_cli \
  --edges output/dm_network/edges.csv \
  --outdir output/dm_communities
```

**Output**: User network showing who messages whom, centrality scores, community structure

### Workflow 2: Semantic Network of Forum Discourse

**Research Question**: What ideological concepts are co-discussed in forum posts?

```bash
# Build semantic network of concepts
python -m src.semantic.build_semantic_network \
  --input iron_march_posts.csv \
  --text-col text \
  --outdir output/semantic_network \
  --min-df 10 \
  --topk 20 \
  --window 10
```

**Output**: Network of ideological terms/concepts, showing which ideas co-occur

### Workflow 3: Knowledge Graph - Entity Extraction

**Research Question**: Which people, organizations, and places are discussed together?

```bash
# Extract entities (people, orgs, locations)
python -m src.semantic.kg_cli \
  --input iron_march_posts.csv \
  --text-col text \
  --outdir output/knowledge_graph \
  --model en_core_web_sm \
  --max-rows 50000
```

**Output**: Network of named entities (Hitler, Mussolini, organizations, locations)

### Workflow 4: Temporal Analysis

**Research Question**: How did discourse evolve over time?

```bash
# Build monthly time-sliced networks
python -m src.semantic.time_slice_cli \
  --input iron_march_posts.csv \
  --text-col text \
  --time-col created_at \
  --outdir output/temporal \
  --freq M

# Track entity emergence/decline
python -m src.semantic.kg_temporal_cli \
  --kg-dir output/temporal \
  --entity "Hitler" \
  --report hitler_timeline.md
```

**Output**: Monthly networks showing discourse evolution, entity timelines

### Workflow 5: User-Entity Bipartite Network

**Research Question**: Which users discuss which topics/entities?

```bash
# First build KG from posts
python -m src.semantic.kg_cli \
  --input iron_march_posts.csv \
  --text-col text \
  --outdir output/kg \
  --max-rows 100000

# Build user-entity network
python -m src.semantic.kg_user_entity_network_cli \
  --kg-dir output/kg \
  --data iron_march_posts.csv \
  --user-col user_id \
  --text-col text \
  --stats \
  --communities \
  --export-all output/user_entity_networks
```

**Output**: Bipartite graph (users ↔ entities), user communities by shared interests

### Workflow 6: Sentiment & Stance Analysis

**Research Question**: What is the sentiment/stance toward specific entities?

```bash
# Analyze sentiment toward entities over time
python -m src.semantic.kg_sentiment_enhanced_cli \
  --input iron_march_posts.csv \
  --text-col text \
  --time-col created_at \
  --entity "Jews" \
  --stance \
  --framing \
  --temporal \
  --outdir output/sentiment
```

**Output**: Sentiment scores, stance detection (pro/anti), framing analysis

### Workflow 7: Thread Participation Network

**Research Question**: Which users participate in the same threads?

```bash
# Build co-participation network
# (requires preprocessing to create user-thread adjacency)

# First, create a CSV with user_id and thread_id
# Then build network where users are connected if they post in same threads
python -m src.semantic.actor_cli \
  --input iron_march_posts.csv \
  --text-col text \
  --thread-col thread_id \
  --author-col user_id \
  --outdir output/thread_network
```

**Output**: Network showing which users interact in same discussion threads

---

## Research Questions You Can Answer

### User Behavior & Influence

1. **Who are the most influential users?**
   - Use: Actor network + centrality analysis
   - Data: Forum posts or direct messages

2. **Are there distinct user communities?**
   - Use: Community detection on user networks
   - Data: Forum posts

3. **Who talks to whom privately vs. publicly?**
   - Use: Compare DM network vs. forum reply network
   - Data: Messages + Posts

### Ideological Content

4. **What are the core ideological concepts?**
   - Use: Semantic network analysis
   - Data: Forum posts

5. **Which historical figures/events are central to the discourse?**
   - Use: Knowledge graph + entity centrality
   - Data: Forum posts

6. **How did extremist narratives evolve over time?**
   - Use: Temporal semantic networks
   - Data: Forum posts with timestamps

### Recruitment & Radicalization

7. **Do new users adopt the language of established members?**
   - Use: Temporal analysis + user timelines
   - Data: Posts + Member join dates

8. **Which topics/threads have highest engagement?**
   - Use: Thread network + participation metrics
   - Data: Thread metadata

9. **Are there recruitment patterns in DMs?**
   - Use: DM network + temporal analysis
   - Data: Direct messages

---

## Code Examples

### Example 1: Complete Analysis Pipeline

```bash
# 1. Export data from R (run in R)
# [See "Exporting from R" section above]

# 2. Build semantic network
python -m src.semantic.build_semantic_network \
  --input iron_march_posts.csv \
  --text-col text \
  --outdir output/semantic \
  --min-df 10 \
  --topk 20

# 3. Extract entities
python -m src.semantic.kg_cli \
  --input iron_march_posts.csv \
  --text-col text \
  --outdir output/entities \
  --model en_core_web_sm

# 4. Build user network
python -m src.semantic.actor_cli \
  --input iron_march_posts.csv \
  --text-col text \
  --thread-col thread_id \
  --author-col user_id \
  --outdir output/users

# 5. Detect communities
python -m src.semantic.community_cli \
  --edges output/users/edges.csv \
  --outdir output/communities

# 6. Analyze results in Gephi or Python
```

### Example 2: Temporal Evolution Analysis

```python
# Run this AFTER time_slice_cli generates monthly networks

import pandas as pd
import glob

# Load all monthly network files
monthly_files = sorted(glob.glob("output/temporal/*/nodes.csv"))

network_sizes = []
for file in monthly_files:
    month = file.split("/")[-2]
    nodes = pd.read_csv(file)
    network_sizes.append({
        "month": month,
        "num_nodes": len(nodes),
        "num_unique_terms": nodes['node'].nunique()
    })

df = pd.DataFrame(network_sizes)
print(df)

# Plot growth over time
import matplotlib.pyplot as plt
plt.plot(df['month'], df['num_nodes'])
plt.xlabel("Month")
plt.ylabel("Network Size (nodes)")
plt.title("Iron March Discourse Growth")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("network_growth.png")
```

### Example 3: User Influence Ranking

```python
import pandas as pd
import networkx as nx

# Load user network
edges = pd.read_csv("output/users/edges.csv")
G = nx.from_pandas_edgelist(edges, 'source', 'target', ['weight'])

# Calculate centrality metrics
betweenness = nx.betweenness_centrality(G, weight='weight')
eigenvector = nx.eigenvector_centrality(G, weight='weight')

# Get top 10 influential users
influence_df = pd.DataFrame({
    'user_id': list(betweenness.keys()),
    'betweenness': list(betweenness.values()),
    'eigenvector': list(eigenvector.values())
}).sort_values('betweenness', ascending=False)

print(influence_df.head(10))
```

---

## Safety Considerations

### Ethical Guidelines

1. **IRB Approval Required**
   - Iron March data involves extremist content
   - Human subjects research protocols apply
   - Consult your institution's IRB

2. **Data Security**
   - Store data on encrypted drives
   - Limit access to research team only
   - Do NOT redistribute raw data

3. **Researcher Safety**
   - Work with a team (don't read alone)
   - Take breaks from disturbing content
   - Access mental health resources if needed

4. **Privacy Protection**
   - Even though data is "leaked," protect user identities
   - Aggregate results when possible
   - Do not dox individuals (even extremists)
   - Use pseudonyms in publications

### Responsible Reporting

**DO:**
- Report aggregate patterns ("35% of users discussed...")
- Use anonymized network visualizations
- Focus on systemic patterns, not individuals
- Paraphrase quotes to prevent searchability

**DON'T:**
- Name specific users unless they're public figures
- Include personally identifiable information
- Publish raw message content
- Create searchable databases linking users to posts

### Platform-Specific Ethics

Iron March differs from Reddit/Twitter because:
- ❌ **Not currently active** (site defunct)
- ❌ **Users had expectation of privacy** (forum required registration)
- ❌ **Contains illegal content** (hate speech, potentially illegal material)
- ✅ **Public interest justification** (understanding extremism)

**Bottom line**: Treat this data MORE cautiously than public Twitter data.

---

## Data Quality Notes

### Known Issues

1. **HTML Artifacts**: Forum posts contain HTML tags
   - **Solution**: Use the R export code above which strips HTML

2. **Encoding Issues**: Some posts have special characters
   - **Solution**: Use UTF-8 encoding when reading CSVs

3. **Deleted Content**: Some posts/threads marked as deleted
   - **Solution**: Filter by `pdelete_time IS NULL` in posts

4. **Spam/Bot Accounts**: Some accounts are spam
   - **Solution**: Filter users with `member_group_id = 14` (banned)

### Recommended Filtering

```r
# In R, before exporting
posts_filtered <- posts %>%
  filter(
    is.na(pdelete_time),           # Not deleted
    queued == 1,                    # Approved/visible
    nchar(post) > 50,              # Substantive content
    !grepl("^<img", post)          # Not just images
  )
```

---

## Additional Resources

- **[Bellingcat Guide](https://www.bellingcat.com/resources/how-tos/2019/11/06/massive-white-supremacist-message-board-leak-how-to-access-and-interpret-the-data/)**: Understanding Iron March data
- **[ETHICS.md](ETHICS.md)**: Full ethics guide for social media research
- **[GETTING_STARTED.md](GETTING_STARTED.md)**: General toolkit introduction
- **[KG_FOR_SOCIAL_SCIENTISTS.md](KG_FOR_SOCIAL_SCIENTISTS.md)**: Knowledge graph research guide

---

## Quick Start Checklist

- [ ] Get IRB approval for extremist data research
- [ ] Install R and `ironmarch` package
- [ ] Export CSVs using code above
- [ ] Install our Python toolkit
- [ ] Choose analysis type (semantic, KG, temporal, etc.)
- [ ] Run analysis pipeline
- [ ] Visualize in Gephi or Python
- [ ] Interpret results responsibly
- [ ] Report findings with privacy protections

---

## Support

For questions about:
- **Iron March R package**: See [github.com/knapply/ironmarch](https://github.com/knapply/ironmarch)
- **Network analysis toolkit**: Open an issue on our GitHub
- **Ethics/IRB**: Consult your institutional review board

**Remember**: This is sensitive data about extremist movements. Handle with care, follow ethics guidelines, and prioritize safety.
