# Training Dataset Documentation

**Realistic social media dataset for learning network analysis**

---

## Overview

This synthetic dataset simulates 3 months of social media activity with realistic patterns:

- **10,000 posts** from 250 users
- **3,469 discussion threads**
- **5 topic categories**: geopolitics, tech, politics, culture, economics
- **Temporal patterns**: Jan 1 - Mar 31, 2024 with clustering around events
- **User diversity**: Casual posters to power users

---

## Files

### training_data.csv (10,000 rows)

Main dataset with columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `post_id` | string | Unique post identifier | "post_000001" |
| `thread_id` | string | Discussion thread ID | "thread_00042" |
| `user_id` | string | Original user ID | "user_0123" |
| `user_id_hashed` | string | SHA256 hashed user ID (16 chars) | "a3f5c89d2e1b7f04" |
| `created_at` | datetime | Post timestamp | "2024-01-15 14:23:00" |
| `text` | string | Post content | "Russia and China..." |
| `is_reply` | boolean | Whether post is a reply | true/false |
| `topic_category` | string | Assigned topic | "geopolitics" |

### training_data_users.csv (249 rows)

User metadata:

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | User identifier |
| `type` | string | User type (casual/active/power/influencer) |
| `posts` | int | Number of posts by user |

---

## User Types & Distribution

| Type | % of Users | Posts per User | Total Posts |
|------|-----------|----------------|-------------|
| Casual | 40% | 5-20 | ~1,200 |
| Active | 35% | 20-60 | ~3,500 |
| Power | 20% | 60-150 | ~4,000 |
| Influencer | 5% | 150-300 | ~1,300 |

---

## Topic Distribution

Posts are distributed across 5 realistic categories:

### Geopolitics (~20%)
- Russia-China alliance, Ukraine conflict
- NATO expansion, Middle East tensions
- UN Security Council actions

### Tech (~20%)
- AI ethics, cryptocurrency crashes
- Cybersecurity breaches, data privacy
- Social media policy changes

### Politics (~20%)
- US elections, immigration policy
- Supreme Court rulings, healthcare reform
- Gun control, police reform

### Culture (~20%)
- Celebrity controversies, cancel culture
- Sports championships, entertainment
- Art exhibitions, book bans

### Economics (~20%)
- Stock market volatility, inflation
- Federal Reserve policy, housing market
- Trade wars, unemployment

---

## Thread Structure

Posts are organized into realistic conversation threads:

- **Thread size distribution**:
  - 30% single posts (no replies)
  - 25% two-post threads
  - 20% three-post threads
  - 15% four-post threads
  - 10% larger threads (5-20 posts)

- **Total threads**: 3,469
- **Average thread size**: 2.9 posts
- **Reply rate**: 65.3%

---

## Temporal Patterns

**Date range**: January 1 - March 31, 2024 (91 days)

**Posting patterns**:
- Chronologically ordered (realistic timeline)
- 20% of posts cluster around "hot" events (±2 days)
- 80% distributed randomly across timespan
- Enables temporal analysis and event detection

**Weekly breakdown**:
- ~770 posts per week
- ~110 posts per day average
- Natural variation in activity

---

## Entities Present

The dataset includes mentions of:

### People (PERSON)
- Politicians: Biden, Trump, Harris, DeSantis
- Tech leaders: Elon Musk
- Celebrities: Taylor Swift, Kanye West, Joe Rogan

### Places (GPE - Geopolitical Entities)
- Countries: Russia, China, Ukraine, US, Iran, North Korea
- Regions: Eastern Europe, Middle East, Asia Pacific
- US States: Texas, California, Florida, New York

### Organizations (ORG)
- International: NATO, UN, EU
- Tech: Google, Meta, Apple, Microsoft, Tesla, Amazon, OpenAI
- Sports: Lakers, Yankees, Patriots

### Events
- Ukraine conflict, Olympics, elections
- Supreme Court rulings, protests

---

## Privacy Features

### User ID Hashing

All posts include `user_id_hashed` for anonymization:

```python
import hashlib
user_id_hashed = hashlib.sha256(user_id.encode()).hexdigest()[:16]
```

**Properties**:
- One-way hash (cannot reverse)
- Consistent (same user → same hash)
- 16-character hexadecimal
- Enables user tracking without exposing identities

### Synthetic Data

All data is **completely synthetic**:
- ✅ No real users
- ✅ No real posts
- ✅ Safe for training and teaching
- ✅ Can be shared publicly

---

## Usage Examples

### Load Dataset

```python
import pandas as pd

# Load posts
df = pd.read_csv('examples/training_data.csv')
print(f"Loaded {len(df)} posts from {df['user_id'].nunique()} users")

# Load user metadata
users = pd.read_csv('examples/training_data_users.csv')
print(f"User types: {users['type'].value_counts()}")
```

### Anonymized Analysis

```bash
# Semantic network (no user IDs needed)
python3 -m src.semantic.build_semantic_network \
  --input examples/training_data.csv \
  --text-col text \
  --outdir output/training_semantic

# Knowledge graph
python3 -m src.semantic.kg_cli \
  --input examples/training_data.csv \
  --text-col text \
  --outdir output/training_kg
```

### Actor-Based Analysis

```bash
# Use hashed IDs for privacy
python3 -m src.semantic.actor_cli \
  --input examples/training_data.csv \
  --thread-col thread_id \
  --author-col user_id_hashed \
  --outdir output/training_actors
```

### Temporal Analysis

```bash
# Weekly knowledge graphs
python3 -m src.semantic.kg_cli \
  --input examples/training_data.csv \
  --text-col text \
  --time-col created_at \
  --group-by-time weekly \
  --outdir output/training_temporal
```

---

## Learning Objectives

This dataset is designed to teach:

1. **Anonymized analysis**: Semantic networks, knowledge graphs, temporal trends
2. **Actor analysis**: User networks, communities, influence
3. **Ethical practices**: When to hash IDs, aggregate reporting
4. **Temporal analysis**: Event detection, trend tracking
5. **Multi-method research**: Combining different network types

---

## Regenerating Data

To create fresh training data with different patterns:

```bash
python3 examples/generate_training_data.py
```

**Configuration variables** in `generate_training_data.py`:
- `NUM_USERS` = 250
- `NUM_POSTS` = 10000
- `START_DATE` = 2024-01-01
- `END_DATE` = 2024-03-31
- `USER_TYPES` = Distribution of user posting patterns
- `TOPICS` = Content templates

Modify these to create custom datasets.

---

## Statistics

**File sizes**:
- `training_data.csv`: ~1.4 MB
- `training_data_users.csv`: ~6 KB

**Content diversity**:
- Unique threads: 3,469
- Unique users: 249
- Topic categories: 5
- Date range: 91 days
- Entities: 50+ people, places, organizations

**Realistic features**:
- Power law distribution of user activity
- Thread size follows natural patterns
- Temporal clustering around events
- Topic mixing (users post about multiple topics)
- Entity co-occurrence patterns

---

## Research Applications

Use this dataset to practice:

- **Discourse analysis**: What topics co-occur?
- **Community detection**: Do users cluster by topic?
- **Influence measurement**: Who are power users?
- **Temporal evolution**: How did topics change over 3 months?
- **Sentiment analysis**: What's the attitude toward entities?
- **Echo chambers**: Do communities have distinct views?

---

## Ethical Training

This dataset demonstrates:

1. **When to anonymize**: Compare `user_id` vs `user_id_hashed` usage
2. **Aggregate reporting**: Report user types, not individuals
3. **K-anonymity**: Filter communities with <5 users
4. **Paraphrasing**: Don't quote synthetic text verbatim in publications

---

## Next Steps

1. Complete [ONBOARDING_TRAINING_GUIDE.md](ONBOARDING_TRAINING_GUIDE.md)
2. Run all exercises with this data
3. Try modifying parameters in `generate_training_data.py`
4. Apply learned techniques to real research data

---

**This is synthetic data for training purposes only. No real users or content.**
