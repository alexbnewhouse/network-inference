# Data Requirements

## Input Data Format

All analysis tools accept **CSV files** with text data. This guide explains column requirements and naming conventions.

---

## Required Columns

### Minimum Requirements

**Every CSV must have ONE text column** containing the data to analyze:

| Column Name | Type | Description | Example |
|------------|------|-------------|---------|
| `text` or `body` | string | Text content to analyze | "This is a post about politics" |

**That's it!** The toolkit works with just a text column.

---

## Optional Columns

Additional columns unlock more features:

### For Temporal Analysis

| Column | Type | Description | Format Example |
|--------|------|-------------|----------------|
| `created_at` or `timestamp` | datetime/string | When text was created | "2024-01-15 14:30:00" or "2024-01-15" |

**Enables**: Time-sliced networks, event detection, trajectory analysis

### For Actor Networks

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `user_id` or `author` | string | Author identifier | "user_12345" or "john_doe" |
| `subject` or `thread_id` | string | Thread/conversation ID | "thread_456" or "Post Title" |

**Enables**: User-entity networks, community detection, reply networks

### For Metadata

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `board` or `subreddit` | string | Source community | "/pol/" or "r/politics" |
| `index` or `post_id` | int/string | Unique post ID | 1234567 or "abc123" |

**Enables**: Cross-community comparison, post tracking

---

## Column Naming Conventions

The toolkit accepts flexible column names. Here are the patterns it recognizes:

### Text Content
- `text` (recommended)
- `body`
- `content`
- `message`

### Timestamps
- `created_at` (recommended)
- `timestamp`
- `date`
- `datetime`

### User Identifiers
- `user_id` (recommended)
- `author`
- `username`
- `tripcode`

### Thread/Conversation
- `subject` (recommended for 4chan-style boards)
- `thread_id`
- `conversation_id`

---

## Example CSV Structures

### Minimal (Text Only)
```csv
text
"This is my first post"
"Here is another post about networks"
"Social science research is fascinating"
```

**Works with**: Semantic networks, transformer networks, knowledge graphs

### With Timestamps
```csv
text,created_at
"Russia announced new policy","2024-01-15 10:30:00"
"China responded to the announcement","2024-01-15 14:25:00"
"Analysis of geopolitical tensions","2024-01-16 09:00:00"
```

**Works with**: Everything + temporal analysis, event detection

### Full Featured
```csv
text,created_at,user_id,board,subject
"Post about topic A","2024-01-15 10:00:00","user_123","/pol/","Thread 1"
"Reply in thread","2024-01-15 10:05:00","user_456","/pol/","Thread 1"
"New topic discussion","2024-01-15 11:00:00","user_789","/int/","Thread 2"
```

**Works with**: All features (temporal, user networks, communities, cross-board comparison)

---

## Common Data Sources

### 4chan/8chan
```csv
text,created_at,user_id,board,subject
```
- Use `body` for text content
- `tripcode` for user identification (can be null/anonymous)
- `subject` for thread grouping
- `board` for community (e.g., "/pol/", "/int/")

### Reddit
```csv
text,created_at,author,subreddit,post_id
```
- `body` or `selftext` for text
- `author` for username
- `subreddit` for community
- `created_utc` often needs conversion to readable format

### Twitter/X
```csv
text,created_at,user_id,tweet_id
```
- `text` or `full_text` for content
- `user_id` or `author_id` for user
- `created_at` for timestamp
- `id` for tweet identifier

### Custom/Survey Data
```csv
text,respondent_id,timestamp,group
```
- `text` for open-ended responses
- Any ID column for user tracking
- `timestamp` for temporal analysis
- `group` for comparison (treatment/control, etc.)

---

## Privacy & Anonymization

### Before Analysis

**If data contains personally identifiable information (PII)**:

1. **Hash user IDs**:
   ```python
   import hashlib
   df['user_id'] = df['user_id'].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16])
   ```

2. **Remove metadata**: Drop columns like email, IP addresses, real names

3. **Aggregate timestamps**: Round to hour/day instead of exact time
   ```python
   df['created_at'] = pd.to_datetime(df['created_at']).dt.floor('H')
   ```

### For Public Data

If analyzing public social media (4chan, public Twitter):
- User IDs are often already anonymized (hashes, tripcodes)
- Still avoid reporting individual users in publications
- Aggregate patterns, not individual behaviors

**See [ETHICS.md](ETHICS.md) for comprehensive privacy guidelines.**

---

## Data Quality Tips

### Text Cleaning

**The toolkit handles most cleaning automatically**, but you can preprocess:

```python
import pandas as pd

# Load data
df = pd.read_csv("data.csv")

# Optional: Remove very short posts
df = df[df['text'].str.len() > 20]

# Optional: Remove duplicates
df = df.drop_duplicates(subset=['text'])

# Optional: Handle missing values
df = df.dropna(subset=['text'])

# Save cleaned data
df.to_csv("data_cleaned.csv", index=False)
```

### Size Recommendations

| Dataset Size | Processing Time | Recommended Workflow |
|-------------|-----------------|---------------------|
| <1K posts | <1 minute | Full analysis, all features |
| 1K-10K | 1-10 minutes | Standard workflow |
| 10K-100K | 10-60 minutes | Use GPU if available |
| 100K-1M | 1-8 hours | Time-sliced processing |
| >1M | Consider sampling | Process monthly chunks |

### Testing with Small Samples

**Always test with small samples first:**

```bash
# Test with first 1000 rows
python -m src.semantic.build_semantic_network \
  --input data.csv \
  --outdir output/test \
  --max-rows 1000
```

Once confirmed working, run on full dataset.

---

## Specifying Columns in Commands

### Explicit Column Names

Most CLIs let you specify column names:

```bash
python -m src.semantic.kg_cli \
  --input data.csv \
  --text-col body \           # Use 'body' instead of 'text'
  --time-col created_at \     # Specify timestamp column
  --outdir output/kg
```

### Actor Networks

```bash
python -m src.semantic.actor_cli \
  --input data.csv \
  --text-col body \
  --thread-col subject \      # Thread grouping
  --post-col index \          # Post IDs
  --author-tripcode-col tripcode \  # User identifier
  --outdir output/actors
```

### User-Entity Networks

```bash
python -m src.semantic.kg_user_entity_network_cli \
  --kg-dir output/kg \
  --data data.csv \
  --user-col user_id \        # User identifier column
  --text-col text \
  --stats
```

---

## Quick Reference

### ✅ What Works Out-of-the-Box

```csv
text
"Your posts here"
```

### ✅ Recommended Structure

```csv
text,created_at,user_id
"Post content","2024-01-15",user_123
```

### ❌ Won't Work

- Empty text column
- Binary data (images, PDFs)
- Non-text data types
- Files other than CSV (convert first)

---

## Troubleshooting

### "Column 'text' not found"
**Solution**: Specify your text column name:
```bash
--text-col body  # or whatever your column is named
```

### "Could not parse timestamp"
**Solution**: Convert timestamps to ISO format (YYYY-MM-DD HH:MM:SS):
```python
df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
```

### "Out of memory"
**Solution**: Process in chunks:
```bash
--max-rows 10000  # Process first 10K rows
```

Or time-slice:
```bash
--group-by-time monthly  # Process one month at a time
```

---

## Need Help?

- **Getting Started**: See [GETTING_STARTED.md](GETTING_STARTED.md)
- **Examples**: See `examples/` directory
- **Advanced Features**: See [KG_FOR_SOCIAL_SCIENTISTS.md](KG_FOR_SOCIAL_SCIENTISTS.md)
- **Ethics**: See [ETHICS.md](ETHICS.md)
