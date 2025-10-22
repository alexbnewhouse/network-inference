# Training Materials Summary

**Complete onboarding package for learning network analysis**

---

## 📦 What Was Created

### 1. ONBOARDING_TRAINING_GUIDE.md
**Comprehensive 60-90 minute hands-on tutorial**

**Structure**:
- **Part 1 (30 min)**: Anonymized Analysis
  - Exercise 1.1: Semantic Network Analysis
  - Exercise 1.2: Knowledge Graph Entity Extraction
  - Exercise 1.3: Temporal Analysis
  
- **Part 2 (30 min)**: Actor-Based Analysis
  - Exercise 2.1: Actor Networks (who talks to whom)
  - Exercise 2.2: User-Entity Networks (communities)
  - Exercise 2.3: Influence and Sentiment
  
- **Part 3 (15 min)**: Comparing Approaches
  - When to use anonymized vs actor-based
  - Decision matrix for research questions
  
- **Part 4 (15 min)**: Ethical Best Practices
  - Privacy protection checklist
  - Anonymization techniques (hashing, k-anonymity)
  - Reporting guidelines

**Features**:
- Step-by-step commands for every exercise
- Expected outputs and findings
- Research questions each method can answer
- Ethical notes for each approach
- Quick reference command cheatsheet

---

### 2. Training Dataset (10,000 posts)

**Files**:
- `examples/training_data.csv` (1.4 MB) - Main dataset
- `examples/training_data_users.csv` (5 KB) - User metadata

**Characteristics**:
- **10,000 posts** from **250 users**
- **3,469 discussion threads**
- **3-month timeline** (Jan-Mar 2024)
- **5 topic categories**:
  - Geopolitics (Russia, China, Ukraine, NATO)
  - Tech (AI, crypto, cybersecurity)
  - Politics (elections, healthcare, immigration)
  - Culture (celebrities, sports, entertainment)
  - Economics (inflation, markets, trade)

**User Types**:
| Type | % | Posts/User | Example |
|------|---|------------|---------|
| Casual | 40% | 5-20 | Occasional lurker |
| Active | 35% | 20-60 | Regular participant |
| Power | 20% | 60-150 | Daily contributor |
| Influencer | 5% | 150-300 | Community leader |

**Privacy Features**:
- Includes `user_id_hashed` (SHA256) for privacy training
- Completely synthetic (no real users/posts)
- Safe for public sharing and teaching

**Entities Present**:
- **People**: Biden, Trump, Putin, Musk, celebrities
- **Places**: Russia, China, Ukraine, US states
- **Organizations**: NATO, UN, Google, Tesla
- **Events**: Elections, Olympics, conflicts

---

### 3. Dataset Generator

**File**: `examples/generate_training_data.py`

**What it does**:
- Generates configurable synthetic social media data
- Realistic posting patterns and temporal clustering
- Customizable topics and entity pools
- User type distributions
- Thread structure simulation

**Configuration**:
```python
NUM_USERS = 250          # Number of users
NUM_POSTS = 10000        # Number of posts
START_DATE = 2024-01-01  # Timeline start
END_DATE = 2024-03-31    # Timeline end
```

**Usage**:
```bash
python3 examples/generate_training_data.py
```

---

### 4. Documentation

**File**: `examples/TRAINING_DATA_README.md`

**Contents**:
- Dataset specifications and column descriptions
- User type distribution details
- Topic and entity breakdowns
- Thread structure statistics
- Temporal patterns explanation
- Privacy features (hashing)
- Usage examples for all analysis types
- Regeneration instructions

---

## 🎯 Learning Objectives

After completing the training, users can:

1. ✅ Build semantic networks to map discourse
2. ✅ Extract knowledge graphs for entity analysis
3. ✅ Conduct actor network analysis with privacy
4. ✅ Perform temporal analysis to track changes
5. ✅ Apply sentiment and stance detection
6. ✅ Decide when to anonymize vs. identify users
7. ✅ Apply ethical best practices
8. ✅ Use hashing, k-anonymity, and aggregation
9. ✅ Report findings responsibly

---

## 📊 Training Data Statistics

**Content Diversity**:
- **Threads**: 3,469 (avg 2.9 posts/thread)
- **Reply rate**: 65.3%
- **Date range**: 91 days
- **Topics**: 5 categories evenly distributed
- **Entities**: 50+ people, places, organizations

**Realistic Features**:
- Power law user activity distribution
- Natural thread size distribution
- Temporal event clustering (20% of posts)
- Multi-topic user participation
- Entity co-occurrence patterns

**File Sizes**:
- Main CSV: 1.4 MB (manageable for training)
- User metadata: 5 KB
- Total: ~1.4 MB (easily shared)

---

## 🔬 Research Applications

The training demonstrates:

**Anonymized Methods**:
- Discourse pattern analysis
- Topic evolution tracking
- Entity co-occurrence networks
- Temporal trend detection
- Aggregate sentiment analysis

**Actor-Based Methods**:
- User influence measurement
- Community detection
- Echo chamber analysis
- Radicalization pathway tracking
- Cross-platform user fingerprinting

**Ethical Practices**:
- When IRB approval is needed
- How to hash user identifiers
- K-anonymity filtering
- Aggregate reporting techniques
- Paraphrasing for privacy

---

## 🚀 Quick Start

### Generate Data
```bash
python3 examples/generate_training_data.py
```

### Start Training
```bash
# Follow ONBOARDING_TRAINING_GUIDE.md
# Exercise 1.1: First semantic network
python3 -m src.semantic.build_semantic_network \
  --input examples/training_data.csv \
  --text-col text \
  --outdir output/training_semantic
```

### Complete Curriculum
1. Read ONBOARDING_TRAINING_GUIDE.md (10 min)
2. Generate training data (1 min)
3. Part 1: Anonymized analysis (30 min)
4. Part 2: Actor-based analysis (30 min)
5. Part 3: Compare approaches (15 min)
6. Part 4: Ethics training (15 min)
7. Bonus: Real-world scenario (15 min)

**Total time**: 90-120 minutes

---

## 📚 Related Documentation

- **GETTING_STARTED.md**: General toolkit introduction
- **ETHICS.md**: Comprehensive ethics guide
- **DATA_REQUIREMENTS.md**: CSV format specifications
- **IRON_MARCH_GUIDE.md**: Real-world application example
- **KG_FOR_SOCIAL_SCIENTISTS.md**: Research applications

---

## ✅ What This Enables

**For Instructors**:
- Turnkey training module for network analysis
- Synthetic data safe for classroom use
- Hands-on exercises with expected outputs
- Ethics integration throughout
- Scalable (modify generator for larger datasets)

**For Self-Learners**:
- Complete guided tutorial
- Safe practice environment
- No IRB needed (synthetic data)
- Concrete examples for each method
- Decision frameworks for real research

**For Researchers**:
- Practice environment before real data
- Method comparison on same dataset
- Privacy technique validation
- Workflow development
- Team training standardization

---

## 🎓 Success Criteria

Students completing this training can:

- [ ] Generate and understand the training dataset
- [ ] Build semantic networks from text
- [ ] Extract and analyze knowledge graphs
- [ ] Conduct temporal analysis with event detection
- [ ] Perform actor network analysis with hashing
- [ ] Detect user communities
- [ ] Apply appropriate anonymization techniques
- [ ] Choose methods based on research questions
- [ ] Report findings ethically and responsibly
- [ ] Apply skills to real research projects

---

## 📦 Files Included

1. **ONBOARDING_TRAINING_GUIDE.md** - Main tutorial
2. **examples/generate_training_data.py** - Data generator
3. **examples/training_data.csv** - 10K post dataset
4. **examples/training_data_users.csv** - User metadata
5. **examples/TRAINING_DATA_README.md** - Dataset documentation

**Total**: ~1.5 MB, fully self-contained

---

## 🔄 Customization

Modify `generate_training_data.py` to create:
- **Larger datasets**: Change `NUM_POSTS` to 50K, 100K
- **Longer timelines**: Extend `END_DATE` to 1 year, 5 years
- **More users**: Increase `NUM_USERS` for denser networks
- **Different topics**: Edit `TOPICS` dictionary
- **Custom entities**: Modify `ENTITIES` pools
- **Platform-specific**: Adjust language for Twitter, Reddit, forums

---

## 🎉 Impact

This training package:

✅ **Reduces barrier to entry** for network analysis  
✅ **Teaches ethical practices** from day one  
✅ **Provides safe practice environment** (synthetic data)  
✅ **Demonstrates multiple methods** on same dataset  
✅ **Includes real-world application** (Iron March scenario)  
✅ **Enables self-paced learning** (60-90 min complete)  
✅ **Supports classroom teaching** (ready-to-use exercises)  

**Ready for immediate use in courses, workshops, and self-study!** 📖🔬🎓
