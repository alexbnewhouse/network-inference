# Knowledge Graph Enhancement: Quick Start Plan

**Goal**: Make KGs production-ready for social science research  
**Timeline**: 4-8 weeks  
**Date**: October 21, 2025

---

## 🎯 Top 3 Immediate Priorities

### 1. Temporal Knowledge Graphs (Week 1-2) ⭐⭐⭐
**Why**: Can't study narrative evolution without time  
**What**: Track entities over time, detect events, time-slice relationships  
**Effort**: 2 weeks  
**Impact**: HIGH - unlocks 80% of social science use cases

### 2. Sentiment & Stance (Week 2-3) ⭐⭐⭐
**Why**: Need to know HOW entities are discussed, not just that they're mentioned  
**What**: Add sentiment scores to entities and edges  
**Effort**: 1 week  
**Impact**: HIGH - enables polarization & framing research

### 3. User-Entity Bridge (Week 3-4) ⭐⭐
**Why**: Connect KG to social network analysis  
**What**: Bipartite network showing who discusses what  
**Effort**: 1 week  
**Impact**: MEDIUM-HIGH - bridges two research paradigms

---

## 🚀 Quick Wins (Do Today/Tomorrow)

### Quick Win #1: Add Basic Time Grouping (2 hours)
```python
# Add to kg_cli.py:
--group-by-time daily|weekly|monthly
--time-col created_at
```
Outputs separate KG files per time period.

### Quick Win #2: Entity Metadata (1 hour)
Add to nodes.csv:
- first_context: sample text where entity appears
- n_contexts: # of unique documents mentioning it

### Quick Win #3: Auto-generated Report (2 hours)
Create kg_report.md with:
- Entity type distributions
- Top entities
- Relation statistics
- Quality warnings

### Quick Win #4: Simple Sentiment (3 hours)
```bash
pip install vaderSentiment
# Add --add-sentiment flag
```
Output: entity_sentiment.csv with per-entity scores

**Total effort: 8 hours (1 day)**

---

## 📈 Current State vs. Desired State

### Current Capabilities ✅
- Extract entities (PERSON, ORG, GPE, etc.)
- Find co-occurrences within character windows
- Extract ~15 dependency relations per 1000 posts
- Output clean CSV files
- 77 entities from 2000 posts (4chan data)
- No NaN issues, no case duplicates, no self-loops

### Critical Gaps ❌
1. **No temporal analysis** - can't track narrative evolution
2. **No sentiment** - don't know how entities are framed
3. **Isolated from social networks** - can't connect "who" to "what"
4. **Limited relation types** - only verbs, no causation/stance/framing
5. **No quality metrics** - can't validate extraction
6. **Weak documentation** - not accessible to non-experts

---

## 📊 What This Enables for Research

### Before (Current State)
**Q**: "Who are the main actors in this dataset?"  
**A**: List of 77 entities with frequencies  
**Limitation**: Static snapshot, no evolution, no context

### After (Week 4)
**Q**: "How did discussion of China evolve during the Ukraine conflict?"  
**A**: 
- Timeline showing China mentions spike Feb 2022
- Sentiment shifts from neutral to negative
- New entity associations: China→Russia, China→Ukraine
- Top users driving the narrative
- Cross-community differences in framing

### After (Week 8)
**Q**: "What conspiracy theories link vaccines to other entities?"  
**A**:
- Entity chain patterns (Vaccine→Pfizer→Gates→Control)
- Sentiment analysis (increasingly negative over time)
- Bridge entities connecting different communities
- Temporal clustering of coordinated mentions
- User network analysis showing spreaders

---

## 🛠️ Technical Implementation

### File Structure
```
src/semantic/
├── kg_pipeline.py          # Core (already good)
├── kg_temporal.py          # NEW - Week 1-2
├── kg_sentiment.py         # NEW - Week 2-3
├── user_entity_network.py  # NEW - Week 3-4
├── kg_validator.py         # NEW - Week 5-6
└── relation_patterns.py    # NEW - Week 5-6

src/semantic/cli/
├── kg_cli.py              # Update with new features
├── kg_temporal_cli.py     # NEW
└── user_entity_cli.py     # NEW
```

### API Design
```python
# Current (simple)
kg = KnowledgeGraphPipeline()
kg.run(df, "output/")

# After (extensible)
from src.semantic.kg_pipeline import KnowledgeGraphPipeline
from src.semantic.kg_temporal import add_temporal_features
from src.semantic.kg_sentiment import add_sentiment

kg = KnowledgeGraphPipeline()
nodes, edges = kg.run(df, "output/")

# Add temporal
nodes, edges = add_temporal_features(nodes, edges, df, time_col='created_at')

# Add sentiment
nodes, edges = add_sentiment(nodes, edges, df)
```

---

## 📚 Documentation Plan

### Week 1-2: Core Guides
1. **KG_FOR_SOCIAL_SCIENTISTS.md**
   - What KGs are (vs semantic networks)
   - When to use what
   - How to interpret outputs
   - Common pitfalls

2. **TEMPORAL_KG_GUIDE.md**
   - Entity lifecycle analysis
   - Event detection
   - Time-sliced networks
   - Worked examples

### Week 3-4: Advanced Usage
3. **SENTIMENT_ANALYSIS_GUIDE.md**
   - Interpreting sentiment scores
   - Controversy detection
   - Polarization analysis
   - Limitations & biases

4. **INTEGRATED_ANALYSIS.md**
   - Combining KG + Actor Network + Semantic Network
   - Multi-level analysis
   - Research workflows

### Week 5-8: Methods & Validation
5. **VALIDATION_GUIDE.md**
   - Manual validation procedures
   - Quality metrics
   - When to trust results

6. **METHODS_PAPER.md**
   - Technical details for publication
   - Benchmarks
   - Limitations

---

## 🎓 Example Use Cases

### Use Case 1: Political Campaign Analysis
**Data**: Social media posts during election  
**Question**: How do candidates frame key issues?  
**Tools**: Temporal KG + Sentiment + User-Entity  
**Output**:
- Entity timeline (which topics emerge when)
- Sentiment evolution per entity
- User communities and their entity preferences
- Bridge topics that unite/divide communities

### Use Case 2: Misinformation Tracking
**Data**: Forum posts about vaccines  
**Question**: How does misinformation spread?  
**Tools**: Temporal KG + Entity Chains + User Network  
**Output**:
- Conspiracy narrative structures
- Temporal clustering of coordinated posts
- Bridge users spreading between communities
- Entity sentiment evolution

### Use Case 3: International Relations
**Data**: News articles about geopolitical events  
**Question**: How is China portrayed in US vs Chinese media?  
**Tools**: KG + Sentiment + Relation Extraction  
**Output**:
- Entity co-occurrence differences
- Sentiment differences per source
- Relation type differences (cooperative vs adversarial)
- Temporal evolution of framing

---

## ✅ Success Criteria

### Technical
- [ ] 50+ dependency relations per 1000 posts (currently: 15)
- [ ] <5% false positive rate on manual validation
- [ ] Process 10K documents in <10 minutes
- [ ] Temporal features work on real data
- [ ] Sentiment scores correlate with manual labels (r > 0.7)

### Usability
- [ ] Non-expert can run full analysis in <1 hour
- [ ] Documentation answers 90% of questions
- [ ] Example notebooks run without errors
- [ ] Clear error messages with solutions

### Research Impact
- [ ] Suitable for publication (methods section complete)
- [ ] Replication-friendly (documented, versioned)
- [ ] Answers real research questions (not just demos)

---

## 🤝 Community & Contribution

### After Week 4 (MVP)
- [ ] Create public demo notebook
- [ ] Write blog post about capabilities
- [ ] Submit to relevant academic mailing lists
- [ ] Create GitHub issues for community features

### After Week 8 (Full Release)
- [ ] Submit methods paper
- [ ] Create tutorial videos
- [ ] Workshop at relevant conferences
- [ ] Seek collaborations for domain-specific validation

---

## 📞 Get Started

### Step 1: Run Current KG (5 min)
```bash
python -m src.semantic.kg_cli \
  --input pol_archive_0.csv \
  --outdir output/kg_test \
  --min-freq 5 \
  --max-rows 2000
```

### Step 2: Identify Your Use Case
- [ ] Temporal analysis? → Priority: Temporal KG
- [ ] Polarization/framing? → Priority: Sentiment
- [ ] User behavior? → Priority: User-Entity Network
- [ ] All of the above? → Follow roadmap in order

### Step 3: Quick Wins (Today)
Pick 1-2 from the Quick Wins section and implement.

### Step 4: Week 1 Kickoff
Start implementing Temporal KG features.

---

**Questions? See**: KG_SOCIAL_SCIENCE_ROADMAP.md (full plan)  
**Status**: Ready to implement  
**Next Update**: After Week 1 (Temporal KG)
