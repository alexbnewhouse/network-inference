# Knowledge Graph Roadmap for Social Science Research

**Date**: October 21, 2025  
**Status**: Planning Document  
**Goal**: Make KGs production-ready for social science research on text data

---

## 📊 Current State Audit

### ✅ What Works Well

**Core Infrastructure:**
- ✅ Solid KG extraction pipeline (`kg_pipeline.py`)
- ✅ Named Entity Recognition with spaCy
- ✅ Entity normalization and filtering
- ✅ Co-occurrence detection (character windows)
- ✅ Dependency-based relation extraction
- ✅ Multiple output formats (CSV, graph formats)
- ✅ Good integration with existing semantic network tools

**Recent Improvements (Oct 2025):**
- ✅ NaN/null handling
- ✅ Case consistency (no duplicates)
- ✅ Self-loop prevention
- ✅ False positive filtering (AI, CEO, etc.)
- ✅ Enhanced entity type filtering

**Quality on Real Data (4chan test):**
- 77 entities from 2000 posts (min_freq=5)
- 311 edges (281 co-occurrence + 30 dependency)
- 0 self-loops, 0 case duplicates
- Avg degree: 8.08 (good connectivity)

### ❌ Critical Gaps for Social Science

**1. No Temporal Analysis** 🕐
- Can't track how narratives evolve over time
- No entity lifecycle tracking (emergence, peak, decline)
- Missing temporal edge weights (relationship strength over time)
- No event detection or timeline construction

**2. Limited Semantic Depth** 🧠
- Only extracts "co-occurrence" and verb-based relations
- Missing: causation, sentiment, stance, framing
- No entity disambiguation (multiple "America" entities)
- No entity linking to knowledge bases (Wikidata, etc.)

**3. No Social Network Integration** 👥
- KG and actor networks are completely separate
- Can't answer: "Who discusses which entities?"
- Missing: user-entity bipartite networks
- No community-entity analysis

**4. Insufficient Validation Tools** ✓
- No inter-rater reliability metrics
- No entity extraction quality scores
- Missing: false positive/negative analysis
- No benchmarking against gold standard datasets

**5. Limited Scale & Performance** ⚡
- Dependency parsing doesn't scale (30 relations from 2000 posts)
- No incremental/streaming processing
- Memory-intensive for >100K documents
- No distributed processing support

**6. Missing Analysis Features** 📈
- No centrality metrics tailored to KGs
- No narrative/conspiracy detection
- No controversy scoring
- No entity polarization analysis
- No cross-community bridging analysis

**7. Weak Documentation for Social Scientists** 📚
- No worked examples with research questions
- Missing: "how to interpret PPMI for entities"
- No replication guides
- Limited troubleshooting for domain-specific data

---

## 🎯 Strategic Priorities

### Tier 1: Must-Have for Social Science (1-2 weeks)

#### 1.1 Temporal Knowledge Graphs ⭐⭐⭐
**Impact**: HIGH - Essential for studying narrative evolution, events, campaigns

**Implementation:**
```python
class TemporalKG:
    """Extend KG with temporal features."""
    
    def extract_temporal_entities(self, df):
        """Track entity first/last appearance, frequency over time."""
        # Output: entity_timeline.csv with columns:
        # [entity, type, first_seen, last_seen, peak_time, 
        #  total_mentions, unique_days, lifespan_days]
    
    def build_temporal_edges(self, df, time_window='1D'):
        """Build time-sliced KG edges."""
        # Output: edges_temporal.csv with columns:
        # [source, target, relation_type, time_period, weight]
    
    def detect_entity_events(self, df, threshold=2.0):
        """Detect sudden spikes in entity mentions (events)."""
        # Output: events.csv with columns:
        # [entity, event_date, baseline_freq, spike_freq, 
        #  z_score, context_sample]
```

**Files to create:**
- `src/semantic/kg_temporal.py` - Core temporal KG class
- `src/semantic/kg_temporal_cli.py` - CLI interface
- `examples/temporal_kg_example.ipynb` - Tutorial notebook

**Tests needed:**
- Entity lifecycle tracking
- Time-sliced edge weights
- Event detection accuracy

---

#### 1.2 Entity Sentiment & Stance ⭐⭐⭐
**Impact**: HIGH - Critical for studying framing, polarization, controversy

**Implementation:**
```python
class SentimentKG:
    """Add sentiment/stance to KG edges."""
    
    def compute_entity_sentiment(self, df):
        """Sentiment of contexts mentioning entity."""
        # Output: entity_sentiment.csv
        # [entity, avg_sentiment, sentiment_std, 
        #  controversy_score, n_contexts]
    
    def compute_edge_sentiment(self, df, edges):
        """Sentiment of contexts containing both entities."""
        # Output: edges_with_sentiment.csv
        # [source, target, relation_type, weight,
        #  avg_sentiment, sentiment_variance]
    
    def detect_controversial_entities(self, threshold=0.3):
        """Find entities with high sentiment variance."""
        # Output: controversial_entities.csv
```

**Method options:**
1. **Simple**: TextBlob or VADER (lexicon-based)
2. **Better**: Fine-tuned transformers (twitter-roberta-base-sentiment)
3. **Best**: Domain-specific fine-tuning on labeled data

**Files to create:**
- `src/semantic/kg_sentiment.py`
- Integration into main `kg_pipeline.py`

---

#### 1.3 User-Entity Bipartite Networks ⭐⭐
**Impact**: MEDIUM-HIGH - Bridge between KG and social network analysis

**Implementation:**
```python
class UserEntityNetwork:
    """Build bipartite networks: users ↔ entities."""
    
    def build_bipartite(self, df):
        """Create user-entity network."""
        # Output: user_entity_edges.csv
        # [user_id, entity, entity_type, n_mentions, 
        #  first_mention, last_mention]
    
    def project_to_user_network(self):
        """Users connected by shared entities."""
        # Output: user_user_edges.csv
        # [source_user, target_user, shared_entities, 
        #  jaccard_similarity]
    
    def project_to_entity_network(self):
        """Entities connected by shared users."""
        # Similar output with entity pairs
```

**Use cases:**
- Who discusses China? (user → entity)
- Which users discuss similar entities? (user projection)
- Which entities are discussed by same communities? (entity projection)

**Files to create:**
- `src/semantic/user_entity_network.py`
- Integration with existing `actor_network.py`

---

### Tier 2: Important Enhancements (2-4 weeks)

#### 2.1 Advanced Relation Extraction ⭐⭐
**Current problem**: Only 30 dependency relations from 2000 posts (0.015 per post)

**Solutions:**
1. **Rule-based patterns** (fast, reliable)
   ```python
   RELATION_PATTERNS = {
       'causal': ['cause', 'lead to', 'result in', 'because of'],
       'opposes': ['against', 'opposes', 'fights', 'versus'],
       'supports': ['supports', 'backs', 'endorses', 'favors'],
       'owns': ['owns', 'possesses', 'has', 'controls'],
       'member': ['member of', 'part of', 'belongs to'],
   }
   ```

2. **OpenIE integration** (more relations, noisier)
   - Use Stanford OpenIE or AllenNLP OpenIE
   - Extract (subject, predicate, object) triples
   - Filter by entity types

3. **Transformer-based relation extraction** (best quality, slowest)
   - Fine-tune BERT for relation classification
   - Use pre-trained models like REBEL

**Priority**: Start with #1 (rule-based), add #2 if needed

---

#### 2.2 Entity Disambiguation & Linking ⭐⭐
**Current problem**: "America" (country) vs "America" (continent) vs "Captain America"

**Implementation:**
```python
class EntityLinker:
    """Link entities to canonical IDs."""
    
    def disambiguate_entities(self, entities, contexts):
        """Resolve ambiguous entities using context."""
        # Use entity types + context to disambiguate
    
    def link_to_wikidata(self, entities):
        """Link entities to Wikidata QIDs."""
        # Optional: requires API calls or local DB
```

**Simple approach:**
- Use entity type + context to create canonical IDs
- "America" (GPE, context: "president") → "United States"
- "America" (LOC, context: "continent") → "Americas"

**Advanced approach:**
- Entity linking to Wikidata/Wikipedia
- Maintains QID for each entity
- Enables cross-dataset entity matching

---

#### 2.3 Narrative & Conspiracy Detection ⭐⭐
**Use case**: Detect coordinated narratives, conspiracy theories, information operations

**Implementation:**
```python
class NarrativeDetector:
    """Detect narrative structures in KGs."""
    
    def find_entity_chains(self, min_length=3):
        """Find multi-hop entity chains."""
        # "CIA → Venezuela → Oil → War"
        # Output: chains.csv
    
    def detect_recurring_patterns(self, min_support=5):
        """Find frequent subgraphs (motifs)."""
        # e.g., "X opposes Y, Y supports Z" pattern
    
    def find_bridge_entities(self):
        """Entities connecting different communities."""
        # High betweenness centrality entities
```

**Applications:**
- Conspiracy theory detection (specific chain patterns)
- Information operation detection (coordinated entity co-mentions)
- Narrative framing analysis (how entities are connected)

---

#### 2.4 Quality Metrics & Validation ⭐
**Critical for publication-quality research**

**Implementation:**
```python
class KGValidator:
    """Quality metrics for KG extraction."""
    
    def compute_precision_recall(self, gold_standard):
        """Compare against hand-labeled data."""
    
    def compute_entity_confidence(self):
        """Per-entity extraction confidence scores."""
        # Based on: frequency, context clarity, NER confidence
    
    def detect_extraction_errors(self):
        """Find likely false positives."""
        # Unusual entity types, very rare entities, etc.
    
    def generate_validation_sample(self, n=100):
        """Sample entities for manual validation."""
        # Stratified by frequency, type, confidence
```

**Outputs:**
- `kg_quality_report.json` - Overall metrics
- `validation_sample.csv` - Entities to manually check
- `confidence_scores.csv` - Per-entity confidence

---

### Tier 3: Advanced Features (4-8 weeks)

#### 3.1 Multilingual Support
- Currently English-only
- Add spaCy models for other languages
- Cross-lingual entity matching

#### 3.2 Interactive Visualization
- Web-based KG explorer
- Filter by time, entity type, sentiment
- Highlight narrative chains

#### 3.3 Integration with Existing Tools
- Export to Gephi with proper attributes
- Export to Neo4j for graph queries
- Integration with network analysis packages (igraph, networkx)

#### 3.4 Scalability Improvements
- Streaming/incremental processing
- Distributed processing (Dask/Spark)
- Database backend for large KGs

---

## 🛠️ Implementation Plan

### Week 1-2: Temporal KG (Tier 1.1)
**Goal**: Track entities and relationships over time

**Tasks:**
1. Create `TemporalKG` class extending `KnowledgeGraphPipeline`
2. Implement entity timeline tracking
3. Implement time-sliced edge extraction
4. Add event detection (z-score spike detection)
5. Create CLI: `kg_temporal_cli.py`
6. Write tutorial notebook
7. Test on pol_archive_0.csv with date ranges

**Deliverables:**
- `src/semantic/kg_temporal.py` (200 lines)
- `src/semantic/kg_temporal_cli.py` (100 lines)
- `examples/temporal_kg_tutorial.ipynb`
- Tests in `tests/test_kg_temporal.py`

---

### Week 2-3: Sentiment & Stance (Tier 1.2)
**Goal**: Add sentiment/stance to KG edges

**Tasks:**
1. Integrate sentiment analysis library (VADER for simplicity)
2. Compute entity-level sentiment scores
3. Compute edge-level sentiment scores
4. Add controversy detection (high variance)
5. Integrate into main pipeline as optional feature
6. Document interpretation guidelines

**Deliverables:**
- `src/semantic/kg_sentiment.py` (150 lines)
- Updated `kg_pipeline.py` with `--add-sentiment` flag
- `docs/INTERPRETING_SENTIMENT.md`
- Tests

---

### Week 3-4: User-Entity Networks (Tier 1.3)
**Goal**: Bridge KG and actor networks

**Tasks:**
1. Create `UserEntityNetwork` class
2. Build bipartite user-entity edges
3. Implement projection methods
4. Add to CLI as separate tool
5. Create integration example with actor networks

**Deliverables:**
- `src/semantic/user_entity_network.py` (200 lines)
- `src/semantic/user_entity_cli.py` (100 lines)
- Example notebook combining all three network types
- Tests

---

### Week 5-6: Advanced Relations (Tier 2.1)
**Goal**: Extract more relation types

**Tasks:**
1. Define relation pattern rules
2. Implement rule-based extraction
3. Integrate with existing dependency parsing
4. Evaluate on sample data
5. Document relation types and patterns

**Deliverables:**
- Enhanced `extract_relations_from_dependencies()` method
- `src/semantic/relation_patterns.py` with pattern definitions
- Evaluation report comparing before/after
- Updated documentation

---

### Week 7-8: Quality Metrics (Tier 2.4)
**Goal**: Provide validation tools

**Tasks:**
1. Create validation sampling tool
2. Implement confidence scoring
3. Add quality report generation
4. Create manual validation interface (simple CSV-based)
5. Write validation best practices guide

**Deliverables:**
- `src/semantic/kg_validator.py` (150 lines)
- Validation workflow documentation
- Example validation session

---

## 📚 Documentation Priorities

### Immediate (Week 1)
1. **"KG for Social Scientists" Guide**
   - What KGs are and aren't
   - When to use KG vs semantic network vs actor network
   - How to interpret entity types, relation types, edge weights
   - Common pitfalls and how to avoid them

2. **Research Workflow Examples**
   - "Tracking political narratives over time"
   - "Detecting coordinated information operations"
   - "Measuring entity polarization across communities"
   - "Finding bridge entities between echo chambers"

3. **Validation & Quality Guide**
   - How to manually validate a sample
   - Expected precision/recall for different data types
   - When extraction quality is "good enough"

### Medium-term (Week 4)
4. **Advanced Analysis Cookbook**
   - Combining KG + semantic network + actor network
   - Multi-level network analysis
   - Temporal narrative analysis
   - Cross-community comparison

5. **Replication Guide**
   - How to document analysis pipeline
   - Sharing extracted KGs
   - Version control for data pipelines

---

## 🎓 Example Research Questions (Post-Implementation)

### Temporal Analysis
- **Q**: How did discussion of "Ukraine" evolve during 2022?
- **Method**: Temporal KG + event detection + sentiment tracking
- **Output**: Timeline of entity emergence, peak discussions, sentiment shifts

### Narrative Analysis
- **Q**: What conspiracy theories link "vaccines" to other entities?
- **Method**: Entity chains + narrative pattern detection
- **Output**: Subgraphs showing conspiratorial connections, frequency analysis

### Polarization
- **Q**: Which entities are most polarizing (discussed differently by different groups)?
- **Method**: User-entity network + community detection + sentiment analysis
- **Output**: Controversy scores, per-community sentiment differences

### Information Operations
- **Q**: Are there coordinated entity co-mentions suggesting manipulation?
- **Method**: Temporal KG + user-entity network + spike detection
- **Output**: Suspicious temporal patterns, coordinated user behavior

### Bridge Analysis
- **Q**: Which entities/topics bridge different ideological communities?
- **Method**: User-entity network + community detection + betweenness
- **Output**: Bridge entities that connect otherwise isolated groups

---

## 🚀 Quick Wins (Can Do This Week)

### 1. Add `--group-by-time` to KG CLI
**Effort**: 2 hours  
**Impact**: Medium

```bash
python -m src.semantic.kg_cli \
  --input data.csv \
  --outdir output \
  --group-by-time daily \
  --time-col created_at
```

Output: `kg_nodes_YYYY-MM-DD.csv`, `kg_edges_YYYY-MM-DD.csv` for each day

### 2. Add Entity Metadata to Output
**Effort**: 1 hour  
**Impact**: Medium

Add to nodes.csv:
- `first_mention_context` - sample text where entity first appears
- `n_unique_contexts` - how many different documents mention it
- `avg_chars_per_mention` - context size (indicates detail level)

### 3. Create "KG Summary Report"
**Effort**: 2 hours  
**Impact**: High (helps users understand their data)

Auto-generate `kg_report.md` with:
- Entity type distribution (bar chart)
- Top entities by frequency (table)
- Relation type distribution (bar chart)
- Sample entity chains (text)
- Quality flags (e.g., "Low entity diversity - 80% are NORP")

### 4. Add Simple Sentiment (VADER)
**Effort**: 3 hours  
**Impact**: High

```python
pip install vaderSentiment
```

Add `--add-sentiment` flag to kg_cli that computes sentiment for each entity context.

Output: `entity_sentiment.csv` with avg/std sentiment per entity

---

## 📊 Success Metrics

### Technical Metrics
- ✅ >50 dependency relations per 1000 posts (currently: 15)
- ✅ <5% case duplicates (currently: 0%)
- ✅ >80% entity precision on manual validation
- ✅ Process 10K docs in <10 minutes

### Usability Metrics
- ✅ Can replicate a research finding in <1 hour
- ✅ Non-experts can run analysis with <30 min training
- ✅ Documentation answers 90% of user questions

### Research Impact
- ✅ Published in peer-reviewed journal (shows validity)
- ✅ Cited by other researchers (shows utility)
- ✅ Used for real research questions (not just demos)

---

## 🤝 Integration with Existing Tools

### Current Architecture
```
Input CSV → [Semantic Network | KG | Actor Network] → Output CSV
```

### Proposed Architecture
```
Input CSV → Preprocessor → [
    Semantic Network (terms)
    Knowledge Graph (entities)  ← NEW: temporal, sentiment
    Actor Network (users)
    User-Entity Network         ← NEW
] → Integrated Analysis → Output CSV + Reports
```

### Key Integration Points
1. **Shared preprocessing**: Tokenization, cleaning (already done)
2. **Shared output format**: All use node/edge CSV format (already done)
3. **Cross-network analysis**: New module to combine networks
4. **Unified CLI**: Wrapper that runs multiple pipelines

---

## 🎯 Next Immediate Actions

### This Week
1. ⬜ Create `kg_temporal.py` with basic entity timeline tracking
2. ⬜ Add `--group-by-time` flag to `kg_cli.py`
3. ⬜ Write "KG for Social Scientists" guide (10 pages)
4. ⬜ Create temporal KG example on pol_archive_0.csv

### Next Week  
5. ⬜ Integrate VADER sentiment analysis
6. ⬜ Add `--add-sentiment` flag to CLI
7. ⬜ Create controversy detection method
8. ⬜ Write sentiment interpretation guide

### Week 3
9. ⬜ Build user-entity bipartite network
10. ⬜ Create projection methods
11. ⬜ Write example combining all three network types
12. ⬜ Create comprehensive tutorial notebook

---

## 📝 Notes & Considerations

### Design Principles
1. **Modularity**: Each feature is optional, can be used independently
2. **Backwards compatibility**: Don't break existing pipelines
3. **Sensible defaults**: Works well out-of-box for most use cases
4. **Clear documentation**: Non-experts can use it

### Known Limitations
- NER quality depends heavily on domain (news > social media)
- Dependency parsing is slow and doesn't scale well
- Sentiment analysis has cultural/domain bias
- Entity linking requires external knowledge bases

### Open Questions
1. Should we support multiple NER models in parallel?
2. How to handle very large KGs (>1M entities)?
3. Should we build a web interface or CLI is enough?
4. What's the minimum viable validation pipeline?

---

**Last Updated**: October 21, 2025  
**Authors**: Network Inference Team  
**Status**: Planning → Implementation Phase 1 starting
