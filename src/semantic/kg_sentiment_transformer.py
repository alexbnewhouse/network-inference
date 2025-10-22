"""
Transformer-based Sentiment Analysis for Knowledge Graphs
=========================================================

GPU-accelerated sentiment analysis using transformer models (BERT, RoBERTa, etc.)
for entity-aware sentiment scoring in knowledge graphs.

Key Features:
- Contextual sentiment understanding (vs lexicon-based VADER)
- Entity-aware analysis (context windows around mentions)
- GPU batch processing (optimized for RTX 5090)
- Multiple pre-trained models supported
- Drop-in replacement for VADER sentiment

Supported Models:
- cardiffnlp/twitter-roberta-base-sentiment (recommended for social media)
- nlptown/bert-base-multilingual-uncased-sentiment (5-class sentiment)
- distilbert-base-uncased-finetuned-sst-2-english (lightweight)

Example Usage:
    analyzer = TransformerSentimentAnalyzer(
        model_name="cardiffnlp/twitter-roberta-base-sentiment",
        device="cuda",
        batch_size=64
    )
    nodes_with_sentiment, sentiment_df = analyzer.analyze_entity_sentiment(
        df, nodes_df, ents_per_doc
    )
"""
import re
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings


class TransformerSentimentAnalyzer:
    """
    GPU-accelerated transformer-based sentiment analysis for knowledge graphs.
    Provides entity-aware sentiment scoring using contextual embeddings.
    """
    
    # Supported models with their configurations
    SUPPORTED_MODELS = {
        "cardiffnlp/twitter-roberta-base-sentiment": {
            "type": "3class",  # negative, neutral, positive
            "labels": ["negative", "neutral", "positive"],
            "description": "RoBERTa trained on 58M tweets (recommended for social media)"
        },
        "nlptown/bert-base-multilingual-uncased-sentiment": {
            "type": "5star",  # 1-5 stars
            "labels": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
            "description": "Multilingual BERT fine-tuned on product reviews"
        },
        "distilbert-base-uncased-finetuned-sst-2-english": {
            "type": "2class",  # negative, positive
            "labels": ["negative", "positive"],
            "description": "Lightweight DistilBERT (40% faster than BERT)"
        }
    }
    
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
        device: str = "cuda",
        batch_size: int = 64,
        max_context_length: int = 200
    ):
        """
        Initialize transformer sentiment analyzer.
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cpu', 'cuda', or 'mps' (Apple Silicon)
            batch_size: Number of texts to process per batch (higher for GPUs)
            max_context_length: Max chars around entity mention for context
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_context_length = max_context_length
        
        # Validate model
        if model_name not in self.SUPPORTED_MODELS:
            warnings.warn(
                f"Model '{model_name}' not in predefined list. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}. "
                f"Will attempt to load anyway."
            )
            self.model_config = {"type": "unknown", "labels": None}
        else:
            self.model_config = self.SUPPORTED_MODELS[model_name]
        
        # Initialize model (lazy loading)
        self._pipeline = None
        
    @property
    def pipeline(self):
        """Lazy load the sentiment pipeline."""
        if self._pipeline is None:
            print(f"Loading transformer model: {self.model_name}")
            print(f"Device: {self.device}, Batch size: {self.batch_size}")
            
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=self.device if self.device != "cpu" else -1,
                    top_k=None  # Return all scores
                )
                print(f"✓ Model loaded successfully")
            except ImportError:
                raise ImportError(
                    "transformers library not found. "
                    "Install with: pip install torch transformers accelerate"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}")
        
        return self._pipeline
    
    def extract_entity_contexts(
        self,
        text: str,
        entity: str,
        window_size: Optional[int] = None
    ) -> List[str]:
        """
        Extract context windows around entity mentions in text.
        
        Args:
            text: Full text document
            entity: Entity to find
            window_size: Characters before/after entity (default: max_context_length)
            
        Returns:
            List of context strings (one per mention)
        """
        if window_size is None:
            window_size = self.max_context_length
        
        contexts = []
        entity_lower = entity.lower()
        text_lower = text.lower()
        
        # Find all mentions
        start = 0
        while True:
            idx = text_lower.find(entity_lower, start)
            if idx == -1:
                break
            
            # Extract context window
            context_start = max(0, idx - window_size)
            context_end = min(len(text), idx + len(entity) + window_size)
            context = text[context_start:context_end]
            
            # Clean up
            context = context.strip()
            if context:
                contexts.append(context)
            
            start = idx + 1
        
        return contexts
    
    def sentiment_to_score(self, sentiment_result: Dict) -> float:
        """
        Convert model output to normalized sentiment score [-1, 1].
        
        Args:
            sentiment_result: HuggingFace pipeline output (list of label/score dicts)
            
        Returns:
            Normalized sentiment score: -1 (negative) to +1 (positive)
        """
        if not sentiment_result:
            return 0.0
        
        model_type = self.model_config["type"]
        
        if model_type == "3class":
            # Extract scores for negative, neutral, positive
            scores = {item['label'].lower(): item['score'] for item in sentiment_result}
            neg = scores.get('negative', 0)
            neu = scores.get('neutral', 0)
            pos = scores.get('positive', 0)
            # Weighted score: positive - negative
            return pos - neg
        
        elif model_type == "5star":
            # Map 1-5 stars to -1 to +1
            scores = {item['label']: item['score'] for item in sentiment_result}
            weighted_sum = sum(
                int(label.split()[0]) * score
                for label, score in scores.items()
            )
            # Normalize: (1-5) -> (-1 to +1)
            # Formula: (weighted_avg - 3) / 2
            return (weighted_sum - 3) / 2
        
        elif model_type == "2class":
            # Binary: negative/positive
            scores = {item['label'].lower(): item['score'] for item in sentiment_result}
            pos = scores.get('positive', 0)
            neg = scores.get('negative', 0)
            return pos - neg
        
        else:
            # Unknown model: use first label's score as proxy
            return sentiment_result[0]['score'] if sentiment_result else 0.0
    
    def analyze_texts_batch(self, texts: List[str]) -> List[float]:
        """
        Batch analyze sentiment for multiple texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of sentiment scores [-1, 1]
        """
        if not texts:
            return []
        
        try:
            # Run inference in batches
            results = self.pipeline(texts, batch_size=self.batch_size, truncation=True)
            
            # Convert to normalized scores
            scores = [self.sentiment_to_score(result) for result in results]
            return scores
        
        except Exception as e:
            warnings.warn(f"Batch sentiment analysis failed: {e}")
            return [0.0] * len(texts)
    
    def analyze_entity_sentiment(
        self,
        df: pd.DataFrame,
        nodes_df: pd.DataFrame,
        ents_per_doc: List[List[Dict]],
        text_col: str = "text"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze sentiment for each entity across all documents.
        
        Args:
            df: Original DataFrame with text
            nodes_df: KG nodes DataFrame
            ents_per_doc: List of entity dicts per document
            text_col: Name of text column in df
            
        Returns:
            Tuple of (nodes_with_sentiment, sentiment_df)
        """
        from collections import defaultdict
        
        # Collect contexts for each entity
        entity_contexts = defaultdict(list)
        entity_doc_ids = defaultdict(list)
        
        print("\nExtracting entity contexts...")
        for doc_id, (text, ents) in enumerate(zip(df[text_col], ents_per_doc)):
            for ent_dict in ents:
                entity = ent_dict.get('text', '')
                if not entity:
                    continue
                
                # Extract context windows for this entity in this doc
                contexts = self.extract_entity_contexts(text, entity)
                
                for context in contexts:
                    entity_contexts[entity].append(context)
                    entity_doc_ids[entity].append(doc_id)
        
        # Prepare all contexts for batch processing
        print(f"\nAnalyzing sentiment for {len(entity_contexts)} entities...")
        all_entities = []
        all_contexts = []
        context_indices = []  # Track which entity each context belongs to
        
        for entity_idx, (entity, contexts) in enumerate(entity_contexts.items()):
            all_entities.extend([entity] * len(contexts))
            all_contexts.extend(contexts)
            context_indices.extend([entity_idx] * len(contexts))
        
        # Batch sentiment analysis
        print(f"Processing {len(all_contexts)} contexts in batches of {self.batch_size}...")
        all_sentiment_scores = []
        
        for i in tqdm(range(0, len(all_contexts), self.batch_size), desc="Sentiment batches"):
            batch = all_contexts[i:i + self.batch_size]
            scores = self.analyze_texts_batch(batch)
            all_sentiment_scores.extend(scores)
        
        # Aggregate sentiment per entity
        entity_sentiment = {}
        entity_list = list(entity_contexts.keys())
        
        for entity_idx, entity in enumerate(entity_list):
            # Get scores for this entity
            mask = [idx == entity_idx for idx in context_indices]
            scores = [score for score, m in zip(all_sentiment_scores, mask) if m]
            
            if scores:
                entity_sentiment[entity] = {
                    'avg_sentiment': float(np.mean(scores)),
                    'sentiment_std': float(np.std(scores)),
                    'n_contexts': len(scores),
                    'min_sentiment': float(np.min(scores)),
                    'max_sentiment': float(np.max(scores))
                }
            else:
                entity_sentiment[entity] = {
                    'avg_sentiment': 0.0,
                    'sentiment_std': 0.0,
                    'n_contexts': 0,
                    'min_sentiment': 0.0,
                    'max_sentiment': 0.0
                }
        
        # Create sentiment DataFrame
        sentiment_df = pd.DataFrame([
            {'entity': entity, **stats}
            for entity, stats in entity_sentiment.items()
        ])
        
        # Add sentiment to nodes
        nodes_with_sentiment = nodes_df.copy()
        nodes_with_sentiment = nodes_with_sentiment.merge(
            sentiment_df[['entity', 'avg_sentiment', 'sentiment_std', 'n_contexts']],
            left_on='label',
            right_on='entity',
            how='left'
        )
        nodes_with_sentiment = nodes_with_sentiment.drop(columns=['entity'], errors='ignore')
        nodes_with_sentiment['avg_sentiment'] = nodes_with_sentiment['avg_sentiment'].fillna(0.0)
        
        # Add sentiment categories
        nodes_with_sentiment['sentiment_category'] = pd.cut(
            nodes_with_sentiment['avg_sentiment'],
            bins=[-float('inf'), -0.3, 0.3, float('inf')],
            labels=['negative', 'neutral', 'positive']
        )
        
        return nodes_with_sentiment, sentiment_df
    
    def analyze_edge_sentiment(
        self,
        df: pd.DataFrame,
        edges_df: pd.DataFrame,
        ents_per_doc: List[List[Dict]],
        text_col: str = "text"
    ) -> pd.DataFrame:
        """
        Analyze sentiment for entity co-occurrences (edges).
        
        Args:
            df: Original DataFrame with text
            edges_df: KG edges DataFrame
            ents_per_doc: List of entity dicts per document
            text_col: Name of text column in df
            
        Returns:
            edges_with_sentiment DataFrame
        """
        print("\nAnalyzing edge sentiment...")
        
        # Collect contexts for each edge
        edge_contexts = []
        
        for doc_id, (text, ents) in enumerate(tqdm(
            zip(df[text_col], ents_per_doc),
            total=len(df),
            desc="Extracting edge contexts"
        )):
            # Get entity pairs in this doc
            entity_texts = [e.get('text', '') for e in ents if e.get('text')]
            
            # For each pair that appears in edges_df
            for i, ent1 in enumerate(entity_texts):
                for ent2 in entity_texts[i+1:]:
                    # Check if this edge exists
                    edge_exists = (
                        ((edges_df['source'] == ent1) & (edges_df['target'] == ent2)) |
                        ((edges_df['source'] == ent2) & (edges_df['target'] == ent1))
                    ).any()
                    
                    if edge_exists:
                        # Extract contexts for both entities
                        contexts_1 = self.extract_entity_contexts(text, ent1)
                        contexts_2 = self.extract_entity_contexts(text, ent2)
                        
                        # Use the full document as edge context (contains both)
                        edge_contexts.append({
                            'source': ent1,
                            'target': ent2,
                            'context': text[:1000],  # Limit to 1000 chars
                            'doc_id': doc_id
                        })
        
        # Batch analyze edge contexts
        if edge_contexts:
            contexts = [ec['context'] for ec in edge_contexts]
            print(f"Analyzing {len(contexts)} edge contexts...")
            sentiment_scores = []
            
            for i in tqdm(range(0, len(contexts), self.batch_size), desc="Edge sentiment batches"):
                batch = contexts[i:i + self.batch_size]
                scores = self.analyze_texts_batch(batch)
                sentiment_scores.extend(scores)
            
            # Add scores to edge contexts
            for ec, score in zip(edge_contexts, sentiment_scores):
                ec['sentiment'] = score
        
        # Aggregate edge sentiment
        edge_sentiment_map = {}
        for ec in edge_contexts:
            key = tuple(sorted([ec['source'], ec['target']]))
            if key not in edge_sentiment_map:
                edge_sentiment_map[key] = []
            edge_sentiment_map[key].append(ec['sentiment'])
        
        # Add to edges DataFrame
        edges_with_sentiment = edges_df.copy()
        
        avg_sentiments = []
        for _, row in edges_with_sentiment.iterrows():
            key = tuple(sorted([row['source'], row['target']]))
            scores = edge_sentiment_map.get(key, [0.0])
            avg_sentiments.append(float(np.mean(scores)))
        
        edges_with_sentiment['avg_sentiment'] = avg_sentiments
        edges_with_sentiment['sentiment_category'] = pd.cut(
            edges_with_sentiment['avg_sentiment'],
            bins=[-float('inf'), -0.3, 0.3, float('inf')],
            labels=['negative', 'neutral', 'positive']
        )
        
        return edges_with_sentiment
    
    def find_controversial_entities(self, sentiment_df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
        """Find entities with high sentiment variance (controversial)."""
        controversial = sentiment_df[sentiment_df['sentiment_std'] > threshold].copy()
        controversial['controversy_score'] = controversial['sentiment_std']
        return controversial.sort_values('controversy_score', ascending=False)
    
    def sentiment_summary(self, sentiment_df: pd.DataFrame, edges_with_sentiment: pd.DataFrame) -> str:
        """Generate sentiment analysis summary."""
        summary = []
        summary.append(f"\n=== TRANSFORMER SENTIMENT ANALYSIS SUMMARY ===")
        summary.append(f"Model: {self.model_name}")
        summary.append(f"Device: {self.device}\n")
        
        # Entity sentiment distribution
        summary.append(f"Entity Sentiment Distribution:")
        summary.append(f"  Average: {sentiment_df['avg_sentiment'].mean():.3f}")
        summary.append(f"  Std Dev: {sentiment_df['avg_sentiment'].std():.3f}")
        summary.append(f"  Range: [{sentiment_df['avg_sentiment'].min():.3f}, {sentiment_df['avg_sentiment'].max():.3f}]")
        
        # Most positive/negative entities
        summary.append(f"\nMost Positive Entities:")
        top_positive = sentiment_df.nlargest(5, 'avg_sentiment')[['entity', 'avg_sentiment', 'n_contexts']]
        for row in top_positive.itertuples():
            summary.append(f"  {row.entity}: {row.avg_sentiment:+.3f} ({row.n_contexts} contexts)")
        
        summary.append(f"\nMost Negative Entities:")
        top_negative = sentiment_df.nsmallest(5, 'avg_sentiment')[['entity', 'avg_sentiment', 'n_contexts']]
        for row in top_negative.itertuples():
            summary.append(f"  {row.entity}: {row.avg_sentiment:+.3f} ({row.n_contexts} contexts)")
        
        # Controversial entities
        controversial = self.find_controversial_entities(sentiment_df)
        if len(controversial) > 0:
            summary.append(f"\nMost Controversial Entities (high sentiment variance):")
            for row in controversial.head(5).itertuples():
                summary.append(f"  {row.entity}: controversy={row.controversy_score:.3f}, avg={row.avg_sentiment:+.3f}")
        
        # Edge sentiment
        if 'sentiment_category' in edges_with_sentiment.columns:
            summary.append(f"\nEdge Sentiment Distribution:")
            sent_counts = edges_with_sentiment['sentiment_category'].value_counts()
            for cat, count in sent_counts.items():
                pct = 100 * count / len(edges_with_sentiment)
                summary.append(f"  {cat}: {count} ({pct:.1f}%)")
        
        return '\n'.join(summary)


def add_sentiment_to_kg_transformer(
    df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    ents_per_doc: List[List[Dict]],
    outdir: str,
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
    device: str = "cuda",
    batch_size: int = 64,
    text_col: str = "text"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add transformer-based sentiment analysis to knowledge graph and save results.
    
    Args:
        df: Original DataFrame with text
        nodes_df: KG nodes DataFrame
        edges_df: KG edges DataFrame
        ents_per_doc: List of entity dicts per document
        outdir: Output directory
        model_name: HuggingFace model identifier
        device: 'cpu', 'cuda', or 'mps'
        batch_size: Batch size for GPU processing
        text_col: Name of text column
        
    Returns:
        Tuple of (nodes_with_sentiment, edges_with_sentiment, sentiment_df)
    """
    import os
    
    analyzer = TransformerSentimentAnalyzer(
        model_name=model_name,
        device=device,
        batch_size=batch_size
    )
    
    print("\n" + "="*70)
    print("TRANSFORMER SENTIMENT ANALYSIS")
    print("="*70)
    
    nodes_with_sentiment, sentiment_df = analyzer.analyze_entity_sentiment(
        df, nodes_df, ents_per_doc, text_col=text_col
    )
    
    edges_with_sentiment = analyzer.analyze_edge_sentiment(
        df, edges_df, ents_per_doc, text_col=text_col
    )
    
    # Save results
    os.makedirs(outdir, exist_ok=True)
    nodes_with_sentiment.to_csv(os.path.join(outdir, "kg_nodes_with_sentiment.csv"), index=False)
    edges_with_sentiment.to_csv(os.path.join(outdir, "kg_edges_with_sentiment.csv"), index=False)
    sentiment_df.to_csv(os.path.join(outdir, "entity_sentiment.csv"), index=False)
    
    # Print summary
    summary = analyzer.sentiment_summary(sentiment_df, edges_with_sentiment)
    print(summary)
    
    # Save summary to file
    with open(os.path.join(outdir, "sentiment_summary.txt"), 'w') as f:
        f.write(summary)
    
    print(f"\n✓ Transformer sentiment analysis saved to {outdir}")
    print("="*70 + "\n")
    
    return nodes_with_sentiment, edges_with_sentiment, sentiment_df
