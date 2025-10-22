"""
Sentiment Analysis for Knowledge Graphs

Add sentiment scores to entities and entity relationships using VADER.
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class KGSentimentAnalyzer:
    """Add sentiment analysis to knowledge graph entities and edges."""
    
    def __init__(self):
        """Initialize VADER sentiment analyzer."""
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_entity_sentiment(self, df, nodes_df, ents_per_doc):
        """
        Analyze sentiment of contexts where each entity appears.
        
        Args:
            df: Original DataFrame with 'text' column
            nodes_df: KG nodes DataFrame
            ents_per_doc: List of entity dicts per document
            
        Returns:
            DataFrame with entity sentiment scores
        """
        texts = df['text'].astype(str).tolist()
        
        # Collect sentiment scores for each entity
        entity_sentiments = defaultdict(list)
        
        for text, doc_ents in zip(texts, ents_per_doc):
            # Get sentiment for the whole text
            if not text or len(text.strip()) == 0:
                continue
                
            scores = self.analyzer.polarity_scores(text)
            compound = scores['compound']
            
            # Assign to all entities in this document
            for ent in doc_ents:
                entity_key = (ent['text'], ent['label'])
                entity_sentiments[entity_key].append(compound)
        
        # Aggregate per entity
        sentiment_data = []
        for (entity, etype), sentiments in entity_sentiments.items():
            if len(sentiments) > 0:
                sentiment_data.append({
                    'entity': entity,
                    'type': etype,
                    'avg_sentiment': np.mean(sentiments),
                    'sentiment_std': np.std(sentiments),
                    'sentiment_min': np.min(sentiments),
                    'sentiment_max': np.max(sentiments),
                    'n_contexts': len(sentiments),
                    'controversy_score': np.std(sentiments)  # High std = controversial
                })
        
        sentiment_df = pd.DataFrame(sentiment_data)
        
        # Merge with nodes
        nodes_with_sentiment = nodes_df.merge(
            sentiment_df[['entity', 'avg_sentiment', 'sentiment_std', 'controversy_score']],
            on='entity',
            how='left'
        )
        
        return nodes_with_sentiment, sentiment_df
    
    def analyze_edge_sentiment(self, df, edges_df, ents_per_doc):
        """
        Analyze sentiment of contexts containing both entities in an edge.
        
        Args:
            df: Original DataFrame with 'text' column
            edges_df: KG edges DataFrame
            ents_per_doc: List of entity dicts per document
            
        Returns:
            DataFrame with edge sentiment scores
        """
        texts = df['text'].astype(str).tolist()
        
        # Find contexts containing both entities
        edge_sentiments = defaultdict(list)
        
        for text, doc_ents in zip(texts, ents_per_doc):
            if not text or len(text.strip()) == 0 or len(doc_ents) < 2:
                continue
            
            # Get sentiment for this document
            scores = self.analyzer.polarity_scores(text)
            compound = scores['compound']
            
            # Find all entity pairs in this document
            for i, e1 in enumerate(doc_ents):
                for e2 in doc_ents[i+1:]:
                    if e1['text'] != e2['text']:
                        # Create ordered pair
                        pair = tuple(sorted([e1['text'], e2['text']]))
                        edge_sentiments[pair].append(compound)
        
        # Add sentiment to edges
        def get_edge_sentiment(row):
            pair = tuple(sorted([row['source_entity'], row['target_entity']]))
            sentiments = edge_sentiments.get(pair, [0.0])
            return np.mean(sentiments) if sentiments else 0.0
        
        def get_edge_sentiment_std(row):
            pair = tuple(sorted([row['source_entity'], row['target_entity']]))
            sentiments = edge_sentiments.get(pair, [0.0])
            return np.std(sentiments) if len(sentiments) > 1 else 0.0
        
        edges_with_sentiment = edges_df.copy()
        edges_with_sentiment['sentiment'] = edges_df.apply(get_edge_sentiment, axis=1)
        edges_with_sentiment['sentiment_std'] = edges_df.apply(get_edge_sentiment_std, axis=1)
        
        # Categorize sentiment
        edges_with_sentiment['sentiment_category'] = pd.cut(
            edges_with_sentiment['sentiment'],
            bins=[-1.0, -0.1, 0.1, 1.0],
            labels=['negative', 'neutral', 'positive']
        )
        
        return edges_with_sentiment
    
    def find_controversial_entities(self, sentiment_df, threshold=0.3):
        """
        Find entities with high sentiment variance (controversial).
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            threshold: Minimum std deviation to be considered controversial
            
        Returns:
            DataFrame of controversial entities sorted by controversy
        """
        controversial = sentiment_df[
            sentiment_df['controversy_score'] > threshold
        ].sort_values('controversy_score', ascending=False)
        
        return controversial
    
    def sentiment_summary(self, sentiment_df, edges_with_sentiment):
        """Generate sentiment analysis summary."""
        summary = []
        summary.append("\n=== SENTIMENT ANALYSIS SUMMARY ===\n")
        
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


def add_sentiment_to_kg(df, nodes_df, edges_df, ents_per_doc, outdir):
    """
    Add sentiment analysis to knowledge graph and save results.
    
    Args:
        df: Original DataFrame with text
        nodes_df: KG nodes DataFrame
        edges_df: KG edges DataFrame
        ents_per_doc: List of entity dicts per document
        outdir: Output directory
        
    Returns:
        Tuple of (nodes_with_sentiment, edges_with_sentiment, sentiment_df)
    """
    import os
    
    analyzer = KGSentimentAnalyzer()
    
    print("\nAnalyzing entity sentiment...")
    nodes_with_sentiment, sentiment_df = analyzer.analyze_entity_sentiment(
        df, nodes_df, ents_per_doc
    )
    
    print("Analyzing edge sentiment...")
    edges_with_sentiment = analyzer.analyze_edge_sentiment(
        df, edges_df, ents_per_doc
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
    
    print(f"\n✓ Sentiment analysis saved to {outdir}")
    
    return nodes_with_sentiment, edges_with_sentiment, sentiment_df
