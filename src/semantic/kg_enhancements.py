"""
Enhanced Knowledge Graph Features for 4chan Data Analysis

This module extends the basic KG pipeline with domain-specific features:
1. Reply network extraction
2. Temporal entity tracking
3. Sentiment on entity relations
4. Quote/meme extraction
5. Entity controversy scoring
"""

import re
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import spacy
from textblob import TextBlob


class Enhanced4chanKG:
    """Enhanced knowledge graph builder for 4chan data."""
    
    def __init__(self, kg_pipeline):
        """
        Args:
            kg_pipeline: Base KnowledgeGraphPipeline instance
        """
        self.kg = kg_pipeline
        self.nlp = kg_pipeline.nlp
        
    def extract_reply_network(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract reply network from >>post_id references.
        
        Returns:
            DataFrame with columns: [source_post, target_post, entities_in_reply]
        """
        reply_pattern = re.compile(r'>>(\d{6,})')
        
        replies = []
        for idx, row in df.iterrows():
            text = str(row.get('body', ''))
            matches = reply_pattern.findall(text)
            
            for target_id in matches:
                replies.append({
                    'source_post': idx,
                    'target_post': target_id,
                    'text': text
                })
        
        return pd.DataFrame(replies)
    
    def add_temporal_features(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, 
                            df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Add temporal features to entity nodes and edges.
        
        Returns:
            Updated (nodes_df, edges_df) with temporal features
        """
        if 'created_at' not in df.columns:
            print("Warning: No 'created_at' column found")
            return nodes_df, edges_df
        
        df['timestamp'] = pd.to_datetime(df['created_at'], errors='coerce')
        
        # Extract entities per document with timestamps
        texts = df['text'].tolist()
        ents_per_doc, entity_counter = self.kg.extract_ner(texts, show_progress=False)
        
        # Track first/last appearance of each entity
        entity_timeline = defaultdict(lambda: {'first': None, 'last': None, 'count': 0})
        
        for idx, doc_ents in enumerate(ents_per_doc):
            timestamp = df.iloc[idx]['timestamp']
            if pd.isna(timestamp):
                continue
                
            for ent in doc_ents:
                key = (ent['text'], ent['label'])
                entity_timeline[key]['count'] += 1
                
                if entity_timeline[key]['first'] is None:
                    entity_timeline[key]['first'] = timestamp
                entity_timeline[key]['last'] = timestamp
        
        # Add temporal features to nodes
        nodes_df['first_seen'] = nodes_df.apply(
            lambda row: entity_timeline.get((row['entity'], row['type']), {}).get('first'),
            axis=1
        )
        nodes_df['last_seen'] = nodes_df.apply(
            lambda row: entity_timeline.get((row['entity'], row['type']), {}).get('last'),
            axis=1
        )
        nodes_df['lifespan_days'] = (nodes_df['last_seen'] - nodes_df['first_seen']).dt.days
        
        return nodes_df, edges_df
    
    def add_sentiment_to_edges(self, edges_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment scores to entity co-occurrence edges.
        
        For each edge, compute average sentiment of sentences containing both entities.
        """
        texts = df['text'].astype(str).tolist()
        ents_per_doc, _ = self.kg.extract_ner(texts, show_progress=False)
        
        # For each edge, collect sentences containing both entities
        edge_sentiments = defaultdict(list)
        
        for text, doc_ents in zip(texts, ents_per_doc):
            # Find all entity pairs in this document
            for i, e1 in enumerate(doc_ents):
                for e2 in doc_ents[i+1:]:
                    if e1['text'] == e2['text']:
                        continue
                    
                    # Get sentiment of the sentence/text containing both
                    context = e1.get('sent', text)
                    sentiment = TextBlob(context).sentiment.polarity
                    
                    # Store for both directions
                    edge_sentiments[(e1['text'], e2['text'])].append(sentiment)
                    edge_sentiments[(e2['text'], e1['text'])].append(sentiment)
        
        # Add average sentiment to edges
        def get_sentiment(row):
            key = (row['source_entity'], row['target_entity'])
            sentiments = edge_sentiments.get(key, [0.0])
            return np.mean(sentiments) if sentiments else 0.0
        
        edges_df['sentiment'] = edges_df.apply(get_sentiment, axis=1)
        edges_df['sentiment_category'] = pd.cut(
            edges_df['sentiment'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['negative', 'neutral', 'positive']
        )
        
        return edges_df
    
    def extract_greentext_quotes(self, df: pd.DataFrame, min_freq: int = 3) -> pd.DataFrame:
        """
        Extract frequently used greentext quotes (>text) as meme nodes.
        
        Returns:
            DataFrame with columns: [quote, frequency, entities_mentioned]
        """
        greentext_pattern = re.compile(r'^>(.+?)$', re.MULTILINE)
        
        quotes = []
        for text in df['body'].astype(str):
            matches = greentext_pattern.findall(text)
            quotes.extend([q.strip() for q in matches if len(q.strip()) > 10])
        
        quote_counter = Counter(quotes)
        
        # Filter by frequency
        frequent_quotes = [
            {'quote': q, 'frequency': c} 
            for q, c in quote_counter.items() 
            if c >= min_freq
        ]
        
        return pd.DataFrame(frequent_quotes)
    
    def compute_entity_controversy(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute controversy score for entities based on sentiment variance.
        
        High controversy = entity appears in both positive and negative contexts.
        """
        if 'sentiment' not in edges_df.columns:
            print("Warning: Run add_sentiment_to_edges first")
            return pd.DataFrame()
        
        # Aggregate sentiments per entity
        entity_sentiments = defaultdict(list)
        
        for _, row in edges_df.iterrows():
            entity_sentiments[row['source_entity']].append(row['sentiment'])
            entity_sentiments[row['target_entity']].append(row['sentiment'])
        
        controversy_scores = []
        for entity, sentiments in entity_sentiments.items():
            if len(sentiments) < 2:
                continue
            
            controversy_scores.append({
                'entity': entity,
                'avg_sentiment': np.mean(sentiments),
                'sentiment_variance': np.var(sentiments),
                'controversy_score': np.std(sentiments),  # Higher = more controversial
                'n_mentions': len(sentiments)
            })
        
        return pd.DataFrame(controversy_scores).sort_values('controversy_score', ascending=False)
    
    def detect_narrative_clusters(self, edges_df: pd.DataFrame, 
                                 method: str = 'louvain') -> Dict[str, int]:
        """
        Detect entity clusters (narratives) using community detection.
        
        Args:
            method: 'louvain' or 'label_propagation'
            
        Returns:
            Dict mapping entity names to cluster IDs
        """
        try:
            import networkx as nx
            from networkx.algorithms import community
        except ImportError:
            print("NetworkX required for community detection")
            return {}
        
        # Build NetworkX graph
        G = nx.Graph()
        for _, row in edges_df.iterrows():
            if row['relation_type'] == 'co-occurrence':
                G.add_edge(
                    row['source_entity'],
                    row['target_entity'],
                    weight=row['weight']
                )
        
        # Detect communities
        if method == 'louvain':
            communities = community.louvain_communities(G, weight='weight')
        else:
            communities = community.label_propagation_communities(G)
        
        # Map entities to cluster IDs
        entity_to_cluster = {}
        for cluster_id, nodes in enumerate(communities):
            for node in nodes:
                entity_to_cluster[node] = cluster_id
        
        return entity_to_cluster
    
    def extract_conspiracy_patterns(self, edges_df: pd.DataFrame, 
                                   min_chain_length: int = 3) -> List[List[str]]:
        """
        Find entity chains that might represent conspiracy narratives.
        
        Example: "CIA → Venezuela → Oil → War" 
        
        Returns:
            List of entity chains (paths through the graph)
        """
        try:
            import networkx as nx
        except ImportError:
            print("NetworkX required for path finding")
            return []
        
        # Build directed graph from dependency relations
        G = nx.DiGraph()
        for _, row in edges_df.iterrows():
            if row['relation_type'] != 'co-occurrence':
                G.add_edge(
                    row['source_entity'],
                    row['target_entity'],
                    relation=row['relation_type']
                )
        
        # Find simple paths of length >= min_chain_length
        chains = []
        for source in G.nodes():
            for target in G.nodes():
                if source == target:
                    continue
                try:
                    paths = nx.all_simple_paths(G, source, target, cutoff=5)
                    for path in paths:
                        if len(path) >= min_chain_length:
                            chains.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        # Return unique chains sorted by length
        unique_chains = [list(x) for x in set(tuple(c) for c in chains)]
        return sorted(unique_chains, key=len, reverse=True)[:20]


def demo_enhancements():
    """Demonstration of enhanced features on sample data."""
    import tempfile
    from src.semantic.kg_pipeline import KnowledgeGraphPipeline
    
    print("=== ENHANCED 4CHAN KG DEMO ===\n")
    
    # Load data
    df = pd.read_csv('pol_archive_0.csv', nrows=2000)
    df['text'] = df['body']
    
    # Build basic KG
    kg = KnowledgeGraphPipeline(min_entity_freq=5, cooccurrence_window=200, use_dependencies=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        nodes, edges = kg.run(df, tmpdir, show_progress=False)
    
    # Apply enhancements
    enhanced = Enhanced4chanKG(kg)
    
    # 1. Reply network
    print("1. REPLY NETWORK")
    replies = enhanced.extract_reply_network(df)
    print(f"   Found {len(replies)} reply relationships")
    print(f"   Most replied-to posts: {replies['target_post'].value_counts().head(3).to_dict()}\n")
    
    # 2. Temporal features
    print("2. TEMPORAL FEATURES")
    nodes_temporal, _ = enhanced.add_temporal_features(nodes, edges, df)
    if 'lifespan_days' in nodes_temporal.columns:
        print(f"   Entities with longest lifespan:")
        print(nodes_temporal.nlargest(5, 'lifespan_days')[['entity', 'lifespan_days']])
    print()
    
    # 3. Sentiment analysis
    print("3. SENTIMENT ANALYSIS")
    edges_sentiment = enhanced.add_sentiment_to_edges(edges, df)
    print(f"   Sentiment distribution:")
    print(edges_sentiment['sentiment_category'].value_counts())
    print(f"\n   Most positive edges:")
    print(edges_sentiment.nlargest(3, 'sentiment')[['source_entity', 'target_entity', 'sentiment']])
    print(f"\n   Most negative edges:")
    print(edges_sentiment.nsmallest(3, 'sentiment')[['source_entity', 'target_entity', 'sentiment']])
    print()
    
    # 4. Controversy scores
    print("4. ENTITY CONTROVERSY")
    controversy = enhanced.compute_entity_controversy(edges_sentiment)
    print("   Most controversial entities:")
    print(controversy.head(10)[['entity', 'controversy_score', 'avg_sentiment']])
    print()
    
    # 5. Greentext quotes
    print("5. POPULAR GREENTEXT QUOTES")
    quotes = enhanced.extract_greentext_quotes(df, min_freq=5)
    print(f"   Found {len(quotes)} popular quotes")
    if len(quotes) > 0:
        print(quotes.head(5))
    print()
    
    # 6. Narrative clusters
    print("6. NARRATIVE CLUSTERS")
    clusters = enhanced.detect_narrative_clusters(edges)
    if clusters:
        cluster_sizes = Counter(clusters.values())
        print(f"   Detected {len(cluster_sizes)} narrative clusters")
        print(f"   Cluster sizes: {dict(cluster_sizes.most_common(5))}")
    print()
    
    # 7. Conspiracy patterns
    print("7. CONSPIRACY PATTERNS")
    patterns = enhanced.extract_conspiracy_patterns(edges, min_chain_length=3)
    if patterns:
        print(f"   Found {len(patterns)} entity chains")
        for chain in patterns[:5]:
            print(f"   {' → '.join(chain)}")


if __name__ == '__main__':
    demo_enhancements()
