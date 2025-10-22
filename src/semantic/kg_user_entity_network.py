"""
User-Entity Network Analysis for Knowledge Graphs.

Creates bipartite graphs connecting users/authors to the entities they mention,
enabling analysis of:
- User similarity based on shared entity interests
- Entity co-occurrence based on shared users
- Community detection around entities
- User clustering by entity mention patterns

Author: AI Assistant
Date: 2025-10-21
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import networkx as nx
from networkx.algorithms import bipartite


class UserEntityNetwork:
    """
    Bipartite network connecting users to entities they mention.
    Supports projection to user-user and entity-entity networks.
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.users = set()
        self.entities = set()
        self.user_entities = defaultdict(set)  # user -> entities
        self.entity_users = defaultdict(set)   # entity -> users
        
    def build_from_texts(
        self, 
        texts: List[str], 
        entities_per_doc: List[List[str]],
        user_ids: List[str]
    ):
        """
        Build bipartite graph from texts, entities, and user IDs.
        
        Args:
            texts: List of text documents
            entities_per_doc: List of entity lists (one per document)
            user_ids: List of user IDs (one per document)
        """
        if len(texts) != len(entities_per_doc) != len(user_ids):
            raise ValueError("texts, entities_per_doc, and user_ids must have same length")
        
        for i, (text, entities, user_id) in enumerate(zip(texts, entities_per_doc, user_ids)):
            if not entities:
                continue
                
            # Add user node
            if user_id not in self.users:
                self.users.add(user_id)
                self.graph.add_node(user_id, bipartite=0, node_type='user')
            
            # Add entity nodes and edges
            for entity in set(entities):  # Deduplicate entities within document
                if entity not in self.entities:
                    self.entities.add(entity)
                    self.graph.add_node(entity, bipartite=1, node_type='entity')
                
                # Add edge (or increment weight if exists)
                if self.graph.has_edge(user_id, entity):
                    self.graph[user_id][entity]['weight'] += 1
                else:
                    self.graph.add_edge(user_id, entity, weight=1)
                
                # Track relationships
                self.user_entities[user_id].add(entity)
                self.entity_users[entity].add(user_id)
    
    def build_from_kg_nodes(
        self,
        nodes_df: pd.DataFrame,
        texts: List[str],
        user_ids: List[str]
    ):
        """
        Build bipartite graph from KG nodes and source texts.
        
        Args:
            nodes_df: DataFrame with KG nodes (must have 'entity' column)
            texts: Original text documents
            user_ids: User IDs corresponding to texts
        """
        if len(texts) != len(user_ids):
            raise ValueError("texts and user_ids must have same length")
        
        # Get entity set from nodes
        valid_entities = set(nodes_df['entity'].unique())
        
        # Extract entities per document from KG metadata
        if 'doc_indices' in nodes_df.columns:
            # Use doc_indices from nodes
            entities_per_doc = [[] for _ in range(len(texts))]
            for _, row in nodes_df.iterrows():
                entity = row['entity']
                doc_indices = eval(row['doc_indices']) if isinstance(row['doc_indices'], str) else row['doc_indices']
                for idx in doc_indices:
                    if 0 <= idx < len(texts):
                        entities_per_doc[idx].append(entity)
        else:
            # Fall back to string matching (less accurate)
            entities_per_doc = []
            for text in texts:
                if pd.isna(text) or not isinstance(text, str):
                    entities_per_doc.append([])
                    continue
                text_lower = text.lower()
                doc_entities = [e for e in valid_entities if e.lower() in text_lower]
                entities_per_doc.append(doc_entities)
        
        self.build_from_texts(texts, entities_per_doc, user_ids)
    
    def get_stats(self) -> Dict:
        """Get network statistics."""
        return {
            'n_users': len(self.users),
            'n_entities': len(self.entities),
            'n_edges': self.graph.number_of_edges(),
            'avg_entities_per_user': np.mean([len(ents) for ents in self.user_entities.values()]),
            'avg_users_per_entity': np.mean([len(users) for users in self.entity_users.values()]),
            'density': nx.density(self.graph)
        }
    
    def get_user_profile(self, user_id: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get entity mention profile for a user.
        
        Args:
            user_id: User to analyze
            top_n: Number of top entities to return
        
        Returns:
            DataFrame with entity, mention_count, sorted by count
        """
        if user_id not in self.users:
            return pd.DataFrame()
        
        entities = []
        for entity in self.user_entities[user_id]:
            weight = self.graph[user_id][entity]['weight']
            entities.append({'entity': entity, 'mention_count': weight})
        
        df = pd.DataFrame(entities)
        return df.sort_values('mention_count', ascending=False).head(top_n).reset_index(drop=True)
    
    def get_entity_audience(self, entity: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get users who mention an entity most frequently.
        
        Args:
            entity: Entity to analyze
            top_n: Number of top users to return
        
        Returns:
            DataFrame with user_id, mention_count
        """
        if entity not in self.entities:
            return pd.DataFrame()
        
        users = []
        for user_id in self.entity_users[entity]:
            weight = self.graph[user_id][entity]['weight']
            users.append({'user_id': user_id, 'mention_count': weight})
        
        df = pd.DataFrame(users)
        return df.sort_values('mention_count', ascending=False).head(top_n).reset_index(drop=True)
    
    def project_to_user_network(
        self, 
        weighted: bool = True,
        min_shared_entities: int = 1
    ) -> nx.Graph:
        """
        Project bipartite graph to user-user network.
        Users are connected if they mention the same entities.
        
        Args:
            weighted: If True, edge weight = number of shared entities
            min_shared_entities: Minimum shared entities required for edge
        
        Returns:
            User-user graph
        """
        user_graph = nx.Graph()
        user_graph.add_nodes_from(self.users)
        
        # Connect users who mention same entities
        user_list = list(self.users)
        for i, user1 in enumerate(user_list):
            for user2 in user_list[i+1:]:
                shared = self.user_entities[user1] & self.user_entities[user2]
                if len(shared) >= min_shared_entities:
                    if weighted:
                        user_graph.add_edge(user1, user2, weight=len(shared), n_shared_entities=len(shared))
                    else:
                        user_graph.add_edge(user1, user2, n_shared_entities=len(shared))
        
        return user_graph
    
    def project_to_entity_network(
        self,
        weighted: bool = True,
        min_shared_users: int = 1
    ) -> nx.Graph:
        """
        Project bipartite graph to entity-entity network.
        Entities are connected if mentioned by the same users.
        
        Args:
            weighted: If True, edge weight = number of shared users
            min_shared_users: Minimum shared users required for edge
        
        Returns:
            Entity-entity graph
        """
        entity_graph = nx.Graph()
        entity_graph.add_nodes_from(self.entities)
        
        # Connect entities mentioned by same users
        entity_list = list(self.entities)
        for i, entity1 in enumerate(entity_list):
            for entity2 in entity_list[i+1:]:
                shared = self.entity_users[entity1] & self.entity_users[entity2]
                if len(shared) >= min_shared_users:
                    if weighted:
                        entity_graph.add_edge(entity1, entity2, weight=len(shared), n_shared_users=len(shared))
                    else:
                        entity_graph.add_edge(entity1, entity2, n_shared_users=len(shared))
        
        return entity_graph
    
    def find_similar_users(
        self,
        user_id: str,
        top_n: int = 10,
        method: str = 'jaccard'
    ) -> pd.DataFrame:
        """
        Find users with similar entity mention patterns.
        
        Args:
            user_id: User to find similarities for
            top_n: Number of similar users to return
            method: Similarity metric ('jaccard', 'cosine', 'overlap')
        
        Returns:
            DataFrame with user_id, similarity_score
        """
        if user_id not in self.users:
            return pd.DataFrame()
        
        user_entities = self.user_entities[user_id]
        similarities = []
        
        for other_id in self.users:
            if other_id == user_id:
                continue
            
            other_entities = self.user_entities[other_id]
            
            if method == 'jaccard':
                intersection = len(user_entities & other_entities)
                union = len(user_entities | other_entities)
                similarity = intersection / union if union > 0 else 0
            
            elif method == 'cosine':
                # Binary cosine similarity
                intersection = len(user_entities & other_entities)
                similarity = intersection / np.sqrt(len(user_entities) * len(other_entities)) if len(user_entities) > 0 and len(other_entities) > 0 else 0
            
            elif method == 'overlap':
                # Overlap coefficient (Szymkiewicz-Simpson)
                intersection = len(user_entities & other_entities)
                min_size = min(len(user_entities), len(other_entities))
                similarity = intersection / min_size if min_size > 0 else 0
            
            else:
                raise ValueError(f"Unknown similarity method: {method}")
            
            if similarity > 0:
                similarities.append({
                    'user_id': other_id,
                    'similarity': similarity,
                    'shared_entities': len(user_entities & other_entities)
                })
        
        df = pd.DataFrame(similarities)
        if df.empty:
            return df
        
        return df.sort_values('similarity', ascending=False).head(top_n).reset_index(drop=True)
    
    def find_related_entities(
        self,
        entity: str,
        top_n: int = 10,
        method: str = 'jaccard'
    ) -> pd.DataFrame:
        """
        Find entities with similar user audiences.
        
        Args:
            entity: Entity to find relations for
            top_n: Number of related entities to return
            method: Similarity metric ('jaccard', 'cosine', 'overlap')
        
        Returns:
            DataFrame with entity, similarity_score
        """
        if entity not in self.entities:
            return pd.DataFrame()
        
        entity_users = self.entity_users[entity]
        similarities = []
        
        for other_entity in self.entities:
            if other_entity == entity:
                continue
            
            other_users = self.entity_users[other_entity]
            
            if method == 'jaccard':
                intersection = len(entity_users & other_users)
                union = len(entity_users | other_users)
                similarity = intersection / union if union > 0 else 0
            
            elif method == 'cosine':
                intersection = len(entity_users & other_users)
                similarity = intersection / np.sqrt(len(entity_users) * len(other_users)) if len(entity_users) > 0 and len(other_users) > 0 else 0
            
            elif method == 'overlap':
                intersection = len(entity_users & other_users)
                min_size = min(len(entity_users), len(other_users))
                similarity = intersection / min_size if min_size > 0 else 0
            
            else:
                raise ValueError(f"Unknown similarity method: {method}")
            
            if similarity > 0:
                similarities.append({
                    'entity': other_entity,
                    'similarity': similarity,
                    'shared_users': len(entity_users & other_users)
                })
        
        df = pd.DataFrame(similarities)
        if df.empty:
            return df
        
        return df.sort_values('similarity', ascending=False).head(top_n).reset_index(drop=True)
    
    def detect_user_communities(
        self,
        method: str = 'louvain',
        resolution: float = 1.0
    ) -> Dict[str, int]:
        """
        Detect communities of users based on entity mention patterns.
        
        Args:
            method: Community detection algorithm ('louvain', 'label_prop')
            resolution: Resolution parameter for modularity (louvain only)
        
        Returns:
            Dictionary mapping user_id to community_id
        """
        user_graph = self.project_to_user_network(weighted=True)
        
        if len(user_graph.nodes) == 0:
            return {}
        
        if method == 'louvain':
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(user_graph, resolution=resolution)
            except ImportError:
                # Fall back to greedy modularity if python-louvain not available
                from networkx.algorithms import community as nx_community
                communities_gen = nx_community.greedy_modularity_communities(user_graph)
                communities = {}
                for i, comm in enumerate(communities_gen):
                    for user_id in comm:
                        communities[user_id] = i
        
        elif method == 'label_prop':
            from networkx.algorithms import community as nx_community
            communities_gen = nx_community.label_propagation_communities(user_graph)
            communities = {}
            for i, comm in enumerate(communities_gen):
                for user_id in comm:
                    communities[user_id] = i
        
        else:
            raise ValueError(f"Unknown community detection method: {method}")
        
        return communities
    
    def export_bipartite_graph(self, output_path: str):
        """Export bipartite graph to GraphML format."""
        nx.write_graphml(self.graph, output_path)
    
    def export_user_network(self, output_path: str, **kwargs):
        """Export projected user-user network to GraphML."""
        user_graph = self.project_to_user_network(**kwargs)
        nx.write_graphml(user_graph, output_path)
    
    def export_entity_network(self, output_path: str, **kwargs):
        """Export projected entity-entity network to GraphML."""
        entity_graph = self.project_to_entity_network(**kwargs)
        nx.write_graphml(entity_graph, output_path)
    
    def export_user_entity_matrix(self, output_path: str):
        """
        Export user-entity incidence matrix to CSV.
        Rows = users, Columns = entities, Values = mention counts.
        """
        # Create matrix
        users_sorted = sorted(self.users)
        entities_sorted = sorted(self.entities)
        
        matrix = np.zeros((len(users_sorted), len(entities_sorted)), dtype=int)
        
        for i, user_id in enumerate(users_sorted):
            for j, entity in enumerate(entities_sorted):
                if self.graph.has_edge(user_id, entity):
                    matrix[i, j] = self.graph[user_id][entity]['weight']
        
        # Convert to DataFrame
        df = pd.DataFrame(matrix, index=users_sorted, columns=entities_sorted)
        df.index.name = 'user_id'
        df.to_csv(output_path)


def load_from_kg_output(
    kg_dir: str,
    data_path: str,
    user_col: str,
    text_col: str
) -> UserEntityNetwork:
    """
    Convenience function to load UserEntityNetwork from KG output.
    
    Args:
        kg_dir: Directory containing kg_nodes.csv
        data_path: Path to original data CSV
        user_col: Name of user ID column
        text_col: Name of text column
    
    Returns:
        UserEntityNetwork instance
    """
    kg_dir = Path(kg_dir)
    
    # Load KG nodes
    nodes_path = kg_dir / "kg_nodes.csv"
    if not nodes_path.exists():
        raise FileNotFoundError(f"KG nodes not found: {nodes_path}")
    
    nodes_df = pd.read_csv(nodes_path)
    
    # Load original data
    data_df = pd.read_csv(data_path)
    
    if user_col not in data_df.columns:
        raise ValueError(f"User column '{user_col}' not found in data")
    if text_col not in data_df.columns:
        raise ValueError(f"Text column '{text_col}' not found in data")
    
    # Build network
    network = UserEntityNetwork()
    network.build_from_kg_nodes(
        nodes_df=nodes_df,
        texts=data_df[text_col].tolist(),
        user_ids=data_df[user_col].astype(str).tolist()
    )
    
    return network
