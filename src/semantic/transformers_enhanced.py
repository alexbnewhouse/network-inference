"""
Enhanced Transformer-based Text Analysis
- Sentence embeddings via transformers (BERT, RoBERTa, etc.)
- Semantic similarity computation
- Enhanced entity recognition with transformer models
- Topic modeling with BERTopic
"""
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm


class TransformerEmbeddings:
    """
    Generate contextualized embeddings using transformer models.
    Supports various models from Hugging Face transformers library.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize transformer embeddings.
        
        Args:
            model_name: Hugging Face model identifier
            device: 'cpu', 'cuda', or 'mps' for Apple Silicon
        """
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            raise ValueError("Input list is empty")
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
    
    def compute_similarity_matrix(self, embeddings_or_texts) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.
        
        Args:
            embeddings_or_texts: Either embedding matrix (n_samples, embedding_dim) 
                                or list of text strings to encode first
            
        Returns:
            Similarity matrix (n_samples, n_samples)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # If input is a list of strings, encode them first
        if isinstance(embeddings_or_texts, (list, tuple)):
            embeddings = self.encode(embeddings_or_texts, show_progress=False)
        else:
            embeddings = embeddings_or_texts
            
        return cosine_similarity(embeddings)


class BERTopicClustering:
    """
    Topic modeling using BERTopic with transformer embeddings.
    """
    
    def __init__(self, embedding_model: Optional[str] = None, nr_topics: Optional[int] = None):
        """
        Initialize BERTopic model.
        
        Args:
            embedding_model: Sentence transformer model name
            nr_topics: Number of topics to extract (None = automatic)
        """
        from bertopic import BERTopic
        self.model = BERTopic(
            embedding_model=embedding_model or "sentence-transformers/all-MiniLM-L6-v2",
            nr_topics=nr_topics,
            verbose=True
        )
        
    def fit_transform(self, texts: List[str]) -> Tuple[List[int], pd.DataFrame]:
        """
        Fit topic model and transform texts.
        
        Args:
            texts: List of documents
            
        Returns:
            topics: List of topic assignments
            topic_info: DataFrame with topic information
        """
        topics, probs = self.model.fit_transform(texts)
        topic_info = self.model.get_topic_info()
        return topics, topic_info
    
    def get_topics(self) -> dict:
        """Get all topics with their top words."""
        return self.model.get_topics()
    
    def visualize_topics(self):
        """Generate interactive topic visualization."""
        return self.model.visualize_topics()


class TransformerSemanticNetwork:
    """
    Build semantic networks using transformer embeddings instead of co-occurrence.
    Creates edges based on semantic similarity between documents or terms.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize transformer-based semantic network builder.
        
        Args:
            model_name: Hugging Face model identifier
            device: Computing device
        """
        self.embedder = TransformerEmbeddings(model_name, device)
        
    def build_document_network(
        self,
        texts: List[str],
        similarity_threshold: float = 0.5,
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Build document similarity network.
        
        Args:
            texts: List of documents
            similarity_threshold: Minimum similarity for edge creation
            top_k: Keep only top-k most similar documents per document
            
        Returns:
            DataFrame with columns: source, target, similarity
        """
        # Handle empty or single document
        if len(texts) == 0:
            return pd.DataFrame(columns=['source', 'target', 'similarity'])
        if len(texts) == 1:
            return pd.DataFrame(columns=['source', 'target', 'similarity'])
            
        print("Encoding documents...")
        embeddings = self.embedder.encode(texts, show_progress=True)
        
        print("Computing similarity matrix...")
        sim_matrix = self.embedder.compute_similarity_matrix(embeddings)
        
        edges = []
        for i in range(len(texts)):
            similarities = sim_matrix[i]
            
            # Get top-k most similar (excluding self)
            if top_k:
                sorted_indices = np.argsort(similarities)[::-1]
                # Skip first (self) and take next top_k
                candidates = sorted_indices[1:top_k+1]
            else:
                candidates = range(len(texts))
                
            for j in candidates:
                if i != j and similarities[j] >= similarity_threshold:
                    edges.append({
                        "source": i,
                        "target": j,
                        "similarity": float(similarities[j])
                    })
                    
        return pd.DataFrame(edges)
    
    def build_term_network(
        self,
        terms: List[str],
        similarity_threshold: float = 0.5,
        top_k: Optional[int] = 20
    ) -> pd.DataFrame:
        """
        Build term similarity network based on contextualized embeddings.
        
        Args:
            terms: List of terms/tokens
            similarity_threshold: Minimum similarity for edge creation
            top_k: Keep only top-k most similar terms per term
            
        Returns:
            DataFrame with columns: source, target, similarity
        """
        # Handle empty or single term
        if len(terms) == 0:
            return pd.DataFrame(columns=['source', 'target', 'similarity'])
        if len(terms) == 1:
            return pd.DataFrame(columns=['source', 'target', 'similarity'])
            
        print("Encoding terms...")
        embeddings = self.embedder.encode(terms, show_progress=True)
        
        print("Computing similarity matrix...")
        sim_matrix = self.embedder.compute_similarity_matrix(embeddings)
        
        edges = []
        for i in range(len(terms)):
            similarities = sim_matrix[i]
            
            # Get top-k most similar (excluding self)
            sorted_indices = np.argsort(similarities)[::-1]
            candidates = sorted_indices[1:top_k+1] if top_k else sorted_indices[1:]
                
            for j in candidates:
                if similarities[j] >= similarity_threshold:
                    edges.append({
                        "source": terms[i],  # Use actual term names
                        "target": terms[j],
                        "similarity": float(similarities[j])
                    })
                    
        return pd.DataFrame(edges)


class TransformerNER:
    """
    Enhanced Named Entity Recognition using transformer models.
    """
    
    def __init__(self, model_name: str = "en_core_web_trf"):
        """
        Initialize transformer-based NER.
        
        Args:
            model_name: spaCy transformer model name
        """
        import spacy
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Downloading {model_name}...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)
            
    def extract_entities(self, text: str) -> pd.DataFrame:
        """
        Extract entities from text using transformer model.
        
        Args:
            text: Input text string
            
        Returns:
            DataFrame with columns: text, label, start, end
        """
        if not text:
            return pd.DataFrame(columns=['text', 'label', 'start', 'end'])
            
        doc = self.nlp(text)
        
        if not doc.ents:
            return pd.DataFrame(columns=['text', 'label', 'start', 'end'])
        
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
            
        return pd.DataFrame(entities)
