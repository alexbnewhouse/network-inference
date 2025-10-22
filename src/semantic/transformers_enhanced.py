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
        Compute pairwise cosine similarity matrix in [0, 1].

        Args:
            embeddings_or_texts: Either embedding matrix (n_samples, embedding_dim)
                                 or list/tuple of text strings to encode first

        Returns:
            Similarity matrix (n_samples, n_samples) with values in [0, 1]
        """
        # If input is a list/tuple of strings, encode them first
        if isinstance(embeddings_or_texts, (list, tuple)):
            embeddings = self.encode(list(embeddings_or_texts), show_progress=False)
        else:
            embeddings = embeddings_or_texts

        # Normalize embeddings to unit norm to compute cosine similarity via dot product
        # Add small epsilon to avoid division by zero in unlikely degenerate cases
        eps = 1e-12
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        emb_norm = embeddings / norms

        # Cosine similarity in [-1, 1]
        sim = emb_norm @ emb_norm.T

        # Numerical stability: clip slight overshoots to [-1, 1]
        sim = np.clip(sim, -1.0, 1.0)

        # Map to [0, 1] as required by tests and downstream expectations
        sim01 = (sim + 1.0) / 2.0

        # Final safety clip to [0, 1]
        sim01 = np.clip(sim01, 0.0, 1.0)

        return sim01


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
        top_k: Optional[int] = None,
        use_faiss: bool = False,  # Disabled by default - FAISS unstable on Python 3.12+
        batch_size: int = 10000
    ) -> pd.DataFrame:
        """
        Build document similarity network.
        
        Args:
            texts: List of documents
            similarity_threshold: Minimum similarity for edge creation
            top_k: Keep only top-k most similar documents per document
            use_faiss: Use FAISS for efficient approximate nearest neighbors 
                      (disabled by default - unstable on Python 3.12+)
            batch_size: Batch size for similarity computation (default=10000)
            
        Returns:
            DataFrame with columns: source, target, similarity
            
        Note:
            For large datasets (>10K docs), batch processing is automatically
            used. FAISS is available but optional due to Python 3.12+ compatibility issues.
        """
        # Handle empty or single document
        if len(texts) == 0:
            return pd.DataFrame(columns=['source', 'target', 'similarity'])
        if len(texts) == 1:
            return pd.DataFrame(columns=['source', 'target', 'similarity'])
            
        print("Encoding documents...")
        embeddings = self.embedder.encode(texts, show_progress=True)
        
        n_docs = len(texts)
        
        # For large datasets (>10K), optionally use FAISS if explicitly requested
        # Note: FAISS has compatibility issues with Python 3.12+ so disabled by default
        if use_faiss and n_docs > 10000:
            try:
                import faiss
                print(f"Using FAISS for efficient similarity search ({n_docs:,} documents)...")
                print("⚠️  Note: FAISS may be unstable on Python 3.12+")
                
                # Make a copy to avoid modifying original
                embeddings_faiss = embeddings.copy()
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings_faiss)
                
                # Build FAISS index
                dimension = embeddings_faiss.shape[1]
                index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity after normalization
                index.add(embeddings_faiss.astype(np.float32))
                
                # Search for top-k similar documents
                k = min(top_k + 1 if top_k else 100, n_docs)  # +1 to account for self
                distances, indices = index.search(embeddings_faiss.astype(np.float32), k)
                
                print("Building edge list...")
                edges = []
                for i in range(n_docs):
                    for j_idx in range(1, k):  # Skip 0 (self)
                        j = indices[i, j_idx]
                        sim = float(distances[i, j_idx])
                        
                        if sim >= similarity_threshold:
                            edges.append({
                                "source": i,
                                "target": j,
                                "similarity": sim
                            })
                        elif top_k is None:
                            # If no top_k limit and below threshold, stop
                            break
                
                print(f"Found {len(edges):,} edges")
                return pd.DataFrame(edges)
                
            except ImportError:
                print("⚠️  FAISS not available. Using memory-efficient batch processing instead.")
                print("   This is recommended for Python 3.12+ due to FAISS compatibility issues.")
                use_faiss = False
            except Exception as e:
                print(f"⚠️  FAISS error: {e}")
                print("   Falling back to memory-efficient batch processing (recommended)...")
                use_faiss = False
        
        # For smaller datasets or if FAISS unavailable, use batch processing
        if n_docs <= 10000:
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
        else:
            # Large dataset without FAISS: use memory-efficient batch processing
            print(f"Using memory-efficient batch processing ({n_docs:,} documents)...")
            edges = []
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_norm = embeddings / np.maximum(norms, 1e-12)
            
            # Process in batches to avoid memory issues
            for i in range(0, n_docs, batch_size):
                batch_end = min(i + batch_size, n_docs)
                print(f"Processing batch {i//batch_size + 1}/{(n_docs + batch_size - 1)//batch_size}...")
                
                # Compute similarities for this batch
                batch_sims = embeddings_norm[i:batch_end] @ embeddings_norm.T
                
                for batch_idx, doc_idx in enumerate(range(i, batch_end)):
                    similarities = batch_sims[batch_idx]
                    
                    # Get top-k most similar (excluding self)
                    if top_k:
                        sorted_indices = np.argsort(similarities)[::-1]
                        # Skip self and take next top_k
                        candidates = [idx for idx in sorted_indices[:top_k+10] if idx != doc_idx][:top_k]
                    else:
                        candidates = [idx for idx in range(n_docs) if idx != doc_idx]
                    
                    for j in candidates:
                        sim = float(similarities[j])
                        if sim >= similarity_threshold:
                            edges.append({
                                "source": doc_idx,
                                "target": j,
                                "similarity": sim
                            })
        
            print(f"Found {len(edges):,} edges")
                    
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
