"""
Unit tests for transformer-enhanced semantic network functionality.
"""

import unittest
import numpy as np
import pandas as pd
from src.semantic.transformers_enhanced import (
    TransformerEmbeddings,
    TransformerSemanticNetwork,
    TransformerNER
)


class TestTransformerEmbeddings(unittest.TestCase):
    """Test TransformerEmbeddings class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedder = TransformerEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.test_texts = [
            "The cat sat on the mat.",
            "A feline rested on the rug.",
            "Python is a programming language.",
        ]
    
    def test_encode_single_text(self):
        """Test encoding a single text."""
        embedding = self.embedder.encode(["Hello world"])
        self.assertEqual(embedding.shape[0], 1)
        self.assertEqual(embedding.shape[1], 384)  # MiniLM dimension
    
    def test_encode_multiple_texts(self):
        """Test encoding multiple texts."""
        embeddings = self.embedder.encode(self.test_texts)
        self.assertEqual(embeddings.shape[0], len(self.test_texts))
        self.assertEqual(embeddings.shape[1], 384)
    
    def test_encode_empty_list(self):
        """Test encoding empty list."""
        with self.assertRaises(ValueError):
            self.embedder.encode([])
    
    def test_similarity_matrix_shape(self):
        """Test similarity matrix has correct shape."""
        sim_matrix = self.embedder.compute_similarity_matrix(self.test_texts)
        self.assertEqual(sim_matrix.shape, (len(self.test_texts), len(self.test_texts)))
    
    def test_similarity_matrix_diagonal(self):
        """Test that diagonal values are 1.0 (self-similarity)."""
        sim_matrix = self.embedder.compute_similarity_matrix(self.test_texts)
        np.testing.assert_array_almost_equal(
            np.diag(sim_matrix),
            np.ones(len(self.test_texts)),
            decimal=5
        )
    
    def test_similarity_matrix_symmetric(self):
        """Test that similarity matrix is symmetric."""
        sim_matrix = self.embedder.compute_similarity_matrix(self.test_texts)
        np.testing.assert_array_almost_equal(
            sim_matrix,
            sim_matrix.T,
            decimal=5
        )
    
    def test_similarity_range(self):
        """Test that similarities are in valid range [0, 1]."""
        sim_matrix = self.embedder.compute_similarity_matrix(self.test_texts)
        self.assertTrue(np.all(sim_matrix >= 0))
        self.assertTrue(np.all(sim_matrix <= 1))
    
    def test_semantic_similarity(self):
        """Test that semantically similar sentences have higher similarity."""
        sim_matrix = self.embedder.compute_similarity_matrix(self.test_texts)
        # Sentences 0 and 1 are semantically similar (cat/feline)
        # Should be more similar than 0 and 2 (cat vs Python)
        self.assertGreater(sim_matrix[0, 1], sim_matrix[0, 2])


class TestTransformerSemanticNetwork(unittest.TestCase):
    """Test TransformerSemanticNetwork class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.builder = TransformerSemanticNetwork(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.test_docs = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Climate change affects global weather patterns.",
            "Renewable energy sources include solar and wind power.",
        ]
    
    def test_build_document_network_returns_dataframe(self):
        """Test that document network returns a DataFrame."""
        edges = self.builder.build_document_network(
            self.test_docs,
            similarity_threshold=0.0,
            top_k=10
        )
        self.assertIsInstance(edges, pd.DataFrame)
    
    def test_build_document_network_columns(self):
        """Test that edges DataFrame has required columns."""
        edges = self.builder.build_document_network(
            self.test_docs,
            similarity_threshold=0.0,
            top_k=10
        )
        required_columns = ['source', 'target', 'similarity']
        for col in required_columns:
            self.assertIn(col, edges.columns)
    
    def test_document_network_threshold(self):
        """Test that threshold filters edges correctly."""
        edges_low = self.builder.build_document_network(
            self.test_docs,
            similarity_threshold=0.1,
            top_k=10
        )
        edges_high = self.builder.build_document_network(
            self.test_docs,
            similarity_threshold=0.5,
            top_k=10
        )
        # Higher threshold should result in fewer edges
        self.assertLessEqual(len(edges_high), len(edges_low))
    
    def test_document_network_topk(self):
        """Test that top_k limits edges per document."""
        edges = self.builder.build_document_network(
            self.test_docs,
            similarity_threshold=0.0,
            top_k=2
        )
        # Count edges per source
        edge_counts = edges['source'].value_counts()
        # No source should have more than top_k edges
        self.assertTrue(all(edge_counts <= 2))
    
    def test_document_network_no_self_loops(self):
        """Test that network has no self-loops."""
        edges = self.builder.build_document_network(
            self.test_docs,
            similarity_threshold=0.0,
            top_k=10
        )
        # Check no edge where source == target
        self.assertTrue(all(edges['source'] != edges['target']))
    
    def test_build_term_network(self):
        """Test building term network."""
        terms = ["artificial intelligence", "machine learning", "climate change"]
        edges = self.builder.build_term_network(
            terms,
            similarity_threshold=0.3,
            top_k=5
        )
        self.assertIsInstance(edges, pd.DataFrame)
        self.assertIn('source', edges.columns)
        self.assertIn('target', edges.columns)
    
    def test_term_network_uses_term_names(self):
        """Test that term network uses actual term strings."""
        terms = ["AI", "ML", "DL"]
        edges = self.builder.build_term_network(
            terms,
            similarity_threshold=0.0,
            top_k=5
        )
        # Sources and targets should be from the term list
        for term in edges['source'].unique():
            self.assertIn(term, terms)
        for term in edges['target'].unique():
            self.assertIn(term, terms)
    
    def test_empty_document_list(self):
        """Test handling of empty document list."""
        edges = self.builder.build_document_network(
            [],
            similarity_threshold=0.5,
            top_k=5
        )
        self.assertEqual(len(edges), 0)
    
    def test_single_document(self):
        """Test handling of single document."""
        edges = self.builder.build_document_network(
            ["Single document"],
            similarity_threshold=0.5,
            top_k=5
        )
        self.assertEqual(len(edges), 0)  # No edges possible with 1 doc


class TestTransformerNER(unittest.TestCase):
    """Test TransformerNER class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use small model for testing
        self.ner = TransformerNER(model_name="en_core_web_sm")
        self.test_text = "Apple Inc. is headquartered in Cupertino, California. Tim Cook is the CEO."
    
    def test_extract_entities_returns_dataframe(self):
        """Test that entity extraction returns a DataFrame."""
        entities = self.ner.extract_entities(self.test_text)
        self.assertIsInstance(entities, pd.DataFrame)
    
    def test_extract_entities_columns(self):
        """Test that entities DataFrame has required columns."""
        entities = self.ner.extract_entities(self.test_text)
        required_columns = ['text', 'label', 'start', 'end']
        for col in required_columns:
            self.assertIn(col, entities.columns)
    
    def test_extract_entities_finds_organizations(self):
        """Test that NER finds organization entities."""
        entities = self.ner.extract_entities(self.test_text)
        org_entities = entities[entities['label'] == 'ORG']
        self.assertGreater(len(org_entities), 0)
        # Should find "Apple Inc."
        self.assertTrue(any('Apple' in text for text in org_entities['text']))
    
    def test_extract_entities_finds_locations(self):
        """Test that NER finds location entities."""
        entities = self.ner.extract_entities(self.test_text)
        loc_entities = entities[entities['label'].isin(['GPE', 'LOC'])]
        self.assertGreater(len(loc_entities), 0)
    
    def test_extract_entities_finds_persons(self):
        """Test that NER finds person entities."""
        entities = self.ner.extract_entities(self.test_text)
        person_entities = entities[entities['label'] == 'PERSON']
        self.assertGreater(len(person_entities), 0)
        # Should find "Tim Cook"
        self.assertTrue(any('Cook' in text for text in person_entities['text']))
    
    def test_empty_text(self):
        """Test handling of empty text."""
        entities = self.ner.extract_entities("")
        self.assertEqual(len(entities), 0)
    
    def test_text_without_entities(self):
        """Test text with no named entities."""
        entities = self.ner.extract_entities("The quick brown fox jumps.")
        # May or may not find entities depending on model, but shouldn't crash
        self.assertIsInstance(entities, pd.DataFrame)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def test_full_pipeline_document_network(self):
        """Test complete pipeline from texts to network."""
        texts = [
            "AI and machine learning are transforming technology.",
            "Deep learning models require large amounts of data.",
            "Climate change is affecting global ecosystems.",
        ]
        
        # Build network
        builder = TransformerSemanticNetwork()
        edges = builder.build_document_network(
            texts,
            similarity_threshold=0.2,
            top_k=5
        )
        
        # Verify output
        self.assertIsInstance(edges, pd.DataFrame)
        self.assertGreater(len(edges), 0)
        self.assertTrue(all(edges['similarity'] >= 0.2))
    
    def test_embedding_consistency(self):
        """Test that embeddings are consistent across calls."""
        embedder = TransformerEmbeddings()
        text = "Test sentence for consistency"
        
        emb1 = embedder.encode([text])
        emb2 = embedder.encode([text])
        
        np.testing.assert_array_almost_equal(emb1, emb2, decimal=5)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    unittest.main()
