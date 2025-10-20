"""
Simple unit tests for core semantic network functionality.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestImports(unittest.TestCase):
    """Test that all modules can be imported."""
    
    def test_import_transformers_enhanced(self):
        """Test importing transformer modules."""
        try:
            from src.semantic.transformers_enhanced import TransformerEmbeddings
            from src.semantic.transformers_enhanced import TransformerSemanticNetwork
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import transformers_enhanced: {e}")
    
    def test_import_build_semantic_network(self):
        """Test importing core semantic network module."""
        try:
            from src.semantic import build_semantic_network
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import build_semantic_network: {e}")


class TestTransformerEmbeddings(unittest.TestCase):
    """Test TransformerEmbeddings basic functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        try:
            from src.semantic.transformers_enhanced import TransformerEmbeddings
            cls.embedder = TransformerEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            cls.test_texts = [
                "The cat sat on the mat.",
                "A dog played in the park.",
                "Python is a programming language.",
            ]
        except Exception as e:
            cls.skipTest(cls, f"Setup failed: {e}")
    
    def test_encode_returns_correct_shape(self):
        """Test that encoding returns expected shape."""
        embeddings = self.embedder.encode(self.test_texts, show_progress=False)
        self.assertEqual(embeddings.shape[0], len(self.test_texts))
        self.assertEqual(embeddings.shape[1], 384)  # MiniLM dimension
    
    def test_similarity_matrix_shape(self):
        """Test similarity matrix dimensions."""
        embeddings = self.embedder.encode(self.test_texts, show_progress=False)
        sim_matrix = self.embedder.compute_similarity_matrix(embeddings)
        expected_shape = (len(self.test_texts), len(self.test_texts))
        self.assertEqual(sim_matrix.shape, expected_shape)
    
    def test_similarity_matrix_properties(self):
        """Test basic properties of similarity matrix."""
        import numpy as np
        embeddings = self.embedder.encode(self.test_texts, show_progress=False)
        sim_matrix = self.embedder.compute_similarity_matrix(embeddings)
        
        # Diagonal should be 1.0 (self-similarity)
        diagonal = np.diag(sim_matrix)
        np.testing.assert_array_almost_equal(
            diagonal, 
            np.ones(len(self.test_texts)),
            decimal=5
        )
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(sim_matrix, sim_matrix.T, decimal=5)
        
        # Values should be in [-1, 1] (cosine similarity range)
        self.assertTrue(np.all(sim_matrix >= -1))
        self.assertTrue(np.all(sim_matrix <= 1))


class TestTransformerSemanticNetwork(unittest.TestCase):
    """Test TransformerSemanticNetwork basic functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from src.semantic.transformers_enhanced import TransformerSemanticNetwork
            cls.builder = TransformerSemanticNetwork()
            cls.test_docs = [
                "Machine learning is part of AI.",
                "Deep learning uses neural networks.",
                "Climate change affects weather.",
                "Solar power is renewable energy.",
            ]
        except Exception as e:
            cls.skipTest(cls, f"Setup failed: {e}")
    
    def test_build_document_network_basic(self):
        """Test basic document network creation."""
        import pandas as pd
        edges = self.builder.build_document_network(
            self.test_docs,
            similarity_threshold=0.0,
            top_k=5
        )
        self.assertIsInstance(edges, pd.DataFrame)
        self.assertIn('source', edges.columns)
        self.assertIn('target', edges.columns)
        self.assertIn('similarity', edges.columns)
    
    def test_build_document_network_threshold(self):
        """Test that threshold parameter works."""
        edges_low = self.builder.build_document_network(
            self.test_docs,
            similarity_threshold=0.1,
            top_k=10
        )
        edges_high = self.builder.build_document_network(
            self.test_docs,
            similarity_threshold=0.9,
            top_k=10
        )
        # Higher threshold should give fewer or equal edges
        self.assertLessEqual(len(edges_high), len(edges_low))
    
    def test_no_self_loops(self):
        """Test that network doesn't contain self-loops."""
        edges = self.builder.build_document_network(
            self.test_docs,
            similarity_threshold=0.0,
            top_k=10
        )
        # No edge should connect a node to itself
        self.assertTrue(all(edges['source'] != edges['target']))
    
    def test_build_term_network(self):
        """Test term network creation."""
        import pandas as pd
        terms = ["artificial intelligence", "machine learning", "climate change"]
        edges = self.builder.build_term_network(
            terms,
            similarity_threshold=0.3,
            top_k=5
        )
        self.assertIsInstance(edges, pd.DataFrame)


def run_quick_tests():
    """Run a quick subset of tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add only import and basic tests
    suite.addTests(loader.loadTestsFromTestCase(TestImports))
    suite.addTests(loader.loadTestsFromTestCase(TestTransformerEmbeddings))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    unittest.main()
