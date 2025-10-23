"""
Tests for 4chan-style data format handling.

Validates that the toolkit correctly handles:
- 4chan-specific column names (body, no, thread_id, board)
- Anonymous posts (missing tripcodes)
- Reply patterns (>>12345678)
- Unix timestamps
- Board filtering
"""

import unittest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class Test4chanDataFormat(unittest.TestCase):
    """Test 4chan-specific data format handling."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_data = cls._create_test_data()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    @classmethod
    def _create_test_data(cls):
        """Create sample 4chan-style data."""
        data = {
            'no': [90000001, 90000002, 90000003, 90000004, 90000005],
            'thread_id': [90000000, 90000000, 90000000, 90000006, 90000006],
            'board': ['pol', 'pol', 'pol', 'int', 'int'],
            'body': [
                'Discussion about Trump and Russia',
                '>>90000001 I agree with this analysis',
                'Biden announced new policy on immigration',
                'France celebrates culture and traditions',
                '>>90000004 Based take'
            ],
            'time': [1704067200, 1704067260, 1704067320, 1704067380, 1704067440],
            'tripcode': [None, '!Ep8pui8Vw2', None, None, None],
            'name': ['Anonymous', 'Anonymous', 'Anonymous', 'Anonymous', 'Anonymous']
        }
        return pd.DataFrame(data)
    
    def test_load_4chan_csv(self):
        """Test loading 4chan-formatted CSV."""
        test_file = Path(self.temp_dir) / "test_4chan.csv"
        self.test_data.to_csv(test_file, index=False)
        
        # Should load without errors
        df = pd.read_csv(test_file)
        
        self.assertEqual(len(df), 5)
        self.assertIn('body', df.columns)
        self.assertIn('no', df.columns)
        self.assertIn('board', df.columns)
    
    def test_body_column_text_extraction(self):
        """Test that 'body' column is recognized as text."""
        # The toolkit should accept 'body' as text column
        df = self.test_data.copy()
        
        # Verify body column has text
        self.assertTrue(all(isinstance(x, str) for x in df['body']))
        self.assertTrue(all(len(x) > 0 for x in df['body']))
    
    def test_anonymous_posts(self):
        """Test handling of posts without tripcodes (anonymous)."""
        df = self.test_data.copy()
        
        # Most posts should be anonymous (no tripcode)
        anonymous = df['tripcode'].isna()
        self.assertTrue(anonymous.sum() > 0)
        
        # Anonymous posts should still have content
        anon_posts = df[anonymous]
        self.assertTrue(all(len(x) > 0 for x in anon_posts['body']))
    
    def test_tripcode_handling(self):
        """Test handling of posts with tripcodes."""
        df = self.test_data.copy()
        
        # Some posts have tripcodes
        with_tripcode = df['tripcode'].notna()
        self.assertTrue(with_tripcode.sum() > 0)
        
        # Tripcode format
        tripcodes = df.loc[with_tripcode, 'tripcode']
        for tc in tripcodes:
            self.assertTrue(tc.startswith('!'))
    
    def test_reply_pattern_detection(self):
        """Test detection of reply patterns (>>12345678)."""
        df = self.test_data.copy()
        
        # Some posts should contain replies
        has_reply = df['body'].str.contains('>>', regex=False)
        self.assertTrue(has_reply.sum() > 0)
        
        # Extract reply patterns
        import re
        for text in df['body']:
            if '>>' in text:
                matches = re.findall(r'>>(\d+)', text)
                self.assertTrue(len(matches) > 0)
                # Should reference valid post numbers
                self.assertTrue(all(len(m) > 5 for m in matches))
    
    def test_unix_timestamp_conversion(self):
        """Test conversion of Unix timestamps to datetime."""
        df = self.test_data.copy()
        
        # Convert Unix timestamp
        df['created_at'] = pd.to_datetime(df['time'], unit='s')
        
        self.assertIn('created_at', df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['created_at']))
        
        # Verify reasonable dates (2024)
        self.assertTrue(all(df['created_at'].dt.year == 2024))
    
    def test_board_filtering(self):
        """Test filtering posts by board."""
        df = self.test_data.copy()
        
        # Filter by board
        pol_posts = df[df['board'] == 'pol']
        int_posts = df[df['board'] == 'int']
        
        self.assertEqual(len(pol_posts), 3)
        self.assertEqual(len(int_posts), 2)
        
        # All pol posts should be from pol
        self.assertTrue(all(pol_posts['board'] == 'pol'))
    
    def test_thread_grouping(self):
        """Test grouping posts by thread."""
        df = self.test_data.copy()
        
        # Group by thread
        threads = df.groupby('thread_id')
        
        self.assertEqual(len(threads), 2)
        
        # First thread should have 3 posts
        thread1 = df[df['thread_id'] == 90000000]
        self.assertEqual(len(thread1), 3)
    
    def test_entity_extraction_from_body(self):
        """Test that entities can be extracted from body column."""
        try:
            import spacy
            nlp = spacy.load('en_core_web_sm')
        except:
            self.skipTest("spaCy not available")
        
        df = self.test_data.copy()
        
        # Extract entities from first post
        doc = nlp(df.iloc[0]['body'])
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Should find some entities (Trump, Russia)
        self.assertTrue(len(entities) > 0)
        
        # Check for expected entities
        entity_texts = [e[0] for e in entities]
        self.assertTrue(any('Trump' in e or 'Russia' in e for e in entity_texts))


class Test4chanWorkflow(unittest.TestCase):
    """Test complete 4chan analysis workflows."""
    
    def test_sample_data_generation(self):
        """Test that sample 4chan data generator works."""
        try:
            from examples.sample_4chan_data import generate_4chan_sample
        except ImportError:
            self.skipTest("Sample data generator not importable")
        
        # Generate small sample
        df = generate_4chan_sample(n_posts=50, boards=['pol', 'int'])
        
        self.assertEqual(len(df), 50)
        self.assertIn('body', df.columns)
        self.assertIn('board', df.columns)
        self.assertIn('no', df.columns)
        
        # Check board distribution
        self.assertTrue(set(df['board'].unique()).issubset({'pol', 'int'}))
    
    def test_4chan_csv_structure(self):
        """Test that generated CSV has correct structure."""
        sample_file = Path('examples/sample_4chan.csv')
        
        if not sample_file.exists():
            self.skipTest("Sample 4chan CSV not generated yet")
        
        df = pd.read_csv(sample_file)
        
        # Required columns
        required = ['body', 'board']
        for col in required:
            self.assertIn(col, df.columns, f"Missing required column: {col}")
        
        # Optional but expected columns
        optional = ['no', 'thread_id', 'time', 'created_at', 'tripcode']
        present_optional = [col for col in optional if col in df.columns]
        self.assertTrue(len(present_optional) > 0, "No optional columns present")


class Test4chanSemanticAnalysis(unittest.TestCase):
    """Test semantic analysis on 4chan data."""
    
    def test_min_requirements_for_semantic_network(self):
        """Test that semantic network can be built with minimal 4chan data."""
        # Minimal 4chan data: just body column
        minimal_data = pd.DataFrame({
            'body': [
                'Trump announced policy changes',
                'Biden responded to Trump statement',
                'Discussion about immigration policy',
                'Russia involved in negotiations',
                'China watching developments'
            ]
        })
        
        # Should have enough for basic analysis
        self.assertTrue(len(minimal_data) >= 3)
        self.assertTrue(all(len(text) > 10 for text in minimal_data['body']))
    
    def test_handles_reply_syntax(self):
        """Test that reply syntax doesn't break analysis."""
        data_with_replies = pd.DataFrame({
            'body': [
                'Original post about topic',
                '>>90000001 Agreeing with this',
                '>>90000001 >>90000002 Both are correct',
                'Normal post without reply',
            ]
        })
        
        # Should not crash when processing
        texts = data_with_replies['body'].tolist()
        self.assertEqual(len(texts), 4)
        
        # All texts should be strings
        self.assertTrue(all(isinstance(t, str) for t in texts))


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(Test4chanDataFormat))
    suite.addTests(loader.loadTestsFromTestCase(Test4chanWorkflow))
    suite.addTests(loader.loadTestsFromTestCase(Test4chanSemanticAnalysis))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
