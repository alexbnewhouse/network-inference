# Tests

Unit and integration tests for the network inference toolkit.

## Running Tests

### Run All Tests

```bash
cd /path/to/network_inference
python3 -m unittest discover tests
```

### Run Specific Test File

```bash
python3 tests/test_basic.py
```

### Run Specific Test Class

```bash
python3 -m unittest tests.test_basic.TestTransformerEmbeddings
```

### Run With Verbose Output

```bash
python3 -m unittest discover tests -v
```

## Test Structure

```
tests/
├── README.md (this file)
├── test_basic.py - Basic functionality tests
└── test_transformers.py - Comprehensive transformer tests (advanced)
```

## Test Coverage

### test_basic.py

**TestImports**
- ✅ Import transformer modules
- ✅ Import core semantic network modules

**TestTransformerEmbeddings**
- ✅ Encoding shape verification
- ✅ Similarity matrix dimensions
- ✅ Similarity matrix properties (diagonal, symmetry, range)

**TestTransformerSemanticNetwork**
- ✅ Document network creation
- ✅ Threshold filtering
- ✅ No self-loops validation
- ✅ Term network creation

## Prerequisites

Install dependencies before running tests:

```bash
pip install -r requirements.txt
```

Minimum required packages:
- `sentence-transformers` - For transformer embeddings
- `scikit-learn` - For similarity computation
- `pandas` - For data handling
- `numpy` - For numerical operations
- `polars` - For efficient data processing

## Writing New Tests

### Test Template

```python
import unittest
from src.semantic.your_module import YourClass

class TestYourFeature(unittest.TestCase):
    """Test YourClass functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once."""
        cls.instance = YourClass()
        cls.test_data = [...]
    
    def test_basic_functionality(self):
        """Test that basic feature works."""
        result = self.instance.method(self.test_data)
        self.assertIsNotNone(result)
    
    def test_edge_case(self):
        """Test edge case handling."""
        result = self.instance.method([])
        self.assertEqual(len(result), 0)

if __name__ == '__main__':
    unittest.main()
```

### Best Practices

1. **One test per behavior** - Each test should verify one specific behavior
2. **Descriptive names** - Use `test_feature_does_something_when_condition`
3. **Arrange-Act-Assert** - Structure tests clearly:
   ```python
   # Arrange
   data = prepare_test_data()
   
   # Act
   result = function_under_test(data)
   
   # Assert
   self.assertEqual(result, expected)
   ```
4. **Use fixtures** - Set up common test data in `setUp()` or `setUpClass()`
5. **Test edge cases** - Empty inputs, invalid data, boundary conditions
6. **Mock expensive operations** - Use mocks for slow operations like model loading

## Troubleshooting

### ImportError: No module named 'sentence_transformers'

```bash
pip install sentence-transformers
```

### Test hangs or runs very slowly

Some tests download transformer models on first run. Subsequent runs will be faster.

To skip slow tests:
```python
@unittest.skip("Slow test - run manually")
def test_large_dataset(self):
    ...
```

### CUDA/GPU Issues

Tests default to CPU. To test GPU functionality:
```python
embedder = TransformerEmbeddings(device='cuda')
```

### Path Issues

Make sure you're running tests from the project root:
```bash
cd /path/to/network_inference
python3 tests/test_basic.py
```

## Continuous Integration

For CI/CD pipelines, create a `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python -m unittest discover tests
```

## Future Test Coverage

Planned test additions:
- [ ] Integration tests for complete pipelines
- [ ] Performance benchmarks
- [ ] Network quality metrics validation
- [ ] Edge case handling for malformed data
- [ ] Multilingual model tests
- [ ] GPU acceleration tests
- [ ] Memory usage tests for large datasets

## Contributing

When adding new features, please add corresponding tests:

1. Create test file: `tests/test_your_feature.py`
2. Write comprehensive tests covering:
   - Happy path scenarios
   - Edge cases
   - Error handling
3. Ensure all tests pass before submitting PR
4. Aim for >80% code coverage

See [CONTRIBUTING.md](../CONTRIBUTING.md) for full guidelines.

---

**Test Status**: ✅ All core tests passing (9/9)

Last updated: October 17, 2025
