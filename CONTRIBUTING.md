# Contributing to Network Inference

Thank you for your interest in contributing to Network Inference! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/network-inference.git
   cd network-inference
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Environment

```bash
# Use Python 3.12
python3.12 -m venv .venv312
source .venv312/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_semantic_pipeline.py
```

### Code Style

We follow PEP 8 guidelines with a few modifications:

- Line length: 100 characters (not 80)
- Use Black for formatting: `black src/`
- Use type hints where practical
- Docstrings: Google style

```python
def example_function(param1: str, param2: int = 5) -> dict:
    """
    Brief description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 5)
        
    Returns:
        Dictionary with results
        
    Example:
        >>> example_function("test", 10)
        {'result': 'test10'}
    """
    return {'result': param1 + str(param2)}
```

## Areas for Contribution

### High Priority

- **Performance optimizations**: Faster tokenization, better memory management
- **Additional transformer models**: Support for more embedding models
- **Entity linking**: Connect entities to knowledge bases (Wikidata, DBpedia)
- **Relation extraction**: More sophisticated pattern matching
- **Temporal metrics**: Network evolution statistics
- **Visualization improvements**: Better interactive visualizations

### Medium Priority

- **Additional languages**: Non-English NLP support
- **Streaming processing**: Handle datasets too large for memory
- **Database backends**: SQLite/PostgreSQL for large networks
- **API server**: REST API for network analysis
- **Web interface**: Simple UI for running pipelines

### Documentation

- **Tutorial notebooks**: Step-by-step examples for specific use cases
- **Video tutorials**: Screencasts demonstrating features
- **API documentation**: Auto-generated docs with Sphinx
- **Performance benchmarks**: Comparative timing studies

## Pull Request Process

1. **Update tests**: Add tests for new features
2. **Update documentation**: Update README and docstrings
3. **Run tests locally**: Ensure all tests pass
4. **Check code style**: Run `black` and `flake8`
5. **Write clear commit messages**: 
   ```
   Add transformer-based community detection
   
   - Implement BERTopic integration
   - Add CLI for topic modeling
   - Update documentation with examples
   ```
6. **Submit PR**: Include description of changes and motivation

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code formatted with Black
- [ ] No new linting errors
- [ ] Commit messages are clear
- [ ] PR description explains the change

## Code Review Process

- Maintainers will review PRs within 1-2 weeks
- Feedback will be provided via GitHub comments
- Once approved, maintainers will merge

## Issue Reporting

### Bug Reports

Include:
- Python version
- Operating system
- Full error traceback
- Minimal reproducible example
- Expected vs actual behavior

### Feature Requests

Include:
- Use case description
- Proposed API/interface
- Example usage
- Alternatives considered

## Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code contributions

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on improving the project
- Help newcomers get started

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Git commit history

Thank you for contributing to Network Inference!
