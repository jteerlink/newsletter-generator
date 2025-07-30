# Contributing to Newsletter Generator

Thank you for your interest in contributing to the Newsletter Generator project! This document provides guidelines and setup instructions for developers.

## 🚀 Development Setup

### Prerequisites

- **Python 3.10+** (required for modern CrewAI)
- **Git** for version control
- **Ollama** for local LLM models
- **Node.js** (optional, for development tools)

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd newsletter-generator
   ```

2. **Create a virtual environment:**
   ```bash
   # Using conda (recommended)
   conda create -n news_env python=3.10 -y
   conda activate news_env
   
   # Using venv
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   # Install main dependencies
   pip install -r requirements.txt
   
   # Install development dependencies
   pip install pytest pytest-cov pytest-asyncio pytest-mock
   pip install black flake8 mypy isort
   
   # Install Streamlit dependencies (for web interface)
   pip install -r streamlit/requirements.txt
   ```

4. **Install Ollama and models:**
   ```bash
   # Install Ollama (follow instructions at https://ollama.com/)
   # Then pull required models:
   ollama pull llama3
   ollama pull gemma3n
   ollama pull deepseek-r1
   ```

5. **Setup environment variables:**
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit .env with your configuration
   nano .env
   ```

## 🧪 Testing

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run specific test categories
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
pytest tests/agents/ -v                  # Agent tests
pytest tests/core/ -v                    # Core functionality tests

# Run tests with markers
pytest tests/ -m "not slow"              # Skip slow tests
pytest tests/ -m "integration"           # Only integration tests

# Run tests in parallel (if pytest-xdist installed)
pytest tests/ -n auto
```

### Test Coverage

We aim for **90%+ test coverage**. To check coverage:

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # On macOS
# or
start htmlcov/index.html  # On Windows
```

### Writing Tests

- Follow the naming convention: `test_*.py` for test files
- Use descriptive test names: `test_function_name_scenario`
- Use fixtures from `tests/conftest.py` for common test data
- Mock external dependencies (APIs, databases, etc.)
- Test both success and error cases

Example test structure:
```python
import pytest
from unittest.mock import Mock, patch

def test_function_success():
    """Test successful function execution."""
    # Arrange
    input_data = "test input"
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == "expected output"

def test_function_error_handling():
    """Test function error handling."""
    with pytest.raises(ValueError):
        function_under_test(None)
```

## 🔧 Code Quality

### Linting and Formatting

We use several tools to maintain code quality:

```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Check code style with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/

# Run all quality checks
make quality-check  # If Makefile is available
```

### Pre-commit Hooks

Install pre-commit hooks to automatically check code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Code Style Guidelines

- **Line length**: 88 characters (Black default)
- **Docstrings**: Use Google style docstrings
- **Type hints**: Use type hints for all function parameters and return values
- **Imports**: Group imports: standard library, third-party, local
- **Naming**: Use descriptive names, follow PEP 8

Example:
```python
from typing import List, Optional
import logging
from pathlib import Path

import requests
from pydantic import BaseModel

from .core.exceptions import NewsletterGeneratorError


def process_newsletter_content(
    content: str, 
    max_length: Optional[int] = None
) -> List[str]:
    """Process newsletter content and return sections.
    
    Args:
        content: Raw newsletter content
        max_length: Maximum length per section
        
    Returns:
        List of processed content sections
        
    Raises:
        NewsletterGeneratorError: If content processing fails
    """
    # Implementation here
    pass
```

## 📝 Development Workflow

### Branching Strategy

- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/***: Feature development branches
- **bugfix/***: Bug fix branches
- **hotfix/***: Critical production fixes

### Commit Guidelines

Use conventional commit messages:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(agents): add new research agent for technical content

fix(core): resolve LLM timeout issues

docs(readme): update installation instructions

test(integration): add end-to-end workflow tests
```

### Pull Request Process

1. **Create a feature branch** from `develop`
2. **Make your changes** following the coding guidelines
3. **Write tests** for new functionality
4. **Update documentation** if needed
5. **Run all tests** and quality checks
6. **Create a pull request** to `develop`
7. **Request review** from maintainers
8. **Address feedback** and make necessary changes
9. **Merge** when approved

### PR Checklist

Before submitting a PR, ensure:

- [ ] All tests pass
- [ ] Code coverage is maintained or improved
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] No new linting errors
- [ ] Type checking passes
- [ ] Performance impact is considered

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment details**: OS, Python version, dependency versions
2. **Steps to reproduce**: Clear, step-by-step instructions
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Error messages**: Full error traceback
6. **Additional context**: Any relevant information

## 💡 Feature Requests

When requesting features, please include:

1. **Problem description**: What problem does this solve?
2. **Proposed solution**: How should it work?
3. **Use cases**: Who would benefit from this?
4. **Implementation ideas**: Any technical suggestions?

## 🔍 Code Analysis

### Running Code Analysis

```bash
# Analyze codebase metrics
python scripts/analyze_codebase.py

# Check for code duplication
# (Install radon or similar tool)
radon cc src/ -a

# Check for security issues
# (Install bandit)
bandit -r src/
```

### Performance Testing

```bash
# Run performance benchmarks
pytest tests/performance/ -v

# Profile specific functions
python -m cProfile -o profile.stats src/main.py
```

## 📚 Documentation

### Writing Documentation

- Use clear, concise language
- Include code examples
- Keep documentation up to date with code changes
- Use proper markdown formatting

### Documentation Structure

- **README.md**: Project overview and quick start
- **docs/**: Detailed documentation
- **docstrings**: Inline code documentation
- **examples/**: Usage examples

## 🤝 Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Code Review**: Request reviews from maintainers
- **Documentation**: Check existing docs first

## 📋 Development Tools

### Recommended IDE Setup

**VS Code:**
- Python extension
- Pylance for type checking
- Black formatter extension
- Flake8 linter extension

**PyCharm:**
- Configure Black as external tool
- Enable type checking
- Setup test runner

### Useful Commands

```bash
# Quick development server
python -m streamlit run streamlit/app_hybrid_minimal.py

# Run specific agent
python -c "from src.agents.agents import ResearchAgent; agent = ResearchAgent(); print(agent.execute_task('test'))"

# Check system health
python scripts/check_system.py

# Generate test data
python scripts/generate_test_data.py
```

## 🎯 Contribution Areas

We welcome contributions in these areas:

- **Core functionality**: LLM integration, agent system
- **Web interface**: Streamlit improvements
- **Testing**: Unit tests, integration tests
- **Documentation**: Guides, examples, API docs
- **Performance**: Optimization, caching
- **Quality**: Linting, type checking, error handling

Thank you for contributing to the Newsletter Generator project! 🚀 