# Newsletter Generator Makefile
# Common development tasks

.PHONY: help install test test-coverage lint format type-check quality-check clean analyze run-dev

# Default target
help:
	@echo "Newsletter Generator - Development Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup:"
	@echo "  install         Install all dependencies"
	@echo "  install-dev     Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test            Run all tests"
	@echo "  test-coverage   Run tests with coverage report"
	@echo "  test-unit       Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint            Run linting checks"
	@echo "  format          Format code with Black and isort"
	@echo "  type-check      Run type checking with mypy"
	@echo "  quality-check   Run all quality checks"
	@echo ""
	@echo "Analysis:"
	@echo "  analyze         Analyze codebase metrics"
	@echo "  clean           Clean up generated files"
	@echo ""
	@echo "Development:"
	@echo "  run-dev         Run development server"
	@echo "  run-streamlit   Run Streamlit application"

# Installation
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -r streamlit/requirements.txt

install-dev: install
	@echo "Installing development dependencies..."
	pip install pytest pytest-cov pytest-asyncio pytest-mock
	pip install black flake8 mypy isort
	pip install pre-commit

# Testing
test:
	@echo "Running all tests..."
	pytest tests/ -v

test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

test-unit:
	@echo "Running unit tests..."
	pytest tests/ -m "unit" -v

test-integration:
	@echo "Running integration tests..."
	pytest tests/ -m "integration" -v

test-agents:
	@echo "Running agent tests..."
	pytest tests/agents/ -v

test-core:
	@echo "Running core tests..."
	pytest tests/core/ -v

# Code Quality
lint:
	@echo "Running linting checks..."
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503

format:
	@echo "Formatting code..."
	black src/ tests/ --line-length=88
	isort src/ tests/ --profile=black

type-check:
	@echo "Running type checks..."
	mypy src/ --ignore-missing-imports

quality-check: lint format type-check
	@echo "All quality checks completed!"

# Analysis
analyze:
	@echo "Analyzing codebase..."
	python scripts/analyze_codebase.py

# Cleanup
clean:
	@echo "Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -f .coverage
	rm -f test.log
	rm -f codebase_analysis.json

# Development
run-dev:
	@echo "Starting development server..."
	python -m streamlit run streamlit/app_hybrid_minimal.py

run-streamlit:
	@echo "Starting Streamlit application..."
	python -m streamlit run streamlit/streamlit_app_hybrid.py

# Pre-commit setup
setup-pre-commit:
	@echo "Setting up pre-commit hooks..."
	pre-commit install

# System check
check-system:
	@echo "Checking system requirements..."
	@python -c "import sys; print(f'Python version: {sys.version}')"
	@python src/core/llm_cli.py doctor 2>/dev/null || echo "LLM Provider validation failed"
	@python -c "import streamlit; print('Streamlit: Available')" 2>/dev/null || echo "Streamlit: Not available"
	@python -c "import crewai; print('CrewAI: Available')" 2>/dev/null || echo "CrewAI: Not available"

# NVIDIA pipeline tests
test-nvidia:
	@echo "Testing NVIDIA pipeline configuration..."
	python tests/test_nvidia_default.py

# Test all pipelines
test-pipelines:
	@echo "Testing all LLM pipelines..."
	@echo "1. Testing NVIDIA (default)..."
	@LLM_PROVIDER=nvidia pytest tests/core/test_nvidia_integration.py -v
	@echo "2. Testing Ollama (fallback)..."
	@LLM_PROVIDER=ollama pytest tests/core/test_core.py -v

# Performance testing
test-performance:
	@echo "Running performance tests..."
	pytest tests/ -m "performance" -v

# Security checks
security-check:
	@echo "Running security checks..."
	@if command -v bandit >/dev/null 2>&1; then \
		bandit -r src/; \
	else \
		echo "Bandit not installed. Install with: pip install bandit"; \
	fi

# Documentation
docs-serve:
	@echo "Serving documentation..."
	@if command -v mkdocs >/dev/null 2>&1; then \
		mkdocs serve; \
	else \
		echo "MkDocs not installed. Install with: pip install mkdocs"; \
	fi

# Database operations
db-migrate:
	@echo "Running database migrations..."
	@echo "No migrations configured yet"

db-reset:
	@echo "Resetting database..."
	@echo "No database reset configured yet"

# Docker operations (if using Docker)
docker-build:
	@echo "Building Docker image..."
	@if [ -f Dockerfile ]; then \
		docker build -t newsletter-generator .; \
	else \
		echo "Dockerfile not found"; \
	fi

docker-run:
	@echo "Running Docker container..."
	@if [ -f Dockerfile ]; then \
		docker run -p 8501:8501 newsletter-generator; \
	else \
		echo "Dockerfile not found"; \
	fi

# Git operations
git-hooks: setup-pre-commit
	@echo "Git hooks configured"

# Full development setup
setup-dev: install-dev setup-pre-commit
	@echo "Development environment setup complete!"
	@echo "Run 'make check-system' to verify all components are available"

# CI/CD helpers
ci-test: test-coverage lint type-check
	@echo "CI tests completed"

ci-quality: quality-check analyze
	@echo "CI quality checks completed"

# Quick development workflow
dev: format test quality-check
	@echo "Development workflow completed"

# Production preparation
prod-prep: clean test-coverage quality-check security-check
	@echo "Production preparation completed"

# Help for specific areas
help-testing:
	@echo "Testing Commands:"
	@echo "  test            - Run all tests"
	@echo "  test-coverage   - Run tests with coverage"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-agents     - Run agent tests only"
	@echo "  test-core       - Run core tests only"
	@echo "  test-performance - Run performance tests"

help-quality:
	@echo "Quality Commands:"
	@echo "  lint            - Run flake8 linting"
	@echo "  format          - Format code with Black and isort"
	@echo "  type-check      - Run mypy type checking"
	@echo "  quality-check   - Run all quality checks"
	@echo "  security-check  - Run security checks with bandit"

help-dev:
	@echo "Development Commands:"
	@echo "  run-dev         - Start development server"
	@echo "  run-streamlit   - Start Streamlit app"
	@echo "  analyze         - Analyze codebase metrics"
	@echo "  check-system    - Check system requirements"
	@echo "  setup-dev       - Complete development setup" 