# Newsletter Generator: Refactoring Implementation Guide

## 📋 Overview

This document provides detailed implementation guidance for each phase of the refactoring plan. It includes specific code examples, step-by-step instructions, and testing procedures.

---

## 🎯 Phase 0: Foundation & Setup

### Step 1: Configure Testing Framework

#### Create `pyproject.toml` configuration:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--cov-fail-under=80"
]

[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*",
    "*/migrations/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod"
]

[tool.black]
line-length = 88
target-version = ['py310']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.flake8]
max-line-length = 88
extend-ignore = ["E203", "W503"]
exclude = [
    ".git",
    "__pycache__",
    "build",
    "dist",
    ".venv",
    ".mypy_cache"
]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = [
    "chromadb.*",
    "crewai.*",
    "ollama.*",
    "streamlit.*"
]
ignore_missing_imports = true
```

#### Create `tests/conftest.py`:

```python
"""Shared test fixtures and configuration."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return "This is a mock LLM response for testing purposes."

@pytest.fixture
def mock_search_results():
    """Mock search results for testing."""
    return [
        {
            "title": "Test Article 1",
            "url": "https://example.com/article1",
            "snippet": "This is a test article about AI."
        },
        {
            "title": "Test Article 2", 
            "url": "https://example.com/article2",
            "snippet": "Another test article about machine learning."
        }
    ]

@pytest.fixture
def sample_newsletter_content():
    """Sample newsletter content for testing."""
    return """
    # AI Newsletter - Test Edition
    
    ## Latest Developments in AI
    
    Artificial intelligence continues to evolve rapidly. Recent developments include:
    
    - New language models with improved capabilities
    - Advances in computer vision
    - Breakthroughs in reinforcement learning
    
    ## Industry Insights
    
    The AI industry is experiencing unprecedented growth, with companies investing heavily in research and development.
    
    ## Technical Deep Dive
    
    ### Transformer Architecture
    
    The transformer architecture has revolutionized natural language processing.
    
    ```python
    def attention_mechanism(query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, value)
    ```
    """

@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock_store = Mock()
    mock_store.query.return_value = [
        {
            "id": "test_id_1",
            "document": "Test document content",
            "metadata": {"source": "test", "timestamp": "2024-01-01"},
            "similarity": 0.95
        }
    ]
    return mock_store

@pytest.fixture
def mock_agent():
    """Mock agent for testing."""
    mock_agent = Mock()
    mock_agent.name = "TestAgent"
    mock_agent.role = "Test Role"
    mock_agent.goal = "Test Goal"
    mock_agent.execute_task.return_value = "Test task result"
    return mock_agent
```

### Step 2: Create Baseline Metrics Script

#### Create `scripts/analyze_codebase.py`:

```python
#!/usr/bin/env python3
"""Codebase analysis script for baseline metrics."""

import os
import ast
import sys
from pathlib import Path
from collections import defaultdict
import json

class CodebaseAnalyzer:
    """Analyze codebase for metrics and issues."""
    
    def __init__(self, src_path: str = "src"):
        self.src_path = Path(src_path)
        self.metrics = defaultdict(int)
        self.issues = []
        
    def analyze_file(self, file_path: Path):
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            tree = ast.parse(content)
            
            # Count lines
            lines = content.split('\n')
            self.metrics['total_lines'] += len(lines)
            self.metrics['code_lines'] += len([l for l in lines if l.strip() and not l.strip().startswith('#')])
            
            # Count functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.metrics['functions'] += 1
                elif isinstance(node, ast.ClassDef):
                    self.metrics['classes'] += 1
                    
            # Check for large files
            if len(lines) > 500:
                self.issues.append(f"Large file: {file_path} ({len(lines)} lines)")
                
        except Exception as e:
            self.issues.append(f"Error analyzing {file_path}: {e}")
    
    def analyze_codebase(self):
        """Analyze entire codebase."""
        for file_path in self.src_path.rglob("*.py"):
            self.analyze_file(file_path)
            
        return {
            'metrics': dict(self.metrics),
            'issues': self.issues,
            'files_analyzed': len(list(self.src_path.rglob("*.py")))
        }

def main():
    """Run codebase analysis."""
    analyzer = CodebaseAnalyzer()
    results = analyzer.analyze_codebase()
    
    print("=== Codebase Analysis Results ===")
    print(f"Files analyzed: {results['files_analyzed']}")
    print(f"Total lines: {results['metrics']['total_lines']}")
    print(f"Code lines: {results['metrics']['code_lines']}")
    print(f"Functions: {results['metrics']['functions']}")
    print(f"Classes: {results['metrics']['classes']}")
    
    if results['issues']:
        print("\n=== Issues Found ===")
        for issue in results['issues']:
            print(f"- {issue}")
    
    # Save results
    with open('codebase_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
```

---

## 🔧 Phase 1: Import & Dependency Cleanup

### Step 1: Create Core Constants

#### Create `src/core/constants.py`:

```python
"""Shared constants for the newsletter generator."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data"
LOGS_PATH = PROJECT_ROOT / "logs"
TESTS_PATH = PROJECT_ROOT / "tests"

# Ensure directories exist
for path in [DATA_PATH, LOGS_PATH]:
    path.mkdir(exist_ok=True)

# LLM Configuration
DEFAULT_LLM_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# Search Configuration
DEFAULT_SEARCH_RESULTS = int(os.getenv("DEFAULT_SEARCH_RESULTS", "5"))
SEARCH_TIMEOUT = int(os.getenv("SEARCH_TIMEOUT", "10"))
SEARCH_MAX_RETRIES = int(os.getenv("SEARCH_MAX_RETRIES", "3"))

# Scraping Configuration
SCRAPER_TIMEOUT = int(os.getenv("SCRAPER_TIMEOUT", "30"))
SCRAPER_MAX_RETRIES = int(os.getenv("SCRAPER_MAX_RETRIES", "3"))
SCRAPER_HEADLESS = os.getenv("SCRAPER_HEADLESS", "true").lower() == "true"

# Quality Configuration
MINIMUM_QUALITY_SCORE = float(os.getenv("MINIMUM_QUALITY_SCORE", "7.0"))
QUALITY_THRESHOLDS = {
    "technical_accuracy": 6.0,
    "readability": 7.0,
    "engagement": 6.5,
    "completeness": 7.0
}

# Newsletter Configuration
DEFAULT_NEWSLETTER_LENGTH = int(os.getenv("DEFAULT_NEWSLETTER_LENGTH", "1500"))
MAX_NEWSLETTER_LENGTH = int(os.getenv("MAX_NEWSLETTER_LENGTH", "3000"))

# Agent Configuration
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "60"))
MAX_AGENT_RETRIES = int(os.getenv("MAX_AGENT_RETRIES", "3"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Error Messages
ERROR_MESSAGES = {
    "llm_timeout": "LLM request timed out",
    "search_failed": "Search operation failed",
    "scraping_failed": "Web scraping failed",
    "validation_failed": "Content validation failed",
    "agent_failed": "Agent execution failed"
}

# Success Messages
SUCCESS_MESSAGES = {
    "newsletter_generated": "Newsletter generated successfully",
    "content_validated": "Content validation passed",
    "agent_completed": "Agent task completed successfully"
}
```

### Step 2: Create Custom Exceptions

#### Create `src/core/exceptions.py`:

```python
"""Custom exceptions for the newsletter generator."""

class NewsletterGeneratorError(Exception):
    """Base exception for newsletter generator."""
    pass

class LLMError(NewsletterGeneratorError):
    """Exception raised for LLM-related errors."""
    pass

class SearchError(NewsletterGeneratorError):
    """Exception raised for search-related errors."""
    pass

class ScrapingError(NewsletterGeneratorError):
    """Exception raised for scraping-related errors."""
    pass

class ValidationError(NewsletterGeneratorError):
    """Exception raised for validation errors."""
    pass

class AgentError(NewsletterGeneratorError):
    """Exception raised for agent-related errors."""
    pass

class ConfigurationError(NewsletterGeneratorError):
    """Exception raised for configuration errors."""
    pass

class QualityGateError(NewsletterGeneratorError):
    """Exception raised for quality gate failures."""
    pass

class ImportError(NewsletterGeneratorError):
    """Exception raised for import-related errors."""
    pass
```

### Step 3: Create Import Helper

#### Create `src/core/import_helper.py`:

```python
"""Helper utilities for managing imports and dependencies."""

import importlib
import logging
from typing import Any, Optional, Dict
from .exceptions import ImportError

logger = logging.getLogger(__name__)

class ImportHelper:
    """Helper class for managing imports and dependencies."""
    
    @staticmethod
    def safe_import(module_name: str, package_name: str = None) -> Optional[Any]:
        """Safely import a module with fallback handling."""
        try:
            if package_name:
                module = importlib.import_module(module_name, package_name)
            else:
                module = importlib.import_module(module_name)
            logger.debug(f"Successfully imported {module_name}")
            return module
        except ImportError as e:
            logger.warning(f"Failed to import {module_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error importing {module_name}: {e}")
            return None
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """Check if all required dependencies are available."""
        dependencies = {
            "ollama": "ollama",
            "chromadb": "chromadb",
            "crewai": "crewai",
            "streamlit": "streamlit",
            "requests": "requests",
            "beautifulsoup4": "bs4",
            "feedparser": "feedparser",
            "python-dotenv": "dotenv",
            "pydantic": "pydantic",
            "scikit-learn": "sklearn"
        }
        
        results = {}
        for name, module in dependencies.items():
            results[name] = ImportHelper.safe_import(module) is not None
            
        return results
    
    @staticmethod
    def get_optional_dependencies() -> Dict[str, bool]:
        """Check optional dependencies."""
        optional_deps = {
            "crawl4ai": "crawl4ai",
            "playwright": "playwright",
            "pytesseract": "pytesseract",
            "PyPDF2": "PyPDF2",
            "PIL": "PIL"
        }
        
        results = {}
        for name, module in optional_deps.items():
            results[name] = ImportHelper.safe_import(module) is not None
            
        return results

def safe_import_core():
    """Safely import core modules with proper error handling."""
    try:
        from .core import query_llm
        return query_llm
    except ImportError as e:
        logger.error(f"Failed to import core module: {e}")
        raise ImportError(f"Core module import failed: {e}")

def safe_import_agents():
    """Safely import agent modules."""
    try:
        from .agents.agents import SimpleAgent, ResearchAgent, WriterAgent, EditorAgent
        return SimpleAgent, ResearchAgent, WriterAgent, EditorAgent
    except ImportError as e:
        logger.error(f"Failed to import agent modules: {e}")
        raise ImportError(f"Agent module import failed: {e}")

def safe_import_tools():
    """Safely import tool modules."""
    try:
        from .tools.tools import search_web, search_knowledge_base
        return search_web, search_knowledge_base
    except ImportError as e:
        logger.error(f"Failed to import tool modules: {e}")
        raise ImportError(f"Tool module import failed: {e}")
```

---

## 🧪 Phase 2: Testing Infrastructure

### Step 1: Create Agent Tests

#### Create `tests/agents/test_agents.py`:

```python
"""Tests for agent system."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.agents import (
    SimpleAgent, ResearchAgent, WriterAgent, EditorAgent, 
    ManagerAgent, Task, EnhancedCrew
)
from src.core.exceptions import AgentError

class TestSimpleAgent:
    """Test the base SimpleAgent class."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = SimpleAgent(
            name="TestAgent",
            role="Test Role",
            goal="Test Goal",
            backstory="Test backstory"
        )
        
        assert agent.name == "TestAgent"
        assert agent.role == "Test Role"
        assert agent.goal == "Test Goal"
        assert agent.backstory == "Test backstory"
        assert isinstance(agent.tools, list)
    
    @patch('src.agents.agents.query_llm')
    def test_execute_task_success(self, mock_query_llm):
        """Test successful task execution."""
        mock_query_llm.return_value = "Test response"
        
        agent = SimpleAgent(
            name="TestAgent",
            role="Test Role",
            goal="Test Goal",
            backstory="Test backstory"
        )
        
        result = agent.execute_task("Test task")
        assert result == "Test response"
        mock_query_llm.assert_called_once()
    
    @patch('src.agents.agents.query_llm')
    def test_execute_task_with_tools(self, mock_query_llm):
        """Test task execution with tools."""
        mock_query_llm.side_effect = [
            "I need to search for information",
            "Based on search results: Test response"
        ]
        
        agent = SimpleAgent(
            name="TestAgent",
            role="Test Role",
            goal="Test Goal",
            backstory="Test backstory",
            tools=["search_web"]
        )
        
        with patch.object(agent, '_execute_tools') as mock_tools:
            mock_tools.return_value = "Search results"
            result = agent.execute_task("Test task")
            
            assert result == "Based on search results: Test response"
            assert mock_query_llm.call_count == 2
    
    def test_execute_task_error_handling(self):
        """Test error handling in task execution."""
        agent = SimpleAgent(
            name="TestAgent",
            role="Test Role",
            goal="Test Goal",
            backstory="Test backstory"
        )
        
        with patch('src.agents.agents.query_llm', side_effect=Exception("Test error")):
            result = agent.execute_task("Test task")
            assert "Error in agent" in result

class TestResearchAgent:
    """Test the ResearchAgent class."""
    
    def test_research_agent_initialization(self):
        """Test research agent initialization."""
        agent = ResearchAgent()
        
        assert agent.name == "ResearchAgent"
        assert "research" in agent.role.lower()
        assert "search" in agent.goal.lower()
        assert "search_web" in agent.tools
        assert "search_knowledge_base" in agent.tools
    
    @patch('src.agents.agents.query_llm')
    def test_research_task_execution(self, mock_query_llm):
        """Test research task execution."""
        mock_query_llm.return_value = "Research findings: AI is advancing rapidly"
        
        agent = ResearchAgent()
        result = agent.execute_task("Research latest AI developments")
        
        assert "Research findings" in result
        mock_query_llm.assert_called_once()

class TestWriterAgent:
    """Test the WriterAgent class."""
    
    def test_writer_agent_initialization(self):
        """Test writer agent initialization."""
        agent = WriterAgent()
        
        assert agent.name == "WriterAgent"
        assert "writer" in agent.role.lower()
        assert "content" in agent.goal.lower()
    
    @patch('src.agents.agents.query_llm')
    def test_writing_task_execution(self, mock_query_llm):
        """Test writing task execution."""
        mock_query_llm.return_value = "Written content about AI developments"
        
        agent = WriterAgent()
        result = agent.execute_task("Write about AI developments")
        
        assert "Written content" in result
        mock_query_llm.assert_called_once()

class TestEditorAgent:
    """Test the EditorAgent class."""
    
    def test_editor_agent_initialization(self):
        """Test editor agent initialization."""
        agent = EditorAgent()
        
        assert agent.name == "EditorAgent"
        assert "editor" in agent.role.lower()
        assert "quality" in agent.goal.lower()
    
    def test_quality_metrics_extraction(self):
        """Test quality metrics extraction."""
        agent = EditorAgent()
        
        content = """
        # Test Content
        
        This is a test newsletter about artificial intelligence.
        It contains multiple paragraphs with detailed information.
        
        ## Key Points
        - Point one with details
        - Point two with more information
        """
        
        metrics = agent.extract_quality_metrics(content)
        
        assert 'word_count' in metrics
        assert 'estimated_reading_time' in metrics
        assert 'content_depth' in metrics
        assert metrics['word_count'] > 0
    
    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        agent = EditorAgent()
        
        sample_scores = {
            'clarity': 8.0,
            'accuracy': 7.5,
            'engagement': 6.5,
            'completeness': 7.0
        }
        
        quality_analysis = agent.calculate_quality_score({'scores': sample_scores})
        
        assert 'overall_score' in quality_analysis
        assert 'grade' in quality_analysis
        assert quality_analysis['overall_score'] > 0

class TestTask:
    """Test the Task class."""
    
    def test_task_creation(self):
        """Test task creation."""
        agent = Mock()
        task = Task("Test task description", agent, "Test context")
        
        assert task.description == "Test task description"
        assert task.agent == agent
        assert task.context == "Test context"
    
    def test_task_execution(self):
        """Test task execution."""
        agent = Mock()
        agent.execute_task.return_value = "Task result"
        
        task = Task("Test task", agent)
        result = task.execute()
        
        assert result == "Task result"
        agent.execute_task.assert_called_once_with("Test task", "")

class TestEnhancedCrew:
    """Test the EnhancedCrew class."""
    
    def test_crew_initialization(self):
        """Test crew initialization."""
        agents = [Mock(), Mock()]
        tasks = [Mock(), Mock()]
        
        crew = EnhancedCrew(agents, tasks)
        
        assert crew.agents == agents
        assert crew.tasks == tasks
        assert crew.workflow_type == "sequential"
    
    @patch('src.agents.agents.query_llm')
    def test_crew_kickoff(self, mock_query_llm):
        """Test crew kickoff."""
        mock_query_llm.return_value = "Agent response"
        
        agent1 = Mock()
        agent1.execute_task.return_value = "Agent 1 result"
        agent2 = Mock()
        agent2.execute_task.return_value = "Agent 2 result"
        
        task1 = Task("Task 1", agent1)
        task2 = Task("Task 2", agent2)
        
        crew = EnhancedCrew([agent1, agent2], [task1, task2])
        result = crew.kickoff()
        
        assert "Agent 1 result" in result
        assert "Agent 2 result" in result
```

### Step 2: Create Core Tests

#### Create `tests/core/test_core.py`:

```python
"""Tests for core functionality."""

import pytest
from unittest.mock import Mock, patch
from src.core.core import query_llm
from src.core.exceptions import LLMError

class TestLLMQuery:
    """Test LLM query functionality."""
    
    @patch('src.core.core.ollama')
    def test_successful_llm_query(self, mock_ollama):
        """Test successful LLM query."""
        mock_response = {
            "message": {
                "content": "This is a test response from the LLM."
            }
        }
        mock_ollama.chat.return_value = mock_response
        
        result = query_llm("Test prompt")
        
        assert result == "This is a test response from the LLM."
        mock_ollama.chat.assert_called_once()
    
    @patch('src.core.core.ollama')
    def test_llm_query_with_custom_model(self, mock_ollama):
        """Test LLM query with custom model."""
        mock_response = {
            "message": {
                "content": "Custom model response."
            }
        }
        mock_ollama.chat.return_value = mock_response
        
        with patch.dict('os.environ', {'OLLAMA_MODEL': 'custom-model'}):
            result = query_llm("Test prompt")
            
            assert result == "Custom model response."
            # Verify the custom model was used
            call_args = mock_ollama.chat.call_args
            assert call_args[1]['model'] == 'custom-model'
    
    @patch('src.core.core.ollama')
    def test_llm_query_error_handling(self, mock_ollama):
        """Test LLM query error handling."""
        from ollama import ResponseError
        mock_ollama.chat.side_effect = ResponseError("Test error")
        
        result = query_llm("Test prompt")
        
        assert "An error occurred while querying the LLM" in result
    
    @patch('src.core.core.ollama')
    def test_llm_query_unexpected_error(self, mock_ollama):
        """Test LLM query unexpected error handling."""
        mock_ollama.chat.side_effect = Exception("Unexpected error")
        
        result = query_llm("Test prompt")
        
        assert "An unexpected error occurred while querying the LLM" in result

class TestLogging:
    """Test logging functionality."""
    
    def test_log_file_creation(self):
        """Test that log files are created."""
        import logging
        from pathlib import Path
        
        # Trigger logging
        logger = logging.getLogger(__name__)
        logger.info("Test log message")
        
        # Check if log file exists
        log_file = Path("logs/interaction.log")
        assert log_file.exists()
    
    def test_log_format(self):
        """Test log format."""
        import logging
        from pathlib import Path
        
        logger = logging.getLogger(__name__)
        logger.info("Test format message")
        
        # Read log file and check format
        log_file = Path("logs/interaction.log")
        if log_file.exists():
            with open(log_file, 'r') as f:
                log_content = f.read()
                assert "Test format message" in log_content
```

### Step 3: Create Integration Tests

#### Create `tests/integration/test_workflows.py`:

```python
"""Integration tests for complete workflows."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.agents import (
    ResearchAgent, WriterAgent, EditorAgent, 
    Task, EnhancedCrew
)
from src.core.exceptions import AgentError

class TestNewsletterWorkflow:
    """Test complete newsletter generation workflow."""
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        research_agent = Mock(spec=ResearchAgent)
        research_agent.name = "ResearchAgent"
        research_agent.execute_task.return_value = "Research findings about AI"
        
        writer_agent = Mock(spec=WriterAgent)
        writer_agent.name = "WriterAgent"
        writer_agent.execute_task.return_value = "Written newsletter content"
        
        editor_agent = Mock(spec=EditorAgent)
        editor_agent.name = "EditorAgent"
        editor_agent.execute_task.return_value = "Edited and improved content"
        
        return research_agent, writer_agent, editor_agent
    
    def test_basic_newsletter_workflow(self, mock_agents):
        """Test basic newsletter generation workflow."""
        research_agent, writer_agent, editor_agent = mock_agents
        
        # Create tasks
        research_task = Task("Research latest AI developments", research_agent)
        writing_task = Task("Write newsletter about AI", writer_agent)
        editing_task = Task("Edit and improve newsletter", editor_agent)
        
        # Create crew
        crew = EnhancedCrew(
            [research_agent, writer_agent, editor_agent],
            [research_task, writing_task, editing_task]
        )
        
        # Execute workflow
        result = crew.kickoff()
        
        # Verify all agents were called
        research_agent.execute_task.assert_called_once()
        writer_agent.execute_task.assert_called_once()
        editor_agent.execute_task.assert_called_once()
        
        # Verify result contains expected content
        assert "Research findings" in result
        assert "Written newsletter" in result
        assert "Edited and improved" in result
    
    def test_workflow_with_context_passing(self, mock_agents):
        """Test workflow with context passing between agents."""
        research_agent, writer_agent, editor_agent = mock_agents
        
        # Mock context-aware responses
        research_agent.execute_task.return_value = "Research: AI trends in 2024"
        writer_agent.execute_task.return_value = "Content based on: Research: AI trends in 2024"
        editor_agent.execute_task.return_value = "Final content: Content based on: Research: AI trends in 2024"
        
        # Create tasks with context
        research_task = Task("Research AI trends", research_agent, "Focus on 2024")
        writing_task = Task("Write newsletter", writer_agent, "Use research findings")
        editing_task = Task("Edit content", editor_agent, "Ensure quality")
        
        # Create crew
        crew = EnhancedCrew(
            [research_agent, writer_agent, editor_agent],
            [research_task, writing_task, editing_task]
        )
        
        # Execute workflow
        result = crew.kickoff()
        
        # Verify context was passed
        assert "Research: AI trends in 2024" in result
        assert "Content based on:" in result
        assert "Final content:" in result
    
    def test_workflow_error_handling(self, mock_agents):
        """Test workflow error handling."""
        research_agent, writer_agent, editor_agent = mock_agents
        
        # Make research agent fail
        research_agent.execute_task.side_effect = Exception("Research failed")
        
        # Create tasks
        research_task = Task("Research AI trends", research_agent)
        writing_task = Task("Write newsletter", writer_agent)
        
        # Create crew
        crew = EnhancedCrew(
            [research_agent, writer_agent],
            [research_task, writing_task]
        )
        
        # Execute workflow
        result = crew.kickoff()
        
        # Verify error handling
        assert "Error in agent" in result
        assert "Research failed" in result
        
        # Verify writer agent was not called due to research failure
        writer_agent.execute_task.assert_not_called()

class TestAgentCoordination:
    """Test agent coordination and communication."""
    
    def test_agent_tool_sharing(self):
        """Test that agents can share tools and results."""
        # This test would verify that agents can access shared tools
        # and that results from one agent can be used by another
        pass
    
    def test_agent_performance_tracking(self):
        """Test agent performance tracking."""
        # This test would verify that agent performance is tracked
        # and can be analyzed
        pass
    
    def test_agent_error_recovery(self):
        """Test agent error recovery mechanisms."""
        # This test would verify that the system can recover from
        # agent failures and continue processing
        pass

class TestEndToEndWorkflow:
    """Test complete end-to-end newsletter generation."""
    
    @patch('src.core.core.query_llm')
    def test_complete_newsletter_generation(self, mock_query_llm):
        """Test complete newsletter generation from start to finish."""
        # Mock LLM responses for different stages
        mock_query_llm.side_effect = [
            "Research findings: AI is advancing rapidly",
            "Written content about AI advancements",
            "Edited and improved newsletter content"
        ]
        
        # Create real agents
        research_agent = ResearchAgent()
        writer_agent = WriterAgent()
        editor_agent = EditorAgent()
        
        # Create tasks
        research_task = Task("Research latest AI developments", research_agent)
        writing_task = Task("Write newsletter about AI", writer_agent)
        editing_task = Task("Edit newsletter", editor_agent)
        
        # Create crew
        crew = EnhancedCrew(
            [research_agent, writer_agent, editor_agent],
            [research_task, writing_task, editing_task]
        )
        
        # Execute workflow
        result = crew.kickoff()
        
        # Verify the complete workflow
        assert "Research findings" in result
        assert "Written content" in result
        assert "Edited and improved" in result
        
        # Verify LLM was called for each stage
        assert mock_query_llm.call_count >= 3
```

---

## 🔍 Phase 3: Search & Scraping Consolidation

### Step 1: Create Unified Search Interface

#### Create `src/tools/search_provider.py`:

```python
"""Unified search provider interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
from functools import lru_cache
import time

from ..core.exceptions import SearchError
from ..core.constants import DEFAULT_SEARCH_RESULTS, SEARCH_TIMEOUT, SEARCH_MAX_RETRIES

logger = logging.getLogger(__name__)

class SearchProvider(ABC):
    """Abstract base class for search providers."""
    
    def __init__(self, max_results: int = DEFAULT_SEARCH_RESULTS, timeout: int = SEARCH_TIMEOUT):
        self.max_results = max_results
        self.timeout = timeout
        self.retry_count = 0
        self.max_retries = SEARCH_MAX_RETRIES
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Perform search and return results."""
        pass
    
    def search_with_retry(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search with retry logic."""
        for attempt in range(self.max_retries):
            try:
                return self.search(query, **kwargs)
            except Exception as e:
                self.retry_count += 1
                logger.warning(f"Search attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise SearchError(f"Search failed after {self.max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results as string."""
        if not results:
            return "No search results found."
        
        formatted = []
        for i, result in enumerate(results[:self.max_results], 1):
            title = result.get('title', 'No title')
            url = result.get('url', 'No URL')
            snippet = result.get('snippet', 'No description')
            
            formatted.append(f"{i}. {title}\n   URL: {url}\n   {snippet}\n")
        
        return "\n".join(formatted)

class SerperSearchProvider(SearchProvider):
    """Serper API search provider."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self._init_serper()
    
    def _init_serper(self):
        """Initialize Serper API client."""
        try:
            from crewai_tools import SerperDevTool
            self.serper_tool = SerperDevTool()
            logger.info("Serper API tool initialized successfully")
        except ImportError:
            logger.error("SerperDevTool not available")
            self.serper_tool = None
        except Exception as e:
            logger.error(f"Failed to initialize Serper tool: {e}")
            self.serper_tool = None
    
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Perform search using Serper API."""
        if not self.serper_tool:
            raise SearchError("Serper tool not initialized")
        
        try:
            raw_results = self.serper_tool.run(search_query=query)
            
            # Parse results
            if isinstance(raw_results, dict):
                parsed_results = raw_results
            elif isinstance(raw_results, str):
                import json
                try:
                    parsed_results = json.loads(raw_results)
                except json.JSONDecodeError:
                    parsed_results = {"organic": [{"title": "Search Result", "snippet": raw_results, "link": ""}]}
            else:
                parsed_results = {"organic": [{"title": "Search Result", "snippet": str(raw_results), "link": ""}]}
            
            # Format results
            return self._format_serper_results(parsed_results)
            
        except Exception as e:
            logger.error(f"Serper API search error: {e}")
            raise SearchError(f"Serper search failed: {e}")
    
    def _format_serper_results(self, serper_results: Any) -> List[Dict[str, Any]]:
        """Format Serper API results."""
        formatted_results = []
        
        try:
            organic_results = serper_results.get("organic", [])
            
            for result in organic_results[:self.max_results]:
                formatted_result = {
                    "title": result.get("title", "No title"),
                    "url": result.get("link", ""),
                    "snippet": result.get("snippet", "No description"),
                    "source": "serper"
                }
                formatted_results.append(formatted_result)
                
        except Exception as e:
            logger.error(f"Error formatting Serper results: {e}")
        
        return formatted_results

class DuckDuckGoSearchProvider(SearchProvider):
    """DuckDuckGo search provider."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_duckduckgo()
    
    def _init_duckduckgo(self):
        """Initialize DuckDuckGo search."""
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
            logger.info("DuckDuckGo search initialized successfully")
        except ImportError:
            logger.error("DuckDuckGo search not available")
            self.ddgs = None
        except Exception as e:
            logger.error(f"Failed to initialize DuckDuckGo search: {e}")
            self.ddgs = None
    
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Perform search using DuckDuckGo."""
        if not self.ddgs:
            raise SearchError("DuckDuckGo search not initialized")
        
        try:
            results = list(self.ddgs.text(query, max_results=self.max_results))
            
            formatted_results = []
            for result in results:
                formatted_result = {
                    "title": result.get("title", "No title"),
                    "url": result.get("link", ""),
                    "snippet": result.get("body", "No description"),
                    "source": "duckduckgo"
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            raise SearchError(f"DuckDuckGo search failed: {e}")

class UnifiedSearchProvider:
    """Unified search provider that can use multiple backends."""
    
    def __init__(self, primary_provider: str = "serper", fallback_providers: List[str] = None):
        self.primary_provider = primary_provider
        self.fallback_providers = fallback_providers or ["duckduckgo"]
        self.providers = self._init_providers()
    
    def _init_providers(self) -> Dict[str, SearchProvider]:
        """Initialize search providers."""
        providers = {}
        
        # Initialize primary provider
        if self.primary_provider == "serper":
            try:
                providers["serper"] = SerperSearchProvider()
            except Exception as e:
                logger.warning(f"Failed to initialize Serper provider: {e}")
        
        # Initialize fallback providers
        for provider_name in self.fallback_providers:
            if provider_name == "duckduckgo":
                try:
                    providers["duckduckgo"] = DuckDuckGoSearchProvider()
                except Exception as e:
                    logger.warning(f"Failed to initialize DuckDuckGo provider: {e}")
        
        return providers
    
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search using available providers with fallback."""
        # Try primary provider first
        if self.primary_provider in self.providers:
            try:
                return self.providers[self.primary_provider].search_with_retry(query, **kwargs)
            except Exception as e:
                logger.warning(f"Primary provider failed: {e}")
        
        # Try fallback providers
        for provider_name in self.fallback_providers:
            if provider_name in self.providers:
                try:
                    return self.providers[provider_name].search_with_retry(query, **kwargs)
                except Exception as e:
                    logger.warning(f"Fallback provider {provider_name} failed: {e}")
        
        raise SearchError("All search providers failed")
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results."""
        if not results:
            return "No search results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('url', 'No URL')
            snippet = result.get('snippet', 'No description')
            source = result.get('source', 'unknown')
            
            formatted.append(f"{i}. {title} [{source}]\n   URL: {url}\n   {snippet}\n")
        
        return "\n".join(formatted)

# Global search provider instance
_search_provider = None

def get_search_provider() -> UnifiedSearchProvider:
    """Get the global search provider instance."""
    global _search_provider
    if _search_provider is None:
        _search_provider = UnifiedSearchProvider()
    return _search_provider

@lru_cache(maxsize=32)
def search_web(query: str, max_results: int = DEFAULT_SEARCH_RESULTS) -> str:
    """Unified web search function."""
    provider = get_search_provider()
    try:
        results = provider.search(query, max_results=max_results)
        return provider.format_results(results)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Search failed: {e}"
```

This implementation guide provides detailed code examples and step-by-step instructions for the first three phases. Each phase builds upon the previous one and includes comprehensive testing procedures. The remaining phases would follow a similar pattern with specific implementations for each component.

The guide emphasizes:
- **Incremental development** with each phase being independently testable
- **Comprehensive testing** with unit, integration, and performance tests
- **Error handling** and fallback mechanisms
- **Code quality** with proper documentation and type hints
- **Performance optimization** with caching and async operations where appropriate 