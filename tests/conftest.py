"""Shared test fixtures and configuration."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response for testing."""
    return {
        "message": {
            "content": "This is a test response from Ollama."
        }
    }

@pytest.fixture
def mock_serper_response():
    """Mock Serper API response for testing."""
    return {
        "organic": [
            {
                "title": "Test Search Result",
                "link": "https://example.com/test",
                "snippet": "This is a test search result snippet."
            }
        ]
    }

@pytest.fixture
def sample_web_content():
    """Sample web content for scraping tests."""
    return """
    <html>
        <head>
            <title>Test Article - AI Developments</title>
        </head>
        <body>
            <article>
                <h1>Latest AI Breakthroughs</h1>
                <p>Artificial intelligence has made significant progress in recent months.</p>
                <h2>Key Developments</h2>
                <ul>
                    <li>Improved language models</li>
                    <li>Better computer vision</li>
                    <li>Enhanced reasoning capabilities</li>
                </ul>
            </article>
        </body>
    </html>
    """

@pytest.fixture
def mock_quality_scores():
    """Mock quality scores for testing."""
    return {
        "technical_accuracy": 8.5,
        "readability": 7.8,
        "engagement": 7.2,
        "completeness": 8.0,
        "overall_score": 7.9,
        "grade": "B+"
    }

@pytest.fixture
def test_config():
    """Test configuration for the application."""
    return {
        "llm_model": "deepseek-r1",
        "max_search_results": 5,
        "quality_threshold": 7.0,
        "newsletter_length": 1500,
        "timeout": 30
    }

@pytest.fixture
def mock_file_system():
    """Mock file system for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test directory structure
        test_dirs = ["data", "logs", "outputs", "configs"]
        for dir_name in test_dirs:
            (Path(temp_dir) / dir_name).mkdir(exist_ok=True)
        
        # Create test files
        test_files = {
            "data/sample.json": '{"test": "data"}',
            "configs/test_config.yaml": "model: test\nversion: 1.0",
            "logs/test.log": "2024-01-01 INFO: Test log entry"
        }
        
        for file_path, content in test_files.items():
            full_path = Path(temp_dir) / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
        
        yield temp_dir

@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for all tests."""
    # Configure logging to capture test output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test.log', mode='w')
        ]
    )
    
    # Suppress noisy loggers during tests
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "LLM_PROVIDER": "nvidia",
        "NVIDIA_API_KEY": "test-nvidia-api-key",
        "NVIDIA_MODEL": "openai/gpt-oss-20b",
        "OLLAMA_MODEL": "deepseek-r1",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "SERPER_API_KEY": "test_api_key",
        "LOG_LEVEL": "INFO"
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars

@pytest.fixture
def performance_benchmark():
    """Performance benchmark fixture for testing."""
    import time
    
    class PerformanceBenchmark:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def end(self):
            self.end_time = time.time()
        
        def duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return PerformanceBenchmark()


# Section-Aware Newsletter Generation Test Fixtures

@pytest.fixture
def section_aware_newsletter_content():
    """Comprehensive sample newsletter content for section-aware testing."""
    return """# Weekly AI Newsletter

Welcome to our comprehensive weekly newsletter covering the latest developments 
in artificial intelligence and machine learning research!

## Latest Industry News

Recent developments have shaped the AI landscape significantly:

- OpenAI announced GPT-4.5 with enhanced reasoning capabilities and 40% efficiency improvement
- Google DeepMind published groundbreaking research on quantum-classical hybrid algorithms  
- Microsoft expanded Azure AI services to include new computer vision capabilities
- Meta introduced open-source multimodal models for research community
- Several AI startups secured $500M+ in Series B funding rounds

## Technical Deep Dive: Transformer Efficiency

Furthermore, recent research has focused extensively on transformer model optimization.
The latest architectural innovations demonstrate remarkable performance improvements.

New attention mechanisms reduce computational complexity from O(n²) to O(n log n).
Implementation studies show 35% reduction in training time across standard benchmarks.
Memory usage optimization enables deployment on edge devices with limited resources.

## Implementation Tutorial: Optimizing Your Models

Next, let's explore practical steps for implementing these efficiency improvements:

### Step 1: Environment Setup
First, install the latest optimization frameworks and compatibility layers.

### Step 2: Model Architecture Updates
Configure your existing models to use sparse attention patterns.

### Step 3: Performance Validation
Benchmark your optimized models against baseline implementations.

## Key Takeaways and Future Outlook

In summary, this week's developments represent substantial progress in AI efficiency and capability.
Organizations should evaluate these developments for competitive advantage and strategic planning.

Try experimenting with these optimization techniques in controlled development environments.
Next week, we'll examine emerging trends in multimodal AI and cross-domain applications.
"""


@pytest.fixture
def section_aware_context():
    """Standard context for section-aware newsletter generation testing."""
    return {
        'topic': 'AI Weekly Newsletter',
        'audience': 'AI/ML Engineers',
        'content_focus': 'Latest AI Developments and Implementation',
        'word_count': 3000,
        'tone': 'professional',
        'technical_level': 'intermediate',
        'special_requirements': [
            'Include practical code examples',
            'Focus on implementation details',
            'Provide performance metrics'
        ]
    }


@pytest.fixture
def high_quality_section_content():
    """High-quality, well-structured content for section testing."""
    return """# Introduction to Advanced Machine Learning

Welcome to our comprehensive examination of cutting-edge machine learning developments.
This systematic analysis explores recent breakthroughs and their practical implications for 
software engineering teams and research organizations.

## Recent Research Breakthroughs

Recent developments in machine learning have demonstrated remarkable progress across multiple domains.
According to the latest research published in Nature Machine Intelligence, new architectures 
achieve 40% improvement in computational efficiency while maintaining model accuracy.

## Technical Implementation Guide

Building on these foundations, let's explore practical implementation strategies:

### Step 1: Environment Configuration
First, establish a robust development environment with the latest frameworks.

### Step 2: Model Architecture Selection
Choose appropriate architectures based on your specific requirements.

## Conclusion and Future Directions

In conclusion, these advances represent substantial progress in machine learning capabilities.
Organizations should evaluate these developments for competitive advantage and strategic planning.
"""


@pytest.fixture
def poor_quality_section_content():
    """Poor-quality content for testing quality detection."""
    return """some intro stuff here

news things happened recently maybe

analysis shows stuff. things are important. data exists.

how to do things:
step 1 do something
step 2 do more
step 3 finish

conclusion: ok thats it
"""


# Utility functions for section-aware tests

def assert_valid_quality_score(score: float, name: str = "score"):
    """Assert that a quality score is valid (between 0.0 and 1.0)."""
    assert isinstance(score, (int, float)), f"{name} must be numeric"
    assert 0.0 <= score <= 1.0, f"{name} must be between 0.0 and 1.0, got {score}"


def assert_valid_content_length(content: str, min_length: int = 10):
    """Assert that content has reasonable length."""
    assert isinstance(content, str), f"Content must be string, got {type(content)}"
    assert len(content) >= min_length, f"Content too short: {len(content)} < {min_length}" 