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