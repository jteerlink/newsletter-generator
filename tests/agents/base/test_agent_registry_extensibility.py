"""
Tests for the extensible agent registry system.
"""

import pytest
import tempfile
import json
import os
from unittest.mock import patch, MagicMock
import inspect

from agents.base.agent_registry import (
    AgentRegistry, AgentPlugin, AgentMetadata, AgentCapability
)
from agents.base.agent_base import AgentBase


class MockAgent(AgentBase):
    """Mock agent for testing."""
    
    def __init__(self, config=None):
        agent_id = config.get('name', 'mock_agent') if config else 'mock_agent'
        super().__init__(agent_id=agent_id)
        self.name = agent_id
    
    def process_query(self, query, context=None):
        return {'result': f'Processed: {query}', 'agent': self.name}
    
    def receive_message(self, message):
        """Implement abstract method."""
        return {'status': 'received', 'agent': self.name}
    
    def send_message(self, recipient_id, message):
        """Implement abstract method."""
        return {'status': 'sent', 'agent': self.name, 'recipient': recipient_id}
    
    def run(self):
        """Implement abstract method."""
        return {'status': 'running', 'agent': self.name}


class MockAgentPlugin(AgentPlugin):
    """Mock agent plugin for testing."""
    
    def get_metadata(self):
        return AgentMetadata(
            name="mock_plugin_agent",
            description="A mock agent plugin for testing",
            capabilities={
                AgentCapability.CONTENT_GENERATION,
                AgentCapability.RESEARCH
            },
            version="1.0.0",
            author="Test Author",
            dependencies=["test.dependency"],
            config_schema={
                "name": {"type": "string", "default": "plugin_agent"}
            }
        )
    
    def create_agent(self, config=None):
        return MockAgent(config=config)
    
    def validate_config(self, config):
        if config and 'name' in config:
            return isinstance(config['name'], str)
        return True


@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    return AgentRegistry()


@pytest.fixture
def sample_metadata():
    """Sample agent metadata for testing."""
    return AgentMetadata(
        name="test_agent",
        description="A test agent",
        capabilities={
            AgentCapability.QUERY_REWRITING,
            AgentCapability.CONTEXT_ASSESSMENT
        },
        version="1.0.0",
        author="Test Author",
        dependencies=["test.dep"],
        config_schema={"param": {"type": "string"}}
    )


class TestAgentRegistryExtensibility:
    """Test the extensible agent registry functionality."""
    
    def test_register_agent_with_metadata(self, registry, sample_metadata):
        """Test registering an agent with metadata."""
        registry.register_agent(MockAgent, sample_metadata)
        
        # Check registration
        assert "test_agent" in registry._agents
        assert registry._agents["test_agent"] == MockAgent
        assert registry._metadata["test_agent"] == sample_metadata
        
        # Check capability indexing
        assert "test_agent" in registry._capability_index[AgentCapability.QUERY_REWRITING]
        assert "test_agent" in registry._capability_index[AgentCapability.CONTEXT_ASSESSMENT]
    
    def test_register_plugin(self, registry):
        """Test registering an agent plugin."""
        plugin = MockAgentPlugin()
        registry.register_plugin(plugin)
        
        # Check registration
        assert "mock_plugin_agent" in registry._plugins
        assert registry._plugins["mock_plugin_agent"] == plugin
        
        # Check metadata
        metadata = registry._metadata["mock_plugin_agent"]
        assert metadata.name == "mock_plugin_agent"
        assert AgentCapability.CONTENT_GENERATION in metadata.capabilities
        
        # Check capability indexing
        assert "mock_plugin_agent" in registry._capability_index[AgentCapability.CONTENT_GENERATION]
        assert "mock_plugin_agent" in registry._capability_index[AgentCapability.RESEARCH]
    
    def test_get_agent_direct_registration(self, registry, sample_metadata):
        """Test getting an agent from direct registration."""
        registry.register_agent(MockAgent, sample_metadata)
        
        agent = registry.get_agent("test_agent", {"name": "custom_name"})
        assert agent is not None
        assert isinstance(agent, MockAgent)
        assert agent.name == "custom_name"
    
    def test_get_agent_plugin(self, registry):
        """Test getting an agent from plugin registration."""
        plugin = MockAgentPlugin()
        registry.register_plugin(plugin)
        
        agent = registry.get_agent("mock_plugin_agent", {"name": "plugin_name"})
        assert agent is not None
        assert isinstance(agent, MockAgent)
        assert agent.name == "plugin_name"
    
    def test_get_agent_not_found(self, registry):
        """Test getting a non-existent agent."""
        agent = registry.get_agent("non_existent")
        assert agent is None
    
    def test_get_agents_by_capability(self, registry, sample_metadata):
        """Test getting agents by capability."""
        registry.register_agent(MockAgent, sample_metadata)
        
        agents = registry.get_agents_by_capability(AgentCapability.QUERY_REWRITING)
        assert "test_agent" in agents
        
        agents = registry.get_agents_by_capability(AgentCapability.CONTENT_GENERATION)
        assert "test_agent" not in agents
    
    def test_get_agents_by_capabilities(self, registry, sample_metadata):
        """Test getting agents by multiple capabilities."""
        registry.register_agent(MockAgent, sample_metadata)
        
        # Test with both capabilities
        agents = registry.get_agents_by_capabilities({
            AgentCapability.QUERY_REWRITING,
            AgentCapability.CONTEXT_ASSESSMENT
        })
        assert "test_agent" in agents
        
        # Test with one capability that doesn't match
        agents = registry.get_agents_by_capabilities({
            AgentCapability.QUERY_REWRITING,
            AgentCapability.CONTENT_GENERATION
        })
        assert "test_agent" not in agents
    
    def test_get_agents_by_capabilities_empty(self, registry):
        """Test getting agents with empty capability set."""
        agents = registry.get_agents_by_capabilities(set())
        assert agents == []
    
    def test_list_agents(self, registry, sample_metadata):
        """Test listing all agents."""
        registry.register_agent(MockAgent, sample_metadata)
        
        agents = registry.list_agents()
        assert len(agents) == 1
        assert agents[0]["name"] == "test_agent"
        assert agents[0]["description"] == "A test agent"
        assert "query_rewriting" in agents[0]["capabilities"]
    
    def test_list_capabilities(self, registry, sample_metadata):
        """Test listing all capabilities."""
        registry.register_agent(MockAgent, sample_metadata)
        
        capabilities = registry.list_capabilities()
        assert "query_rewriting" in capabilities
        assert "context_assessment" in capabilities
        assert "test_agent" in capabilities["query_rewriting"]
        assert "test_agent" in capabilities["context_assessment"]
    
    def test_plugin_config_validation(self, registry):
        """Test plugin configuration validation."""
        plugin = MockAgentPlugin()
        registry.register_plugin(plugin)
        
        # Valid config
        agent = registry.get_agent("mock_plugin_agent", {"name": "valid_name"})
        assert agent is not None
        
        # Invalid config
        agent = registry.get_agent("mock_plugin_agent", {"name": 123})
        assert agent is None
    
    def test_unregister_agent(self, registry, sample_metadata):
        """Test unregistering an agent."""
        registry.register_agent(MockAgent, sample_metadata)
        
        # Verify registration
        assert "test_agent" in registry._agents
        
        # Unregister
        result = registry.unregister_agent("test_agent")
        assert result is True
        
        # Verify removal
        assert "test_agent" not in registry._agents
        assert "test_agent" not in registry._metadata
        assert "test_agent" not in registry._capability_index[AgentCapability.QUERY_REWRITING]
    
    def test_unregister_plugin(self, registry):
        """Test unregistering a plugin."""
        plugin = MockAgentPlugin()
        registry.register_plugin(plugin)
        
        # Verify registration
        assert "mock_plugin_agent" in registry._plugins
        
        # Unregister
        result = registry.unregister_agent("mock_plugin_agent")
        assert result is True
        
        # Verify removal
        assert "mock_plugin_agent" not in registry._plugins
        assert "mock_plugin_agent" not in registry._metadata
        assert "mock_plugin_agent" not in registry._capability_index[AgentCapability.CONTENT_GENERATION]
    
    def test_unregister_nonexistent(self, registry):
        """Test unregistering a non-existent agent."""
        result = registry.unregister_agent("non_existent")
        assert result is False
    
    def test_get_agent_metadata(self, registry, sample_metadata):
        """Test getting agent metadata."""
        registry.register_agent(MockAgent, sample_metadata)
        
        metadata = registry.get_agent_metadata("test_agent")
        assert metadata is not None
        assert metadata.name == "test_agent"
        assert metadata.description == "A test agent"
    
    def test_get_agent_metadata_not_found(self, registry):
        """Test getting metadata for non-existent agent."""
        metadata = registry.get_agent_metadata("non_existent")
        assert metadata is None


class TestAgentRegistryPersistence:
    """Test registry persistence functionality."""
    
    def test_export_registry(self, registry, sample_metadata):
        """Test exporting registry to JSON."""
        registry.register_agent(MockAgent, sample_metadata)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            registry.export_registry(filepath)
            
            # Verify file was created and contains expected data
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert "agents" in data
            assert "capabilities" in data
            assert "total_agents" in data
            assert data["total_agents"] == 1
            
            agents = data["agents"]
            assert len(agents) == 1
            assert agents[0]["name"] == "test_agent"
            
        finally:
            os.unlink(filepath)
    
    def test_import_registry(self, registry):
        """Test importing registry from JSON."""
        # Create test data
        test_data = {
            "agents": [
                {
                    "name": "imported_agent",
                    "description": "An imported agent",
                    "capabilities": ["query_rewriting"],
                    "version": "1.0.0",
                    "author": "Test Author",
                    "dependencies": [],
                    "config_schema": {}
                }
            ],
            "capabilities": {
                "query_rewriting": ["imported_agent"]
            },
            "total_agents": 1
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            filepath = f.name
        
        try:
            registry.import_registry(filepath)
            
            # Verify metadata was imported
            metadata = registry.get_agent_metadata("imported_agent")
            assert metadata is not None
            assert metadata.name == "imported_agent"
            assert metadata.description == "An imported agent"
            
            # Verify capability indexing
            assert "imported_agent" in registry._capability_index[AgentCapability.QUERY_REWRITING]
            
        finally:
            os.unlink(filepath)


class TestAgentRegistryPluginDiscovery:
    """Test plugin discovery functionality."""
    
    def test_discover_plugins_basic(self, registry):
        """Test basic plugin discovery functionality."""
        # Test that discover_plugins doesn't crash with non-existent paths
        registry.discover_plugins(["non_existent_path"])
        
        # Test that discover_plugins doesn't crash with empty paths
        registry.discover_plugins([])
        
        # Test that discover_plugins doesn't crash with None paths
        registry.discover_plugins(None)
        
        # Verify the registry is still functional
        assert len(registry._plugins) == 0
        assert len(registry._agents) == 0


class TestAgentMetadata:
    """Test agent metadata functionality."""
    
    def test_metadata_creation(self):
        """Test creating agent metadata."""
        metadata = AgentMetadata(
            name="test_agent",
            description="Test description",
            capabilities={AgentCapability.QUERY_REWRITING}
        )
        
        assert metadata.name == "test_agent"
        assert metadata.description == "Test description"
        assert AgentCapability.QUERY_REWRITING in metadata.capabilities
        assert metadata.version == "1.0.0"
        assert metadata.author == "Unknown"
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = AgentMetadata(
            name="test_agent",
            description="Test description",
            capabilities={AgentCapability.QUERY_REWRITING, AgentCapability.CONTENT_GENERATION}
        )
        
        data = metadata.to_dict()
        assert data["name"] == "test_agent"
        assert data["description"] == "Test description"
        assert "query_rewriting" in data["capabilities"]
        assert "content_generation" in data["capabilities"]
    
    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "name": "test_agent",
            "description": "Test description",
            "capabilities": ["query_rewriting", "content_generation"],
            "version": "2.0.0",
            "author": "Test Author"
        }
        
        metadata = AgentMetadata.from_dict(data)
        assert metadata.name == "test_agent"
        assert metadata.description == "Test description"
        assert AgentCapability.QUERY_REWRITING in metadata.capabilities
        assert AgentCapability.CONTENT_GENERATION in metadata.capabilities
        assert metadata.version == "2.0.0"
        assert metadata.author == "Test Author" 