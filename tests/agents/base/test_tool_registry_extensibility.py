"""
Tests for the extensible tool registry system.
"""

import pytest
import tempfile
import json
import os
from unittest.mock import patch, MagicMock
import inspect

from agents.base.tool_registry import (
    ToolRegistry, ToolPlugin, ToolMetadata, ToolCapability
)


class MockTool:
    """Mock tool for testing."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.name = config.get('name', 'mock_tool') if config else 'mock_tool'
    
    def execute(self, inputs):
        return {'result': f'Processed: {inputs}', 'tool': self.name}


class MockToolPlugin(ToolPlugin):
    """Mock tool plugin for testing."""
    
    def get_metadata(self):
        return ToolMetadata(
            name="mock_plugin_tool",
            description="A mock tool plugin for testing",
            capabilities={
                ToolCapability.DATA_EXTRACTION,
                ToolCapability.DOCUMENT_PROCESSING
            },
            version="1.0.0",
            author="Test Author",
            dependencies=["test.dependency"],
            config_schema={
                "name": {"type": "string", "default": "plugin_tool"}
            },
            input_schema={
                "data": {"type": "string", "description": "Input data"}
            },
            output_schema={
                "result": {"type": "string", "description": "Output result"}
            }
        )
    
    def create_tool(self, config=None):
        return MockTool(config=config)
    
    def validate_config(self, config):
        if config and 'name' in config:
            return isinstance(config['name'], str)
        return True
    
    def execute(self, tool_instance, inputs):
        return tool_instance.execute(inputs)


@pytest.fixture
def tool_registry():
    """Create a fresh tool registry for each test."""
    return ToolRegistry()


@pytest.fixture
def sample_tool_metadata():
    """Sample tool metadata for testing."""
    return ToolMetadata(
        name="test_tool",
        description="A test tool",
        capabilities={
            ToolCapability.WEB_SEARCH,
            ToolCapability.VECTOR_SEARCH
        },
        version="1.0.0",
        author="Test Author",
        dependencies=["test.dep"],
        config_schema={"param": {"type": "string"}},
        input_schema={"query": {"type": "string"}},
        output_schema={"results": {"type": "array"}}
    )


class TestToolRegistryExtensibility:
    """Test the extensible tool registry functionality."""
    
    def test_register_tool_with_metadata(self, tool_registry, sample_tool_metadata):
        """Test registering a tool with metadata."""
        tool_registry.register_tool(MockTool, sample_tool_metadata)
        
        # Check registration
        assert "test_tool" in tool_registry._tools
        assert tool_registry._tools["test_tool"] == MockTool
        assert tool_registry._metadata["test_tool"] == sample_tool_metadata
        
        # Check capability indexing
        assert "test_tool" in tool_registry._capability_index[ToolCapability.WEB_SEARCH]
        assert "test_tool" in tool_registry._capability_index[ToolCapability.VECTOR_SEARCH]
    
    def test_register_plugin(self, tool_registry):
        """Test registering a tool plugin."""
        plugin = MockToolPlugin()
        tool_registry.register_plugin(plugin)
        
        # Check registration
        assert "mock_plugin_tool" in tool_registry._plugins
        assert tool_registry._plugins["mock_plugin_tool"] == plugin
        
        # Check metadata
        metadata = tool_registry._metadata["mock_plugin_tool"]
        assert metadata.name == "mock_plugin_tool"
        assert ToolCapability.DATA_EXTRACTION in metadata.capabilities
        
        # Check capability indexing
        assert "mock_plugin_tool" in tool_registry._capability_index[ToolCapability.DATA_EXTRACTION]
        assert "mock_plugin_tool" in tool_registry._capability_index[ToolCapability.DOCUMENT_PROCESSING]
    
    def test_get_tool_direct_registration(self, tool_registry, sample_tool_metadata):
        """Test getting a tool from direct registration."""
        tool_registry.register_tool(MockTool, sample_tool_metadata)
        
        tool = tool_registry.get_tool("test_tool", {"name": "custom_name"})
        assert tool is not None
        assert isinstance(tool, MockTool)
        assert tool.name == "custom_name"
    
    def test_get_tool_plugin(self, tool_registry):
        """Test getting a tool from plugin registration."""
        plugin = MockToolPlugin()
        tool_registry.register_plugin(plugin)
        
        tool = tool_registry.get_tool("mock_plugin_tool", {"name": "plugin_name"})
        assert tool is not None
        assert isinstance(tool, MockTool)
        assert tool.name == "plugin_name"
    
    def test_get_tool_not_found(self, tool_registry):
        """Test getting a non-existent tool."""
        tool = tool_registry.get_tool("non_existent")
        assert tool is None
    
    def test_execute_tool_direct_registration(self, tool_registry, sample_tool_metadata):
        """Test executing a tool from direct registration."""
        tool_registry.register_tool(MockTool, sample_tool_metadata)
        
        result = tool_registry.execute_tool("test_tool", {"query": "test"}, {"name": "custom_name"})
        assert result is not None
        assert "Processed: {'query': 'test'}" in result["result"]
        assert result["tool"] == "custom_name"
    
    def test_execute_tool_plugin(self, tool_registry):
        """Test executing a tool from plugin registration."""
        plugin = MockToolPlugin()
        tool_registry.register_plugin(plugin)
        
        result = tool_registry.execute_tool("mock_plugin_tool", {"data": "test"}, {"name": "plugin_name"})
        assert result is not None
        assert "Processed: {'data': 'test'}" in result["result"]
        assert result["tool"] == "plugin_name"
    
    def test_execute_tool_not_found(self, tool_registry):
        """Test executing a non-existent tool."""
        result = tool_registry.execute_tool("non_existent", {})
        assert result is None
    
    def test_get_tools_by_capability(self, tool_registry, sample_tool_metadata):
        """Test getting tools by capability."""
        tool_registry.register_tool(MockTool, sample_tool_metadata)
        
        tools = tool_registry.get_tools_by_capability(ToolCapability.WEB_SEARCH)
        assert "test_tool" in tools
        
        tools = tool_registry.get_tools_by_capability(ToolCapability.DATA_EXTRACTION)
        assert "test_tool" not in tools
    
    def test_get_tools_by_capabilities(self, tool_registry, sample_tool_metadata):
        """Test getting tools by multiple capabilities."""
        tool_registry.register_tool(MockTool, sample_tool_metadata)
        
        # Test with both capabilities
        tools = tool_registry.get_tools_by_capabilities({
            ToolCapability.WEB_SEARCH,
            ToolCapability.VECTOR_SEARCH
        })
        assert "test_tool" in tools
        
        # Test with one capability that doesn't match
        tools = tool_registry.get_tools_by_capabilities({
            ToolCapability.WEB_SEARCH,
            ToolCapability.DATA_EXTRACTION
        })
        assert "test_tool" not in tools
    
    def test_get_tools_by_capabilities_empty(self, tool_registry):
        """Test getting tools with empty capability set."""
        tools = tool_registry.get_tools_by_capabilities(set())
        assert tools == []
    
    def test_list_tools(self, tool_registry, sample_tool_metadata):
        """Test listing all tools."""
        tool_registry.register_tool(MockTool, sample_tool_metadata)
        
        tools = tool_registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"
        assert tools[0]["description"] == "A test tool"
        assert "web_search" in tools[0]["capabilities"]
    
    def test_list_capabilities(self, tool_registry, sample_tool_metadata):
        """Test listing all capabilities."""
        tool_registry.register_tool(MockTool, sample_tool_metadata)
        
        capabilities = tool_registry.list_capabilities()
        assert "web_search" in capabilities
        assert "vector_search" in capabilities
        assert "test_tool" in capabilities["web_search"]
        assert "test_tool" in capabilities["vector_search"]
    
    def test_plugin_config_validation(self, tool_registry):
        """Test plugin configuration validation."""
        plugin = MockToolPlugin()
        tool_registry.register_plugin(plugin)
        
        # Valid config
        tool = tool_registry.get_tool("mock_plugin_tool", {"name": "valid_name"})
        assert tool is not None
        
        # Invalid config
        tool = tool_registry.get_tool("mock_plugin_tool", {"name": 123})
        assert tool is None
    
    def test_unregister_tool(self, tool_registry, sample_tool_metadata):
        """Test unregistering a tool."""
        tool_registry.register_tool(MockTool, sample_tool_metadata)
        
        # Verify registration
        assert "test_tool" in tool_registry._tools
        
        # Unregister
        result = tool_registry.unregister_tool("test_tool")
        assert result is True
        
        # Verify removal
        assert "test_tool" not in tool_registry._tools
        assert "test_tool" not in tool_registry._metadata
        assert "test_tool" not in tool_registry._capability_index[ToolCapability.WEB_SEARCH]
    
    def test_unregister_plugin(self, tool_registry):
        """Test unregistering a plugin."""
        plugin = MockToolPlugin()
        tool_registry.register_plugin(plugin)
        
        # Verify registration
        assert "mock_plugin_tool" in tool_registry._plugins
        
        # Unregister
        result = tool_registry.unregister_tool("mock_plugin_tool")
        assert result is True
        
        # Verify removal
        assert "mock_plugin_tool" not in tool_registry._plugins
        assert "mock_plugin_tool" not in tool_registry._metadata
        assert "mock_plugin_tool" not in tool_registry._capability_index[ToolCapability.DATA_EXTRACTION]
    
    def test_unregister_nonexistent(self, tool_registry):
        """Test unregistering a non-existent tool."""
        result = tool_registry.unregister_tool("non_existent")
        assert result is False
    
    def test_get_tool_metadata(self, tool_registry, sample_tool_metadata):
        """Test getting tool metadata."""
        tool_registry.register_tool(MockTool, sample_tool_metadata)
        
        metadata = tool_registry.get_tool_metadata("test_tool")
        assert metadata is not None
        assert metadata.name == "test_tool"
        assert metadata.description == "A test tool"
    
    def test_get_tool_metadata_not_found(self, tool_registry):
        """Test getting metadata for non-existent tool."""
        metadata = tool_registry.get_tool_metadata("non_existent")
        assert metadata is None
    
    def test_validate_tool_inputs(self, tool_registry, sample_tool_metadata):
        """Test validating tool inputs."""
        tool_registry.register_tool(MockTool, sample_tool_metadata)
        
        # Valid inputs
        validation = tool_registry.validate_tool_inputs("test_tool", {"query": "test"})
        assert validation["query"] is True
        
        # Missing required input
        validation = tool_registry.validate_tool_inputs("test_tool", {})
        assert validation["query"] is False
        
        # Non-existent tool
        validation = tool_registry.validate_tool_inputs("non_existent", {})
        assert validation == {}


class TestToolRegistryPersistence:
    """Test tool registry persistence functionality."""
    
    def test_export_registry(self, tool_registry, sample_tool_metadata):
        """Test exporting tool registry to JSON."""
        tool_registry.register_tool(MockTool, sample_tool_metadata)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            tool_registry.export_registry(filepath)
            
            # Verify file was created and contains expected data
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert "tools" in data
            assert "capabilities" in data
            assert "total_tools" in data
            assert data["total_tools"] == 1
            
            tools = data["tools"]
            assert len(tools) == 1
            assert tools[0]["name"] == "test_tool"
            
        finally:
            os.unlink(filepath)
    
    def test_import_registry(self, tool_registry):
        """Test importing tool registry from JSON."""
        # Create test data
        test_data = {
            "tools": [
                {
                    "name": "imported_tool",
                    "description": "An imported tool",
                    "capabilities": ["web_search"],
                    "version": "1.0.0",
                    "author": "Test Author",
                    "dependencies": [],
                    "config_schema": {},
                    "input_schema": {},
                    "output_schema": {}
                }
            ],
            "capabilities": {
                "web_search": ["imported_tool"]
            },
            "total_tools": 1
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            filepath = f.name
        
        try:
            tool_registry.import_registry(filepath)
            
            # Verify metadata was imported
            metadata = tool_registry.get_tool_metadata("imported_tool")
            assert metadata is not None
            assert metadata.name == "imported_tool"
            assert metadata.description == "An imported tool"
            
            # Verify capability indexing
            assert "imported_tool" in tool_registry._capability_index[ToolCapability.WEB_SEARCH]
            
        finally:
            os.unlink(filepath)


class TestToolRegistryPluginDiscovery:
    """Test plugin discovery functionality."""
    
    def test_discover_plugins_basic(self, tool_registry):
        """Test basic plugin discovery functionality."""
        # Test that discover_plugins doesn't crash with non-existent paths
        tool_registry.discover_plugins(["non_existent_path"])
        
        # Test that discover_plugins doesn't crash with empty paths
        tool_registry.discover_plugins([])
        
        # Test that discover_plugins doesn't crash with None paths
        tool_registry.discover_plugins(None)
        
        # Verify the registry is still functional
        assert len(tool_registry._plugins) == 0
        assert len(tool_registry._tools) == 0


class TestToolMetadata:
    """Test tool metadata functionality."""
    
    def test_metadata_creation(self):
        """Test creating tool metadata."""
        metadata = ToolMetadata(
            name="test_tool",
            description="Test description",
            capabilities={ToolCapability.WEB_SEARCH}
        )
        
        assert metadata.name == "test_tool"
        assert metadata.description == "Test description"
        assert ToolCapability.WEB_SEARCH in metadata.capabilities
        assert metadata.version == "1.0.0"
        assert metadata.author == "Unknown"
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = ToolMetadata(
            name="test_tool",
            description="Test description",
            capabilities={ToolCapability.WEB_SEARCH, ToolCapability.VECTOR_SEARCH}
        )
        
        data = metadata.to_dict()
        assert data["name"] == "test_tool"
        assert data["description"] == "Test description"
        assert "web_search" in data["capabilities"]
        assert "vector_search" in data["capabilities"]
    
    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "name": "test_tool",
            "description": "Test description",
            "capabilities": ["web_search", "vector_search"],
            "version": "2.0.0",
            "author": "Test Author"
        }
        
        metadata = ToolMetadata.from_dict(data)
        assert metadata.name == "test_tool"
        assert metadata.description == "Test description"
        assert ToolCapability.WEB_SEARCH in metadata.capabilities
        assert ToolCapability.VECTOR_SEARCH in metadata.capabilities
        assert metadata.version == "2.0.0"
        assert metadata.author == "Test Author" 