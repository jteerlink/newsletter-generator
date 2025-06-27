"""
Integration tests for the extensibility system.
Demonstrates how the plugin/registration system works in practice.
"""

import pytest
import tempfile
import json
import os
from typing import Dict, Any

from agents.base.agent_registry import (
    AgentRegistry, AgentPlugin, AgentMetadata, AgentCapability
)
from agents.base.tool_registry import (
    ToolRegistry, ToolPlugin, ToolMetadata, ToolCapability
)
from agents.base.agent_base import AgentBase
from agents.base.agent_memory import AgentMemory


class TestSpecializedAgent(AgentBase):
    """A specialized agent for testing extensibility."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id=agent_id)
        self.specialization = "test_specialization"
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a query with specialized knowledge."""
        enhanced_query = f"[{self.specialization.upper()}] {query}"
        
        # Store in memory
        query_data = {
            'query': enhanced_query,
            'context': context,
            'timestamp': self.memory.get_current_timestamp()
        }
        self.memory.add_query(query_data)
        
        result = {
            'query': enhanced_query,
            'specialization': self.specialization,
            'processed_at': self.memory.get_current_timestamp(),
            'confidence': 0.85
        }
        
        self.memory.add_response(result)
        return result
    
    def run(self):
        """Main execution loop."""
        return {'status': 'running', 'agent': self.agent_id}
    
    def receive_message(self, message: dict):
        """Handle incoming messages."""
        return {'status': 'received', 'agent': self.agent_id, 'message': message}
    
    def send_message(self, recipient_id: str, message: dict):
        """Send a message to another agent."""
        return {'status': 'sent', 'agent': self.agent_id, 'recipient': recipient_id}


class TestDataProcessor:
    """A specialized data processing tool for testing extensibility."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.processing_mode = config.get('processing_mode', 'standard')
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the data processing tool."""
        data = inputs.get('data', [])
        operation = inputs.get('operation', 'filter')
        
        if operation == 'filter':
            result = self._filter_data(data, inputs.get('filter_criteria', {}))
        elif operation == 'transform':
            result = self._transform_data(data, inputs.get('transform_rules', {}))
        else:
            result = {'error': f'Unknown operation: {operation}'}
        
        return {
            'processed_data': result,
            'processing_mode': self.processing_mode,
            'operation': operation,
            'input_count': len(data),
            'output_count': len(result) if isinstance(result, list) else 1
        }
    
    def _filter_data(self, data: list, criteria: Dict[str, Any]) -> list:
        """Filter data based on criteria."""
        filtered = []
        for item in data:
            if isinstance(item, dict):
                include = True
                for key, value in criteria.items():
                    if key in item and item[key] != value:
                        include = False
                        break
                if include:
                    filtered.append(item)
        return filtered
    
    def _transform_data(self, data: list, rules: Dict[str, Any]) -> list:
        """Transform data based on rules."""
        transformed = []
        for item in data:
            if isinstance(item, dict):
                new_item = item.copy()
                for field, transformation in rules.items():
                    if field in new_item:
                        if transformation == 'uppercase':
                            new_item[field] = str(new_item[field]).upper()
                        elif transformation == 'lowercase':
                            new_item[field] = str(new_item[field]).lower()
                transformed.append(new_item)
        return transformed


class TestAgentPlugin(AgentPlugin):
    """Test agent plugin for integration testing."""
    
    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        return AgentMetadata(
            name="test_specialized_agent",
            description="A test specialized agent for integration testing",
            capabilities={
                AgentCapability.CONTENT_GENERATION,
                AgentCapability.RESEARCH,
                AgentCapability.TECHNICAL_WRITING
            },
            version="1.0.0",
            author="Integration Test",
            dependencies=["agents.base.agent_base"],
            config_schema={
                "specialization": {
                    "type": "string",
                    "default": "general",
                    "description": "Agent specialization area"
                }
            }
        )
    
    def create_agent(self, config: Dict[str, Any] = None) -> AgentBase:
        """Create and return an agent instance."""
        if config is None:
            config = {}
        
        agent_id = config.get('agent_id', 'test_agent')
        return TestSpecializedAgent(agent_id=agent_id)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        if not config:
            return True
        
        # Validate agent_id
        agent_id = config.get('agent_id')
        if agent_id and not isinstance(agent_id, str):
            return False
        
        return True


class TestToolPlugin(ToolPlugin):
    """Test tool plugin for integration testing."""
    
    def get_metadata(self) -> ToolMetadata:
        """Return tool metadata."""
        return ToolMetadata(
            name="test_data_processor",
            description="A test data processing tool for integration testing",
            capabilities={
                ToolCapability.DATA_EXTRACTION,
                ToolCapability.DOCUMENT_PROCESSING,
                ToolCapability.STATISTICAL_ANALYSIS
            },
            version="1.0.0",
            author="Integration Test",
            dependencies=["agents.base.tool_registry"],
            config_schema={
                "processing_mode": {
                    "type": "string",
                    "enum": ["standard", "fast", "thorough"],
                    "default": "standard",
                    "description": "Processing mode for the tool"
                }
            },
            input_schema={
                "data": {
                    "type": "array",
                    "description": "Data to be processed"
                },
                "operation": {
                    "type": "string",
                    "enum": ["filter", "transform"],
                    "description": "Operation to perform"
                }
            },
            output_schema={
                "processed_data": {
                    "type": "any",
                    "description": "Processed data result"
                },
                "processing_mode": {
                    "type": "string",
                    "description": "Mode used for processing"
                }
            }
        )
    
    def create_tool(self, config: Dict[str, Any] = None) -> Any:
        """Create and return a tool instance."""
        if config is None:
            config = {}
        
        return TestDataProcessor(config=config)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate tool configuration."""
        if not config:
            return True
        
        # Validate processing mode
        processing_mode = config.get('processing_mode')
        valid_modes = ['standard', 'fast', 'thorough']
        if processing_mode and processing_mode not in valid_modes:
            return False
        
        return True
    
    def execute(self, tool_instance: Any, inputs: Dict[str, Any]) -> Any:
        """Execute the tool with given inputs."""
        return tool_instance.execute(inputs)


class TestExtensibilityIntegration:
    """Integration tests for the extensibility system."""
    
    def test_agent_plugin_integration(self):
        """Test complete agent plugin integration workflow."""
        # Create registry
        registry = AgentRegistry()
        
        # Register plugin
        plugin = TestAgentPlugin()
        registry.register_plugin(plugin)
        
        # Verify registration
        assert "test_specialized_agent" in registry._plugins
        assert registry._plugins["test_specialized_agent"] == plugin
        
        # Get agent metadata
        metadata = registry.get_agent_metadata("test_specialized_agent")
        assert metadata is not None
        assert metadata.name == "test_specialized_agent"
        assert AgentCapability.CONTENT_GENERATION in metadata.capabilities
        
        # Create agent instance
        agent = registry.get_agent("test_specialized_agent", {"agent_id": "test_agent_1"})
        assert agent is not None
        assert isinstance(agent, TestSpecializedAgent)
        assert agent.agent_id == "test_agent_1"
        
        # Test agent functionality
        result = agent.process_query("test query", {"context": "test"})
        assert result["query"] == "[TEST_SPECIALIZATION] test query"
        assert result["specialization"] == "test_specialization"
        assert result["confidence"] == 0.85
        
        # Test memory integration
        memory = agent.get_memory()
        assert memory is not None
        assert len(memory.get_queries()) > 0
        
        # Test capability-based lookup
        agents = registry.get_agents_by_capability(AgentCapability.CONTENT_GENERATION)
        assert "test_specialized_agent" in agents
        
        # Test unregistration
        success = registry.unregister_agent("test_specialized_agent")
        assert success is True
        assert "test_specialized_agent" not in registry._plugins
    
    def test_tool_plugin_integration(self):
        """Test complete tool plugin integration workflow."""
        # Create registry
        tool_registry = ToolRegistry()
        
        # Register plugin
        plugin = TestToolPlugin()
        tool_registry.register_plugin(plugin)
        
        # Verify registration
        assert "test_data_processor" in tool_registry._plugins
        assert tool_registry._plugins["test_data_processor"] == plugin
        
        # Get tool metadata
        metadata = tool_registry.get_tool_metadata("test_data_processor")
        assert metadata is not None
        assert metadata.name == "test_data_processor"
        assert ToolCapability.DATA_EXTRACTION in metadata.capabilities
        
        # Create tool instance
        tool = tool_registry.get_tool("test_data_processor", {"processing_mode": "fast"})
        assert tool is not None
        assert isinstance(tool, TestDataProcessor)
        assert tool.processing_mode == "fast"
        
        # Test tool execution
        test_data = [
            {"name": "Alice", "age": 30, "city": "New York"},
            {"name": "Bob", "age": 25, "city": "Boston"},
            {"name": "Charlie", "age": 35, "city": "New York"}
        ]
        
        # Test filtering
        filter_result = tool_registry.execute_tool(
            "test_data_processor",
            {
                "data": test_data,
                "operation": "filter",
                "filter_criteria": {"city": "New York"}
            },
            {"processing_mode": "standard"}
        )
        
        assert filter_result is not None
        assert filter_result["processing_mode"] == "standard"
        assert filter_result["operation"] == "filter"
        assert filter_result["input_count"] == 3
        assert filter_result["output_count"] == 2
        
        # Test transformation
        transform_result = tool_registry.execute_tool(
            "test_data_processor",
            {
                "data": test_data,
                "operation": "transform",
                "transform_rules": {"name": "uppercase"}
            },
            {"processing_mode": "thorough"}
        )
        
        assert transform_result is not None
        assert transform_result["processing_mode"] == "thorough"
        assert transform_result["operation"] == "transform"
        assert transform_result["input_count"] == 3
        assert transform_result["output_count"] == 3
        
        # Test capability-based lookup
        tools = tool_registry.get_tools_by_capability(ToolCapability.DATA_EXTRACTION)
        assert "test_data_processor" in tools
        
        # Test input validation
        validation = tool_registry.validate_tool_inputs("test_data_processor", {"data": [], "operation": "filter"})
        assert validation["data"] is True
        assert validation["operation"] is True
        
        # Test unregistration
        success = tool_registry.unregister_tool("test_data_processor")
        assert success is True
        assert "test_data_processor" not in tool_registry._plugins
    
    def test_registry_persistence_integration(self):
        """Test registry persistence integration."""
        # Create registries
        agent_registry = AgentRegistry()
        tool_registry = ToolRegistry()
        
        # Register plugins
        agent_plugin = TestAgentPlugin()
        tool_plugin = TestToolPlugin()
        
        agent_registry.register_plugin(agent_plugin)
        tool_registry.register_plugin(tool_plugin)
        
        # Export registries
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            agent_filepath = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            tool_filepath = f.name
        
        try:
            agent_registry.export_registry(agent_filepath)
            tool_registry.export_registry(tool_filepath)
            
            # Verify files were created
            assert os.path.exists(agent_filepath)
            assert os.path.exists(tool_filepath)
            
            # Load exported data
            with open(agent_filepath, 'r') as f:
                agent_data = json.load(f)
            
            with open(tool_filepath, 'r') as f:
                tool_data = json.load(f)
            
            # Verify data structure
            assert "agents" in agent_data
            assert "capabilities" in agent_data
            assert "total_agents" in agent_data
            assert agent_data["total_agents"] == 1
            
            assert "tools" in tool_data
            assert "capabilities" in tool_data
            assert "total_tools" in tool_data
            assert tool_data["total_tools"] == 1
            
            # Create new registries and import
            new_agent_registry = AgentRegistry()
            new_tool_registry = ToolRegistry()
            
            new_agent_registry.import_registry(agent_filepath)
            new_tool_registry.import_registry(tool_filepath)
            
            # Verify metadata was imported
            agent_metadata = new_agent_registry.get_agent_metadata("test_specialized_agent")
            assert agent_metadata is not None
            assert agent_metadata.name == "test_specialized_agent"
            
            tool_metadata = new_tool_registry.get_tool_metadata("test_data_processor")
            assert tool_metadata is not None
            assert tool_metadata.name == "test_data_processor"
            
        finally:
            os.unlink(agent_filepath)
            os.unlink(tool_filepath)
    
    def test_multi_capability_integration(self):
        """Test multi-capability agent and tool integration."""
        # Create registries
        agent_registry = AgentRegistry()
        tool_registry = ToolRegistry()
        
        # Register plugins
        agent_plugin = TestAgentPlugin()
        tool_plugin = TestToolPlugin()
        
        agent_registry.register_plugin(agent_plugin)
        tool_registry.register_plugin(tool_plugin)
        
        # Test multi-capability lookup
        content_agents = agent_registry.get_agents_by_capability(AgentCapability.CONTENT_GENERATION)
        research_agents = agent_registry.get_agents_by_capability(AgentCapability.RESEARCH)
        
        assert "test_specialized_agent" in content_agents
        assert "test_specialized_agent" in research_agents
        
        # Test intersection of capabilities
        multi_capable_agents = agent_registry.get_agents_by_capabilities({
            AgentCapability.CONTENT_GENERATION,
            AgentCapability.RESEARCH
        })
        assert "test_specialized_agent" in multi_capable_agents
        
        # Test tool capabilities
        data_tools = tool_registry.get_tools_by_capability(ToolCapability.DATA_EXTRACTION)
        processing_tools = tool_registry.get_tools_by_capability(ToolCapability.DOCUMENT_PROCESSING)
        
        assert "test_data_processor" in data_tools
        assert "test_data_processor" in processing_tools
        
        # Test tool capability intersection
        multi_capable_tools = tool_registry.get_tools_by_capabilities({
            ToolCapability.DATA_EXTRACTION,
            ToolCapability.DOCUMENT_PROCESSING
        })
        assert "test_data_processor" in multi_capable_tools
    
    def test_config_validation_integration(self):
        """Test configuration validation integration."""
        # Create registries
        agent_registry = AgentRegistry()
        tool_registry = ToolRegistry()
        
        # Register plugins
        agent_plugin = TestAgentPlugin()
        tool_plugin = TestToolPlugin()
        
        agent_registry.register_plugin(agent_plugin)
        tool_registry.register_plugin(tool_plugin)
        
        # Test valid agent config
        valid_agent = agent_registry.get_agent("test_specialized_agent", {"agent_id": "valid_agent"})
        assert valid_agent is not None
        
        # Test invalid agent config
        invalid_agent = agent_registry.get_agent("test_specialized_agent", {"agent_id": 123})
        assert invalid_agent is None
        
        # Test valid tool config
        valid_tool = tool_registry.get_tool("test_data_processor", {"processing_mode": "fast"})
        assert valid_tool is not None
        
        # Test invalid tool config
        invalid_tool = tool_registry.get_tool("test_data_processor", {"processing_mode": "invalid_mode"})
        assert invalid_tool is None 