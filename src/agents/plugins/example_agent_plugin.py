"""
Example Agent Plugin
Demonstrates how to create a custom agent plugin for the extensible agent system.
"""

from typing import Dict, Any, Set
from agents.base.agent_registry import AgentPlugin, AgentMetadata, AgentCapability
from agents.base.agent_base import AgentBase
from agents.base.agent_memory import AgentMemory


class ExampleSpecializedAgent(AgentBase):
    """
    Example specialized agent that demonstrates plugin functionality.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config=config)
        self.specialization = config.get('specialization', 'general')
        self.expertise_level = config.get('expertise_level', 'intermediate')
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query with specialized knowledge.
        """
        # Add specialized processing logic here
        enhanced_query = f"[{self.specialization.upper()}] {query}"
        
        # Store in memory
        self.memory.add_query(enhanced_query, context)
        
        # Simulate specialized processing
        result = {
            'query': enhanced_query,
            'specialization': self.specialization,
            'expertise_level': self.expertise_level,
            'processed_at': self.memory.get_current_timestamp(),
            'confidence': 0.85
        }
        
        self.memory.add_response(enhanced_query, result)
        return result
    
    def get_capabilities(self) -> Set[str]:
        """Return agent capabilities."""
        return {self.specialization, self.expertise_level}


class ExampleAgentPlugin(AgentPlugin):
    """
    Example agent plugin that demonstrates the plugin system.
    """
    
    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        return AgentMetadata(
            name="example_specialized_agent",
            description="An example specialized agent for demonstration purposes",
            capabilities={
                AgentCapability.CONTENT_GENERATION,
                AgentCapability.RESEARCH,
                AgentCapability.TECHNICAL_WRITING
            },
            version="1.0.0",
            author="Example Author",
            dependencies=["agents.base.agent_base"],
            config_schema={
                "specialization": {
                    "type": "string",
                    "default": "general",
                    "description": "Agent specialization area"
                },
                "expertise_level": {
                    "type": "string",
                    "enum": ["beginner", "intermediate", "expert"],
                    "default": "intermediate",
                    "description": "Agent expertise level"
                }
            }
        )
    
    def create_agent(self, config: Dict[str, Any] = None) -> AgentBase:
        """Create and return an agent instance."""
        if config is None:
            config = {}
        
        return ExampleSpecializedAgent(config=config)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        if not config:
            return True
        
        # Validate specialization
        specialization = config.get('specialization')
        if specialization and not isinstance(specialization, str):
            return False
        
        # Validate expertise level
        expertise_level = config.get('expertise_level')
        valid_levels = ['beginner', 'intermediate', 'expert']
        if expertise_level and expertise_level not in valid_levels:
            return False
        
        return True 