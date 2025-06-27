"""
Enhanced Agent Registry with Plugin/Registration System
Supports dynamic agent discovery, capability-based registration, and extensibility.
"""

import importlib
import inspect
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Type, Any, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from .agent_base import AgentBase
from .agent_memory import AgentMemory


class AgentCapability(Enum):
    """Enumeration of agent capabilities for dynamic registration."""
    QUERY_REWRITING = "query_rewriting"
    CONTEXT_ASSESSMENT = "context_assessment"
    SOURCE_SELECTION = "source_selection"
    PROMPT_BUILDING = "prompt_building"
    RESPONSE_EVALUATION = "response_evaluation"
    CONTENT_GENERATION = "content_generation"
    RESEARCH = "research"
    FACT_CHECKING = "fact_checking"
    QUALITY_ASSESSMENT = "quality_assessment"
    TREND_ANALYSIS = "trend_analysis"
    TECHNICAL_WRITING = "technical_writing"
    NEWS_WRITING = "news_writing"
    INTERVIEW_WRITING = "interview_writing"
    SUMMARY_WRITING = "summary_writing"


@dataclass
class AgentMetadata:
    """Metadata for agent registration."""
    name: str
    description: str
    capabilities: Set[AgentCapability]
    version: str = "1.0.0"
    author: str = "Unknown"
    dependencies: List[str] = None
    config_schema: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.config_schema is None:
            self.config_schema = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['capabilities'] = [cap.value for cap in self.capabilities]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMetadata':
        """Create from dictionary."""
        data['capabilities'] = {AgentCapability(cap) for cap in data.get('capabilities', [])}
        return cls(**data)


class AgentPlugin(ABC):
    """Abstract base class for agent plugins."""
    
    @abstractmethod
    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        pass
    
    @abstractmethod
    def create_agent(self, config: Dict[str, Any] = None) -> AgentBase:
        """Create and return an agent instance."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        pass


class AgentRegistry:
    """
    Enhanced agent registry with plugin support and dynamic registration.
    """
    
    def __init__(self):
        self._agents: Dict[str, Type[AgentBase]] = {}
        self._plugins: Dict[str, AgentPlugin] = {}
        self._metadata: Dict[str, AgentMetadata] = {}
        self._capability_index: Dict[AgentCapability, Set[str]] = {}
        self._logger = logging.getLogger(__name__)
        
        # Initialize capability index
        for capability in AgentCapability:
            self._capability_index[capability] = set()
    
    def register_agent(self, agent_class: Type[AgentBase], metadata: AgentMetadata) -> None:
        """
        Register an agent class with metadata.
        
        Args:
            agent_class: The agent class to register
            metadata: Agent metadata including capabilities
        """
        agent_name = metadata.name
        
        if agent_name in self._agents:
            self._logger.warning(f"Agent '{agent_name}' already registered, overwriting")
        
        self._agents[agent_name] = agent_class
        self._metadata[agent_name] = metadata
        
        # Update capability index
        for capability in metadata.capabilities:
            self._capability_index[capability].add(agent_name)
        
        self._logger.info(f"Registered agent '{agent_name}' with capabilities: {[cap.value for cap in metadata.capabilities]}")
    
    def register_plugin(self, plugin: AgentPlugin) -> None:
        """
        Register an agent plugin.
        
        Args:
            plugin: The agent plugin to register
        """
        metadata = plugin.get_metadata()
        plugin_name = metadata.name
        
        if plugin_name in self._plugins:
            self._logger.warning(f"Plugin '{plugin_name}' already registered, overwriting")
        
        self._plugins[plugin_name] = plugin
        self._metadata[plugin_name] = metadata
        
        # Update capability index
        for capability in metadata.capabilities:
            self._capability_index[capability].add(plugin_name)
        
        self._logger.info(f"Registered plugin '{plugin_name}' with capabilities: {[cap.value for cap in metadata.capabilities]}")
    
    def get_agent(self, name: str, config: Dict[str, Any] = None) -> Optional[AgentBase]:
        """
        Get an agent instance by name.
        
        Args:
            name: Agent name
            config: Configuration for the agent
            
        Returns:
            Agent instance or None if not found
        """
        if name in self._agents:
            # Direct agent class registration
            agent_class = self._agents[name]
            return agent_class(config=config)
        elif name in self._plugins:
            # Plugin-based agent
            plugin = self._plugins[name]
            if config and not plugin.validate_config(config):
                self._logger.error(f"Invalid config for agent '{name}'")
                return None
            return plugin.create_agent(config)
        
        self._logger.error(f"Agent '{name}' not found")
        return None
    
    def get_agents_by_capability(self, capability: AgentCapability) -> List[str]:
        """
        Get all agents that have a specific capability.
        
        Args:
            capability: The capability to search for
            
        Returns:
            List of agent names with the capability
        """
        return list(self._capability_index.get(capability, set()))
    
    def get_agents_by_capabilities(self, capabilities: Set[AgentCapability]) -> List[str]:
        """
        Get all agents that have all specified capabilities.
        
        Args:
            capabilities: Set of capabilities to search for
            
        Returns:
            List of agent names with all capabilities
        """
        if not capabilities:
            return list(self._agents.keys()) + list(self._plugins.keys())
        
        # Find intersection of agents with all capabilities
        agent_sets = [self._capability_index.get(cap, set()) for cap in capabilities]
        if not agent_sets:
            return []
        
        return list(set.intersection(*agent_sets))
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all registered agents with their metadata.
        
        Returns:
            List of agent metadata dictionaries
        """
        return [metadata.to_dict() for metadata in self._metadata.values()]
    
    def list_capabilities(self) -> Dict[str, List[str]]:
        """
        List all capabilities and the agents that have them.
        
        Returns:
            Dictionary mapping capabilities to agent names
        """
        return {
            capability.value: list(agent_names)
            for capability, agent_names in self._capability_index.items()
            if agent_names
        }
    
    def discover_plugins(self, plugin_paths: List[str] = None) -> None:
        """
        Discover and load agent plugins from specified paths.
        
        Args:
            plugin_paths: List of paths to search for plugins
        """
        if plugin_paths is None:
            plugin_paths = ["plugins", "src/agents/plugins"]
        
        for path in plugin_paths:
            self._discover_plugins_in_path(Path(path))
    
    def _discover_plugins_in_path(self, path: Path) -> None:
        """Discover plugins in a specific path."""
        if not path.exists():
            return
        
        for plugin_file in path.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            
            try:
                # Import the module
                module_name = f"{path.name}.{plugin_file.stem}"
                module = importlib.import_module(module_name)
                
                # Look for classes that inherit from AgentPlugin
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, AgentPlugin) and 
                        obj != AgentPlugin):
                        
                        # Create plugin instance and register
                        plugin = obj()
                        self.register_plugin(plugin)
                        
            except Exception as e:
                self._logger.error(f"Failed to load plugin from {plugin_file}: {e}")
    
    def export_registry(self, filepath: str) -> None:
        """
        Export registry metadata to a JSON file.
        
        Args:
            filepath: Path to save the registry data
        """
        registry_data = {
            "agents": self.list_agents(),
            "capabilities": self.list_capabilities(),
            "total_agents": len(self._agents) + len(self._plugins)
        }
        
        with open(filepath, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        self._logger.info(f"Registry exported to {filepath}")
    
    def import_registry(self, filepath: str) -> None:
        """
        Import registry metadata from a JSON file.
        Note: This only imports metadata, not actual agent classes.
        
        Args:
            filepath: Path to the registry data file
        """
        try:
            with open(filepath, 'r') as f:
                registry_data = json.load(f)
            
            # Import metadata (agents must be registered separately)
            for agent_data in registry_data.get("agents", []):
                metadata = AgentMetadata.from_dict(agent_data)
                self._metadata[metadata.name] = metadata
                
                # Update capability index
                for capability in metadata.capabilities:
                    self._capability_index[capability].add(metadata.name)
            
            self._logger.info(f"Registry metadata imported from {filepath}")
            
        except Exception as e:
            self._logger.error(f"Failed to import registry from {filepath}: {e}")
    
    def get_agent_metadata(self, name: str) -> Optional[AgentMetadata]:
        """
        Get metadata for a specific agent.
        
        Args:
            name: Agent name
            
        Returns:
            Agent metadata or None if not found
        """
        return self._metadata.get(name)
    
    def unregister_agent(self, name: str) -> bool:
        """
        Unregister an agent.
        
        Args:
            name: Agent name to unregister
            
        Returns:
            True if agent was unregistered, False if not found
        """
        if name in self._agents:
            # Remove from direct agents
            del self._agents[name]
            
            # Remove from metadata and capability index
            if name in self._metadata:
                metadata = self._metadata[name]
                for capability in metadata.capabilities:
                    self._capability_index[capability].discard(name)
                del self._metadata[name]
            
            self._logger.info(f"Unregistered agent '{name}'")
            return True
        
        elif name in self._plugins:
            # Remove from plugins
            del self._plugins[name]
            
            # Remove from metadata and capability index
            if name in self._metadata:
                metadata = self._metadata[name]
                for capability in metadata.capabilities:
                    self._capability_index[capability].discard(name)
                del self._metadata[name]
            
            self._logger.info(f"Unregistered plugin '{name}'")
            return True
        
        return False


# Global registry instance
registry = AgentRegistry() 