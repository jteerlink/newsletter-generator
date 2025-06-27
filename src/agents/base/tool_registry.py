"""
Tool Registry with Plugin/Registration System
Supports dynamic tool discovery, capability-based registration, and extensibility.
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


class ToolCapability(Enum):
    """Enumeration of tool capabilities for dynamic registration."""
    WEB_SEARCH = "web_search"
    VECTOR_SEARCH = "vector_search"
    DOCUMENT_PROCESSING = "document_processing"
    DATA_EXTRACTION = "data_extraction"
    CONTENT_ANALYSIS = "content_analysis"
    TEXT_GENERATION = "text_generation"
    IMAGE_PROCESSING = "image_processing"
    FILE_OPERATIONS = "file_operations"
    DATABASE_OPERATIONS = "database_operations"
    API_INTEGRATION = "api_integration"
    MACHINE_LEARNING = "machine_learning"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    NETWORK_OPERATIONS = "network_operations"
    SYSTEM_OPERATIONS = "system_operations"


@dataclass
class ToolMetadata:
    """Metadata for tool registration."""
    name: str
    description: str
    capabilities: Set[ToolCapability]
    version: str = "1.0.0"
    author: str = "Unknown"
    dependencies: List[str] = None
    config_schema: Dict[str, Any] = None
    input_schema: Dict[str, Any] = None
    output_schema: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.config_schema is None:
            self.config_schema = {}
        if self.input_schema is None:
            self.input_schema = {}
        if self.output_schema is None:
            self.output_schema = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['capabilities'] = [cap.value for cap in self.capabilities]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolMetadata':
        """Create from dictionary."""
        data['capabilities'] = {ToolCapability(cap) for cap in data.get('capabilities', [])}
        return cls(**data)


class ToolPlugin(ABC):
    """Abstract base class for tool plugins."""
    
    @abstractmethod
    def get_metadata(self) -> ToolMetadata:
        """Return tool metadata."""
        pass
    
    @abstractmethod
    def create_tool(self, config: Dict[str, Any] = None) -> Any:
        """Create and return a tool instance."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate tool configuration."""
        pass
    
    @abstractmethod
    def execute(self, tool_instance: Any, inputs: Dict[str, Any]) -> Any:
        """Execute the tool with given inputs."""
        pass


class ToolRegistry:
    """
    Enhanced tool registry with plugin support and dynamic registration.
    """
    
    def __init__(self):
        self._tools: Dict[str, Type] = {}
        self._plugins: Dict[str, ToolPlugin] = {}
        self._metadata: Dict[str, ToolMetadata] = {}
        self._capability_index: Dict[ToolCapability, Set[str]] = {}
        self._logger = logging.getLogger(__name__)
        
        # Initialize capability index
        for capability in ToolCapability:
            self._capability_index[capability] = set()
    
    def register_tool(self, tool_class: Type, metadata: ToolMetadata) -> None:
        """
        Register a tool class with metadata.
        
        Args:
            tool_class: The tool class to register
            metadata: Tool metadata including capabilities
        """
        tool_name = metadata.name
        
        if tool_name in self._tools:
            self._logger.warning(f"Tool '{tool_name}' already registered, overwriting")
        
        self._tools[tool_name] = tool_class
        self._metadata[tool_name] = metadata
        
        # Update capability index
        for capability in metadata.capabilities:
            self._capability_index[capability].add(tool_name)
        
        self._logger.info(f"Registered tool '{tool_name}' with capabilities: {[cap.value for cap in metadata.capabilities]}")
    
    def register_plugin(self, plugin: ToolPlugin) -> None:
        """
        Register a tool plugin.
        
        Args:
            plugin: The tool plugin to register
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
    
    def get_tool(self, name: str, config: Dict[str, Any] = None) -> Optional[Any]:
        """
        Get a tool instance by name.
        
        Args:
            name: Tool name
            config: Configuration for the tool
            
        Returns:
            Tool instance or None if not found
        """
        if name in self._tools:
            # Direct tool class registration
            tool_class = self._tools[name]
            return tool_class(config=config)
        elif name in self._plugins:
            # Plugin-based tool
            plugin = self._plugins[name]
            if config and not plugin.validate_config(config):
                self._logger.error(f"Invalid config for tool '{name}'")
                return None
            return plugin.create_tool(config)
        
        self._logger.error(f"Tool '{name}' not found")
        return None
    
    def execute_tool(self, name: str, inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Optional[Any]:
        """
        Execute a tool by name with inputs.
        
        Args:
            name: Tool name
            inputs: Input parameters for the tool
            config: Configuration for the tool
            
        Returns:
            Tool execution result or None if failed
        """
        if name in self._plugins:
            plugin = self._plugins[name]
            if config and not plugin.validate_config(config):
                self._logger.error(f"Invalid config for tool '{name}'")
                return None
            
            tool_instance = plugin.create_tool(config)
            return plugin.execute(tool_instance, inputs)
        elif name in self._tools:
            # For direct tool classes, assume they have an execute method
            tool_instance = self.get_tool(name, config)
            if tool_instance and hasattr(tool_instance, 'execute'):
                return tool_instance.execute(inputs)
            else:
                self._logger.error(f"Tool '{name}' does not have an execute method")
                return None
        
        self._logger.error(f"Tool '{name}' not found")
        return None
    
    def get_tools_by_capability(self, capability: ToolCapability) -> List[str]:
        """
        Get all tools that have a specific capability.
        
        Args:
            capability: The capability to search for
            
        Returns:
            List of tool names with the capability
        """
        return list(self._capability_index.get(capability, set()))
    
    def get_tools_by_capabilities(self, capabilities: Set[ToolCapability]) -> List[str]:
        """
        Get all tools that have all specified capabilities.
        
        Args:
            capabilities: Set of capabilities to search for
            
        Returns:
            List of tool names with all capabilities
        """
        if not capabilities:
            return list(self._tools.keys()) + list(self._plugins.keys())
        
        # Find intersection of tools with all capabilities
        tool_sets = [self._capability_index.get(cap, set()) for cap in capabilities]
        if not tool_sets:
            return []
        
        return list(set.intersection(*tool_sets))
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all registered tools with their metadata.
        
        Returns:
            List of tool metadata dictionaries
        """
        return [metadata.to_dict() for metadata in self._metadata.values()]
    
    def list_capabilities(self) -> Dict[str, List[str]]:
        """
        List all capabilities and the tools that have them.
        
        Returns:
            Dictionary mapping capabilities to tool names
        """
        return {
            capability.value: list(tool_names)
            for capability, tool_names in self._capability_index.items()
            if tool_names
        }
    
    def discover_plugins(self, plugin_paths: List[str] = None) -> None:
        """
        Discover and load tool plugins from specified paths.
        
        Args:
            plugin_paths: List of paths to search for plugins
        """
        if plugin_paths is None:
            plugin_paths = ["plugins", "src/agents/tools/plugins"]
        
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
                
                # Look for classes that inherit from ToolPlugin
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, ToolPlugin) and 
                        obj != ToolPlugin):
                        
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
            "tools": self.list_tools(),
            "capabilities": self.list_capabilities(),
            "total_tools": len(self._tools) + len(self._plugins)
        }
        
        with open(filepath, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        self._logger.info(f"Tool registry exported to {filepath}")
    
    def import_registry(self, filepath: str) -> None:
        """
        Import registry metadata from a JSON file.
        Note: This only imports metadata, not actual tool classes.
        
        Args:
            filepath: Path to the registry data file
        """
        try:
            with open(filepath, 'r') as f:
                registry_data = json.load(f)
            
            # Import metadata (tools must be registered separately)
            for tool_data in registry_data.get("tools", []):
                metadata = ToolMetadata.from_dict(tool_data)
                self._metadata[metadata.name] = metadata
                
                # Update capability index
                for capability in metadata.capabilities:
                    self._capability_index[capability].add(metadata.name)
            
            self._logger.info(f"Tool registry metadata imported from {filepath}")
            
        except Exception as e:
            self._logger.error(f"Failed to import tool registry from {filepath}: {e}")
    
    def get_tool_metadata(self, name: str) -> Optional[ToolMetadata]:
        """
        Get metadata for a specific tool.
        
        Args:
            name: Tool name
            
        Returns:
            Tool metadata or None if not found
        """
        return self._metadata.get(name)
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            name: Tool name to unregister
            
        Returns:
            True if tool was unregistered, False if not found
        """
        if name in self._tools:
            # Remove from direct tools
            del self._tools[name]
            
            # Remove from metadata and capability index
            if name in self._metadata:
                metadata = self._metadata[name]
                for capability in metadata.capabilities:
                    self._capability_index[capability].discard(name)
                del self._metadata[name]
            
            self._logger.info(f"Unregistered tool '{name}'")
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
    
    def validate_tool_inputs(self, name: str, inputs: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate tool inputs against the tool's input schema.
        
        Args:
            name: Tool name
            inputs: Input parameters to validate
            
        Returns:
            Dictionary mapping input field names to validation status
        """
        metadata = self.get_tool_metadata(name)
        if not metadata or not metadata.input_schema:
            return {}
        
        validation_results = {}
        for field, schema in metadata.input_schema.items():
            if field in inputs:
                # Basic validation - could be enhanced with JSON Schema validation
                validation_results[field] = True
            else:
                validation_results[field] = False
        
        return validation_results


# Global tool registry instance
tool_registry = ToolRegistry() 