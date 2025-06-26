from .web_search_tool import WebSearchTool
from .vector_search_tool import VectorSearchTool

class ToolFactory:
    """Factory for creating MCP tool instances by name."""
    TOOL_REGISTRY = {
        "web_search": WebSearchTool,
        "vector_search": VectorSearchTool,
    }

    @staticmethod
    def create_tool(tool_name, config=None):
        """Create and return a tool instance by name."""
        tool_cls = ToolFactory.TOOL_REGISTRY.get(tool_name)
        if tool_cls is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return tool_cls(config=config)
