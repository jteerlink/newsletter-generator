"""
Tool registry for MCP server.
"""

class ToolRegistry:
    """Manages registration and lookup of MCP tools."""
    def __init__(self):
        self.tools = {}

    def register_tool(self, tool_name, tool_instance):
        """Register a tool by name."""
        self.tools[tool_name] = tool_instance

    def unregister_tool(self, tool_name):
        """Remove a tool from the registry."""
        if tool_name in self.tools:
            del self.tools[tool_name]

    def get_tool(self, tool_name):
        """Retrieve a tool instance by name."""
        return self.tools.get(tool_name)
