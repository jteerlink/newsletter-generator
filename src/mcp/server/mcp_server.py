"""
MCP Server main entry point.
"""

from .protocol_handler import ProtocolHandler

class MCPServer:
    """Main class for the MCP server lifecycle management."""
    def __init__(self, config=None):
        self.config = config
        self.tools = {}
        self.protocol_handler = ProtocolHandler(self)

    def register_tool(self, tool_name, tool_instance):
        """Register a tool with the server."""
        self.tools[tool_name] = tool_instance

    def start(self):
        """Start the MCP server."""
        # Placeholder for server start logic
        print("MCP Server started.")

    def stop(self):
        """Stop the MCP server."""
        # Placeholder for server stop logic
        print("MCP Server stopped.")

    def handle_message(self, message):
        """Handle a message using the protocol handler."""
        return self.protocol_handler.handle_message(message)
