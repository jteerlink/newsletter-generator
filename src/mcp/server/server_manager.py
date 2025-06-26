"""
Server manager for MCP server lifecycle and coordination.
"""

from mcp.config.mcp_config import MCPConfig
from mcp.tools.tool_factory import ToolFactory

class ServerManager:
    """Manages the MCP server lifecycle and coordinates protocol handler and tool registry."""
    def __init__(self, server, protocol_handler, tool_registry):
        self.server = server
        self.protocol_handler = protocol_handler
        self.tool_registry = tool_registry

    def start(self):
        """Start the MCP server and initialize components."""
        self.server.start()
        print("ServerManager: MCP server started.")

    def stop(self):
        """Stop the MCP server and clean up resources."""
        self.server.stop()
        print("ServerManager: MCP server stopped.")

    def restart(self):
        """Restart the MCP server."""
        self.stop()
        self.start()
        print("ServerManager: MCP server restarted.")

    @staticmethod
    def run():
        """Basic server startup: load config, register tools, start server."""
        config = MCPConfig()
        from mcp.server.mcp_server import MCPServer
        server = MCPServer(config=config.config)
        # Register enabled tools
        for tool_name, tool_cfg in config.config.get("tools", {}).items():
            if tool_cfg.get("enabled", False):
                tool = ToolFactory.create_tool(tool_name)
                server.register_tool(tool_name, tool)
        server.start()
        print("MCP Server is running with tools:", list(server.tools.keys()))
        return server
