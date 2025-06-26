import os

class MCPConfig:
    """Loads and stores MCP server configuration."""
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from file or environment (placeholder)."""
        # Placeholder: In a real implementation, load from file or env vars
        return {
            "server": {
                "host": os.getenv("MCP_SERVER_HOST", "localhost"),
                "port": int(os.getenv("MCP_SERVER_PORT", 8001)),
                "protocol": "mcp",
                "version": "1.0"
            },
            "tools": {
                "web_search": {"enabled": True},
                "vector_search": {"enabled": True}
            }
        }
