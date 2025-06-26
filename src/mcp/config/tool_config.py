import os

class ToolConfig:
    """Loads and stores configuration for individual MCP tools."""
    def __init__(self, tool_name, config_path=None):
        self.tool_name = tool_name
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """Load tool configuration from file or environment (placeholder)."""
        # Placeholder: In a real implementation, load from file or env vars
        if self.tool_name == "web_search":
            return {
                "providers": ["google", "duckduckgo"],
                "rate_limit": "100/hour",
                "cache_ttl": 3600
            }
        elif self.tool_name == "vector_search":
            return {
                "max_results": 50,
                "default_threshold": 0.7
            }
        else:
            return {}
