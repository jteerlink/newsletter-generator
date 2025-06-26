from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Abstract base class for MCP tools."""
    name = None
    description = None
    input_schema = None

    def __init__(self, config=None):
        self.config = config

    @abstractmethod
    def run(self, **kwargs):
        """Run the tool with the given parameters."""
        pass
