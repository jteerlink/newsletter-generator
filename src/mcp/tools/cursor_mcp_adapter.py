import json

class CursorMCPAdapter:
    """
    Adapter for registering and exposing custom MCP tools to the Cursor MCP system.
    Handles tool registration and tool call dispatching.
    """
    def __init__(self, tool, tool_name, description, input_schema):
        self.tool = tool
        self.tool_name = tool_name
        self.description = description
        self.input_schema = input_schema

    def get_registration_payload(self):
        """Return the JSON payload for tool registration with Cursor MCP."""
        return {
            "name": self.tool_name,
            "description": self.description,
            "input_schema": self.input_schema
        }

    def handle_tool_call(self, parameters):
        """Handle a tool call from Cursor MCP (parameters is a dict)."""
        # Call the underlying tool with parameters
        return self.tool.run(**parameters) 