"""
Protocol handler for MCP server.
"""

class ProtocolHandler:
    """Handles incoming protocol messages and dispatches tool calls."""
    def __init__(self, server):
        self.server = server

    def handle_message(self, message):
        """Process an incoming message and dispatch to the appropriate tool or handler.
        Expects message dict with 'tool_call': {'tool': tool_name, 'parameters': {...}}
        Returns a dict with 'response' or 'error'.
        """
        try:
            tool_call = message.get('tool_call')
            if not tool_call:
                return {"error": "No tool_call in message"}
            tool_name = tool_call.get('tool')
            parameters = tool_call.get('parameters', {})
            if not tool_name:
                return {"error": "No tool specified in tool_call"}
            tool = self.server.tools.get(tool_name)
            if not tool:
                return {"error": f"Tool '{tool_name}' not registered"}
            response = tool.run(**parameters)
            return {"response": response}
        except Exception as e:
            return {"error": str(e)}
