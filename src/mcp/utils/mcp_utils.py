import requests
import json

def register_tool_with_cursor(tool_adapter, cursor_url="http://localhost:8001/mcp/tools/register"):
    """
    Register a custom tool with the Cursor MCP system via HTTP POST.
    tool_adapter: instance of CursorMCPAdapter
    cursor_url: registration endpoint (default is placeholder)
    Returns the response from Cursor MCP.
    """
    payload = tool_adapter.get_registration_payload()
    headers = {"Content-Type": "application/json"}
    response = requests.post(cursor_url, data=json.dumps(payload), headers=headers)
    return response.json()
