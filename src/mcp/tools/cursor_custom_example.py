from mcp.tools.web_search_tool import WebSearchTool
from mcp.tools.cursor_mcp_adapter import CursorMCPAdapter
from mcp.utils.mcp_utils import register_tool_with_cursor

if __name__ == "__main__":
    # Example: Register WebSearchTool as a custom tool with Cursor MCP
    tool = WebSearchTool()
    adapter = CursorMCPAdapter(
        tool=tool,
        tool_name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema
    )
    response = register_tool_with_cursor(adapter)
    print("Cursor MCP registration response:", response) 