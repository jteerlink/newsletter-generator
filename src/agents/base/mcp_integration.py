"""
MCP Client Integration for Agent Tool Access.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
import aiohttp

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Client for interacting with MCP server tools.
    """
    
    def __init__(self, server_url: str = "http://localhost:8001"):
        self.server_url = server_url
        self.session = None
        self.available_tools = []
        self._initialized = False
    
    async def initialize(self):
        """Initialize the MCP client connection."""
        if self._initialized:
            return
        
        try:
            self.session = aiohttp.ClientSession()
            
            # Get available tools from server
            async with self.session.get(f"{self.server_url}/tools") as response:
                if response.status == 200:
                    tools_data = await response.json()
                    self.available_tools = [tool["name"] for tool in tools_data.get("tools", [])]
                    logger.info(f"MCP client initialized with {len(self.available_tools)} tools")
                else:
                    logger.warning(f"Failed to get tools from MCP server: {response.status}")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            self._initialized = False
    
    async def close(self):
        """Close the MCP client connection."""
        if self.session:
            await self.session.close()
            self.session = None
        self._initialized = False
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return self.available_tools.copy()
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Call an MCP tool with given parameters.
        
        Args:
            tool_name: Name of the tool to call
            parameters: Tool parameters
            
        Returns:
            Tool response or None if failed
        """
        if not self._initialized:
            await self.initialize()
        
        if tool_name not in self.available_tools:
            logger.error(f"Tool {tool_name} not available. Available tools: {self.available_tools}")
            return None
        
        try:
            payload = {
                "tool": tool_name,
                "parameters": parameters
            }
            
            async with self.session.post(
                f"{self.server_url}/call_tool",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Successfully called MCP tool {tool_name}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"MCP tool {tool_name} call failed: {response.status} - {error_text}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"MCP tool {tool_name} call timed out")
            return None
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return None
    
    async def health_check(self) -> bool:
        """Check if MCP server is healthy."""
        try:
            if not self._initialized:
                await self.initialize()
            
            async with self.session.get(f"{self.server_url}/health") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"MCP health check failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if MCP client is available and initialized."""
        return self._initialized and self.session is not None


class MCPToolManager:
    """
    Manager for coordinating MCP tool usage across agents.
    """
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.tool_usage_stats = {}
        self.rate_limits = {}
    
    async def call_tool_with_retry(self, tool_name: str, parameters: Dict[str, Any], 
                                 max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Call MCP tool with retry logic.
        
        Args:
            tool_name: Name of the tool to call
            parameters: Tool parameters
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tool response or None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                result = await self.mcp_client.call_tool(tool_name, parameters)
                if result is not None:
                    # Update usage stats
                    if tool_name not in self.tool_usage_stats:
                        self.tool_usage_stats[tool_name] = {"calls": 0, "successes": 0, "failures": 0}
                    
                    self.tool_usage_stats[tool_name]["calls"] += 1
                    self.tool_usage_stats[tool_name]["successes"] += 1
                    return result
                
                # Update failure stats
                if tool_name not in self.tool_usage_stats:
                    self.tool_usage_stats[tool_name] = {"calls": 0, "successes": 0, "failures": 0}
                self.tool_usage_stats[tool_name]["calls"] += 1
                self.tool_usage_stats[tool_name]["failures"] += 1
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Retrying MCP tool {tool_name} in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    
            except Exception as e:
                logger.error(f"Error in retry attempt {attempt + 1} for tool {tool_name}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
        
        logger.error(f"All retry attempts failed for MCP tool {tool_name}")
        return None
    
    def get_tool_stats(self) -> Dict[str, Dict[str, int]]:
        """Get usage statistics for all tools."""
        return self.tool_usage_stats.copy()
    
    def get_tool_success_rate(self, tool_name: str) -> float:
        """Get success rate for a specific tool."""
        if tool_name not in self.tool_usage_stats:
            return 0.0
        
        stats = self.tool_usage_stats[tool_name]
        total_calls = stats["calls"]
        if total_calls == 0:
            return 0.0
        
        return stats["successes"] / total_calls
    
    async def batch_call_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """
        Make multiple tool calls in parallel.
        
        Args:
            tool_calls: List of tool call specifications
                       [{"tool": "tool_name", "parameters": {...}}, ...]
        
        Returns:
            List of results in same order as input
        """
        tasks = []
        for call in tool_calls:
            task = self.call_tool_with_retry(call["tool"], call["parameters"])
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to None
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch tool call failed: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results 