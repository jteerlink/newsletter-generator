from .base_tool import BaseTool
import os
import requests
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

class WebSearchTool(BaseTool):
    """Tool for searching the web for current information on AI topics."""
    name = "web_search"
    description = "Search the web for current information on AI topics"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query for AI-related content"},
            "max_results": {"type": "integer", "default": 10, "description": "Maximum number of results to return"},
            "date_range": {"type": "string", "enum": ["day", "week", "month", "year"], "default": "week", "description": "Time range for search results"},
            "content_type": {"type": "string", "enum": ["news", "research", "blog", "all"], "default": "all", "description": "Type of content to search for"}
        },
        "required": ["query"]
    }

    def run(self, query, max_results=10, date_range="week", content_type="all"):
        """Run the web search tool using DuckDuckGo by default, Google Custom Search only if enabled."""
        results = []
        metadata = {"search_time": None, "results_count": 0, "cache_hit": False}
        # Check if Google Custom Search is explicitly enabled
        use_google = os.getenv("WEB_SEARCH_USE_GOOGLE", "false").lower() == "true"
        api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        cx = os.getenv("GOOGLE_SEARCH_CX")
        if use_google and api_key and cx:
            try:
                params = {
                    "key": api_key,
                    "cx": cx,
                    "q": query,
                    "num": min(max_results, 10),
                }
                url = "https://www.googleapis.com/customsearch/v1"
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                for item in data.get("items", []):
                    results.append({
                        "title": item.get("title"),
                        "url": item.get("link"),
                        "snippet": item.get("snippet"),
                        "source": "google"
                    })
                metadata["results_count"] = len(results)
                return {"results": results, "metadata": metadata}
            except Exception as e:
                pass  # Fallback to DuckDuckGo
        # Default: DuckDuckGo
        if DDGS:
            try:
                with DDGS() as ddgs:
                    ddg_results = ddgs.text(query, max_results=max_results)
                    for r in ddg_results:
                        results.append({
                            "title": r.get("title"),
                            "url": r.get("href"),
                            "snippet": r.get("body"),
                            "source": "duckduckgo"
                        })
                metadata["results_count"] = len(results)
                return {"results": results, "metadata": metadata}
            except Exception as e:
                pass
        # If all fails, return empty
        return {"results": [], "metadata": metadata}
