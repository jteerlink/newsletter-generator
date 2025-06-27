"""
Industry News Agent with MCP Tool Integration.

This agent covers company announcements, product launches, and market trends using web and vector search tools.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from ..base.agent_base import AgentBase
from ..base.communication import Message, MessageType

logger = logging.getLogger(__name__)

class IndustryNewsAgent(AgentBase):
    """
    Agent specialized in covering industry news and market trends using MCP tools.
    """
    def __init__(self, agent_id: str = "industry_news_agent"):
        super().__init__(agent_id)
        self.section_template = self._default_section_template

    async def run(self):
        logger.info(f"Industry News Agent {self.agent_id} started")
        while True:
            await asyncio.sleep(1)

    def receive_message(self, message: dict):
        try:
            msg = Message.from_dict(message)
            if msg.message_type == MessageType.TASK:
                asyncio.create_task(self._handle_news_task(msg.payload))
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def send_message(self, recipient_id: str, message: dict):
        logger.info(f"Sending message to {recipient_id}: {message}")

    async def _handle_news_task(self, task_data: Dict[str, Any]):
        try:
            topic = task_data.get("topic", "AI industry news")
            max_length = task_data.get("max_length", 600)
            logger.info(f"Industry news task for topic: {topic}")
            # Step 1: Vector search for local/archived news
            local_news = await self._search_local_news(topic)
            emerging = False
            # Step 2: If not enough local results, trigger web search
            if not local_news.get("results") or len(local_news["results"]) < 3:
                emerging = True
                news_results = await self._search_company_news(topic)
            else:
                news_results = {"results": [], "sources": []}
            # Step 3: Synthesize section
            section = self.section_template(topic, news_results, local_news, max_length)
            if emerging:
                section = f"[Emerging Topic: Web search required for up-to-date info]\n\n" + section
            result_message = {
                "task_id": task_data.get("task_id"),
                "section": section,
                "sources": (news_results.get("sources", []) + local_news.get("sources", [])),
                "status": "completed"
            }
            self.send_message(task_data.get("sender_id", "master_planner"), result_message)
        except Exception as e:
            logger.error(f"Error handling industry news task: {e}")
            error_message = {
                "task_id": task_data.get("task_id"),
                "error": str(e),
                "status": "failed"
            }
            self.send_message(task_data.get("sender_id", "master_planner"), error_message)

    async def _search_company_news(self, topic: str) -> Dict[str, Any]:
        if not self.can_use_tool("web_search"):
            logger.warning("Web search tool not available")
            return {"results": [], "sources": []}
        try:
            query = f"{topic} company announcement product launch market trend"
            results = await self.web_search(
                query=query,
                max_results=10,
                date_range="week",
                content_type="news"
            )
            sources = [r.get("url", "") for r in results.get("results", [])] if results else []
            return {"results": results.get("results", []), "sources": sources}
        except Exception as e:
            logger.error(f"Error searching company news: {e}")
            return {"results": [], "sources": [], "error": str(e)}

    async def _search_local_news(self, topic: str) -> Dict[str, Any]:
        if not self.can_use_tool("vector_search"):
            logger.warning("Vector search tool not available")
            return {"results": [], "sources": []}
        try:
            results = await self.vector_search(
                query=topic,
                max_results=10,
                similarity_threshold=0.6,
                filters={"topics": [topic]}
            )
            sources = [r.get("url", "") for r in results.get("results", [])] if results else []
            return {"results": results.get("results", []), "sources": sources}
        except Exception as e:
            logger.error(f"Error searching local news: {e}")
            return {"results": [], "sources": [], "error": str(e)}

    def _default_section_template(self, topic: str, news_results: Dict[str, Any], local_news: Dict[str, Any], max_length: int) -> str:
        section = f"# Industry News: {topic}\n\n"
        all_news = news_results.get("results", []) + local_news.get("results", [])
        if not all_news:
            section += "No recent industry news available.\n"
            return section
        section += "## Highlights\n\n"
        for i, item in enumerate(all_news[:5], 1):
            title = item.get("title", "Untitled")
            snippet = item.get("snippet", item.get("content", ""))[:200]
            url = item.get("url", "")
            section += f"{i}. **{title}**\n   {snippet}...\n   [Read more]({url})\n\n"
        section += "## Market Trends\n"
        section += f"The AI industry continues to evolve with new announcements and product launches.\n"
        return section[:max_length] 