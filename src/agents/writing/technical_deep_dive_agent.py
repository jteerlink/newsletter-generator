"""
Technical Deep-Dive Agent with MCP Tool Integration.

This agent explains complex AI concepts, creates tutorials, and provides code examples using web and vector search tools.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from ..base.agent_base import AgentBase
from ..base.communication import Message, MessageType

logger = logging.getLogger(__name__)

class TechnicalDeepDiveAgent(AgentBase):
    """
    Agent specialized in technical deep-dives, tutorials, and code explanations using MCP tools.
    """
    def __init__(self, agent_id: str = "technical_deep_dive_agent"):
        super().__init__(agent_id)
        self.section_template = self._default_section_template

    async def run(self):
        logger.info(f"Technical Deep-Dive Agent {self.agent_id} started")
        while True:
            await asyncio.sleep(1)

    def receive_message(self, message: dict):
        try:
            msg = Message.from_dict(message)
            if msg.message_type == MessageType.TASK:
                asyncio.create_task(self._handle_deep_dive_task(msg.payload))
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def send_message(self, recipient_id: str, message: dict):
        logger.info(f"Sending message to {recipient_id}: {message}")

    async def _handle_deep_dive_task(self, task_data: Dict[str, Any]):
        try:
            topic = task_data.get("topic", "AI technical deep-dive")
            max_length = task_data.get("max_length", 800)
            logger.info(f"Technical deep-dive task for topic: {topic}")
            # Step 1: Vector search for local technical content
            local_tech = await self._search_local_technical(topic)
            emerging = False
            # Step 2: If not enough local results, trigger web search
            if not local_tech.get("results") or len(local_tech["results"]) < 3:
                emerging = True
                tech_results = await self._search_technical_articles(topic)
            else:
                tech_results = {"results": [], "sources": []}
            # Step 3: Synthesize section
            section = self.section_template(topic, tech_results, local_tech, max_length)
            if emerging:
                section = f"[Emerging Topic: Web search required for up-to-date info]\n\n" + section
            result_message = {
                "task_id": task_data.get("task_id"),
                "section": section,
                "sources": (tech_results.get("sources", []) + local_tech.get("sources", [])),
                "status": "completed"
            }
            self.send_message(task_data.get("sender_id", "master_planner"), result_message)
        except Exception as e:
            logger.error(f"Error handling technical deep-dive task: {e}")
            error_message = {
                "task_id": task_data.get("task_id"),
                "error": str(e),
                "status": "failed"
            }
            self.send_message(task_data.get("sender_id", "master_planner"), error_message)

    async def _search_technical_articles(self, topic: str) -> Dict[str, Any]:
        if not self.can_use_tool("web_search"):
            logger.warning("Web search tool not available")
            return {"results": [], "sources": []}
        try:
            query = f"{topic} tutorial code example technical explanation"
            results = await self.web_search(
                query=query,
                max_results=10,
                date_range="month",
                content_type="research"
            )
            sources = [r.get("url", "") for r in results.get("results", [])] if results else []
            return {"results": results.get("results", []), "sources": sources}
        except Exception as e:
            logger.error(f"Error searching technical articles: {e}")
            return {"results": [], "sources": [], "error": str(e)}

    async def _search_local_technical(self, topic: str) -> Dict[str, Any]:
        if not self.can_use_tool("vector_search"):
            logger.warning("Vector search tool not available")
            return {"results": [], "sources": []}
        try:
            results = await self.vector_search(
                query=topic,
                max_results=10,
                similarity_threshold=0.7,
                filters={"topics": [topic]}
            )
            sources = [r.get("url", "") for r in results.get("results", [])] if results else []
            return {"results": results.get("results", []), "sources": sources}
        except Exception as e:
            logger.error(f"Error searching local technical content: {e}")
            return {"results": [], "sources": [], "error": str(e)}

    def _default_section_template(self, topic: str, tech_results: Dict[str, Any], local_tech: Dict[str, Any], max_length: int) -> str:
        section = f"# Technical Deep-Dive: {topic}\n\n"
        all_tech = tech_results.get("results", []) + local_tech.get("results", [])
        if not all_tech:
            section += "No recent technical deep-dives available.\n"
            return section
        section += "## Tutorials & Explanations\n\n"
        for i, item in enumerate(all_tech[:4], 1):
            title = item.get("title", "Untitled")
            snippet = item.get("snippet", item.get("content", ""))[:250]
            url = item.get("url", "")
            section += f"{i}. **{title}**\n   {snippet}...\n   [Read more]({url})\n\n"
        section += "## Technical Insights\n"
        section += f"This section provides a deep-dive into {topic.lower()} with tutorials and code examples.\n"
        return section[:max_length] 