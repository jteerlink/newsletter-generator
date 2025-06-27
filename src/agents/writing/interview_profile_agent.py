"""
Interview & Profile Agent with MCP Tool Integration.

This agent covers key figures, interviews, and conference highlights using web and vector search tools.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from ..base.agent_base import AgentBase
from ..base.communication import Message, MessageType

logger = logging.getLogger(__name__)

class InterviewProfileAgent(AgentBase):
    """
    Agent specialized in interviews, profiles, and conference highlights using MCP tools.
    """
    def __init__(self, agent_id: str = "interview_profile_agent"):
        super().__init__(agent_id)
        self.section_template = self._default_section_template

    async def run(self):
        logger.info(f"Interview & Profile Agent {self.agent_id} started")
        while True:
            await asyncio.sleep(1)

    def receive_message(self, message: dict):
        try:
            msg = Message.from_dict(message)
            if msg.message_type == MessageType.TASK:
                asyncio.create_task(self._handle_interview_task(msg.payload))
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def send_message(self, recipient_id: str, message: dict):
        logger.info(f"Sending message to {recipient_id}: {message}")

    async def _handle_interview_task(self, task_data: Dict[str, Any]):
        try:
            topic = task_data.get("topic", "AI interviews and profiles")
            max_length = task_data.get("max_length", 600)
            logger.info(f"Interview/profile task for topic: {topic}")
            # Step 1: Vector search for local interviews/profiles
            local_interviews = await self._search_local_interviews(topic)
            emerging = False
            # Step 2: If not enough local results, trigger web search
            if not local_interviews.get("results") or len(local_interviews["results"]) < 3:
                emerging = True
                interview_results = await self._search_interviews(topic)
            else:
                interview_results = {"results": [], "sources": []}
            # Step 3: Synthesize section
            section = self.section_template(topic, interview_results, local_interviews, max_length)
            if emerging:
                section = f"[Emerging Topic: Web search required for up-to-date info]\n\n" + section
            result_message = {
                "task_id": task_data.get("task_id"),
                "section": section,
                "sources": (interview_results.get("sources", []) + local_interviews.get("sources", [])),
                "status": "completed"
            }
            self.send_message(task_data.get("sender_id", "master_planner"), result_message)
        except Exception as e:
            logger.error(f"Error handling interview/profile task: {e}")
            error_message = {
                "task_id": task_data.get("task_id"),
                "error": str(e),
                "status": "failed"
            }
            self.send_message(task_data.get("sender_id", "master_planner"), error_message)

    async def _search_interviews(self, topic: str) -> Dict[str, Any]:
        if not self.can_use_tool("web_search"):
            logger.warning("Web search tool not available")
            return {"results": [], "sources": []}
        try:
            query = f"{topic} interview profile conference keynote"
            results = await self.web_search(
                query=query,
                max_results=10,
                date_range="month",
                content_type="news"
            )
            sources = [r.get("url", "") for r in results.get("results", [])] if results else []
            return {"results": results.get("results", []), "sources": sources}
        except Exception as e:
            logger.error(f"Error searching interviews/profiles: {e}")
            return {"results": [], "sources": [], "error": str(e)}

    async def _search_local_interviews(self, topic: str) -> Dict[str, Any]:
        if not self.can_use_tool("vector_search"):
            logger.warning("Vector search tool not available")
            return {"results": [], "sources": []}
        try:
            results = await self.vector_search(
                query=topic,
                max_results=10,
                similarity_threshold=0.65,
                filters={"topics": [topic]}
            )
            sources = [r.get("url", "") for r in results.get("results", [])] if results else []
            return {"results": results.get("results", []), "sources": sources}
        except Exception as e:
            logger.error(f"Error searching local interviews/profiles: {e}")
            return {"results": [], "sources": [], "error": str(e)}

    def _default_section_template(self, topic: str, interview_results: Dict[str, Any], local_interviews: Dict[str, Any], max_length: int) -> str:
        section = f"# Interviews & Profiles: {topic}\n\n"
        all_interviews = interview_results.get("results", []) + local_interviews.get("results", [])
        if not all_interviews:
            section += "No recent interviews or profiles available.\n"
            return section
        section += "## Key Figures & Highlights\n\n"
        for i, item in enumerate(all_interviews[:4], 1):
            title = item.get("title", "Untitled")
            snippet = item.get("snippet", item.get("content", ""))[:200]
            url = item.get("url", "")
            section += f"{i}. **{title}**\n   {snippet}...\n   [Read more]({url})\n\n"
        section += "## Conference Insights\n"
        section += f"This section features interviews, profiles, and highlights from recent AI events.\n"
        return section[:max_length] 