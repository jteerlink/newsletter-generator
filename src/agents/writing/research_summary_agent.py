"""
Research Summary Agent with MCP Tool Integration.

This agent demonstrates how to use MCP tools (web_search and vector_search)
to gather information and create research summaries.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from ..base.agent_base import AgentBase
from ..base.communication import Message, MessageType

logger = logging.getLogger(__name__)


class ResearchSummaryAgent(AgentBase):
    """
    Agent specialized in creating research summaries using MCP tools.
    """
    
    def __init__(self, agent_id: str = "research_summary_agent"):
        super().__init__(agent_id)
        self.research_cache = {}
        self.summary_templates = {
            "academic": self._academic_summary_template,
            "business": self._business_summary_template,
            "technical": self._technical_summary_template
        }
    
    async def run(self):
        """Main execution loop for the research summary agent."""
        logger.info(f"Research Summary Agent {self.agent_id} started")
        
        # Example: Monitor for research summary requests
        while True:
            try:
                # Check for incoming messages/tasks
                # This would typically be handled by the message queue
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in research summary agent loop: {e}")
                await asyncio.sleep(5)
    
    def receive_message(self, message: dict):
        """Handle incoming messages."""
        try:
            msg = Message.from_dict(message)
            
            if msg.message_type == MessageType.TASK:
                asyncio.create_task(self._handle_research_task(msg.payload))
            elif msg.message_type == MessageType.STATUS:
                logger.info(f"Received status message: {msg.payload}")
            else:
                logger.warning(f"Unknown message type: {msg.message_type}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def send_message(self, recipient_id: str, message: dict):
        """Send a message to another agent."""
        # This would typically send to a message queue
        logger.info(f"Sending message to {recipient_id}: {message}")
    
    async def _handle_research_task(self, task_data: Dict[str, Any]):
        """Handle a research summary task."""
        try:
            topic = task_data.get("topic")
            audience = task_data.get("audience", "technical")
            max_length = task_data.get("max_length", 500)
            
            logger.info(f"Starting research task for topic: {topic}")
            
            # Step 1: Search for recent research using MCP web search
            research_results = await self._search_recent_research(topic)
            
            # Step 2: Search local vector database for related content
            local_content = await self._search_local_content(topic)
            
            # Step 3: Synthesize information
            summary = await self._create_research_summary(
                topic, research_results, local_content, audience, max_length
            )
            
            # Step 4: Send result back
            result_message = {
                "task_id": task_data.get("task_id"),
                "summary": summary,
                "sources": research_results.get("sources", []),
                "status": "completed"
            }
            
            self.send_message(task_data.get("sender_id", "master_planner"), result_message)
            
        except Exception as e:
            logger.error(f"Error handling research task: {e}")
            error_message = {
                "task_id": task_data.get("task_id"),
                "error": str(e),
                "status": "failed"
            }
            self.send_message(task_data.get("sender_id", "master_planner"), error_message)
    
    async def _search_recent_research(self, topic: str) -> Dict[str, Any]:
        """
        Search for recent research using MCP web search tool.
        
        Args:
            topic: Research topic to search for
            
        Returns:
            Search results with sources
        """
        if not self.can_use_tool("web_search"):
            logger.warning("Web search tool not available")
            return {"results": [], "sources": []}
        
        try:
            # Search for academic papers and research
            academic_query = f"{topic} research paper academic"
            academic_results = await self.web_search(
                query=academic_query,
                max_results=10,
                date_range="month",
                content_type="research"
            )
            
            # Search for recent news and developments
            news_query = f"{topic} latest developments news"
            news_results = await self.web_search(
                query=news_query,
                max_results=5,
                date_range="week",
                content_type="news"
            )
            
            # Combine and process results
            all_results = []
            sources = []
            
            if academic_results and "results" in academic_results:
                all_results.extend(academic_results["results"])
                sources.extend([r.get("url", "") for r in academic_results["results"]])
            
            if news_results and "results" in news_results:
                all_results.extend(news_results["results"])
                sources.extend([r.get("url", "") for r in news_results["results"]])
            
            logger.info(f"Found {len(all_results)} research results for topic: {topic}")
            
            return {
                "results": all_results,
                "sources": sources,
                "search_metadata": {
                    "academic_results": len(academic_results.get("results", [])),
                    "news_results": len(news_results.get("results", [])),
                    "total_sources": len(sources)
                }
            }
            
        except Exception as e:
            logger.error(f"Error searching recent research: {e}")
            return {"results": [], "sources": [], "error": str(e)}
    
    async def _search_local_content(self, topic: str) -> Dict[str, Any]:
        """
        Search local vector database for related content.
        
        Args:
            topic: Topic to search for
            
        Returns:
            Local content results
        """
        if not self.can_use_tool("vector_search"):
            logger.warning("Vector search tool not available")
            return {"results": [], "sources": []}
        
        try:
            # Search for related content in local database
            local_results = await self.vector_search(
                query=topic,
                max_results=15,
                similarity_threshold=0.6,
                filters={
                    "date_range": {
                        "start_date": "2024-01-01"  # Last year
                    }
                }
            )
            
            if local_results and "results" in local_results:
                logger.info(f"Found {len(local_results['results'])} local content items for topic: {topic}")
                return local_results
            else:
                logger.info(f"No local content found for topic: {topic}")
                return {"results": [], "sources": []}
                
        except Exception as e:
            logger.error(f"Error searching local content: {e}")
            return {"results": [], "sources": [], "error": str(e)}
    
    async def _create_research_summary(self, topic: str, research_results: Dict[str, Any], 
                                     local_content: Dict[str, Any], audience: str, 
                                     max_length: int) -> str:
        """
        Create a research summary from gathered information.
        
        Args:
            topic: Research topic
            research_results: Results from web search
            local_content: Results from vector search
            audience: Target audience (academic, business, technical)
            max_length: Maximum summary length
            
        Returns:
            Formatted research summary
        """
        try:
            # Extract key information from results
            key_findings = self._extract_key_findings(research_results, local_content)
            
            # Choose appropriate template
            template_func = self.summary_templates.get(audience, self._technical_summary_template)
            
            # Generate summary using template
            summary = template_func(topic, key_findings, max_length)
            
            logger.info(f"Generated {len(summary)} character summary for topic: {topic}")
            return summary
            
        except Exception as e:
            logger.error(f"Error creating research summary: {e}")
            return f"Error generating summary for {topic}: {str(e)}"
    
    def _extract_key_findings(self, research_results: Dict[str, Any], 
                             local_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key findings from search results."""
        findings = []
        
        # Process web search results
        for result in research_results.get("results", []):
            finding = {
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "url": result.get("url", ""),
                "source_type": "web_search",
                "relevance_score": result.get("relevance", 0.0)
            }
            findings.append(finding)
        
        # Process local content results
        for result in local_content.get("results", []):
            finding = {
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "url": result.get("url", ""),
                "source_type": "local_content",
                "similarity_score": result.get("similarity", 0.0)
            }
            findings.append(finding)
        
        # Sort by relevance/similarity
        findings.sort(key=lambda x: x.get("relevance_score", x.get("similarity_score", 0)), reverse=True)
        
        return findings[:10]  # Top 10 findings
    
    def _academic_summary_template(self, topic: str, findings: List[Dict[str, Any]], 
                                 max_length: int) -> str:
        """Academic summary template."""
        summary = f"# Research Summary: {topic}\n\n"
        
        if not findings:
            summary += "No recent research findings available for this topic.\n"
            return summary
        
        summary += "## Key Findings\n\n"
        
        for i, finding in enumerate(findings[:5], 1):
            title = finding.get("title", "Untitled")
            snippet = finding.get("snippet", finding.get("content", ""))[:200]
            source = finding.get("source_type", "unknown")
            
            summary += f"{i}. **{title}**\n"
            summary += f"   {snippet}...\n"
            summary += f"   *Source: {source}*\n\n"
        
        summary += f"## Summary\n"
        summary += f"This research summary covers recent developments in {topic.lower()}. "
        summary += f"Based on {len(findings)} sources, the field shows significant activity "
        summary += f"with both academic and industry contributions.\n"
        
        return summary[:max_length]
    
    def _business_summary_template(self, topic: str, findings: List[Dict[str, Any]], 
                                 max_length: int) -> str:
        """Business-focused summary template."""
        summary = f"# Business Impact Analysis: {topic}\n\n"
        
        if not findings:
            summary += "No recent business developments available for this topic.\n"
            return summary
        
        summary += "## Market Developments\n\n"
        
        for i, finding in enumerate(findings[:3], 1):
            title = finding.get("title", "Untitled")
            snippet = finding.get("snippet", finding.get("content", ""))[:150]
            
            summary += f"{i}. **{title}**\n"
            summary += f"   {snippet}...\n\n"
        
        summary += f"## Business Implications\n"
        summary += f"The developments in {topic.lower()} present both opportunities and challenges "
        summary += f"for businesses. Companies should monitor these trends closely.\n"
        
        return summary[:max_length]
    
    def _technical_summary_template(self, topic: str, findings: List[Dict[str, Any]], 
                                  max_length: int) -> str:
        """Technical summary template."""
        summary = f"# Technical Overview: {topic}\n\n"
        
        if not findings:
            summary += "No recent technical developments available for this topic.\n"
            return summary
        
        summary += "## Recent Developments\n\n"
        
        for i, finding in enumerate(findings[:4], 1):
            title = finding.get("title", "Untitled")
            content = finding.get("snippet", finding.get("content", ""))[:180]
            
            summary += f"{i}. **{title}**\n"
            summary += f"   {content}...\n\n"
        
        summary += f"## Technical Insights\n"
        summary += f"The field of {topic.lower()} continues to evolve rapidly with new "
        summary += f"approaches and methodologies emerging regularly.\n"
        
        return summary[:max_length]
    
    async def get_research_capabilities(self) -> Dict[str, Any]:
        """Get agent's research capabilities."""
        return {
            "agent_id": self.agent_id,
            "capabilities": [
                "academic_research_summary",
                "business_impact_analysis", 
                "technical_overview",
                "web_search_integration",
                "vector_search_integration"
            ],
            "available_tools": self.get_available_tools(),
            "templates": list(self.summary_templates.keys())
        } 