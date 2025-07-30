"""
Research Agent for Newsletter Generation

This module provides the ResearchAgent class, which is responsible for gathering
information and conducting research for newsletter content.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

from .base import SimpleAgent, AgentType
from src.core.core import query_llm
from src.tools.tools import search_web, search_knowledge_base

logger = logging.getLogger(__name__)


class ResearchAgent(SimpleAgent):
    """Agent specialized in research and information gathering."""
    
    def __init__(self, name: str = "ResearchAgent", **kwargs):
        super().__init__(
            name=name,
            role="Research Specialist",
            goal="Gather comprehensive, accurate, and up-to-date information on given topics",
            backstory="""You are an expert research specialist with years of experience in gathering 
            information from various sources. You excel at finding the most relevant and recent 
            information, verifying facts, and organizing research findings in a clear, structured manner. 
            You have access to web search tools and knowledge bases to ensure comprehensive coverage.""",
            agent_type=AgentType.RESEARCH,
            tools=["search_web", "search_knowledge_base"],
            **kwargs
        )
    
    def execute_task(self, task: str, context: str = "", **kwargs) -> str:
        """Execute research task with enhanced research capabilities."""
        logger.info(f"ResearchAgent executing research task: {task}")
        
        # First, try to get information from knowledge base
        knowledge_results = self._generate_knowledge_based_research(task, context)
        
        # Then, enhance with web search
        web_results = self._execute_web_research(task, context)
        
        # Combine and synthesize results
        combined_results = self._synthesize_research_results(knowledge_results, web_results, task)
        
        return combined_results
    
    def _generate_knowledge_based_research(self, task: str, context: str) -> str:
        """Generate research using knowledge base."""
        try:
            knowledge_prompt = f"""
            Research Task: {task}
            Context: {context}
            
            Please search the knowledge base for relevant information on this topic.
            Focus on finding:
            1. Key facts and data points
            2. Recent developments
            3. Expert opinions and analysis
            4. Related topics and connections
            
            Provide a structured summary of your findings.
            """
            
            # Use knowledge base search
            knowledge_results = search_knowledge_base(task)
            
            if knowledge_results and knowledge_results.strip():
                return f"Knowledge Base Research:\n{knowledge_results}"
            else:
                return "No relevant information found in knowledge base."
                
        except Exception as e:
            logger.error(f"Error in knowledge-based research: {e}")
            return f"Error accessing knowledge base: {e}"
    
    def _execute_web_research(self, task: str, context: str) -> str:
        """Execute web research for the task."""
        try:
            # Extract search queries from task
            search_queries = self._generate_search_queries(task, context)
            
            web_results = []
            for query in search_queries:
                try:
                    result = search_web(query)
                    if result and result.strip():
                        web_results.append(f"Search Query: {query}\nResults:\n{result}")
                except Exception as e:
                    logger.warning(f"Web search failed for query '{query}': {e}")
            
            return "\n\n".join(web_results) if web_results else "No web search results found."
            
        except Exception as e:
            logger.error(f"Error in web research: {e}")
            return f"Error in web research: {e}"
    
    def _generate_search_queries(self, task: str, context: str) -> List[str]:
        """Generate multiple search queries for comprehensive research."""
        # Create a prompt to generate search queries
        query_generation_prompt = f"""
        Task: {task}
        Context: {context}
        
        Generate 3-5 specific search queries to gather comprehensive information on this topic.
        Make the queries:
        1. Specific and focused
        2. Cover different aspects of the topic
        3. Include recent developments
        4. Target authoritative sources
        
        Return only the search queries, one per line.
        """
        
        try:
            response = query_llm(query_generation_prompt)
            queries = [line.strip() for line in response.split('\n') if line.strip()]
            
            # Fallback to simple query if LLM fails
            if not queries:
                queries = [self._extract_search_query(task)]
            
            return queries[:5]  # Limit to 5 queries
            
        except Exception as e:
            logger.warning(f"Failed to generate search queries: {e}")
            return [self._extract_search_query(task)]
    
    def _synthesize_research_results(self, knowledge_results: str, web_results: str, task: str) -> str:
        """Synthesize and organize research results."""
        synthesis_prompt = f"""
        Research Task: {task}
        
        Knowledge Base Findings:
        {knowledge_results}
        
        Web Search Findings:
        {web_results}
        
        Please synthesize these research findings into a comprehensive, well-organized summary.
        Structure your response with:
        1. Executive Summary
        2. Key Findings
        3. Supporting Evidence
        4. Recent Developments
        5. Expert Opinions
        6. Related Topics
        7. Sources and References
        
        Ensure the information is:
        - Accurate and factual
        - Well-organized and easy to understand
        - Comprehensive but concise
        - Focused on the research task
        """
        
        try:
            return query_llm(synthesis_prompt)
        except Exception as e:
            logger.error(f"Error synthesizing research results: {e}")
            return f"""
            Research Synthesis Error: {e}
            
            Knowledge Base Results:
            {knowledge_results}
            
            Web Search Results:
            {web_results}
            """
    
    def _build_research_prompt_with_tools(self, task: str, context: str, tool_output: str) -> str:
        """Build enhanced prompt for research with tool results."""
        prompt_parts = [
            f"You are a {self.role}.",
            f"Your goal: {self.goal}",
            f"Background: {self.backstory}",
            "",
            f"Research Task: {task}"
        ]
        
        if context:
            prompt_parts.extend([
                "",
                f"Context: {context}"
            ])
        
        prompt_parts.extend([
            "",
            "Research Findings:",
            tool_output,
            "",
            "Based on the research findings above, please provide:",
            "1. A comprehensive summary of the key information",
            "2. Analysis of the findings and their implications",
            "3. Identification of trends or patterns",
            "4. Recommendations for further research if needed",
            "5. A structured format suitable for newsletter content",
            "",
            "Ensure your response is well-organized, factual, and provides valuable insights."
        ])
        
        return "\n".join(prompt_parts)
    
    def get_research_analytics(self) -> Dict[str, Any]:
        """Get research-specific analytics."""
        analytics = self.get_tool_usage_analytics()
        
        # Add research-specific metrics
        research_metrics = {
            "research_sessions": len(self.execution_history),
            "avg_research_time": sum(r.execution_time for r in self.execution_history) / len(self.execution_history) if self.execution_history else 0,
            "success_rate": sum(1 for r in self.execution_history if r.status.value == "completed") / len(self.execution_history) if self.execution_history else 0,
            "tool_usage_breakdown": {
                "search_web": sum(1 for entry in analytics.get("agent_tool_usage", []) if entry.get("tool_name") == "search_web"),
                "search_knowledge_base": sum(1 for entry in analytics.get("agent_tool_usage", []) if entry.get("tool_name") == "search_knowledge_base")
            }
        }
        
        analytics.update(research_metrics)
        return analytics 