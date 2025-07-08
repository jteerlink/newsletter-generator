"""
MCP Orchestrator for Newsletter Generation System

This orchestrator coordinates multiple MCP tools to provide enhanced
functionality for newsletter generation, including:
- Content research and aggregation
- Multi-source data integration
- Automated publishing workflows
- Analytics and performance tracking
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from src.agents.agentic_rag_agent import AgenticRAGAgent, AgenticRAGSession
from src.storage.enhanced_vector_store import EnhancedVectorStore
from src.core.feedback_system import FeedbackLearningSystem
from src.tools.notion_integration import NotionNewsletterPublisher

logger = logging.getLogger(__name__)

@dataclass
class MCPWorkflowStep:
    """Represents a single step in an MCP workflow."""
    step_id: str
    mcp_tool: str
    action: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: Optional[datetime] = None

@dataclass
class MCPWorkflow:
    """Represents a complete MCP workflow."""
    workflow_id: str
    name: str
    description: str
    steps: List[MCPWorkflowStep]
    context: Dict[str, Any]
    status: str = "pending"
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    results: Dict[str, Any] = None

class MCPOrchestrator:
    """
    Orchestrates multiple MCP tools for enhanced newsletter generation.
    """
    
    def __init__(self, 
                 vector_store: EnhancedVectorStore,
                 feedback_system: FeedbackLearningSystem,
                 notion_publisher: NotionNewsletterPublisher):
        self.vector_store = vector_store
        self.feedback_system = feedback_system
        self.notion_publisher = notion_publisher
        self.agentic_rag = AgenticRAGAgent(vector_store)
        
        # MCP tool registry
        self.mcp_tools = {
            "notion": self._get_notion_tools(),
            "github": self._get_github_tools(),
            "slack": self._get_slack_tools(),
            "linear": self._get_linear_tools(),
            "google_drive": self._get_google_drive_tools()
        }
        
        # Workflow templates
        self.workflow_templates = {
            "research_workflow": self._create_research_workflow_template(),
            "content_aggregation": self._create_content_aggregation_template(),
            "publishing_workflow": self._create_publishing_workflow_template(),
            "analytics_workflow": self._create_analytics_workflow_template()
        }
        
        # Active workflows
        self.active_workflows = {}
        
        # Workflow history
        self.workflow_history = []
    
    def create_research_workflow(self, 
                               topic: str, 
                               sources: List[str] = None,
                               time_range: Optional[Dict[str, datetime]] = None) -> MCPWorkflow:
        """
        Create a comprehensive research workflow using multiple MCP tools.
        
        Args:
            topic: The research topic
            sources: List of source types to include
            time_range: Time range for temporal research
        """
        
        workflow_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Default sources if not provided
        if not sources:
            sources = ["notion", "github", "web_search", "vector_store"]
        
        steps = []
        
        # Step 1: Agentic RAG query
        steps.append(MCPWorkflowStep(
            step_id="agentic_rag_query",
            mcp_tool="internal",
            action="agentic_query",
            parameters={
                "query": topic,
                "time_range": time_range,
                "max_iterations": 3
            },
            dependencies=[]
        ))
        
        # Step 2: Notion search (if available)
        if "notion" in sources:
            steps.append(MCPWorkflowStep(
                step_id="notion_search",
                mcp_tool="notion",
                action="search",
                parameters={
                    "query": topic,
                    "query_type": "internal"
                },
                dependencies=[]
            ))
        
        # Step 3: GitHub search (if available)
        if "github" in sources:
            steps.append(MCPWorkflowStep(
                step_id="github_search",
                mcp_tool="github",
                action="search_repositories",
                parameters={
                    "query": topic,
                    "sort": "updated",
                    "per_page": 10
                },
                dependencies=[]
            ))
        
        # Step 4: Content synthesis
        steps.append(MCPWorkflowStep(
            step_id="content_synthesis",
            mcp_tool="internal",
            action="synthesize_research",
            parameters={
                "topic": topic,
                "source_results": ["agentic_rag_query", "notion_search", "github_search"]
            },
            dependencies=["agentic_rag_query"]
        ))
        
        # Step 5: Quality assessment
        steps.append(MCPWorkflowStep(
            step_id="quality_assessment",
            mcp_tool="internal",
            action="assess_quality",
            parameters={
                "content": "content_synthesis",
                "topic": topic
            },
            dependencies=["content_synthesis"]
        ))
        
        workflow = MCPWorkflow(
            workflow_id=workflow_id,
            name=f"Research: {topic}",
            description=f"Comprehensive research workflow for {topic}",
            steps=steps,
            context={
                "topic": topic,
                "sources": sources,
                "time_range": time_range
            },
            created_at=datetime.now()
        )
        
        return workflow
    
    def create_publishing_workflow(self, 
                                 newsletter_content: str,
                                 title: str,
                                 target_platforms: List[str] = None) -> MCPWorkflow:
        """
        Create a publishing workflow for newsletter distribution.
        
        Args:
            newsletter_content: The newsletter content
            title: Newsletter title
            target_platforms: List of platforms to publish to
        """
        
        workflow_id = f"publish_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if not target_platforms:
            target_platforms = ["notion", "file_system"]
        
        steps = []
        
        # Step 1: Content preparation
        steps.append(MCPWorkflowStep(
            step_id="content_preparation",
            mcp_tool="internal",
            action="prepare_content",
            parameters={
                "content": newsletter_content,
                "title": title,
                "format": "markdown"
            },
            dependencies=[]
        ))
        
        # Step 2: Notion publishing
        if "notion" in target_platforms:
            steps.append(MCPWorkflowStep(
                step_id="notion_publish",
                mcp_tool="notion",
                action="create_page",
                parameters={
                    "parent": {"page_id": "226b1384-d996-813f-bc9c-c540b498df90"},
                    "title": title,
                    "content": newsletter_content
                },
                dependencies=["content_preparation"]
            ))
        
        # Step 3: File system save
        if "file_system" in target_platforms:
            steps.append(MCPWorkflowStep(
                step_id="file_save",
                mcp_tool="internal",
                action="save_file",
                parameters={
                    "content": newsletter_content,
                    "filename": f"newsletter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    "directory": "output"
                },
                dependencies=["content_preparation"]
            ))
        
        # Step 4: Analytics setup
        steps.append(MCPWorkflowStep(
            step_id="analytics_setup",
            mcp_tool="internal",
            action="setup_analytics",
            parameters={
                "newsletter_id": workflow_id,
                "title": title,
                "platforms": target_platforms
            },
            dependencies=["notion_publish", "file_save"]
        ))
        
        workflow = MCPWorkflow(
            workflow_id=workflow_id,
            name=f"Publish: {title}",
            description=f"Publishing workflow for {title}",
            steps=steps,
            context={
                "title": title,
                "platforms": target_platforms,
                "content_length": len(newsletter_content)
            },
            created_at=datetime.now()
        )
        
        return workflow
    
    async def execute_workflow(self, workflow: MCPWorkflow) -> Dict[str, Any]:
        """
        Execute a complete MCP workflow.
        
        Args:
            workflow: The workflow to execute
            
        Returns:
            Execution results
        """
        
        logger.info(f"Starting workflow execution: {workflow.workflow_id}")
        
        # Track workflow
        self.active_workflows[workflow.workflow_id] = workflow
        workflow.status = "running"
        
        try:
            # Execute steps in dependency order
            execution_order = self._calculate_execution_order(workflow.steps)
            
            for step_id in execution_order:
                step = next(s for s in workflow.steps if s.step_id == step_id)
                
                logger.info(f"Executing step: {step.step_id}")
                step.status = "running"
                step.timestamp = datetime.now()
                
                try:
                    # Execute the step
                    result = await self._execute_step(step, workflow.context)
                    step.result = result
                    step.status = "completed"
                    
                    # Update workflow context with results
                    workflow.context[step.step_id] = result
                    
                except Exception as e:
                    logger.error(f"Step {step.step_id} failed: {str(e)}")
                    step.status = "failed"
                    step.error = str(e)
                    
                    # Decide whether to continue or abort
                    if step.step_id in ["content_synthesis", "notion_publish"]:
                        # Critical steps - abort workflow
                        workflow.status = "failed"
                        break
                    else:
                        # Non-critical steps - continue
                        logger.warning(f"Continuing workflow despite step failure: {step.step_id}")
            
            # Finalize workflow
            if workflow.status != "failed":
                workflow.status = "completed"
                workflow.completed_at = datetime.now()
                
                # Compile results
                workflow.results = self._compile_workflow_results(workflow)
            
            # Move to history
            del self.active_workflows[workflow.workflow_id]
            self.workflow_history.append(workflow)
            
            logger.info(f"Workflow {workflow.workflow_id} completed with status: {workflow.status}")
            
            return {
                "workflow_id": workflow.workflow_id,
                "status": workflow.status,
                "results": workflow.results,
                "execution_time": (workflow.completed_at - workflow.created_at).total_seconds() if workflow.completed_at else None
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            workflow.status = "failed"
            workflow.completed_at = datetime.now()
            
            # Move to history
            if workflow.workflow_id in self.active_workflows:
                del self.active_workflows[workflow.workflow_id]
            self.workflow_history.append(workflow)
            
            return {
                "workflow_id": workflow.workflow_id,
                "status": "failed",
                "error": str(e),
                "execution_time": (workflow.completed_at - workflow.created_at).total_seconds()
            }
    
    async def _execute_step(self, step: MCPWorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step."""
        
        if step.mcp_tool == "internal":
            return await self._execute_internal_step(step, context)
        elif step.mcp_tool == "notion":
            return await self._execute_notion_step(step, context)
        elif step.mcp_tool == "github":
            return await self._execute_github_step(step, context)
        else:
            raise ValueError(f"Unknown MCP tool: {step.mcp_tool}")
    
    async def _execute_internal_step(self, step: MCPWorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute internal system steps."""
        
        action = step.action
        params = step.parameters
        
        if action == "agentic_query":
            # Execute agentic RAG query
            session = self.agentic_rag.process_query(
                query=params["query"],
                context=context,
                max_iterations=params.get("max_iterations", 3)
            )
            
            return {
                "session_id": len(self.agentic_rag.session_history),
                "response": session.synthesized_response,
                "confidence": session.confidence_score,
                "sources": session.sources_used,
                "reasoning": session.reasoning_chain
            }
        
        elif action == "synthesize_research":
            # Synthesize research from multiple sources
            source_results = []
            for source_step in params["source_results"]:
                if source_step in context:
                    source_results.append(context[source_step])
            
            synthesis = self._synthesize_multi_source_research(
                params["topic"], source_results
            )
            
            return {
                "synthesized_content": synthesis,
                "source_count": len(source_results),
                "topic": params["topic"]
            }
        
        elif action == "assess_quality":
            # Assess content quality
            content_key = params["content"]
            content = context.get(content_key, {}).get("synthesized_content", "")
            
            quality_assessment = self._assess_content_quality(
                content, params["topic"]
            )
            
            return quality_assessment
        
        elif action == "prepare_content":
            # Prepare content for publishing
            return {
                "prepared_content": params["content"],
                "title": params["title"],
                "format": params["format"],
                "word_count": len(params["content"].split())
            }
        
        elif action == "save_file":
            # Save file to system
            filepath = Path(params["directory"]) / params["filename"]
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(params["content"])
            
            return {
                "file_path": str(filepath),
                "saved_at": datetime.now().isoformat(),
                "file_size": len(params["content"])
            }
        
        elif action == "setup_analytics":
            # Setup analytics tracking
            return {
                "analytics_id": params["newsletter_id"],
                "tracking_enabled": True,
                "platforms": params["platforms"]
            }
        
        else:
            raise ValueError(f"Unknown internal action: {action}")
    
    async def _execute_notion_step(self, step: MCPWorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Notion MCP steps."""
        
        action = step.action
        params = step.parameters
        
        if action == "search":
            # This would be handled by the MCP system
            # For now, return mock data
            return {
                "results": [
                    {
                        "title": f"Mock Notion result for {params['query']}",
                        "url": "https://notion.so/mock-page",
                        "content": "Mock content from Notion search"
                    }
                ],
                "query": params["query"]
            }
        
        elif action == "create_page":
            # This would be handled by the MCP system
            # For now, return mock data
            return {
                "page_id": "mock-page-id",
                "url": "https://notion.so/mock-page",
                "title": params["title"],
                "created_at": datetime.now().isoformat()
            }
        
        else:
            raise ValueError(f"Unknown Notion action: {action}")
    
    async def _execute_github_step(self, step: MCPWorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GitHub MCP steps."""
        
        # This would be handled by the MCP system
        # For now, return mock data
        return {
            "repositories": [
                {
                    "name": f"mock-repo-{step.parameters['query']}",
                    "description": "Mock repository description",
                    "url": "https://github.com/mock/repo",
                    "stars": 100
                }
            ],
            "query": step.parameters["query"]
        }
    
    def _calculate_execution_order(self, steps: List[MCPWorkflowStep]) -> List[str]:
        """Calculate the correct execution order based on dependencies."""
        
        # Simple topological sort
        visited = set()
        order = []
        
        def visit(step_id: str):
            if step_id in visited:
                return
            
            step = next(s for s in steps if s.step_id == step_id)
            for dep in step.dependencies:
                visit(dep)
            
            visited.add(step_id)
            order.append(step_id)
        
        for step in steps:
            visit(step.step_id)
        
        return order
    
    def _compile_workflow_results(self, workflow: MCPWorkflow) -> Dict[str, Any]:
        """Compile final results from all workflow steps."""
        
        results = {
            "workflow_id": workflow.workflow_id,
            "status": workflow.status,
            "step_results": {},
            "summary": {}
        }
        
        # Collect step results
        for step in workflow.steps:
            if step.result:
                results["step_results"][step.step_id] = step.result
        
        # Create summary based on workflow type
        if "research" in workflow.name.lower():
            results["summary"] = self._create_research_summary(workflow)
        elif "publish" in workflow.name.lower():
            results["summary"] = self._create_publishing_summary(workflow)
        
        return results
    
    def _create_research_summary(self, workflow: MCPWorkflow) -> Dict[str, Any]:
        """Create summary for research workflows."""
        
        summary = {
            "topic": workflow.context.get("topic", "Unknown"),
            "sources_used": workflow.context.get("sources", []),
            "total_results": 0,
            "quality_score": 0.0
        }
        
        # Count total results
        for step in workflow.steps:
            if step.result and isinstance(step.result, dict):
                if "results" in step.result:
                    summary["total_results"] += len(step.result["results"])
                elif "repositories" in step.result:
                    summary["total_results"] += len(step.result["repositories"])
        
        # Get quality score
        quality_step = next((s for s in workflow.steps if s.step_id == "quality_assessment"), None)
        if quality_step and quality_step.result:
            summary["quality_score"] = quality_step.result.get("overall_score", 0.0)
        
        return summary
    
    def _create_publishing_summary(self, workflow: MCPWorkflow) -> Dict[str, Any]:
        """Create summary for publishing workflows."""
        
        summary = {
            "title": workflow.context.get("title", "Unknown"),
            "platforms": workflow.context.get("platforms", []),
            "published_urls": [],
            "analytics_enabled": False
        }
        
        # Collect published URLs
        for step in workflow.steps:
            if step.result and isinstance(step.result, dict):
                if "url" in step.result:
                    summary["published_urls"].append(step.result["url"])
                elif "page_id" in step.result:
                    summary["published_urls"].append(f"notion://{step.result['page_id']}")
        
        # Check analytics
        analytics_step = next((s for s in workflow.steps if s.step_id == "analytics_setup"), None)
        if analytics_step and analytics_step.result:
            summary["analytics_enabled"] = analytics_step.result.get("tracking_enabled", False)
        
        return summary
    
    # Helper methods
    def _synthesize_multi_source_research(self, topic: str, source_results: List[Dict[str, Any]]) -> str:
        """Synthesize research from multiple sources."""
        
        # Collect all content
        all_content = []
        for result in source_results:
            if isinstance(result, dict):
                if "response" in result:
                    all_content.append(result["response"])
                elif "synthesized_content" in result:
                    all_content.append(result["synthesized_content"])
        
        # Simple synthesis (could be enhanced with LLM)
        if all_content:
            return f"Synthesized research on {topic}:\n\n" + "\n\n".join(all_content)
        else:
            return f"No content found for topic: {topic}"
    
    def _assess_content_quality(self, content: str, topic: str) -> Dict[str, Any]:
        """Assess the quality of generated content."""
        
        # Simple quality metrics
        word_count = len(content.split())
        
        # Basic quality scoring
        quality_score = 0.0
        
        if word_count > 100:
            quality_score += 0.3
        if word_count > 500:
            quality_score += 0.2
        if topic.lower() in content.lower():
            quality_score += 0.3
        if len(content) > 1000:
            quality_score += 0.2
        
        return {
            "overall_score": quality_score,
            "word_count": word_count,
            "content_length": len(content),
            "topic_relevance": topic.lower() in content.lower(),
            "quality_level": "high" if quality_score > 0.8 else "medium" if quality_score > 0.5 else "low"
        }
    
    # MCP tool definitions (placeholders)
    def _get_notion_tools(self) -> List[str]:
        return ["search", "create_page", "update_page", "get_page"]
    
    def _get_github_tools(self) -> List[str]:
        return ["search_repositories", "get_repository", "get_issues", "get_releases"]
    
    def _get_slack_tools(self) -> List[str]:
        return ["search_messages", "get_channels", "post_message"]
    
    def _get_linear_tools(self) -> List[str]:
        return ["search_issues", "get_projects", "get_roadmap"]
    
    def _get_google_drive_tools(self) -> List[str]:
        return ["search_files", "create_file", "update_file", "get_file"]
    
    # Workflow templates
    def _create_research_workflow_template(self) -> Dict[str, Any]:
        return {
            "name": "Research Workflow",
            "description": "Comprehensive research using multiple sources",
            "steps": ["agentic_rag", "external_search", "synthesis", "quality_check"]
        }
    
    def _create_content_aggregation_template(self) -> Dict[str, Any]:
        return {
            "name": "Content Aggregation",
            "description": "Aggregate content from multiple platforms",
            "steps": ["platform_search", "content_extraction", "deduplication", "formatting"]
        }
    
    def _create_publishing_workflow_template(self) -> Dict[str, Any]:
        return {
            "name": "Publishing Workflow",
            "description": "Multi-platform content publishing",
            "steps": ["content_prep", "platform_publish", "analytics_setup", "notification"]
        }
    
    def _create_analytics_workflow_template(self) -> Dict[str, Any]:
        return {
            "name": "Analytics Workflow",
            "description": "Performance tracking and analysis",
            "steps": ["data_collection", "analysis", "reporting", "insights"]
        }

# Factory function
def create_mcp_orchestrator(vector_store: EnhancedVectorStore, 
                          feedback_system: FeedbackLearningSystem,
                          notion_publisher: NotionNewsletterPublisher) -> MCPOrchestrator:
    """Create an MCP orchestrator instance."""
    return MCPOrchestrator(vector_store, feedback_system, notion_publisher) 