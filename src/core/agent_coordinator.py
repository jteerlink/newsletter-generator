"""
Agent Coordinator: Multi-Agent Orchestration and Fallback Handling (Phase 2 FR2.4)

Coordinates specialized agents with simple dependency ordering and circuit breaker
awareness. Provides a unified interface to run multiple agents against content and
aggregate their results. Keeps backward compatibility by remaining optional.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import (
    BaseSpecializedAgent,
    ProcessingContext,
    ProcessingMode,
    ProcessingResult,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentExecutionSpec:
    name: str
    agent: BaseSpecializedAgent
    depends_on: List[str]


class AgentCoordinator:
    """Manages multi-agent orchestration and basic dependency scheduling."""

    def __init__(self, agents: Optional[Dict[str, BaseSpecializedAgent]] = None):
        self.agents = agents or {}
        self._execution_cache: Dict[str, Any] = {}  # Cache for expensive operations
        self._performance_thresholds = {
            'max_agent_time_ms': 30000,  # 30 seconds
            'max_total_time_ms': 120000,  # 2 minutes
            'max_memory_usage_mb': 500    # 500 MB
        }

    def register(self, name: str, agent: BaseSpecializedAgent, depends_on: Optional[List[str]] = None):
        self.agents[name] = agent

    def execute_pipeline(
        self,
        ordered_specs: List[AgentExecutionSpec],
        base_context: ProcessingContext,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {"__errors__": [], "__metrics__": {}}
        execution_start = time.time()

        for spec in ordered_specs:
            agent_start = time.time()
            
            # Check dependencies
            missing = [d for d in spec.depends_on if d not in results or results[d].get('success') is False]
            if missing:
                logger.warning("Dependencies not satisfied for %s: %s", spec.name, missing)
                # Skip this agent if critical dependencies failed
                if any(results.get(dep, {}).get('success') is False for dep in missing):
                    logger.error("Skipping %s due to failed dependencies: %s", spec.name, missing)
                    results["__errors__"].append({
                        "agent": spec.name, 
                        "error": f"Skipped due to failed dependencies: {missing}",
                        "type": "dependency_failure"
                    })
                    continue

            try:
                # Check agent health before execution
                health = spec.agent.get_health_status()
                if health['status'] == 'failed':
                    logger.warning("Agent %s is in failed state, attempting anyway", spec.name)
                
                context = ProcessingContext(
                    content=base_context.content,
                    section_type=base_context.section_type,
                    audience=base_context.audience,
                    technical_level=base_context.technical_level,
                    word_count_target=base_context.word_count_target,
                    processing_mode=base_context.processing_mode,
                    metadata=base_context.metadata,
                )
                
                agent_result: ProcessingResult = spec.agent.process(context)
                results[spec.name] = agent_result.to_dict()

                # Track execution metrics
                agent_time = time.time() - agent_start
                results["__metrics__"][spec.name] = {
                    "execution_time_ms": agent_time * 1000,
                    "success": agent_result.success,
                    "quality_score": agent_result.quality_score,
                    "confidence_score": agent_result.confidence_score
                }

                # Optionally propagate improved content if provided and agent succeeded
                if agent_result.success and agent_result.processed_content and len(agent_result.processed_content.strip()) > 0:
                    base_context = ProcessingContext(
                        content=agent_result.processed_content,
                        section_type=base_context.section_type,
                        audience=base_context.audience,
                        technical_level=base_context.technical_level,
                        word_count_target=base_context.word_count_target,
                        processing_mode=base_context.processing_mode,
                        metadata=base_context.metadata,
                    )
                    logger.debug("Content updated by agent %s", spec.name)

            except Exception as e:
                agent_time = time.time() - agent_start
                logger.error("Agent %s failed in coordinator: %s", spec.name, e)
                
                results["__errors__"].append({
                    "agent": spec.name, 
                    "error": str(e),
                    "type": "execution_failure",
                    "execution_time_ms": agent_time * 1000
                })
                
                # Add failed agent metrics
                results["__metrics__"][spec.name] = {
                    "execution_time_ms": agent_time * 1000,
                    "success": False,
                    "error": str(e)
                }

        total_time = time.time() - execution_start
        results["final_content"] = base_context.content
        results["__metrics__"]["total_execution_time_ms"] = total_time * 1000
        results["__metrics__"]["agents_executed"] = len(ordered_specs)
        results["__metrics__"]["agents_succeeded"] = len([m for m in results["__metrics__"].values() if isinstance(m, dict) and m.get('success')])
        
        return results

    def execute_parallel_best_effort(
        self,
        specs: List[AgentExecutionSpec],
        base_context: ProcessingContext,
    ) -> Dict[str, Any]:
        """
        Best-effort parallel execution (coarse), then aggregate. This avoids hard concurrency
        dependencies to keep it simple and deterministic for now; real parallelism can be added
        later. We execute independent agents first and combine their processed_content by last-writer-wins.
        """
        # Split independent (no depends_on) vs dependent
        independent = [s for s in specs if not s.depends_on]
        dependent = [s for s in specs if s.depends_on]
        combined: Dict[str, Any] = {"__errors__": []}
        context = base_context
        for s in independent + dependent:
            try:
                res = self.execute_pipeline([s], context)
                combined[s.name] = res.get(s.name) or {}
                # propagate content
                if isinstance(combined[s.name], dict):
                    improved = combined[s.name].get('processed_content')
                    if improved:
                        context = ProcessingContext(
                            content=improved,
                            section_type=context.section_type,
                            audience=context.audience,
                            technical_level=context.technical_level,
                            word_count_target=context.word_count_target,
                            processing_mode=context.processing_mode,
                            metadata=context.metadata,
                        )
            except Exception as e:
                combined["__errors__"].append({"agent": s.name, "error": str(e)})
        combined['final_content'] = context.content
        return combined

    def _should_skip_agent_for_performance(self, agent_name: str, spec: AgentExecutionSpec) -> bool:
        """Determine if agent should be skipped for performance reasons."""
        # Check agent health and recent performance
        health = spec.agent.get_health_status()
        avg_response_time = health['metrics']['average_response_time_ms']
        
        # Skip if agent is consistently slow
        if avg_response_time > self._performance_thresholds['max_agent_time_ms']:
            logger.warning(f"Skipping {agent_name} due to slow performance: {avg_response_time}ms average")
            return True
        
        # Skip if agent is in failed state and has multiple consecutive failures
        if health['status'] == 'failed' and health['metrics']['consecutive_failures'] >= 3:
            logger.warning(f"Skipping {agent_name} due to repeated failures")
            return True
        
        return False

    def _optimize_processing_mode(self, context: ProcessingContext, remaining_time_ms: float) -> ProcessingMode:
        """Optimize processing mode based on remaining time and performance requirements."""
        if remaining_time_ms < 10000:  # Less than 10 seconds remaining
            logger.info("Switching to FAST mode due to time constraints")
            return ProcessingMode.FAST
        elif remaining_time_ms < 30000:  # Less than 30 seconds remaining
            return ProcessingMode.FAST if context.processing_mode == ProcessingMode.FULL else context.processing_mode
        else:
            return context.processing_mode

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all managed agents."""
        metrics = {}
        for name, agent in self.agents.items():
            health = agent.get_health_status()
            metrics[name] = {
                'status': health['status'],
                'success_rate': health['metrics']['success_rate'],
                'average_response_time_ms': health['metrics']['average_response_time_ms'],
                'consecutive_failures': health['metrics']['consecutive_failures']
            }
        return metrics

    def reset_all_agent_metrics(self):
        """Reset metrics for all agents (useful for testing or maintenance)."""
        for agent in self.agents.values():
            agent.reset_metrics()
        self._execution_cache.clear()
        logger.info("Reset metrics for all agents")

    def get_coordination_health(self) -> Dict[str, Any]:
        """Get overall coordination system health."""
        agent_health = self.get_performance_metrics()
        total_agents = len(self.agents)
        healthy_agents = sum(1 for metrics in agent_health.values() if metrics['status'] == 'healthy')
        
        return {
            'total_agents': total_agents,
            'healthy_agents': healthy_agents,
            'health_percentage': (healthy_agents / total_agents * 100) if total_agents > 0 else 0,
            'cache_size': len(self._execution_cache),
            'agent_details': agent_health
        }


# Cross-Agent Coordination Patterns (Phase 2 Week 6)

def transfer_research_context(research_results: Dict, writer_agent):
    """Transfer validated search results and claims to writer."""
    from core.tool_cache import get_tool_cache
    
    cache = get_tool_cache()
    
    # Extract validated claims and sources
    validated_claims = research_results.get('validated_claims', [])
    search_results = research_results.get('search_results', [])
    vector_context = research_results.get('vector_context', '')
    
    # Package coordination data
    coordination_data = {
        'validated_claims': validated_claims,
        'search_results': search_results,
        'vector_context': vector_context,
        'research_quality_score': research_results.get('quality_score', 0.0),
        'research_metadata': {
            'sources_count': len(search_results),
            'claims_count': len(validated_claims),
            'research_timestamp': research_results.get('timestamp'),
            'research_agent': research_results.get('agent_name', 'ResearchAgent')
        }
    }
    
    # Share with writer agent
    cache.share_between_agents(
        from_agent="ResearchAgent",
        to_agent=writer_agent.name if hasattr(writer_agent, 'name') else "WriterAgent",
        data=coordination_data,
        message_type="research_context",
        session_id=research_results.get('session_id'),
        workflow_id=research_results.get('workflow_id')
    )
    
    logger.info(f"Transferred research context: {len(validated_claims)} claims, {len(search_results)} sources")
    return coordination_data


def transfer_content_metadata(content: str, tool_usage: Dict, editor_agent,
                            writer_agent_name: str = "WriterAgent",
                            session_id: Optional[str] = None,
                            workflow_id: Optional[str] = None):
    """Transfer content with tool usage metadata for validation."""
    from core.tool_cache import get_tool_cache
    
    cache = get_tool_cache()
    
    # Extract content analysis metadata
    content_metadata = {
        'content': content,
        'content_length': len(content),
        'tool_usage_metrics': tool_usage,
        'content_analysis': {
            'word_count': len(content.split()),
            'paragraph_count': len([p for p in content.split('\n\n') if p.strip()]),
            'has_headers': '##' in content or '#' in content,
            'has_links': '[' in content and '](' in content,
            'estimated_reading_time': len(content.split()) / 200  # 200 WPM average
        },
        'quality_indicators': {
            'tool_integration': tool_usage.get('vector_queries', 0) > 0,
            'claim_validation': len(tool_usage.get('verified_claims', [])),
            'source_diversity': len(set(tool_usage.get('search_providers', [])))
        },
        'writer_metadata': {
            'generation_timestamp': time.time(),
            'writer_agent': writer_agent_name
        }
    }
    
    # Share with editor agent
    cache.share_between_agents(
        from_agent=writer_agent_name,
        to_agent=editor_agent.name if hasattr(editor_agent, 'name') else "EditorAgent",
        data=content_metadata,
        message_type="content_validation",
        session_id=session_id,
        workflow_id=workflow_id
    )
    
    logger.info(f"Transferred content metadata: {len(content)} chars, "
                f"{tool_usage.get('vector_queries', 0)} vector queries, "
                f"{len(tool_usage.get('verified_claims', []))} verified claims")
    return content_metadata


def coordinate_iterative_refinement(content: str, agents: List[Any], 
                                  max_iterations: int = 3,
                                  quality_threshold: float = 0.8,
                                  session_id: Optional[str] = None,
                                  workflow_id: Optional[str] = None) -> Dict[str, Any]:
    """Coordinate iterative refinement across multiple agents."""
    from core.tool_cache import get_tool_cache
    
    cache = get_tool_cache()
    current_content = content
    iteration_results = []
    
    for iteration in range(max_iterations):
        logger.info(f"Starting refinement iteration {iteration + 1}/{max_iterations}")
        
        iteration_start = time.time()
        agent_results = {}
        
        # Run all agents on current content
        for agent in agents:
            try:
                agent_name = getattr(agent, 'name', agent.__class__.__name__)
                
                # Get any pending coordination messages for this agent
                messages = cache.get_agent_messages(agent_name, clear_after_read=False)
                
                # Process content with agent
                if hasattr(agent, 'process'):
                    result = agent.process(current_content)
                elif hasattr(agent, 'execute_task'):
                    result = agent.execute_task(f"Refine content (iteration {iteration + 1})", current_content)
                else:
                    logger.warning(f"Agent {agent_name} has no recognized processing method")
                    continue
                
                agent_results[agent_name] = {
                    'result': result,
                    'messages_received': len(messages),
                    'processing_time': time.time() - iteration_start
                }
                
                # Update content if agent provided improvements
                if isinstance(result, str) and len(result.strip()) > len(current_content.strip()) * 0.8:
                    current_content = result
                elif hasattr(result, 'processed_content') and result.processed_content:
                    current_content = result.processed_content
                
            except Exception as e:
                logger.error(f"Agent {agent_name} failed in iteration {iteration + 1}: {e}")
                agent_results[agent_name] = {'error': str(e)}
        
        # Evaluate iteration quality
        iteration_quality = _evaluate_iteration_quality(current_content, agent_results)
        
        iteration_result = {
            'iteration': iteration + 1,
            'content': current_content,
            'quality_score': iteration_quality,
            'agent_results': agent_results,
            'duration_seconds': time.time() - iteration_start
        }
        
        iteration_results.append(iteration_result)
        
        # Cache iteration results
        cache.cache_analysis_results(
            f"refinement_iteration_{iteration + 1}",
            iteration_result,
            session_id=session_id,
            workflow_id=workflow_id
        )
        
        # Check if quality threshold is met
        if iteration_quality >= quality_threshold:
            logger.info(f"Quality threshold {quality_threshold} met at iteration {iteration + 1}")
            break
        
        # Share iteration results between agents for next iteration
        for agent in agents:
            agent_name = getattr(agent, 'name', agent.__class__.__name__)
            
            coordination_data = {
                'iteration_number': iteration + 1,
                'current_content': current_content,
                'quality_score': iteration_quality,
                'peer_results': {k: v for k, v in agent_results.items() if k != agent_name}
            }
            
            cache.share_between_agents(
                from_agent="IterativeCoordinator",
                to_agent=agent_name,
                data=coordination_data,
                message_type="iteration_feedback",
                session_id=session_id,
                workflow_id=workflow_id
            )
    
    return {
        'final_content': current_content,
        'iterations': iteration_results,
        'total_iterations': len(iteration_results),
        'final_quality_score': iteration_results[-1]['quality_score'] if iteration_results else 0.0,
        'improvement_achieved': len(iteration_results) > 1,
        'coordination_metadata': {
            'agents_involved': [getattr(a, 'name', a.__class__.__name__) for a in agents],
            'session_id': session_id,
            'workflow_id': workflow_id
        }
    }


def _evaluate_iteration_quality(content: str, agent_results: Dict[str, Any]) -> float:
    """Evaluate the quality of a refinement iteration."""
    quality_score = 0.5  # Base score
    
    # Content length and structure improvements
    word_count = len(content.split())
    if word_count > 100:  # Substantial content
        quality_score += 0.1
    
    if '##' in content or '#' in content:  # Has headers
        quality_score += 0.1
    
    if '[' in content and '](' in content:  # Has links
        quality_score += 0.1
    
    # Agent success rate
    successful_agents = sum(1 for result in agent_results.values() 
                          if isinstance(result, dict) and 'error' not in result)
    total_agents = len(agent_results)
    
    if total_agents > 0:
        success_rate = successful_agents / total_agents
        quality_score += success_rate * 0.2
    
    # Processing efficiency (faster is better)
    avg_processing_time = sum(
        result.get('processing_time', 0) for result in agent_results.values()
        if isinstance(result, dict)
    ) / max(total_agents, 1)
    
    if avg_processing_time < 5.0:  # Under 5 seconds average
        quality_score += 0.1
    
    return min(1.0, quality_score)


__all__ = ["AgentCoordinator", "AgentExecutionSpec", "transfer_research_context", 
           "transfer_content_metadata", "coordinate_iterative_refinement"]


