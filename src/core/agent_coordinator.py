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

from src.agents.base_agent import (
    BaseSpecializedAgent,
    ProcessingContext,
    ProcessingResult,
    ProcessingMode,
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


__all__ = ["AgentCoordinator", "AgentExecutionSpec"]


