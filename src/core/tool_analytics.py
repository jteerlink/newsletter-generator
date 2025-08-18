"""
Tool Effectiveness Analytics

Analyzes and optimizes tool usage effectiveness as specified in PRD Phase 3 Week 8.
Provides comprehensive analytics on tool performance, impact measurement, and 
optimization recommendations for intelligent tool selection.
"""

from __future__ import annotations

import hashlib
import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from core.tool_cache import get_tool_cache

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """Types of tools in the system."""
    VECTOR_SEARCH = "vector_search"
    WEB_SEARCH = "web_search"
    CLAIM_VALIDATION = "claim_validation"
    INFORMATION_ENRICHMENT = "information_enrichment"
    SOURCE_RANKING = "source_ranking"
    CITATION_GENERATION = "citation_generation"


class ImpactCategory(Enum):
    """Categories of tool impact on content quality."""
    CONTENT_QUALITY = "content_quality"
    FACTUAL_ACCURACY = "factual_accuracy"
    SOURCE_DIVERSITY = "source_diversity"
    INFORMATION_FRESHNESS = "information_freshness"
    TECHNICAL_DEPTH = "technical_depth"
    USER_ENGAGEMENT = "user_engagement"


@dataclass
class ToolUsageMetric:
    """Metrics for a specific tool usage instance."""
    tool_type: ToolType
    agent_name: str
    execution_time_ms: float
    success: bool
    input_size: int
    output_size: int
    quality_score: float
    timestamp: datetime
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImpactMeasurement:
    """Measurement of tool impact on content quality."""
    tool_type: ToolType
    before_content: str
    after_content: str
    quality_improvement: float
    impact_categories: Dict[ImpactCategory, float]
    confidence_score: float
    measurement_timestamp: datetime


@dataclass
class ToolPerformanceReport:
    """Performance report for a specific tool."""
    tool_type: ToolType
    total_usage_count: int
    success_rate: float
    average_execution_time_ms: float
    average_quality_impact: float
    usage_trend: str  # "increasing", "stable", "decreasing"
    efficiency_score: float
    recommendations: List[str]
    time_period: Tuple[datetime, datetime]


@dataclass
class OptimizationRecommendation:
    """Recommendation for tool usage optimization."""
    agent_type: str
    task_type: str
    recommended_tools: List[ToolType]
    confidence: float
    reasoning: str
    expected_improvement: float
    implementation_effort: str  # "low", "medium", "high"


class ToolEffectivenessAnalyzer:
    """Analyze and optimize tool usage effectiveness."""
    
    def __init__(self):
        """Initialize tool effectiveness analyzer."""
        self.cache = get_tool_cache()
        self.usage_metrics: List[ToolUsageMetric] = []
        self.impact_measurements: List[ImpactMeasurement] = []
        
        # Tool performance baselines
        self.performance_baselines = {
            ToolType.VECTOR_SEARCH: {"execution_time_ms": 1000, "quality_impact": 0.15},
            ToolType.WEB_SEARCH: {"execution_time_ms": 3000, "quality_impact": 0.20},
            ToolType.CLAIM_VALIDATION: {"execution_time_ms": 5000, "quality_impact": 0.25},
            ToolType.INFORMATION_ENRICHMENT: {"execution_time_ms": 4000, "quality_impact": 0.18},
            ToolType.SOURCE_RANKING: {"execution_time_ms": 500, "quality_impact": 0.10},
            ToolType.CITATION_GENERATION: {"execution_time_ms": 800, "quality_impact": 0.12}
        }
        
        logger.info("Tool effectiveness analyzer initialized")
    
    def record_tool_usage(self, tool_type: ToolType, agent_name: str,
                         execution_time_ms: float, success: bool,
                         input_size: int, output_size: int,
                         quality_score: float = 0.0,
                         session_id: Optional[str] = None,
                         workflow_id: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None):
        """Record a tool usage instance for analysis."""
        metric = ToolUsageMetric(
            tool_type=tool_type,
            agent_name=agent_name,
            execution_time_ms=execution_time_ms,
            success=success,
            input_size=input_size,
            output_size=output_size,
            quality_score=quality_score,
            timestamp=datetime.now(),
            session_id=session_id,
            workflow_id=workflow_id,
            metadata=metadata or {}
        )
        
        self.usage_metrics.append(metric)
        
        # Cache the metric for persistence
        self.cache.cache_analysis_results(
            f"tool_usage_metric_{tool_type.value}_{datetime.now().timestamp()}",
            metric,
            session_id=session_id,
            workflow_id=workflow_id
        )
        
        logger.debug(f"Recorded tool usage: {tool_type.value} by {agent_name}")
    
    def analyze_tool_impact(self, before_content: str, after_content: str, 
                          tool_usage: Dict[str, Any]) -> ImpactMeasurement:
        """Measure impact of tool usage on content quality."""
        # Determine primary tool used
        tool_type = self._identify_primary_tool(tool_usage)
        
        # Calculate quality improvement
        quality_improvement = self._calculate_quality_improvement(before_content, after_content)
        
        # Analyze impact by category
        impact_categories = self._analyze_impact_categories(before_content, after_content, tool_usage)
        
        # Calculate confidence score
        confidence_score = self._calculate_impact_confidence(
            before_content, after_content, tool_usage)
        
        impact = ImpactMeasurement(
            tool_type=tool_type,
            before_content=before_content,
            after_content=after_content,
            quality_improvement=quality_improvement,
            impact_categories=impact_categories,
            confidence_score=confidence_score,
            measurement_timestamp=datetime.now()
        )
        
        self.impact_measurements.append(impact)
        
        # Cache impact measurement
        self.cache.cache_analysis_results(
            f"tool_impact_{tool_type.value}_{datetime.now().timestamp()}",
            impact,
            session_id=tool_usage.get('session_id'),
            workflow_id=tool_usage.get('workflow_id')
        )
        
        logger.info(f"Tool impact analyzed: {tool_type.value} -> {quality_improvement:+.3f} quality improvement")
        return impact
    
    def optimize_tool_selection(self, task_type: str, 
                               historical_data: List[Dict[str, Any]]) -> List[ToolType]:
        """Recommend optimal tools based on historical effectiveness."""
        # Analyze historical performance by tool
        tool_performance = defaultdict(list)
        
        for data in historical_data:
            if data.get('task_type') == task_type:
                tools_used = data.get('tools_used', [])
                quality_score = data.get('quality_score', 0.0)
                execution_time = data.get('execution_time_ms', 0)
                
                for tool in tools_used:
                    tool_performance[tool].append({
                        'quality': quality_score,
                        'time': execution_time,
                        'efficiency': quality_score / max(execution_time, 1) * 1000  # Quality per second
                    })
        
        # Rank tools by effectiveness
        tool_scores = {}
        for tool, performances in tool_performance.items():
            if len(performances) >= 3:  # Minimum data points
                avg_quality = statistics.mean(p['quality'] for p in performances)
                avg_efficiency = statistics.mean(p['efficiency'] for p in performances)
                consistency = 1.0 - (statistics.stdev(p['quality'] for p in performances) / max(avg_quality, 0.1))
                
                # Combined effectiveness score
                effectiveness = (avg_quality * 0.4) + (avg_efficiency * 0.4) + (consistency * 0.2)
                tool_scores[tool] = effectiveness
        
        # Return top tools
        recommended_tools = []
        for tool, score in sorted(tool_scores.items(), key=lambda x: x[1], reverse=True):
            try:
                tool_enum = ToolType(tool)
                recommended_tools.append(tool_enum)
            except ValueError:
                logger.warning(f"Unknown tool type in optimization: {tool}")
        
        return recommended_tools[:5]  # Top 5 recommendations
    
    def generate_usage_recommendations(self, agent_type: str) -> List[OptimizationRecommendation]:
        """Generate personalized tool usage recommendations."""
        recommendations = []
        
        # Analyze current usage patterns for agent
        agent_metrics = [m for m in self.usage_metrics if m.agent_name.lower().startswith(agent_type.lower())]
        
        if not agent_metrics:
            # No historical data - provide general recommendations
            recommendations.extend(self._get_general_recommendations(agent_type))
            return recommendations
        
        # Analyze tool usage patterns
        tool_usage_stats = defaultdict(list)
        for metric in agent_metrics:
            tool_usage_stats[metric.tool_type].append(metric)
        
        # Identify underutilized high-impact tools
        for tool_type, baseline in self.performance_baselines.items():
            if tool_type not in tool_usage_stats:
                # Tool not used at all
                recommendations.append(OptimizationRecommendation(
                    agent_type=agent_type,
                    task_type="general",
                    recommended_tools=[tool_type],
                    confidence=0.7,
                    reasoning=f"Tool {tool_type.value} not currently used but has high impact potential",
                    expected_improvement=baseline["quality_impact"],
                    implementation_effort="low"
                ))
            else:
                # Analyze usage effectiveness
                metrics = tool_usage_stats[tool_type]
                avg_quality = statistics.mean(m.quality_score for m in metrics if m.quality_score > 0)
                success_rate = sum(1 for m in metrics if m.success) / len(metrics)
                
                if success_rate < 0.8:
                    recommendations.append(OptimizationRecommendation(
                        agent_type=agent_type,
                        task_type="general",
                        recommended_tools=[tool_type],
                        confidence=0.8,
                        reasoning=f"Improve {tool_type.value} reliability (current success rate: {success_rate:.1%})",
                        expected_improvement=0.1,
                        implementation_effort="medium"
                    ))
        
        # Identify tool combinations that work well together
        combo_recommendations = self._analyze_tool_combinations(agent_type)
        recommendations.extend(combo_recommendations)
        
        return recommendations
    
    def get_tool_performance_report(self, tool_type: ToolType, 
                                  days_back: int = 30) -> ToolPerformanceReport:
        """Generate performance report for a specific tool."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        relevant_metrics = [m for m in self.usage_metrics 
                          if m.tool_type == tool_type and m.timestamp >= cutoff_date]
        
        if not relevant_metrics:
            return ToolPerformanceReport(
                tool_type=tool_type,
                total_usage_count=0,
                success_rate=0.0,
                average_execution_time_ms=0.0,
                average_quality_impact=0.0,
                usage_trend="no_data",
                efficiency_score=0.0,
                recommendations=["No usage data available"],
                time_period=(cutoff_date, datetime.now())
            )
        
        # Calculate metrics
        total_usage_count = len(relevant_metrics)
        success_rate = sum(1 for m in relevant_metrics if m.success) / total_usage_count
        average_execution_time_ms = statistics.mean(m.execution_time_ms for m in relevant_metrics)
        
        quality_scores = [m.quality_score for m in relevant_metrics if m.quality_score > 0]
        average_quality_impact = statistics.mean(quality_scores) if quality_scores else 0.0
        
        # Calculate usage trend
        usage_trend = self._calculate_usage_trend(relevant_metrics)
        
        # Calculate efficiency score
        baseline = self.performance_baselines.get(tool_type, {"execution_time_ms": 1000, "quality_impact": 0.1})
        time_efficiency = min(1.0, baseline["execution_time_ms"] / max(average_execution_time_ms, 1))
        quality_efficiency = average_quality_impact / max(baseline["quality_impact"], 0.01)
        efficiency_score = (time_efficiency + quality_efficiency) / 2
        
        # Generate recommendations
        recommendations = self._generate_tool_recommendations(
            tool_type, success_rate, average_execution_time_ms, average_quality_impact)
        
        return ToolPerformanceReport(
            tool_type=tool_type,
            total_usage_count=total_usage_count,
            success_rate=success_rate,
            average_execution_time_ms=average_execution_time_ms,
            average_quality_impact=average_quality_impact,
            usage_trend=usage_trend,
            efficiency_score=efficiency_score,
            recommendations=recommendations,
            time_period=(cutoff_date, datetime.now())
        )
    
    def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get comprehensive tool usage analytics."""
        if not self.usage_metrics:
            return {"error": "No usage data available"}
        
        # Overall statistics
        total_usage = len(self.usage_metrics)
        overall_success_rate = sum(1 for m in self.usage_metrics if m.success) / total_usage
        
        # Tool usage distribution
        tool_usage_dist = Counter(m.tool_type.value for m in self.usage_metrics)
        
        # Agent usage distribution
        agent_usage_dist = Counter(m.agent_name for m in self.usage_metrics)
        
        # Performance trends
        recent_metrics = [m for m in self.usage_metrics 
                         if m.timestamp >= datetime.now() - timedelta(days=7)]
        
        performance_trends = {}
        for tool_type in ToolType:
            tool_metrics = [m for m in recent_metrics if m.tool_type == tool_type]
            if tool_metrics:
                avg_time = statistics.mean(m.execution_time_ms for m in tool_metrics)
                success_rate = sum(1 for m in tool_metrics if m.success) / len(tool_metrics)
                performance_trends[tool_type.value] = {
                    "average_time_ms": avg_time,
                    "success_rate": success_rate,
                    "usage_count": len(tool_metrics)
                }
        
        # Quality impact analysis
        quality_impacts = {}
        for impact in self.impact_measurements:
            tool = impact.tool_type.value
            if tool not in quality_impacts:
                quality_impacts[tool] = []
            quality_impacts[tool].append(impact.quality_improvement)
        
        quality_impact_summary = {}
        for tool, impacts in quality_impacts.items():
            quality_impact_summary[tool] = {
                "average_improvement": statistics.mean(impacts),
                "measurement_count": len(impacts),
                "max_improvement": max(impacts),
                "consistency": 1.0 - (statistics.stdev(impacts) / max(statistics.mean(impacts), 0.1))
            }
        
        return {
            "overview": {
                "total_tool_usage": total_usage,
                "overall_success_rate": overall_success_rate,
                "tools_analyzed": len(ToolType),
                "agents_tracked": len(agent_usage_dist),
                "analysis_period_days": (datetime.now() - min(m.timestamp for m in self.usage_metrics)).days
            },
            "tool_usage_distribution": dict(tool_usage_dist),
            "agent_usage_distribution": dict(agent_usage_dist),
            "performance_trends": performance_trends,
            "quality_impact_analysis": quality_impact_summary,
            "top_performing_tools": self._get_top_performing_tools(),
            "optimization_opportunities": self._identify_optimization_opportunities()
        }
    
    def _identify_primary_tool(self, tool_usage: Dict[str, Any]) -> ToolType:
        """Identify the primary tool used based on usage data."""
        # Simple heuristic - most usage indicates primary tool
        if tool_usage.get('vector_queries', 0) > 0:
            return ToolType.VECTOR_SEARCH
        elif tool_usage.get('web_searches', 0) > 0:
            return ToolType.WEB_SEARCH
        elif tool_usage.get('verified_claims'):
            return ToolType.CLAIM_VALIDATION
        else:
            return ToolType.VECTOR_SEARCH  # Default
    
    def _calculate_quality_improvement(self, before: str, after: str) -> float:
        """Calculate quality improvement between before and after content."""
        # Simple metrics-based approach
        before_words = len(before.split())
        after_words = len(after.split())
        
        # Length improvement (normalized)
        length_improvement = min(1.0, (after_words - before_words) / max(before_words, 1))
        
        # Structure improvement
        before_structure = before.count('##') + before.count('- ') + before.count('* ')
        after_structure = after.count('##') + after.count('- ') + after.count('* ')
        structure_improvement = (after_structure - before_structure) / max(before_structure + 1, 1)
        
        # Link improvement
        before_links = before.count('[') + before.count('](')
        after_links = after.count('[') + after.count('](')
        link_improvement = (after_links - before_links) / max(before_links + 1, 1)
        
        # Combined improvement score
        improvement = (length_improvement * 0.4 + 
                      structure_improvement * 0.3 + 
                      link_improvement * 0.3)
        
        return max(0.0, min(1.0, improvement))
    
    def _analyze_impact_categories(self, before: str, after: str, 
                                 tool_usage: Dict[str, Any]) -> Dict[ImpactCategory, float]:
        """Analyze impact by different categories."""
        impacts = {}
        
        # Content quality impact
        word_diff = len(after.split()) - len(before.split())
        impacts[ImpactCategory.CONTENT_QUALITY] = min(1.0, max(0.0, word_diff / 100))
        
        # Factual accuracy impact (based on claim verification)
        verified_claims = len(tool_usage.get('verified_claims', []))
        impacts[ImpactCategory.FACTUAL_ACCURACY] = min(1.0, verified_claims * 0.2)
        
        # Source diversity impact
        search_providers = len(set(tool_usage.get('search_providers', [])))
        impacts[ImpactCategory.SOURCE_DIVERSITY] = min(1.0, search_providers * 0.25)
        
        # Information freshness impact
        fresh_indicators = sum(1 for indicator in ['2024', '2025', 'recent', 'latest'] 
                              if indicator in after.lower() and indicator not in before.lower())
        impacts[ImpactCategory.INFORMATION_FRESHNESS] = min(1.0, fresh_indicators * 0.3)
        
        # Technical depth impact
        tech_terms = ['algorithm', 'implementation', 'architecture', 'framework']
        tech_added = sum(1 for term in tech_terms 
                        if term in after.lower() and term not in before.lower())
        impacts[ImpactCategory.TECHNICAL_DEPTH] = min(1.0, tech_added * 0.2)
        
        # User engagement impact (based on structure)
        engagement_features = ['##', '- ', '* ', '[', '```']
        engagement_added = sum(after.count(feature) - before.count(feature) 
                              for feature in engagement_features)
        impacts[ImpactCategory.USER_ENGAGEMENT] = min(1.0, max(0.0, engagement_added * 0.1))
        
        return impacts
    
    def _calculate_impact_confidence(self, before: str, after: str, 
                                   tool_usage: Dict[str, Any]) -> float:
        """Calculate confidence in impact measurement."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence with more tool usage
        tool_usage_score = (tool_usage.get('vector_queries', 0) * 0.1 + 
                           tool_usage.get('web_searches', 0) * 0.15 + 
                           len(tool_usage.get('verified_claims', [])) * 0.1)
        confidence += min(0.3, tool_usage_score)
        
        # Higher confidence with significant content changes
        content_change_ratio = abs(len(after) - len(before)) / max(len(before), 1)
        if content_change_ratio > 0.1:  # Significant change
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _calculate_usage_trend(self, metrics: List[ToolUsageMetric]) -> str:
        """Calculate usage trend over time."""
        if len(metrics) < 5:
            return "insufficient_data"
        
        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        
        # Split into early and recent periods
        mid_point = len(sorted_metrics) // 2
        early_usage = sorted_metrics[:mid_point]
        recent_usage = sorted_metrics[mid_point:]
        
        early_rate = len(early_usage) / max((early_usage[-1].timestamp - early_usage[0].timestamp).days, 1)
        recent_rate = len(recent_usage) / max((recent_usage[-1].timestamp - recent_usage[0].timestamp).days, 1)
        
        if recent_rate > early_rate * 1.2:
            return "increasing"
        elif recent_rate < early_rate * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _generate_tool_recommendations(self, tool_type: ToolType, success_rate: float,
                                     avg_time: float, avg_quality: float) -> List[str]:
        """Generate recommendations for tool improvement."""
        recommendations = []
        
        baseline = self.performance_baselines.get(tool_type, {})
        
        if success_rate < 0.9:
            recommendations.append(f"Improve {tool_type.value} reliability (current: {success_rate:.1%})")
        
        if avg_time > baseline.get("execution_time_ms", 1000) * 1.5:
            recommendations.append(f"Optimize {tool_type.value} performance (current: {avg_time:.0f}ms)")
        
        if avg_quality < baseline.get("quality_impact", 0.1):
            recommendations.append(f"Enhance {tool_type.value} quality impact")
        
        if not recommendations:
            recommendations.append(f"{tool_type.value} performing within expected parameters")
        
        return recommendations
    
    def _get_general_recommendations(self, agent_type: str) -> List[OptimizationRecommendation]:
        """Get general recommendations for agent types without historical data."""
        recommendations = []
        
        if agent_type.lower() in ['writer', 'writing']:
            recommendations.append(OptimizationRecommendation(
                agent_type=agent_type,
                task_type="content_generation",
                recommended_tools=[ToolType.VECTOR_SEARCH, ToolType.CLAIM_VALIDATION],
                confidence=0.8,
                reasoning="Writers benefit from vector search for context and claim validation for accuracy",
                expected_improvement=0.25,
                implementation_effort="low"
            ))
        
        elif agent_type.lower() in ['research', 'researcher']:
            recommendations.append(OptimizationRecommendation(
                agent_type=agent_type,
                task_type="research",
                recommended_tools=[ToolType.WEB_SEARCH, ToolType.SOURCE_RANKING, ToolType.INFORMATION_ENRICHMENT],
                confidence=0.9,
                reasoning="Researchers need comprehensive search and source validation capabilities",
                expected_improvement=0.35,
                implementation_effort="medium"
            ))
        
        elif agent_type.lower() in ['editor', 'editing']:
            recommendations.append(OptimizationRecommendation(
                agent_type=agent_type,
                task_type="content_review",
                recommended_tools=[ToolType.CLAIM_VALIDATION, ToolType.CITATION_GENERATION],
                confidence=0.7,
                reasoning="Editors focus on accuracy and proper citations",
                expected_improvement=0.20,
                implementation_effort="low"
            ))
        
        return recommendations
    
    def _analyze_tool_combinations(self, agent_type: str) -> List[OptimizationRecommendation]:
        """Analyze effective tool combinations."""
        # Find workflows with multiple tools used together
        workflow_combinations = defaultdict(list)
        
        for metric in self.usage_metrics:
            if metric.agent_name.lower().startswith(agent_type.lower()) and metric.workflow_id:
                workflow_combinations[metric.workflow_id].append(metric)
        
        # Analyze combinations that led to high quality
        effective_combinations = []
        for workflow_id, metrics in workflow_combinations.items():
            if len(metrics) >= 2:  # Multiple tools used
                avg_quality = statistics.mean(m.quality_score for m in metrics if m.quality_score > 0)
                if avg_quality > 0.7:  # High quality outcome
                    tools_used = [m.tool_type for m in metrics]
                    effective_combinations.append((tools_used, avg_quality))
        
        # Generate recommendations for effective combinations
        recommendations = []
        if effective_combinations:
            # Find most common effective combination
            combo_counter = Counter(tuple(sorted(combo[0])) for combo in effective_combinations)
            top_combo = combo_counter.most_common(1)[0]
            
            recommendations.append(OptimizationRecommendation(
                agent_type=agent_type,
                task_type="workflow",
                recommended_tools=list(top_combo[0]),
                confidence=0.8,
                reasoning=f"Tool combination used in {top_combo[1]} high-quality workflows",
                expected_improvement=0.15,
                implementation_effort="medium"
            ))
        
        return recommendations
    
    def _get_top_performing_tools(self) -> List[Dict[str, Any]]:
        """Get top performing tools by efficiency."""
        tool_performance = {}
        
        for tool_type in ToolType:
            metrics = [m for m in self.usage_metrics if m.tool_type == tool_type]
            if metrics:
                avg_quality = statistics.mean(m.quality_score for m in metrics if m.quality_score > 0)
                avg_time = statistics.mean(m.execution_time_ms for m in metrics)
                success_rate = sum(1 for m in metrics if m.success) / len(metrics)
                
                efficiency = (avg_quality * success_rate) / max(avg_time, 1) * 1000
                tool_performance[tool_type.value] = {
                    "efficiency": efficiency,
                    "quality": avg_quality,
                    "success_rate": success_rate,
                    "usage_count": len(metrics)
                }
        
        # Sort by efficiency and return top performers
        sorted_tools = sorted(tool_performance.items(), key=lambda x: x[1]["efficiency"], reverse=True)
        return [{"tool": tool, **stats} for tool, stats in sorted_tools[:5]]
    
    def _identify_optimization_opportunities(self) -> List[str]:
        """Identify optimization opportunities."""
        opportunities = []
        
        # Low success rate tools
        for tool_type in ToolType:
            metrics = [m for m in self.usage_metrics if m.tool_type == tool_type]
            if metrics:
                success_rate = sum(1 for m in metrics if m.success) / len(metrics)
                if success_rate < 0.8:
                    opportunities.append(f"Improve {tool_type.value} reliability ({success_rate:.1%} success rate)")
        
        # Underutilized tools
        usage_counts = Counter(m.tool_type for m in self.usage_metrics)
        total_usage = sum(usage_counts.values())
        
        for tool_type in ToolType:
            usage_rate = usage_counts[tool_type] / max(total_usage, 1)
            if usage_rate < 0.1:  # Less than 10% usage
                opportunities.append(f"Consider increasing {tool_type.value} utilization (current: {usage_rate:.1%})")
        
        # Performance opportunities
        for tool_type, baseline in self.performance_baselines.items():
            metrics = [m for m in self.usage_metrics if m.tool_type == tool_type]
            if metrics:
                avg_time = statistics.mean(m.execution_time_ms for m in metrics)
                if avg_time > baseline["execution_time_ms"] * 1.5:
                    opportunities.append(f"Optimize {tool_type.value} performance ({avg_time:.0f}ms vs {baseline['execution_time_ms']}ms baseline)")
        
        return opportunities[:10]  # Top 10 opportunities


# Export main classes
__all__ = [
    'ToolEffectivenessAnalyzer',
    'ToolUsageMetric',
    'ImpactMeasurement', 
    'ToolPerformanceReport',
    'OptimizationRecommendation',
    'ToolType',
    'ImpactCategory'
]