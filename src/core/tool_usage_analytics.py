"""
Tool Usage Analytics Dashboard

Provides comprehensive analytics and reporting capabilities for tool usage tracking
across the newsletter generation system. Includes dashboards, reports, and insights.
"""

import json
import logging
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .tool_usage_tracker import ToolUsageLogger, get_tool_tracker

# from .feedback_system import FeedbackLearningSystem  # Not available,
# commenting out for now

logger = logging.getLogger(__name__)


class ToolUsageAnalyticsDashboard:
    """
    Comprehensive analytics dashboard for tool usage tracking and performance analysis.
    """

    def __init__(self,
                 tool_tracker: Optional[ToolUsageLogger] = None,
                 feedback_system: Optional[Any] = None):
        self.tool_tracker = tool_tracker or get_tool_tracker()
        self.feedback_system = feedback_system

        # Initialize dashboard data
        self.dashboard_data = {}
        self.last_refresh = None

    def refresh_dashboard_data(self, hours_back: int = 24) -> Dict[str, Any]:
        """Refresh all dashboard data with latest analytics."""
        logger.info(f"Refreshing dashboard data for last {hours_back} hours")

        # Get core analytics and convert to dictionary
        analytics_obj = self.tool_tracker.generate_usage_analytics(hours_back)
        system_analytics = asdict(analytics_obj)

        # Get agent-specific analytics
        agent_analytics = self._get_agent_analytics(hours_back)

        # Get performance insights
        performance_insights = self._get_performance_insights(hours_back)

        # Get workflow analytics if available
        workflow_analytics = self._get_workflow_analytics(hours_back)

        # Get quality correlations if feedback system is available
        quality_correlations = {}
        if self.feedback_system:
            quality_correlations = self.feedback_system.analyze_tool_usage_correlations()

        self.dashboard_data = {
            "last_refresh": datetime.now().isoformat(),
            "time_range_hours": hours_back,
            "system_analytics": system_analytics,
            "agent_analytics": agent_analytics,
            "performance_insights": performance_insights,
            "workflow_analytics": workflow_analytics,
            "quality_correlations": quality_correlations,
            "summary": self._generate_dashboard_summary(
                system_analytics, agent_analytics, performance_insights
            )
        }

        self.last_refresh = datetime.now()
        return self.dashboard_data

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get a high-level dashboard summary."""
        if not self.dashboard_data or self._needs_refresh():
            self.refresh_dashboard_data()

        return self.dashboard_data.get("summary", {})

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for live dashboard display."""
        recent_entries = self.tool_tracker.get_tool_usage_history(hours_back=1)

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "last_hour": {
                "total_tool_calls": len(recent_entries),
                "unique_tools_used": len(set(entry.tool_name for entry in recent_entries)),
                "unique_agents": len(set(entry.agent_name for entry in recent_entries)),
                "average_execution_time": self._calculate_average_time(recent_entries),
                "success_rate": self._calculate_success_rate(recent_entries)
            },
            "active_sessions": len(set(entry.session_id for entry in recent_entries if entry.session_id)),
            "most_used_tool_last_hour": self._get_most_used_tool(recent_entries),
            "system_status": self._get_system_status()
        }

        return metrics

    def get_tool_performance_report(
            self, tool_name: str, hours_back: int = 168) -> Dict[str, Any]:
        """Generate detailed performance report for a specific tool."""
        tool_entries = [entry for entry in self.tool_tracker.get_tool_usage_history(
            hours_back=hours_back) if entry.tool_name == tool_name]

        if not tool_entries:
            return {"error": f"No usage data found for tool: {tool_name}"}

        # Performance metrics
        execution_times = [
            entry.execution_time for entry in tool_entries if entry.execution_time]
        success_count = len(
            [entry for entry in tool_entries if entry.status == "success"])

        # Usage patterns
        usage_by_agent = defaultdict(int)
        usage_by_hour = defaultdict(int)

        for entry in tool_entries:
            usage_by_agent[entry.agent_name] += 1
            hour = datetime.fromisoformat(entry.timestamp).hour
            usage_by_hour[hour] += 1

        # Error analysis
        error_types = defaultdict(int)
        for entry in tool_entries:
            if entry.status == "failure" and entry.error_details:
                error_type = entry.error_details.get("error_type", "unknown")
                error_types[error_type] += 1

        report = {
            "tool_name": tool_name,
            "analysis_period": f"Last {hours_back} hours",
            "usage_statistics": {
                "total_calls": len(tool_entries),
                "success_rate": success_count /
                len(tool_entries) if tool_entries else 0,
                "average_execution_time": sum(execution_times) /
                len(execution_times) if execution_times else 0,
                "fastest_execution": min(execution_times) if execution_times else 0,
                "slowest_execution": max(execution_times) if execution_times else 0},
            "usage_patterns": {
                "by_agent": dict(usage_by_agent),
                "by_hour_of_day": dict(usage_by_hour),
                "peak_usage_hour": max(
                    usage_by_hour.items(),
                    key=lambda x: x[1])[0] if usage_by_hour else None},
            "error_analysis": {
                "total_errors": len(tool_entries) -
                success_count,
                "error_rate": (
                    len(tool_entries) -
                    success_count) /
                len(tool_entries) if tool_entries else 0,
                "error_types": dict(error_types)},
            "recommendations": self._generate_tool_recommendations(
                tool_name,
                tool_entries)}

        return report

    def get_agent_performance_comparison(
            self, hours_back: int = 168) -> Dict[str, Any]:
        """Compare performance across different agents."""
        all_entries = self.tool_tracker.get_tool_usage_history(
            hours_back=hours_back)

        agent_metrics = defaultdict(lambda: {
            "total_calls": 0,
            "successful_calls": 0,
            "total_execution_time": 0,
            "tools_used": set(),
            "sessions": set(),
            "errors": 0
        })

        # Aggregate metrics by agent
        for entry in all_entries:
            metrics = agent_metrics[entry.agent_name]
            metrics["total_calls"] += 1
            if entry.status == "success":
                metrics["successful_calls"] += 1
            if entry.execution_time:
                metrics["total_execution_time"] += entry.execution_time
            metrics["tools_used"].add(entry.tool_name)
            if entry.session_id:
                metrics["sessions"].add(entry.session_id)
            if entry.status == "failure":
                metrics["errors"] += 1

        # Calculate derived metrics
        comparison = {}
        for agent_name, metrics in agent_metrics.items():
            comparison[agent_name] = {
                "total_tool_calls": metrics["total_calls"],
                "success_rate": metrics["successful_calls"] /
                metrics["total_calls"] if metrics["total_calls"] > 0 else 0,
                "average_execution_time": metrics["total_execution_time"] /
                metrics["successful_calls"] if metrics["successful_calls"] > 0 else 0,
                "unique_tools_used": len(
                    metrics["tools_used"]),
                "active_sessions": len(
                    metrics["sessions"]),
                "error_rate": metrics["errors"] /
                metrics["total_calls"] if metrics["total_calls"] > 0 else 0,
                "efficiency_score": self._calculate_efficiency_score(metrics)}

        # Rank agents
        ranked_agents = sorted(
            comparison.items(),
            key=lambda x: x[1]["efficiency_score"],
            reverse=True
        )

        return {
            "analysis_period": f"Last {hours_back} hours",
            "agent_comparison": comparison,
            "rankings": {
                "by_efficiency": [
                    {
                        "agent": name,
                        "score": data["efficiency_score"]} for name,
                    data in ranked_agents],
                "by_success_rate": sorted(
                    comparison.items(),
                    key=lambda x: x[1]["success_rate"],
                    reverse=True),
                "by_tool_diversity": sorted(
                    comparison.items(),
                    key=lambda x: x[1]["unique_tools_used"],
                    reverse=True)},
            "insights": self._generate_agent_insights(comparison)}

    def generate_executive_summary(
            self, hours_back: int = 168) -> Dict[str, Any]:
        """Generate executive summary for stakeholders."""
        if not self.dashboard_data or self._needs_refresh():
            self.refresh_dashboard_data(hours_back)

        system_analytics = self.dashboard_data.get("system_analytics", {})
        agent_analytics = self.dashboard_data.get("agent_analytics", {})

        # Key metrics
        total_calls = system_analytics.get("total_tool_calls", 0)
        success_rate = system_analytics.get("overall_success_rate", 0)
        avg_response_time = system_analytics.get("average_execution_time", 0)

        # Performance trends
        performance_trend = self._analyze_performance_trend(hours_back)

        # Quality insights
        quality_insights = {}
        if self.feedback_system and self.dashboard_data.get(
                "quality_correlations"):
            quality_insights = self.dashboard_data["quality_correlations"].get(
                "analysis_summary", {})

        summary = {
            "report_period": f"Last {hours_back} hours",
            "generated_at": datetime.now().isoformat(),
            "key_metrics": {
                "total_tool_calls": total_calls,
                "system_success_rate": f"{success_rate:.1%}",
                "average_response_time": f"{avg_response_time:.2f}s",
                "active_agents": len(agent_analytics.get("agent_breakdown", {})),
                "tools_in_use": len(system_analytics.get("tool_breakdown", {}))
            },
            "performance_trend": performance_trend,
            "top_insights": [
                f"System processed {total_calls} tool calls with {success_rate:.1%} success rate",
                f"Average response time: {avg_response_time:.2f} seconds",
                self._get_top_performing_insight(),
                self._get_improvement_opportunity_insight()
            ],
            "quality_insights": quality_insights,
            "recommendations": self._generate_executive_recommendations()
        }

        return summary

    def export_analytics_data(self,
                              format_type: str = "json",
                              output_dir: str = "logs/analytics",
                              hours_back: int = 168) -> str:
        """Export analytics data in various formats."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not self.dashboard_data or self._needs_refresh():
            self.refresh_dashboard_data(hours_back)

        if format_type.lower() == "json":
            filename = f"tool_analytics_{timestamp}.json"
            filepath = output_path / filename

            with open(filepath, 'w') as f:
                json.dump(self.dashboard_data, f, indent=2)

        elif format_type.lower() == "csv":
            filename = f"tool_usage_{timestamp}.csv"
            filepath = output_path / filename

            # Convert tool usage data to CSV
            tool_entries = self.tool_tracker.get_tool_usage_history(
                hours_back=hours_back)
            data = [asdict(entry) for entry in tool_entries]

            if data:
                df = pd.DataFrame(data)
                df.to_csv(filepath, index=False)

        else:
            raise ValueError(f"Unsupported export format: {format_type}")

        logger.info(f"Analytics data exported to: {filepath}")
        return str(filepath)

    # Helper methods
    def _needs_refresh(self, max_age_minutes: int = 5) -> bool:
        """Check if dashboard data needs refresh."""
        if not self.last_refresh:
            return True

        age = datetime.now() - self.last_refresh
        return age > timedelta(minutes=max_age_minutes)

    def _get_agent_analytics(self, hours_back: int) -> Dict[str, Any]:
        """Get agent-specific analytics."""
        entries = self.tool_tracker.get_tool_usage_history(
            hours_back=hours_back)

        agent_breakdown = defaultdict(
            lambda: {
                "calls": 0,
                "success": 0,
                "avg_time": 0,
                "tools": set()})

        for entry in entries:
            agent_breakdown[entry.agent_name]["calls"] += 1
            if entry.status == "success":
                agent_breakdown[entry.agent_name]["success"] += 1
            if entry.execution_time:
                current_avg = agent_breakdown[entry.agent_name]["avg_time"]
                current_count = agent_breakdown[entry.agent_name]["calls"]
                agent_breakdown[entry.agent_name]["avg_time"] = (
                    current_avg * (current_count - 1) + entry.execution_time) / current_count
            agent_breakdown[entry.agent_name]["tools"].add(entry.tool_name)

        # Convert to serializable format
        result = {}
        for agent, data in agent_breakdown.items():
            result[agent] = {
                "total_calls": data["calls"],
                "success_rate": data["success"] /
                data["calls"] if data["calls"] > 0 else 0,
                "average_execution_time": data["avg_time"],
                "unique_tools": len(
                    data["tools"])}

        return {"agent_breakdown": result}

    def _get_performance_insights(self, hours_back: int) -> Dict[str, Any]:
        """Generate performance insights."""
        entries = self.tool_tracker.get_tool_usage_history(
            hours_back=hours_back)

        if not entries:
            return {"no_data": True}

        # Performance analysis
        execution_times = [
            entry.execution_time for entry in entries if entry.execution_time]
        success_count = len(
            [entry for entry in entries if entry.status == "success"])

        # Identify bottlenecks
        slow_tools = defaultdict(list)
        for entry in entries:
            if entry.execution_time and entry.execution_time > 5.0:  # Slow threshold
                slow_tools[entry.tool_name].append(entry.execution_time)

        bottlenecks = {
            tool: {
                "average_slow_time": sum(times) / len(times),
                "slow_call_count": len(times)
            }
            for tool, times in slow_tools.items()
        }

        return {
            "total_entries_analyzed": len(entries),
            "overall_success_rate": success_count /
            len(entries),
            "average_execution_time": sum(execution_times) /
            len(execution_times) if execution_times else 0,
            "performance_bottlenecks": bottlenecks,
            "fast_tools": self._identify_fast_tools(entries),
            "reliable_tools": self._identify_reliable_tools(entries)}

    def _get_workflow_analytics(self, hours_back: int) -> Dict[str, Any]:
        """Get workflow-specific analytics if available."""
        entries = self.tool_tracker.get_tool_usage_history(
            hours_back=hours_back)

        # Group by workflow_id if available
        workflow_data = defaultdict(
            lambda: {
                "calls": 0,
                "tools": set(),
                "duration": 0})

        for entry in entries:
            if entry.workflow_id:
                workflow_data[entry.workflow_id]["calls"] += 1
                workflow_data[entry.workflow_id]["tools"].add(entry.tool_name)

        if not workflow_data:
            return {"message": "No workflow data available"}

        return {
            "total_workflows": len(workflow_data),
            "workflow_breakdown": {
                wf_id: {
                    "total_tool_calls": data["calls"],
                    "unique_tools": len(data["tools"])
                }
                for wf_id, data in workflow_data.items()
            }
        }

    def _generate_dashboard_summary(self,
                                    system_analytics: Dict,
                                    agent_analytics: Dict,
                                    performance_insights: Dict) -> Dict[str,
                                                                        Any]:
        """Generate high-level dashboard summary."""
        return {
            "health_status": self._determine_system_health(system_analytics, performance_insights),
            "key_metrics": {
                "total_tool_calls": system_analytics.get("total_tool_calls", 0),
                "success_rate": system_analytics.get("overall_success_rate", 0),
                "active_agents": len(agent_analytics.get("agent_breakdown", {})),
                "avg_response_time": system_analytics.get("average_execution_time", 0)
            },
            "alerts": self._generate_alerts(system_analytics, performance_insights),
            "top_performers": self._identify_top_performers(system_analytics),
            "improvement_areas": self._identify_improvement_areas(performance_insights)
        }

    def _calculate_average_time(self, entries: List) -> float:
        """Calculate average execution time."""
        times = [entry.execution_time for entry in entries if entry.execution_time]
        return sum(times) / len(times) if times else 0.0

    def _calculate_success_rate(self, entries: List) -> float:
        """Calculate success rate."""
        if not entries:
            return 0.0
        success_count = len(
            [entry for entry in entries if entry.status == "success"])
        return success_count / len(entries)

    def _get_most_used_tool(self, entries: List) -> str:
        """Get most used tool from entries."""
        if not entries:
            return "None"

        tool_counts = defaultdict(int)
        for entry in entries:
            tool_counts[entry.tool_name] += 1

        return max(tool_counts.items(), key=lambda x: x[1])[
            0] if tool_counts else "None"

    def _get_system_status(self) -> str:
        """Determine current system status."""
        recent_entries = self.tool_tracker.get_tool_usage_history(
            hours_back=0.5)  # Last 30 minutes

        if not recent_entries:
            return "idle"

        success_rate = self._calculate_success_rate(recent_entries)
        if success_rate > 0.9:
            return "healthy"
        elif success_rate > 0.7:
            return "warning"
        else:
            return "critical"

    def _calculate_efficiency_score(self, metrics: Dict) -> float:
        """Calculate efficiency score for an agent."""
        success_rate = metrics["successful_calls"] / \
            metrics["total_calls"] if metrics["total_calls"] > 0 else 0
        tool_diversity = len(metrics["tools_used"])
        error_penalty = metrics["errors"] / \
            metrics["total_calls"] if metrics["total_calls"] > 0 else 0

        # Weighted efficiency score
        score = (success_rate * 0.5) + \
            (min(tool_diversity / 5, 1.0) * 0.3) - (error_penalty * 0.2)
        return max(0, min(1, score))  # Clamp between 0 and 1

    def _generate_tool_recommendations(
            self, tool_name: str, entries: List) -> List[str]:
        """Generate recommendations for tool optimization."""
        recommendations = []

        success_rate = self._calculate_success_rate(entries)
        avg_time = self._calculate_average_time(entries)

        if success_rate < 0.8:
            recommendations.append(
                f"Investigate reliability issues - success rate is {success_rate:.1%}")

        if avg_time > 10.0:
            recommendations.append(
                f"Optimize performance - average execution time is {avg_time:.1f}s")

        return recommendations if recommendations else [
            "Tool performance is within acceptable parameters"]

    def _generate_agent_insights(self, comparison: Dict) -> List[str]:
        """Generate insights from agent comparison."""
        insights = []

        if not comparison:
            return ["No agent data available for analysis"]

        # Find highest and lowest performers
        best_agent = max(
            comparison.items(),
            key=lambda x: x[1]["efficiency_score"])
        worst_agent = min(
            comparison.items(),
            key=lambda x: x[1]["efficiency_score"])

        insights.append(
            f"Top performing agent: {
                best_agent[0]} (efficiency: {
                best_agent[1]['efficiency_score']:.1%})")

        if best_agent[1]["efficiency_score"] - \
                worst_agent[1]["efficiency_score"] > 0.2:
            insights.append(
                f"Significant performance gap detected between {
                    best_agent[0]} and {
                    worst_agent[0]}")

        return insights

    def _analyze_performance_trend(self, hours_back: int) -> str:
        """Analyze performance trend."""
        # Simple trend analysis - compare first and second half of time period
        half_period = hours_back // 2

        recent_entries = self.tool_tracker.get_tool_usage_history(
            hours_back=half_period)
        older_entries = self.tool_tracker.get_tool_usage_history(
            hours_back=hours_back)
        older_entries = older_entries[len(recent_entries):]  # Get older half

        if not recent_entries or not older_entries:
            return "insufficient_data"

        recent_success = self._calculate_success_rate(recent_entries)
        older_success = self._calculate_success_rate(older_entries)

        if recent_success > older_success + 0.05:
            return "improving"
        elif recent_success < older_success - 0.05:
            return "declining"
        else:
            return "stable"

    def _get_top_performing_insight(self) -> str:
        """Get insight about top performing tool."""
        system_analytics = self.dashboard_data.get("system_analytics", {})
        most_reliable = system_analytics.get("most_reliable_tools", [])

        if most_reliable:
            return f"Most reliable tool: {
                most_reliable[0].get(
                    'tool',
                    'Unknown')} with {
                most_reliable[0].get(
                    'success_rate',
                    0):.1%} success rate"
        return "Performance data insufficient for insights"

    def _get_improvement_opportunity_insight(self) -> str:
        """Get insight about improvement opportunities."""
        performance_insights = self.dashboard_data.get(
            "performance_insights", {})
        bottlenecks = performance_insights.get("performance_bottlenecks", {})

        if bottlenecks:
            slowest_tool = max(
                bottlenecks.items(),
                key=lambda x: x[1]["average_slow_time"])
            return f"Optimization opportunity: {
                slowest_tool[0]} averaging {
                slowest_tool[1]['average_slow_time']:.1f}s"
        return "No significant performance bottlenecks identified"

    def _generate_executive_recommendations(self) -> List[str]:
        """Generate high-level recommendations for executives."""
        recommendations = []

        system_analytics = self.dashboard_data.get("system_analytics", {})
        success_rate = system_analytics.get("overall_success_rate", 0)

        if success_rate < 0.9:
            recommendations.append(
                "Investigate and address system reliability issues")

        performance_insights = self.dashboard_data.get(
            "performance_insights", {})
        if performance_insights.get("performance_bottlenecks"):
            recommendations.append(
                "Optimize performance bottlenecks for better user experience")

        quality_correlations = self.dashboard_data.get(
            "quality_correlations", {})
        if quality_correlations and quality_correlations.get(
                "analysis_summary", {}).get("recommendations"):
            recommendations.extend(
                quality_correlations["analysis_summary"]["recommendations"][:2])

        return recommendations if recommendations else [
            "System operating within optimal parameters"]

    def _determine_system_health(
            self,
            system_analytics: Dict,
            performance_insights: Dict) -> str:
        """Determine overall system health status."""
        success_rate = system_analytics.get("overall_success_rate", 0)
        avg_time = system_analytics.get("average_execution_time", 0)
        bottlenecks = len(
            performance_insights.get(
                "performance_bottlenecks", {}))

        if success_rate > 0.95 and avg_time < 3.0 and bottlenecks == 0:
            return "excellent"
        elif success_rate > 0.9 and avg_time < 5.0 and bottlenecks < 3:
            return "good"
        elif success_rate > 0.8 and avg_time < 10.0:
            return "fair"
        else:
            return "needs_attention"

    def _generate_alerts(
            self,
            system_analytics: Dict,
            performance_insights: Dict) -> List[str]:
        """Generate system alerts."""
        alerts = []

        success_rate = system_analytics.get("overall_success_rate", 0)
        if success_rate < 0.8:
            alerts.append(
                f"LOW SUCCESS RATE: {
                    success_rate:.1%} - investigate immediately")

        avg_time = system_analytics.get("average_execution_time", 0)
        if avg_time > 10.0:
            alerts.append(
                f"HIGH RESPONSE TIME: {
                    avg_time:.1f}s - performance issue detected")

        bottlenecks = performance_insights.get("performance_bottlenecks", {})
        if len(bottlenecks) > 5:
            alerts.append(
                f"MULTIPLE BOTTLENECKS: {
                    len(bottlenecks)} tools showing performance issues")

        return alerts

    def _identify_top_performers(self, system_analytics: Dict) -> List[str]:
        """Identify top performing tools."""
        most_reliable = system_analytics.get("most_reliable_tools", [])
        fastest = system_analytics.get("fastest_tools", [])

        performers = []
        if most_reliable:
            # most_reliable_tools is a List[str], not List[Dict]
            performers.append(f"Most Reliable: {most_reliable[0]}")
        if fastest:
            # fastest_tools is a List[str], not List[Dict]
            performers.append(f"Fastest: {fastest[0]}")

        return performers

    def _identify_improvement_areas(
            self, performance_insights: Dict) -> List[str]:
        """Identify areas for improvement."""
        areas = []

        bottlenecks = performance_insights.get("performance_bottlenecks", {})
        if bottlenecks:
            areas.append(
                f"Performance: {
                    len(bottlenecks)} tools need optimization")

        success_rate = performance_insights.get("overall_success_rate", 1.0)
        if success_rate < 0.9:
            areas.append(f"Reliability: Success rate at {success_rate:.1%}")

        return areas

    def _identify_fast_tools(self, entries: List) -> List[Dict[str, Any]]:
        """Identify fastest tools."""
        tool_times = defaultdict(list)
        for entry in entries:
            if entry.execution_time and entry.status == "success":
                tool_times[entry.tool_name].append(entry.execution_time)

        fast_tools = []
        for tool, times in tool_times.items():
            if len(times) >= 3:  # Minimum sample size
                avg_time = sum(times) / len(times)
                fast_tools.append({"tool": tool, "avg_time": avg_time})

        return sorted(fast_tools, key=lambda x: x["avg_time"])[:5]

    def _identify_reliable_tools(self, entries: List) -> List[Dict[str, Any]]:
        """Identify most reliable tools."""
        tool_stats = defaultdict(lambda: {"total": 0, "success": 0})

        for entry in entries:
            tool_stats[entry.tool_name]["total"] += 1
            if entry.status == "success":
                tool_stats[entry.tool_name]["success"] += 1

        reliable_tools = []
        for tool, stats in tool_stats.items():
            if stats["total"] >= 3:  # Minimum sample size
                success_rate = stats["success"] / stats["total"]
                reliable_tools.append({
                    "tool": tool,
                    "success_rate": success_rate,
                    "total_calls": stats["total"]
                })

        return sorted(
            reliable_tools,
            key=lambda x: x["success_rate"],
            reverse=True)[
            :5]


def create_analytics_dashboard(
        tool_tracker: Optional[ToolUsageLogger] = None,
        feedback_system: Optional[Any] = None) -> ToolUsageAnalyticsDashboard:
    """Factory function to create analytics dashboard."""
    return ToolUsageAnalyticsDashboard(tool_tracker, feedback_system)
