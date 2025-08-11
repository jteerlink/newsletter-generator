"""
Learning System: Performance tracking and continuous improvement.

This module defines the learning system that tracks performance metrics,
analyzes patterns, and generates improvement recommendations for the
enhanced agent architecture.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .campaign_context import CampaignContext


@dataclass
class PerformanceMetrics:
    """Performance metrics for tracking agent and content performance."""
    execution_time: float
    quality_score: float
    revision_cycles: int
    user_satisfaction: Optional[float] = None
    engagement_metrics: Optional[Dict[str, Any]] = None
    error_count: int = 0
    success_rate: float = 1.0
    timestamp: float = None

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()


class PerformanceTracker:
    """Tracks performance metrics over time."""

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.performance_trends: Dict[str, List[float]] = {}

    def add_metrics(self, metrics: PerformanceMetrics) -> None:
        """Add performance metrics to history."""
        self.metrics_history.append(metrics)
        self._update_trends(metrics)

    def _update_trends(self, metrics: PerformanceMetrics) -> None:
        """Update performance trends with new metrics."""
        trends = ['execution_time', 'quality_score', 'revision_cycles']

        for trend in trends:
            if trend not in self.performance_trends:
                self.performance_trends[trend] = []

            value = getattr(metrics, trend)
            if isinstance(value, (int, float)):
                self.performance_trends[trend].append(value)

    def get_recent_metrics(self, limit: int = 10) -> List[PerformanceMetrics]:
        """Get recent performance metrics."""
        return self.metrics_history[-limit:] if self.metrics_history else []

    def get_average_metrics(self, window: int = 10) -> Dict[str, float]:
        """Calculate average metrics over a window."""
        recent_metrics = self.get_recent_metrics(window)
        if not recent_metrics:
            return {}

        averages = {}
        for trend in ['execution_time', 'quality_score', 'revision_cycles']:
            values = [getattr(m, trend)
                      for m in recent_metrics if hasattr(m, trend)]
            if values:
                averages[trend] = sum(values) / len(values)

        return averages

    def get_trend_analysis(
            self, metric: str, window: int = 10) -> Dict[str, Any]:
        """Analyze trends for a specific metric."""
        if metric not in self.performance_trends:
            return {'trend': 'stable', 'change': 0.0}

        values = self.performance_trends[metric][-window:]
        if len(values) < 2:
            return {'trend': 'stable', 'change': 0.0}

        # Calculate trend
        first_half = values[:len(values) // 2]
        second_half = values[len(values) // 2:]

        if not first_half or not second_half:
            return {'trend': 'stable', 'change': 0.0}

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        change = second_avg - first_avg
        change_percent = (change / first_avg * 100) if first_avg != 0 else 0

        if change_percent > 5:
            trend = 'improving'
        elif change_percent < -5:
            trend = 'declining'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'change': change,
            'change_percent': change_percent,
            'current_value': values[-1] if values else 0,
            'average_value': sum(values) / len(values)
        }


class ImprovementAnalyzer:
    """Analyzes performance data to generate improvement recommendations."""

    def __init__(self):
        self.recommendation_patterns = {
            'low_quality': 'Focus on content quality improvement',
            'high_revisions': 'Optimize initial content generation',
            'slow_execution': 'Improve agent efficiency',
            'user_dissatisfaction': 'Enhance user experience',
            'high_errors': 'Strengthen error handling'
        }

    def analyze(self, learning_data: Dict[str, Any]) -> List[str]:
        """Analyze learning data and generate recommendations."""
        recommendations = []

        # Analyze performance trends
        if 'performance_trends' in learning_data:
            trends = learning_data['performance_trends']
            recommendations.extend(self._analyze_performance_trends(trends))

        # Analyze successful patterns
        if 'successful_patterns' in learning_data:
            patterns = learning_data['successful_patterns']
            recommendations.extend(self._analyze_successful_patterns(patterns))

        # Analyze failed patterns
        if 'failed_patterns' in learning_data:
            patterns = learning_data['failed_patterns']
            recommendations.extend(self._analyze_failed_patterns(patterns))

        # Analyze audience feedback
        if 'audience_feedback' in learning_data:
            feedback = learning_data['audience_feedback']
            recommendations.extend(self._analyze_audience_feedback(feedback))

        return recommendations

    def _analyze_performance_trends(self, trends: Dict[str, Any]) -> List[str]:
        """Analyze performance trends and generate recommendations."""
        recommendations = []

        # Check for declining quality scores
        if 'quality_score' in trends:
            quality_trend = trends['quality_score']
            if isinstance(quality_trend, list) and len(quality_trend) >= 2:
                recent_avg = sum(quality_trend[-5:]) / len(quality_trend[-5:])
                if recent_avg < 0.7:
                    recommendations.append(
                        "Quality scores are below target. Consider enhancing content validation.")

        # Check for high revision cycles
        if 'revision_cycles' in trends:
            revision_trend = trends['revision_cycles']
            if isinstance(revision_trend, list) and len(revision_trend) >= 2:
                recent_avg = sum(
                    revision_trend[-5:]) / len(revision_trend[-5:])
                if recent_avg > 2:
                    recommendations.append(
                        "High revision cycles detected. Optimize initial content generation.")

        # Check for slow execution times
        if 'execution_time' in trends:
            time_trend = trends['execution_time']
            if isinstance(time_trend, list) and len(time_trend) >= 2:
                recent_avg = sum(time_trend[-5:]) / len(time_trend[-5:])
                if recent_avg > 300:  # 5 minutes
                    recommendations.append(
                        "Slow execution times detected. Consider optimizing agent workflows.")

        return recommendations

    def _analyze_successful_patterns(
            self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Analyze successful patterns and generate recommendations."""
        recommendations = []

        if not patterns:
            return recommendations

        # Identify common characteristics of successful content
        successful_themes = []
        successful_styles = []

        for pattern in patterns:
            if 'theme' in pattern:
                successful_themes.append(pattern['theme'])
            if 'style' in pattern:
                successful_styles.append(pattern['style'])

        if successful_themes:
            most_common_theme = max(
                set(successful_themes),
                key=successful_themes.count)
            recommendations.append(
                f"Focus on {most_common_theme} themes as they show high success rates.")

        if successful_styles:
            most_common_style = max(
                set(successful_styles),
                key=successful_styles.count)
            recommendations.append(
                f"Emphasize {most_common_style} writing style for better engagement.")

        return recommendations

    def _analyze_failed_patterns(
            self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Analyze failed patterns and generate recommendations."""
        recommendations = []

        if not patterns:
            return recommendations

        # Identify common characteristics of failed content
        failed_themes = []
        failed_styles = []
        common_errors = []

        for pattern in patterns:
            if 'theme' in pattern:
                failed_themes.append(pattern['theme'])
            if 'style' in pattern:
                failed_styles.append(pattern['style'])
            if 'error' in pattern:
                common_errors.append(pattern['error'])

        if failed_themes:
            most_common_failed_theme = max(
                set(failed_themes), key=failed_themes.count)
            recommendations.append(f"Avoid or improve approach to {
                                   most_common_failed_theme} themes.")

        if failed_styles:
            most_common_failed_style = max(
                set(failed_styles), key=failed_styles.count)
            recommendations.append(
                f"Reconsider {most_common_failed_style} writing style approach.")

        if common_errors:
            most_common_error = max(
                set(common_errors),
                key=common_errors.count)
            recommendations.append(
                f"Address recurring issue: {most_common_error}")

        return recommendations

    def _analyze_audience_feedback(
            self, feedback: List[Dict[str, Any]]) -> List[str]:
        """Analyze audience feedback and generate recommendations."""
        recommendations = []

        if not feedback:
            return recommendations

        # Analyze feedback themes
        feedback_themes = []
        sentiment_scores = []

        for item in feedback:
            if 'theme' in item:
                feedback_themes.append(item['theme'])
            if 'sentiment' in item:
                sentiment_scores.append(item['sentiment'])

        if feedback_themes:
            most_common_theme = max(
                set(feedback_themes),
                key=feedback_themes.count)
            recommendations.append(
                f"Address audience concerns about {most_common_theme}.")

        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            if avg_sentiment < 0.5:
                recommendations.append(
                    "Overall audience sentiment is low. Consider content strategy adjustments.")

        return recommendations


class LearningSystem:
    """Main learning system that coordinates performance tracking and improvement analysis."""

    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.improvement_analyzer = ImprovementAnalyzer()
        self.learning_data: Dict[str, Any] = {
            'successful_patterns': [],
            'failed_patterns': [],
            'audience_feedback': [],
            'performance_trends': {}
        }

    def update_campaign_context(self,
                                context: CampaignContext,
                                performance_data: Dict[str,
                                                       Any]) -> None:
        """Update campaign context with learning data."""
        # Create performance metrics
        metrics = PerformanceMetrics(
            execution_time=performance_data.get('execution_time', 0.0),
            quality_score=performance_data.get('quality_score', 0.0),
            revision_cycles=performance_data.get('revision_cycles', 0),
            user_satisfaction=performance_data.get('user_satisfaction'),
            engagement_metrics=performance_data.get('engagement_metrics'),
            error_count=performance_data.get('error_count', 0),
            success_rate=performance_data.get('success_rate', 1.0)
        )

        # Add to performance tracker
        self.performance_tracker.add_metrics(metrics)

        # Update learning data
        self._update_learning_data(performance_data)

        # Update campaign context
        context.learning_data.update(self.learning_data)
        context.updated_at = time.time()

    def _update_learning_data(self, performance_data: Dict[str, Any]) -> None:
        """Update learning data with new performance information."""
        # Update performance trends
        trends = self.performance_tracker.performance_trends
        self.learning_data['performance_trends'] = trends

        # Update patterns based on performance
        if performance_data.get('quality_score', 0) >= 0.8:
            # Successful pattern
            pattern = {
                'timestamp': time.time(),
                'quality_score': performance_data.get('quality_score'),
                'theme': performance_data.get('theme', 'unknown'),
                'style': performance_data.get('style', 'unknown'),
                'execution_time': performance_data.get('execution_time')
            }
            self.learning_data['successful_patterns'].append(pattern)
        elif performance_data.get('quality_score', 0) < 0.6:
            # Failed pattern
            pattern = {
                'timestamp': time.time(),
                'quality_score': performance_data.get('quality_score'),
                'theme': performance_data.get('theme', 'unknown'),
                'style': performance_data.get('style', 'unknown'),
                'error': performance_data.get('error_message', 'unknown'),
                'execution_time': performance_data.get('execution_time')
            }
            self.learning_data['failed_patterns'].append(pattern)

        # Update audience feedback if available
        if 'audience_feedback' in performance_data:
            feedback = performance_data['audience_feedback']
            if isinstance(feedback, dict):
                feedback['timestamp'] = time.time()
                self.learning_data['audience_feedback'].append(feedback)

    def generate_improvement_recommendations(
            self, context: CampaignContext) -> List[str]:
        """Generate improvement recommendations based on learning data."""
        return self.improvement_analyzer.analyze(context.learning_data)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of current performance metrics."""
        recent_metrics = self.performance_tracker.get_recent_metrics(10)
        if not recent_metrics:
            return {'status': 'no_data'}

        avg_metrics = self.performance_tracker.get_average_metrics(10)
        quality_trend = self.performance_tracker.get_trend_analysis(
            'quality_score')

        return {
            'status': 'active',
            'average_metrics': avg_metrics,
            'quality_trend': quality_trend,
            'total_sessions': len(self.performance_tracker.metrics_history),
            'recent_performance': {
                'avg_quality': avg_metrics.get('quality_score', 0.0),
                'avg_execution_time': avg_metrics.get('execution_time', 0.0),
                'avg_revisions': avg_metrics.get('revision_cycles', 0)
            }
        }

    def export_learning_data(self) -> Dict[str, Any]:
        """Export learning data for external analysis."""
        return {
            'learning_data': self.learning_data,
            'performance_history': [
                {
                    'execution_time': m.execution_time,
                    'quality_score': m.quality_score,
                    'revision_cycles': m.revision_cycles,
                    'timestamp': m.timestamp
                }
                for m in self.performance_tracker.metrics_history
            ],
            'performance_trends': self.performance_tracker.performance_trends
        }
