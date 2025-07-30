"""
Quality Monitor

This module provides monitoring, tracking, and reporting capabilities for the
unified quality validation system.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict
from collections import defaultdict, Counter

from .base import QualityReport, QualityMetrics, QualityStatus, UnifiedQualitySystem

logger = logging.getLogger(__name__)


class QualityMonitor:
    """Monitor and track quality validation performance and trends."""
    
    def __init__(self, storage_path: str = "logs/quality_metrics.json"):
        self.storage_path = storage_path
        self.logger = logging.getLogger(__name__)
        self.metrics_history: List[Dict[str, Any]] = []
        self.performance_data: Dict[str, List[float]] = defaultdict(list)
        self.quality_trends: Dict[str, List[float]] = defaultdict(list)
        
        # Load existing metrics if available
        self._load_metrics_history()
    
    def record_validation(self, report: QualityReport, content_type: str = "newsletter",
                         pipeline_type: str = "daily", processing_time: float = 0.0):
        """Record a quality validation result."""
        timestamp = datetime.now()
        
        # Convert report to serializable format
        metrics_dict = asdict(report.metrics)
        metrics_dict['validation_timestamp'] = timestamp.isoformat()
        
        record = {
            'timestamp': timestamp.isoformat(),
            'content_type': content_type,
            'pipeline_type': pipeline_type,
            'status': report.status.value,
            'metrics': metrics_dict,
            'issues_count': len(report.issues),
            'warnings_count': len(report.warnings),
            'blocking_issues_count': len(report.blocking_issues),
            'overall_score': report.metrics.overall_score,
            'grade': report.get_grade(),
            'processing_time_seconds': processing_time,
            'is_publishable': report.is_publishable()
        }
        
        self.metrics_history.append(record)
        
        # Update performance data
        self.performance_data[f"{content_type}_{pipeline_type}"].append(processing_time)
        
        # Update quality trends
        self.quality_trends[f"{content_type}_{pipeline_type}"].append(report.metrics.overall_score)
        
        # Save to storage
        self._save_metrics_history()
        
        self.logger.info(f"Recorded quality validation: {report.status.value} - Score: {report.metrics.overall_score:.2f}")
    
    def get_performance_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Get performance summary for the specified time period."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Filter records by date
        recent_records = [
            record for record in self.metrics_history
            if datetime.fromisoformat(record['timestamp']) >= cutoff_date
        ]
        
        if not recent_records:
            return {
                'total_validations': 0,
                'average_score': 0.0,
                'success_rate': 0.0,
                'average_processing_time': 0.0,
                'status_distribution': {},
                'grade_distribution': {},
                'content_type_distribution': {},
                'pipeline_type_distribution': {}
            }
        
        # Calculate summary statistics
        total_validations = len(recent_records)
        average_score = sum(r['overall_score'] for r in recent_records) / total_validations
        success_rate = sum(1 for r in recent_records if r['is_publishable']) / total_validations
        average_processing_time = sum(r['processing_time_seconds'] for r in recent_records) / total_validations
        
        # Status distribution
        status_counts = Counter(r['status'] for r in recent_records)
        status_distribution = dict(status_counts)
        
        # Grade distribution
        grade_counts = Counter(r['grade'] for r in recent_records)
        grade_distribution = dict(grade_counts)
        
        # Content type distribution
        content_type_counts = Counter(r['content_type'] for r in recent_records)
        content_type_distribution = dict(content_type_counts)
        
        # Pipeline type distribution
        pipeline_type_counts = Counter(r['pipeline_type'] for r in recent_records)
        pipeline_type_distribution = dict(pipeline_type_counts)
        
        return {
            'total_validations': total_validations,
            'average_score': round(average_score, 2),
            'success_rate': round(success_rate, 3),
            'average_processing_time': round(average_processing_time, 3),
            'status_distribution': status_distribution,
            'grade_distribution': grade_distribution,
            'content_type_distribution': content_type_distribution,
            'pipeline_type_distribution': pipeline_type_distribution,
            'time_period_days': days_back
        }
    
    def get_quality_trends(self, days_back: int = 30) -> Dict[str, Any]:
        """Get quality trends over time."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Filter records by date
        recent_records = [
            record for record in self.metrics_history
            if datetime.fromisoformat(record['timestamp']) >= cutoff_date
        ]
        
        if not recent_records:
            return {
                'trends': {},
                'improvements': {},
                'regressions': {}
            }
        
        # Group by content type and pipeline type
        trends = {}
        for record in recent_records:
            key = f"{record['content_type']}_{record['pipeline_type']}"
            if key not in trends:
                trends[key] = []
            trends[key].append({
                'timestamp': record['timestamp'],
                'score': record['overall_score'],
                'status': record['status']
            })
        
        # Calculate trend analysis for each group
        trend_analysis = {}
        improvements = {}
        regressions = {}
        
        for key, records in trends.items():
            if len(records) < 2:
                continue
            
            # Sort by timestamp
            sorted_records = sorted(records, key=lambda x: x['timestamp'])
            
            # Calculate trend
            scores = [r['score'] for r in sorted_records]
            if len(scores) >= 2:
                trend = (scores[-1] - scores[0]) / len(scores)
                trend_analysis[key] = {
                    'trend': round(trend, 3),
                    'start_score': scores[0],
                    'end_score': scores[-1],
                    'improvement': scores[-1] - scores[0],
                    'total_records': len(records)
                }
                
                # Identify improvements and regressions
                if trend > 0.1:  # Significant improvement
                    improvements[key] = trend_analysis[key]
                elif trend < -0.1:  # Significant regression
                    regressions[key] = trend_analysis[key]
        
        return {
            'trends': trend_analysis,
            'improvements': improvements,
            'regressions': regressions
        }
    
    def get_quality_insights(self, days_back: int = 7) -> Dict[str, Any]:
        """Get insights and recommendations based on quality data."""
        summary = self.get_performance_summary(days_back)
        trends = self.get_quality_trends(days_back)
        
        insights = {
            'overall_performance': self._analyze_overall_performance(summary),
            'quality_trends': self._analyze_quality_trends(trends),
            'recommendations': self._generate_recommendations(summary, trends),
            'alerts': self._generate_alerts(summary, trends)
        }
        
        return insights
    
    def generate_quality_report(self, days_back: int = 7) -> str:
        """Generate a comprehensive quality report."""
        summary = self.get_performance_summary(days_back)
        trends = self.get_quality_trends(days_back)
        insights = self.get_quality_insights(days_back)
        
        report = f"""
# Quality Validation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Period: Last {days_back} days

## Performance Summary
- Total Validations: {summary['total_validations']}
- Average Score: {summary['average_score']}/10
- Success Rate: {summary['success_rate']:.1%}
- Average Processing Time: {summary['average_processing_time']:.3f}s

## Status Distribution
"""
        
        for status, count in summary['status_distribution'].items():
            percentage = (count / summary['total_validations']) * 100
            report += f"- {status.title()}: {count} ({percentage:.1f}%)\n"
        
        report += f"""
## Grade Distribution
"""
        
        for grade, count in summary['grade_distribution'].items():
            percentage = (count / summary['total_validations']) * 100
            report += f"- {grade}: {count} ({percentage:.1f}%)\n"
        
        report += f"""
## Quality Trends
"""
        
        for key, trend_data in trends['trends'].items():
            direction = "📈" if trend_data['trend'] > 0 else "📉" if trend_data['trend'] < 0 else "➡️"
            report += f"- {key}: {direction} {trend_data['trend']:.3f} (Score: {trend_data['start_score']:.1f} → {trend_data['end_score']:.1f})\n"
        
        report += f"""
## Insights
{insights['overall_performance']}

## Recommendations
"""
        
        for rec in insights['recommendations']:
            report += f"- {rec}\n"
        
        if insights['alerts']:
            report += f"""
## Alerts
"""
            for alert in insights['alerts']:
                report += f"- ⚠️ {alert}\n"
        
        return report
    
    def _analyze_overall_performance(self, summary: Dict[str, Any]) -> str:
        """Analyze overall performance and provide insights."""
        insights = []
        
        if summary['average_score'] >= 8.0:
            insights.append("Excellent overall quality performance")
        elif summary['average_score'] >= 7.0:
            insights.append("Good quality performance with room for improvement")
        elif summary['average_score'] >= 6.0:
            insights.append("Acceptable quality but needs attention")
        else:
            insights.append("Quality performance needs immediate attention")
        
        if summary['success_rate'] >= 0.9:
            insights.append("High success rate - most content is publishable")
        elif summary['success_rate'] >= 0.7:
            insights.append("Good success rate with some content needing revision")
        else:
            insights.append("Low success rate - many content items need significant improvement")
        
        if summary['average_processing_time'] <= 1.0:
            insights.append("Fast processing times")
        elif summary['average_processing_time'] <= 3.0:
            insights.append("Acceptable processing times")
        else:
            insights.append("Slow processing times - consider optimization")
        
        return " ".join(insights)
    
    def _analyze_quality_trends(self, trends: Dict[str, Any]) -> str:
        """Analyze quality trends and provide insights."""
        insights = []
        
        improvements = len(trends['improvements'])
        regressions = len(trends['regressions'])
        
        if improvements > regressions:
            insights.append(f"Positive trend: {improvements} improvements vs {regressions} regressions")
        elif regressions > improvements:
            insights.append(f"Concerning trend: {regressions} regressions vs {improvements} improvements")
        else:
            insights.append("Stable quality trends")
        
        if trends['trends']:
            best_improvement = max(trends['improvements'].values(), key=lambda x: x['improvement']) if trends['improvements'] else None
            worst_regression = min(trends['regressions'].values(), key=lambda x: x['improvement']) if trends['regressions'] else None
            
            if best_improvement:
                insights.append(f"Best improvement: {best_improvement['improvement']:.2f} points")
            if worst_regression:
                insights.append(f"Worst regression: {worst_regression['improvement']:.2f} points")
        
        return " ".join(insights)
    
    def _generate_recommendations(self, summary: Dict[str, Any], trends: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality data."""
        recommendations = []
        
        # Performance-based recommendations
        if summary['average_score'] < 7.0:
            recommendations.append("Focus on improving content quality standards")
        
        if summary['success_rate'] < 0.8:
            recommendations.append("Review and improve quality validation criteria")
        
        if summary['average_processing_time'] > 2.0:
            recommendations.append("Optimize validation processing performance")
        
        # Trend-based recommendations
        if trends['regressions']:
            recommendations.append("Investigate causes of quality regressions")
        
        if not trends['improvements']:
            recommendations.append("Implement quality improvement initiatives")
        
        # Status-based recommendations
        if summary['status_distribution'].get('failed', 0) > summary['total_validations'] * 0.1:
            recommendations.append("Address high failure rate in quality validation")
        
        return recommendations
    
    def _generate_alerts(self, summary: Dict[str, Any], trends: Dict[str, Any]) -> List[str]:
        """Generate alerts for concerning trends."""
        alerts = []
        
        # Performance alerts
        if summary['average_score'] < 6.0:
            alerts.append("Critical: Average quality score below acceptable threshold")
        
        if summary['success_rate'] < 0.5:
            alerts.append("Critical: Less than 50% of content is publishable")
        
        if summary['average_processing_time'] > 5.0:
            alerts.append("Warning: Processing times are significantly slow")
        
        # Trend alerts
        if len(trends['regressions']) > len(trends['improvements']):
            alerts.append("Warning: More quality regressions than improvements detected")
        
        # Status alerts
        failed_count = summary['status_distribution'].get('failed', 0)
        if failed_count > summary['total_validations'] * 0.2:
            alerts.append("Warning: High failure rate in quality validation")
        
        return alerts
    
    def _load_metrics_history(self):
        """Load metrics history from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                self.metrics_history = json.load(f)
            self.logger.info(f"Loaded {len(self.metrics_history)} quality metrics from storage")
        except (FileNotFoundError, json.JSONDecodeError):
            self.logger.info("No existing quality metrics found, starting fresh")
            self.metrics_history = []
    
    def _save_metrics_history(self):
        """Save metrics history to storage."""
        try:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            with open(self.storage_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save quality metrics: {e}")


class QualityDashboard:
    """Simple dashboard for quality monitoring."""
    
    def __init__(self, monitor: QualityMonitor):
        self.monitor = monitor
    
    def display_summary(self, days_back: int = 7):
        """Display a summary of quality metrics."""
        summary = self.monitor.get_performance_summary(days_back)
        
        print(f"\n{'='*50}")
        print(f"QUALITY VALIDATION SUMMARY (Last {days_back} days)")
        print(f"{'='*50}")
        print(f"Total Validations: {summary['total_validations']}")
        print(f"Average Score: {summary['average_score']}/10")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Average Processing Time: {summary['average_processing_time']:.3f}s")
        
        print(f"\nStatus Distribution:")
        for status, count in summary['status_distribution'].items():
            percentage = (count / summary['total_validations']) * 100 if summary['total_validations'] > 0 else 0
            print(f"  {status.title()}: {count} ({percentage:.1f}%)")
        
        print(f"\nGrade Distribution:")
        for grade, count in summary['grade_distribution'].items():
            percentage = (count / summary['total_validations']) * 100 if summary['total_validations'] > 0 else 0
            print(f"  {grade}: {count} ({percentage:.1f}%)")
    
    def display_trends(self, days_back: int = 30):
        """Display quality trends."""
        trends = self.monitor.get_quality_trends(days_back)
        
        print(f"\n{'='*50}")
        print(f"QUALITY TRENDS (Last {days_back} days)")
        print(f"{'='*50}")
        
        if not trends['trends']:
            print("No trend data available")
            return
        
        for key, trend_data in trends['trends'].items():
            direction = "📈" if trend_data['trend'] > 0 else "📉" if trend_data['trend'] < 0 else "➡️"
            print(f"{key}: {direction} {trend_data['trend']:.3f} (Score: {trend_data['start_score']:.1f} → {trend_data['end_score']:.1f})")
        
        if trends['improvements']:
            print(f"\nImprovements:")
            for key, data in trends['improvements'].items():
                print(f"  {key}: +{data['improvement']:.2f} points")
        
        if trends['regressions']:
            print(f"\nRegressions:")
            for key, data in trends['regressions'].items():
                print(f"  {key}: {data['improvement']:.2f} points")
    
    def display_insights(self, days_back: int = 7):
        """Display quality insights and recommendations."""
        insights = self.monitor.get_quality_insights(days_back)
        
        print(f"\n{'='*50}")
        print(f"QUALITY INSIGHTS (Last {days_back} days)")
        print(f"{'='*50}")
        
        print(f"\nPerformance Analysis:")
        print(f"  {insights['overall_performance']}")
        
        print(f"\nTrend Analysis:")
        print(f"  {insights['quality_trends']}")
        
        if insights['recommendations']:
            print(f"\nRecommendations:")
            for rec in insights['recommendations']:
                print(f"  • {rec}")
        
        if insights['alerts']:
            print(f"\nAlerts:")
            for alert in insights['alerts']:
                print(f"  ⚠️ {alert}")
    
    def generate_full_report(self, days_back: int = 7):
        """Generate and display a full quality report."""
        report = self.monitor.generate_quality_report(days_back)
        print(report)