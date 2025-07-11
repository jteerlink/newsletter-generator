"""
Feedback System for Newsletter Generation

This module handles user feedback collection, analysis, and learning to improve
agent performance over time. Now includes tool usage correlation analysis.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class FeedbackEntry:
    """Structure for individual feedback entries."""
    timestamp: str
    newsletter_topic: str
    content_preview: str  # First 200 chars
    user_rating: str  # 'approved', 'rejected', 'needs_revision'
    quality_scores: Dict[str, float]
    specific_feedback: str
    agent_performance: Dict[str, Any]
    suggestions: List[str]
    metadata: Dict[str, Any]

class FeedbackLogger:
    """Handles logging and storage of user feedback."""
    
    def __init__(self, feedback_file: str = "logs/feedback_history.json"):
        self.feedback_file = Path(feedback_file)
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file if it doesn't exist
        if not self.feedback_file.exists():
            self._initialize_feedback_file()
    
    def _initialize_feedback_file(self):
        """Initialize the feedback file with empty structure."""
        initial_data = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "total_entries": 0,
            "feedback_entries": []
        }
        
        with open(self.feedback_file, 'w') as f:
            json.dump(initial_data, f, indent=2)
        
        logger.info(f"Initialized feedback file: {self.feedback_file}")
    
    def log_feedback(self, 
                    topic: str,
                    content: str, 
                    user_rating: str,
                    quality_scores: Dict[str, float],
                    specific_feedback: str = "",
                    agent_performance: Dict[str, Any] = None,
                    suggestions: List[str] = None) -> str:
        """Log user feedback for a newsletter generation session."""
        
        feedback_entry = FeedbackEntry(
            timestamp=datetime.now().isoformat(),
            newsletter_topic=topic,
            content_preview=content[:200] + "..." if len(content) > 200 else content,
            user_rating=user_rating,
            quality_scores=quality_scores or {},
            specific_feedback=specific_feedback,
            agent_performance=agent_performance or {},
            suggestions=suggestions or [],
            metadata={
                "content_length": len(content),
                "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
        )
        
        # Load existing data
        with open(self.feedback_file, 'r') as f:
            data = json.load(f)
        
        # Add new entry
        data["feedback_entries"].append(asdict(feedback_entry))
        data["total_entries"] += 1
        data["last_updated"] = datetime.now().isoformat()
        
        # Save updated data
        with open(self.feedback_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Logged feedback: {user_rating} for topic '{topic}'")
        return feedback_entry.metadata["session_id"]
    
    def get_feedback_history(self, limit: Optional[int] = None) -> List[FeedbackEntry]:
        """Retrieve feedback history."""
        try:
            with open(self.feedback_file, 'r') as f:
                data = json.load(f)
            
            entries = data.get("feedback_entries", [])
            if limit:
                entries = entries[-limit:]  # Get most recent entries
            
            return [FeedbackEntry(**entry) for entry in entries]
        
        except FileNotFoundError:
            logger.warning("Feedback file not found, returning empty history")
            return []
        except Exception as e:
            logger.error(f"Error loading feedback history: {e}")
            return []

class FeedbackAnalyzer:
    """Analyzes feedback patterns and generates improvement insights."""
    
    def __init__(self, feedback_logger: FeedbackLogger):
        self.feedback_logger = feedback_logger
    
    def analyze_rejection_patterns(self, days: int = 30) -> Dict[str, Any]:
        """Analyze patterns in rejected newsletters to identify common issues."""
        feedback_history = self.feedback_logger.get_feedback_history()
        
        # Filter rejected entries
        rejected_entries = [
            entry for entry in feedback_history 
            if entry.user_rating == 'rejected'
        ]
        
        if not rejected_entries:
            return {"message": "No rejected entries found for analysis"}
        
        analysis = {
            "total_rejected": len(rejected_entries),
            "rejection_rate": len(rejected_entries) / len(feedback_history) if feedback_history else 0,
            "common_issues": defaultdict(int),
            "quality_score_patterns": defaultdict(list),
            "topic_analysis": defaultdict(int),
            "agent_performance_issues": defaultdict(int)
        }
        
        # Analyze quality scores and issues
        for entry in rejected_entries:
            # Quality score analysis
            for dimension, score in entry.quality_scores.items():
                analysis["quality_score_patterns"][dimension].append(score)
            
            # Topic analysis
            analysis["topic_analysis"][entry.newsletter_topic[:50]] += 1
            
            # Agent performance issues
            for agent, perf_data in entry.agent_performance.items():
                if isinstance(perf_data, dict) and 'issues' in perf_data:
                    for issue in perf_data['issues']:
                        analysis["agent_performance_issues"][f"{agent}:{issue}"] += 1
        
        # Calculate averages for quality scores
        avg_quality_scores = {}
        for dimension, scores in analysis["quality_score_patterns"].items():
            if scores:
                avg_quality_scores[dimension] = {
                    "average": sum(scores) / len(scores),
                    "lowest": min(scores),
                    "count": len(scores)
                }
        
        analysis["average_quality_scores"] = avg_quality_scores
        
        return dict(analysis)
    
    def generate_improvement_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on feedback analysis."""
        recommendations = []
        
        if not analysis or analysis.get("total_rejected", 0) == 0:
            recommendations.append("No rejection patterns identified. Continue monitoring feedback.")
            return recommendations
        
        # Quality score recommendations
        avg_scores = analysis.get("average_quality_scores", {})
        for dimension, score_data in avg_scores.items():
            if score_data["average"] < 6.0:
                if dimension == "clarity":
                    recommendations.append(f"Focus on improving content clarity - current average: {score_data['average']:.1f}/10")
                elif dimension == "accuracy":
                    recommendations.append(f"Enhance fact-checking processes - current average: {score_data['average']:.1f}/10")
                elif dimension == "engagement":
                    recommendations.append(f"Improve content engagement strategies - current average: {score_data['average']:.1f}/10")
                elif dimension == "completeness":
                    recommendations.append(f"Ensure more comprehensive coverage - current average: {score_data['average']:.1f}/10")
        
        # Agent performance recommendations  
        agent_issues = analysis.get("agent_performance_issues", {})
        if agent_issues:
            top_issues = sorted(agent_issues.items(), key=lambda x: x[1], reverse=True)[:3]
            for issue, count in top_issues:
                agent_name, issue_desc = issue.split(':', 1)
                recommendations.append(f"Address {agent_name} performance: {issue_desc} (occurred {count} times)")
        
        # High rejection rate warning
        rejection_rate = analysis.get("rejection_rate", 0)
        if rejection_rate > 0.3:  # More than 30% rejection rate
            recommendations.append(f"High rejection rate ({rejection_rate:.1%}) - consider comprehensive agent prompt revision")
        
        return recommendations
    
    def get_performance_trends(self, agent_name: str = None) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        feedback_history = self.feedback_logger.get_feedback_history()
        
        if not feedback_history:
            return {"message": "No feedback history available"}
        
        # Group by month for trend analysis
        monthly_performance = defaultdict(lambda: {
            "total": 0, "approved": 0, "rejected": 0, 
            "quality_scores": defaultdict(list)
        })
        
        for entry in feedback_history:
            month_key = entry.timestamp[:7]  # YYYY-MM format
            monthly_performance[month_key]["total"] += 1
            
            if entry.user_rating == "approved":
                monthly_performance[month_key]["approved"] += 1
            elif entry.user_rating == "rejected":
                monthly_performance[month_key]["rejected"] += 1
            
            # Aggregate quality scores
            for dimension, score in entry.quality_scores.items():
                monthly_performance[month_key]["quality_scores"][dimension].append(score)
        
        # Calculate trends
        trends = {}
        for month, data in monthly_performance.items():
            approval_rate = data["approved"] / data["total"] if data["total"] > 0 else 0
            avg_quality = {}
            
            for dimension, scores in data["quality_scores"].items():
                if scores:
                    avg_quality[dimension] = sum(scores) / len(scores)
            
            trends[month] = {
                "approval_rate": approval_rate,
                "total_newsletters": data["total"],
                "average_quality_scores": avg_quality
            }
        
        return trends

class FeedbackLearningSystem:
    """Coordinates the complete feedback learning system with tool usage analysis."""
    
    def __init__(self, feedback_file: str = "logs/feedback_history.json"):
        self.logger = FeedbackLogger(feedback_file)
        self.analyzer = FeedbackAnalyzer(self.logger)
        
        # Tool usage integration - will be set by system components
        self.tool_tracker = None
        
    def set_tool_tracker(self, tool_tracker):
        """Set the tool tracker instance for correlation analysis."""
        self.tool_tracker = tool_tracker
    
    def collect_user_feedback(self, topic: str, content: str, interactive: bool = True) -> str:
        """Collect feedback from user with interactive prompts."""
        
        if interactive:
            print("\n" + "="*60)
            print("📊 NEWSLETTER QUALITY REVIEW")
            print("="*60)
            print(f"Topic: {topic}")
            print(f"Content Length: {len(content)} characters")
            print("\nContent Preview:")
            print("-" * 40)
            print(content[:300] + "..." if len(content) > 300 else content)
            print("-" * 40)
            
            # Get user rating
            while True:
                rating = input("\n✅ Overall Rating (approved/rejected/needs_revision): ").strip().lower()
                if rating in ['approved', 'rejected', 'needs_revision']:
                    break
                print("Please enter 'approved', 'rejected', or 'needs_revision'")
            
            # Get quality scores
            quality_scores = {}
            dimensions = ['clarity', 'accuracy', 'engagement', 'completeness']
            
            print("\n📋 Quality Scores (1-10):")
            for dimension in dimensions:
                while True:
                    try:
                        score = float(input(f"  {dimension.title()}: "))
                        if 1 <= score <= 10:
                            quality_scores[dimension] = score
                            break
                        else:
                            print("Score must be between 1 and 10")
                    except ValueError:
                        print("Please enter a valid number")
            
            # Get specific feedback
            specific_feedback = input("\n💬 Specific feedback (optional): ").strip()
            
            # Get suggestions
            suggestions_input = input("\n💡 Suggestions for improvement (optional): ").strip()
            suggestions = [s.strip() for s in suggestions_input.split(',') if s.strip()] if suggestions_input else []
        
        else:
            # Non-interactive mode - use defaults
            rating = "approved"
            quality_scores = {"clarity": 7.0, "accuracy": 7.0, "engagement": 7.0, "completeness": 7.0}
            specific_feedback = "Automated feedback collection"
            suggestions = []
        
        # Log the feedback
        session_id = self.logger.log_feedback(
            topic=topic,
            content=content,
            user_rating=rating,
            quality_scores=quality_scores,
            specific_feedback=specific_feedback,
            suggestions=suggestions
        )
        
        if interactive:
            print(f"\n✅ Feedback logged successfully (Session: {session_id})")
        
        return session_id
    
    def generate_learning_insights(self) -> Dict[str, Any]:
        """Generate comprehensive learning insights from all feedback."""
        rejection_analysis = self.analyzer.analyze_rejection_patterns()
        recommendations = self.analyzer.generate_improvement_recommendations(rejection_analysis)
        trends = self.analyzer.get_performance_trends()
        
        # Add tool usage correlation analysis if available
        tool_correlations = {}
        if self.tool_tracker:
            tool_correlations = self.analyze_tool_usage_correlations()
        
        insights = {
            "rejection_analysis": rejection_analysis,
            "improvement_recommendations": recommendations,
            "performance_trends": trends,
            "tool_usage_correlations": tool_correlations,
            "summary": {
                "total_feedback_entries": len(self.logger.get_feedback_history()),
                "key_focus_areas": recommendations[:3],  # Top 3 recommendations
                "learning_status": "active" if recommendations else "stable",
                "tool_insights_available": bool(self.tool_tracker)
            }
        }
        
        return insights
    
    def analyze_tool_usage_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between tool usage patterns and feedback quality."""
        if not self.tool_tracker:
            return {"error": "Tool tracker not available"}
        
        feedback_history = self.logger.get_feedback_history()
        if not feedback_history:
            return {"error": "No feedback history available"}
        
        # Analyze correlations between tool usage and quality scores
        tool_quality_correlations = defaultdict(lambda: {"quality_scores": [], "usage_counts": [], "sessions": []})
        
        for feedback in feedback_history[-50:]:  # Analyze last 50 entries
            session_id = feedback.metadata.get("session_id")
            if not session_id:
                continue
            
            # Get tool usage for this session timeframe
            feedback_time = datetime.fromisoformat(feedback.timestamp)
            tool_entries = self.tool_tracker.get_tool_usage_history(
                hours_back=24,  # Look within 24 hours of feedback
                session_id=session_id
            )
            
            if not tool_entries:
                continue
                
            # Count tool usage by type
            tool_usage_counts = defaultdict(int)
            for entry in tool_entries:
                tool_usage_counts[entry.tool_name] += 1
            
            # Average quality score for this session
            avg_quality = sum(feedback.quality_scores.values()) / len(feedback.quality_scores) if feedback.quality_scores else 0
            
            # Store correlations
            for tool_name, count in tool_usage_counts.items():
                tool_quality_correlations[tool_name]["quality_scores"].append(avg_quality)
                tool_quality_correlations[tool_name]["usage_counts"].append(count)
                tool_quality_correlations[tool_name]["sessions"].append(session_id)
        
        # Calculate correlation insights
        correlation_insights = {}
        for tool_name, data in tool_quality_correlations.items():
            if len(data["quality_scores"]) < 3:  # Need at least 3 data points
                continue
                
            avg_quality = sum(data["quality_scores"]) / len(data["quality_scores"])
            avg_usage = sum(data["usage_counts"]) / len(data["usage_counts"])
            
            # Simple correlation analysis
            high_usage_sessions = [i for i, count in enumerate(data["usage_counts"]) if count > avg_usage]
            high_usage_quality = [data["quality_scores"][i] for i in high_usage_sessions]
            
            low_usage_sessions = [i for i, count in enumerate(data["usage_counts"]) if count <= avg_usage]
            low_usage_quality = [data["quality_scores"][i] for i in low_usage_sessions]
            
            high_avg = sum(high_usage_quality) / len(high_usage_quality) if high_usage_quality else 0
            low_avg = sum(low_usage_quality) / len(low_usage_quality) if low_usage_quality else 0
            
            correlation_insights[tool_name] = {
                "average_quality_with_tool": avg_quality,
                "average_usage_per_session": avg_usage,
                "high_usage_avg_quality": high_avg,
                "low_usage_avg_quality": low_avg,
                "quality_difference": high_avg - low_avg,
                "total_sessions": len(data["sessions"]),
                "recommendation": self._generate_tool_recommendation(tool_name, high_avg, low_avg, avg_usage)
            }
        
        return {
            "tool_correlations": correlation_insights,
            "analysis_summary": self._summarize_tool_correlations(correlation_insights),
            "analysis_metadata": {
                "feedback_entries_analyzed": len([f for f in feedback_history[-50:] if f.metadata.get("session_id")]),
                "tools_analyzed": len(correlation_insights)
            }
        }
    
    def _generate_tool_recommendation(self, tool_name: str, high_usage_quality: float, low_usage_quality: float, avg_usage: float) -> str:
        """Generate recommendations based on tool usage correlation."""
        quality_difference = high_usage_quality - low_usage_quality
        
        if quality_difference > 1.0:  # Significant positive correlation
            if avg_usage < 2:
                return f"Increase usage of {tool_name} - shows strong positive correlation with quality (+{quality_difference:.1f})"
            else:
                return f"Continue current usage pattern for {tool_name} - showing positive results"
        elif quality_difference < -1.0:  # Negative correlation
            return f"Review usage patterns for {tool_name} - may be overused or ineffective (-{abs(quality_difference):.1f})"
        else:
            return f"Neutral impact for {tool_name} - maintain current usage level"
    
    def _summarize_tool_correlations(self, correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize overall tool correlation patterns."""
        if not correlations:
            return {"message": "No tool correlations available"}
        
        # Find best and worst performing tools
        best_tools = sorted(
            correlations.items(), 
            key=lambda x: x[1]["quality_difference"], 
            reverse=True
        )[:3]
        
        worst_tools = sorted(
            correlations.items(), 
            key=lambda x: x[1]["quality_difference"]
        )[:3]
        
        # Overall insights
        total_tools = len(correlations)
        positive_correlations = len([t for t in correlations.values() if t["quality_difference"] > 0.5])
        
        return {
            "total_tools_analyzed": total_tools,
            "tools_with_positive_correlation": positive_correlations,
            "positive_correlation_rate": positive_correlations / total_tools if total_tools > 0 else 0,
            "best_performing_tools": [{"tool": name, "quality_impact": data["quality_difference"]} for name, data in best_tools],
            "underperforming_tools": [{"tool": name, "quality_impact": data["quality_difference"]} for name, data in worst_tools],
            "recommendations": self._generate_overall_tool_recommendations(correlations)
        }
    
    def _generate_overall_tool_recommendations(self, correlations: Dict[str, Any]) -> List[str]:
        """Generate system-wide tool usage recommendations."""
        recommendations = []
        
        if not correlations:
            return ["No tool correlation data available for recommendations"]
        
        # Identify high-impact tools
        high_impact_tools = [
            name for name, data in correlations.items() 
            if data["quality_difference"] > 1.0
        ]
        
        if high_impact_tools:
            recommendations.append(f"Prioritize usage of high-impact tools: {', '.join(high_impact_tools[:3])}")
        
        # Identify problematic tools
        problematic_tools = [
            name for name, data in correlations.items() 
            if data["quality_difference"] < -1.0
        ]
        
        if problematic_tools:
            recommendations.append(f"Review and optimize usage patterns for: {', '.join(problematic_tools[:3])}")
        
        # Usage efficiency recommendations
        overused_tools = [
            name for name, data in correlations.items() 
            if data["average_usage_per_session"] > 5 and data["quality_difference"] < 0
        ]
        
        if overused_tools:
            recommendations.append(f"Consider reducing usage frequency for: {', '.join(overused_tools[:2])}")
        
        return recommendations if recommendations else ["Current tool usage patterns appear optimal"]
    
    def get_tool_effectiveness_report(self) -> Dict[str, Any]:
        """Generate a detailed tool effectiveness report."""
        if not self.tool_tracker:
            return {"error": "Tool tracker not available"}
        
        # Get overall tool analytics
        tool_analytics = self.tool_tracker.generate_usage_analytics(hours_back=24*7)  # Last week
        
        # Get correlation analysis
        correlations = self.analyze_tool_usage_correlations()
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "tool_usage_analytics": tool_analytics,
            "quality_correlations": correlations,
            "effectiveness_insights": {
                "most_used_tools": tool_analytics.get("most_used_tools", [])[:5],
                "fastest_tools": tool_analytics.get("fastest_tools", [])[:5],
                "most_reliable_tools": tool_analytics.get("most_reliable_tools", [])[:5],
                "quality_correlated_tools": correlations.get("analysis_summary", {}).get("best_performing_tools", [])[:3]
            }
        }

    def save_learning_report(self, output_file: str = "logs/learning_report.json"):
        """Generate and save a comprehensive learning report."""
        report = {
            "report_generated": datetime.now().isoformat(),
            "learning_insights": self.generate_learning_insights(),
            "feedback_statistics": {
                "total_entries": len(self.logger.get_feedback_history()),
                "recent_entries": len(self.logger.get_feedback_history(limit=10))
            }
        }
        
        # Add tool effectiveness report if available
        if self.tool_tracker:
            report["tool_effectiveness"] = self.get_tool_effectiveness_report()
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Learning report saved to: {output_path}")
        return output_path 