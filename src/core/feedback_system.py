"""
Feedback System for Newsletter Generation

This module handles user feedback collection, analysis, and learning to improve
agent performance over time.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
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
    """Coordinates the complete feedback learning system."""
    
    def __init__(self, feedback_file: str = "logs/feedback_history.json"):
        self.logger = FeedbackLogger(feedback_file)
        self.analyzer = FeedbackAnalyzer(self.logger)
    
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
        
        insights = {
            "rejection_analysis": rejection_analysis,
            "improvement_recommendations": recommendations,
            "performance_trends": trends,
            "summary": {
                "total_feedback_entries": len(self.logger.get_feedback_history()),
                "key_focus_areas": recommendations[:3],  # Top 3 recommendations
                "learning_status": "active" if recommendations else "stable"
            }
        }
        
        return insights
    
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
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Learning report saved to: {output_path}")
        return output_path 