"""
Simplified Feedback System for Newsletter Generation

This module provides basic feedback logging functionality without complex
analysis or learning systems.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class SimpleFeedbackEntry:
    """Simple structure for feedback entries."""
    timestamp: str
    newsletter_topic: str
    user_rating: str  # 'approved', 'rejected', 'needs_revision'
    quality_score: float
    feedback_notes: str


class SimpleFeedbackLogger:
    """Simple feedback logging without complex analysis."""
    
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
            "feedback_entries": []
        }
        
        with open(self.feedback_file, 'w') as f:
            json.dump(initial_data, f, indent=2)
        
        logger.info(f"Initialized feedback file: {self.feedback_file}")
    
    def log_feedback(self, 
                    topic: str,
                    user_rating: str,
                    quality_score: float,
                    feedback_notes: str = "") -> str:
        """Log simple user feedback for a newsletter generation session."""
        
        feedback_entry = SimpleFeedbackEntry(
            timestamp=datetime.now().isoformat(),
            newsletter_topic=topic,
            user_rating=user_rating,
            quality_score=quality_score,
            feedback_notes=feedback_notes
        )
        
        try:
            # Load existing data
            with open(self.feedback_file, 'r') as f:
                data = json.load(f)
            
            # Add new entry
            data["feedback_entries"].append(asdict(feedback_entry))
            data["last_updated"] = datetime.now().isoformat()
            
            # Save updated data
            with open(self.feedback_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Logged feedback: {user_rating} for topic '{topic}'")
            return feedback_entry.timestamp
            
        except Exception as e:
            logger.error(f"Error logging feedback: {e}")
            return ""
    
    def get_feedback_history(self, limit: Optional[int] = None) -> List[SimpleFeedbackEntry]:
        """Retrieve feedback history."""
        try:
            with open(self.feedback_file, 'r') as f:
                data = json.load(f)
            
            entries = data.get("feedback_entries", [])
            if limit:
                entries = entries[-limit:]  # Get most recent entries
            
            return [SimpleFeedbackEntry(**entry) for entry in entries]
        
        except FileNotFoundError:
            logger.warning("Feedback file not found, returning empty history")
            return []
        except Exception as e:
            logger.error(f"Error loading feedback history: {e}")
            return []


# Legacy compatibility aliases
FeedbackLogger = SimpleFeedbackLogger
FeedbackEntry = SimpleFeedbackEntry


class FeedbackLearningSystem:
    """Legacy compatibility class for simple feedback operations."""
    
    def __init__(self):
        self.feedback_logger = SimpleFeedbackLogger()
    
    def log_feedback(self, topic: str, user_rating: str, quality_score: float, feedback_notes: str = "") -> str:
        """Legacy method for logging feedback."""
        return self.feedback_logger.log_feedback(topic, user_rating, quality_score, feedback_notes)
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get simple feedback summary."""
        history = self.feedback_logger.get_feedback_history()
        if not history:
            return {"total_entries": 0, "average_score": 0.0}
        
        total_score = sum(entry.quality_score for entry in history)
        return {
            "total_entries": len(history),
            "average_score": total_score / len(history),
            "recent_feedback": [entry.user_rating for entry in history[-5:]]
        }