"""
CampaignContext: Long-term campaign configuration and learning data.

This module defines the CampaignContext dataclass that stores persistent
brand voice, strategic goals, audience persona, and learning data for
the enhanced agent architecture.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CampaignContext:
    """Long-term campaign configuration and learning data."""
    content_style: Dict[str, Any]
    strategic_goals: List[str]
    audience_persona: Dict[str, Any]
    performance_analytics: Dict[str, Any]
    quality_thresholds: Dict[str, float]
    forbidden_terminology: List[str]
    preferred_terminology: List[str]
    learning_data: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def update_learning_data(self, new_data: Dict[str, Any]) -> None:
        """Update learning data and timestamp."""
        self.learning_data.update(new_data)
        self.updated_at = time.time()

    def update_performance_analytics(self, analytics: Dict[str, Any]) -> None:
        """Update performance analytics and timestamp."""
        self.performance_analytics.update(analytics)
        self.updated_at = time.time()

    def get_quality_threshold(self, content_type: str) -> float:
        """Get quality threshold for specific content type."""
        return self.quality_thresholds.get(content_type, 0.7)

    def is_forbidden_term(self, term: str) -> bool:
        """Check if a term is in the forbidden terminology list."""
        return term.lower() in [t.lower() for t in self.forbidden_terminology]

    def get_preferred_alternatives(self, term: str) -> List[str]:
        """Get preferred alternatives for a given term."""
        # This could be enhanced with a mapping system
        return [t for t in self.preferred_terminology if term.lower()
                in t.lower()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert CampaignContext to dictionary for serialization."""
        return {
            'content_style': self.content_style,
            'strategic_goals': self.strategic_goals,
            'audience_persona': self.audience_persona,
            'performance_analytics': self.performance_analytics,
            'quality_thresholds': self.quality_thresholds,
            'forbidden_terminology': self.forbidden_terminology,
            'preferred_terminology': self.preferred_terminology,
            'learning_data': self.learning_data,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CampaignContext':
        """Create CampaignContext from dictionary."""
        return cls(**data)

    @classmethod
    def create_default_context(cls) -> 'CampaignContext':
        """Create a default campaign context."""
        return cls(
            content_style={
                'tone': 'professional',
                'style': 'informative',
                'personality': 'friendly',
                'reading_level': 'college',
                'target_length': 'medium'
            },
            strategic_goals=[
                'Increase reader engagement',
                'Establish thought leadership',
                'Drive website traffic'
            ],
            audience_persona={
                'demographics': 'professionals aged 25-45',
                'interests': 'technology, business, innovation',
                'pain_points': 'information overload, time constraints',
                'preferred_format': 'scannable, actionable content'
            },
            performance_analytics={
                'average_engagement_rate': 0.0,
                'click_through_rate': 0.0,
                'open_rate': 0.0,
                'total_sent': 0
            },
            quality_thresholds={
                'minimum': 0.7,
                'target': 0.85,
                'excellent': 0.95
            },
            forbidden_terminology=[],
            preferred_terminology=[],
            learning_data={
                'successful_patterns': [],
                'failed_patterns': [],
                'audience_feedback': [],
                'performance_trends': []
            }
        )
