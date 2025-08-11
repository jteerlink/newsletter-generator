"""
Configuration Manager: Campaign context and configuration management.

This module provides centralized configuration management for campaign
contexts, default settings, and configuration persistence for the
enhanced agent architecture.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .campaign_context import CampaignContext


class ConfigManager:
    """Manages campaign contexts and configuration settings."""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        self.campaign_contexts: Dict[str, CampaignContext] = {}
        self.default_context_id = "default"

        # Load existing contexts
        self._load_existing_contexts()

    def _load_existing_contexts(self) -> None:
        """Load existing campaign contexts from storage."""
        context_files = list(self.config_dir.glob("context_*.json"))

        for context_file in context_files:
            try:
                context_id = context_file.stem.replace("context_", "")
                with open(context_file, 'r') as f:
                    context_data = json.load(f)

                context = CampaignContext.from_dict(context_data)
                self.campaign_contexts[context_id] = context

            except Exception as e:
                print(f"Warning: Could not load context from {
                      context_file}: {e}")

        # Ensure default context exists
        if self.default_context_id not in self.campaign_contexts:
            self.campaign_contexts[self.default_context_id] = self._create_default_context(
                self.default_context_id)

    def load_campaign_context(self, context_id: str) -> CampaignContext:
        """Load campaign context from storage."""
        if context_id not in self.campaign_contexts:
            # Create default context if it doesn't exist
            self.campaign_contexts[context_id] = self._create_default_context(
                context_id)

        return self.campaign_contexts[context_id]

    def save_campaign_context(
            self,
            context_id: str,
            context: CampaignContext) -> None:
        """Save campaign context to storage."""
        self.campaign_contexts[context_id] = context
        self._persist_context(context_id, context)

    def _persist_context(
            self,
            context_id: str,
            context: CampaignContext) -> None:
        """Persist campaign context to file."""
        context_file = self.config_dir / f"context_{context_id}.json"

        try:
            with open(context_file, 'w') as f:
                json.dump(context.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Error saving context {context_id}: {e}")

    def _create_default_context(self, context_id: str) -> CampaignContext:
        """Create a default campaign context."""
        if context_id == "default":
            return CampaignContext.create_default_context()
        else:
            # Create a customized default context based on context_id
            return self._create_customized_context(context_id)

    def _create_customized_context(self, context_id: str) -> CampaignContext:
        """Create a customized context based on the context_id."""
        # Define different context types
        context_templates = {
            "technical": {
                "content_style": {
                    "tone": "technical",
                    "style": "informative",
                    "personality": "precise",
                    "reading_level": "expert",
                    "target_length": "long"
                },
                "strategic_goals": [
                    "Establish technical expertise",
                    "Share industry insights",
                    "Drive technical engagement"
                ],
                "audience_persona": {
                    "demographics": "technical professionals aged 25-50",
                    "interests": "technology, engineering, innovation",
                    "pain_points": "complex technical information, staying current",
                    "preferred_format": "detailed, technical content with examples"
                },
                "quality_thresholds": {
                    "minimum": 0.8,
                    "target": 0.9,
                    "excellent": 0.95
                }
            },
            "business": {
                "content_style": {
                    "tone": "professional",
                    "style": "strategic",
                    "personality": "authoritative",
                    "reading_level": "business",
                    "target_length": "medium"
                },
                "strategic_goals": [
                    "Drive business insights",
                    "Establish thought leadership",
                    "Generate leads"
                ],
                "audience_persona": {
                    "demographics": "business professionals aged 30-55",
                    "interests": "business strategy, market trends, leadership",
                    "pain_points": "information overload, decision making",
                    "preferred_format": "actionable insights with clear takeaways"
                },
                "quality_thresholds": {
                    "minimum": 0.75,
                    "target": 0.85,
                    "excellent": 0.92
                }
            },
            "casual": {
                "content_style": {
                    "tone": "casual",
                    "style": "conversational",
                    "personality": "friendly",
                    "reading_level": "general",
                    "target_length": "short"
                },
                "strategic_goals": [
                    "Increase reader engagement",
                    "Build community",
                    "Share personal insights"
                ],
                "audience_persona": {
                    "demographics": "general audience aged 18-45",
                    "interests": "lifestyle, personal development, entertainment",
                    "pain_points": "time constraints, information overload",
                    "preferred_format": "easy-to-read, engaging content"
                },
                "quality_thresholds": {
                    "minimum": 0.7,
                    "target": 0.8,
                    "excellent": 0.9
                }
            }
        }

        # Get template based on context_id prefix
        template_key = None
        for key in context_templates.keys():
            if context_id.startswith(key):
                template_key = key
                break

        if template_key:
            template = context_templates[template_key]
            return CampaignContext(
                content_style=template["content_style"],
                strategic_goals=template["strategic_goals"],
                audience_persona=template["audience_persona"],
                performance_analytics={
                    "average_engagement_rate": 0.0,
                    "click_through_rate": 0.0,
                    "open_rate": 0.0,
                    "total_sent": 0
                },
                quality_thresholds=template["quality_thresholds"],
                forbidden_terminology=[],
                preferred_terminology=[],
                learning_data={
                    "successful_patterns": [],
                    "failed_patterns": [],
                    "audience_feedback": [],
                    "performance_trends": []
                }
            )
        else:
            # Fall back to default context
            return CampaignContext.create_default_context()

    def list_campaign_contexts(self) -> List[str]:
        """List all available campaign context IDs."""
        return list(self.campaign_contexts.keys())

    def delete_campaign_context(self, context_id: str) -> bool:
        """Delete a campaign context."""
        if context_id == self.default_context_id:
            print(f"Cannot delete default context: {context_id}")
            return False

        if context_id in self.campaign_contexts:
            del self.campaign_contexts[context_id]

            # Remove from file system
            context_file = self.config_dir / f"context_{context_id}.json"
            if context_file.exists():
                context_file.unlink()

            return True

        return False

    def copy_campaign_context(self, source_id: str, target_id: str) -> bool:
        """Copy a campaign context to a new ID."""
        if source_id not in self.campaign_contexts:
            return False

        source_context = self.campaign_contexts[source_id]

        # Create a copy with new timestamps
        import time
        copied_context = CampaignContext(
            content_style=source_context.content_style.copy(),
            strategic_goals=source_context.strategic_goals.copy(),
            audience_persona=source_context.audience_persona.copy(),
            performance_analytics=source_context.performance_analytics.copy(),
            quality_thresholds=source_context.quality_thresholds.copy(),
            forbidden_terminology=source_context.forbidden_terminology.copy(),
            preferred_terminology=source_context.preferred_terminology.copy(),
            learning_data=source_context.learning_data.copy(),
            created_at=time.time(),
            updated_at=time.time()
        )

        self.campaign_contexts[target_id] = copied_context
        self._persist_context(target_id, copied_context)

        return True

    def update_context_field(
            self,
            context_id: str,
            field: str,
            value: Any) -> bool:
        """Update a specific field in a campaign context."""
        if context_id not in self.campaign_contexts:
            return False

        context = self.campaign_contexts[context_id]

        if hasattr(context, field):
            setattr(context, field, value)
            context.updated_at = time.time()
            self._persist_context(context_id, context)
            return True

        return False

    def get_context_summary(self, context_id: str) -> Dict[str, Any]:
        """Get a summary of a campaign context."""
        if context_id not in self.campaign_contexts:
            return {"error": "Context not found"}

        context = self.campaign_contexts[context_id]

        return {
            "context_id": context_id,
            "content_style": context.content_style,
            "strategic_goals": context.strategic_goals,
            "audience_persona": context.audience_persona,
            "quality_thresholds": context.quality_thresholds,
            "created_at": context.created_at,
            "updated_at": context.updated_at,
            "learning_data_keys": list(context.learning_data.keys()),
            "performance_analytics": context.performance_analytics
        }

    def export_all_contexts(self) -> Dict[str, Any]:
        """Export all campaign contexts for backup."""
        return {
            "export_timestamp": time.time(),
            "contexts": {
                context_id: context.to_dict()
                for context_id, context in self.campaign_contexts.items()
            }
        }

    def import_contexts(self, import_data: Dict[str, Any]) -> List[str]:
        """Import campaign contexts from backup data."""
        imported_ids = []

        if "contexts" not in import_data:
            return imported_ids

        for context_id, context_data in import_data["contexts"].items():
            try:
                context = CampaignContext.from_dict(context_data)
                self.campaign_contexts[context_id] = context
                self._persist_context(context_id, context)
                imported_ids.append(context_id)
            except Exception as e:
                print(f"Error importing context {context_id}: {e}")

        return imported_ids

    def validate_context(self, context_id: str) -> Dict[str, Any]:
        """Validate a campaign context for completeness and consistency."""
        if context_id not in self.campaign_contexts:
            return {"valid": False, "error": "Context not found"}

        context = self.campaign_contexts[context_id]
        issues = []

        # Check required fields
        required_fields = [
            "content_style", "strategic_goals", "audience_persona",
            "performance_analytics", "quality_thresholds", "learning_data"
        ]

        for field in required_fields:
            if not hasattr(context, field) or getattr(context, field) is None:
                issues.append(f"Missing required field: {field}")

        # Check content style completeness
        if hasattr(context, 'content_style') and context.content_style:
            required_style_fields = ["tone", "style", "personality"]
            for field in required_style_fields:
                if field not in context.content_style:
                    issues.append(f"Missing content style field: {field}")

        # Check quality thresholds
        if hasattr(
                context,
                'quality_thresholds') and context.quality_thresholds:
            required_thresholds = ["minimum", "target"]
            for threshold in required_thresholds:
                if threshold not in context.quality_thresholds:
                    issues.append(f"Missing quality threshold: {threshold}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "context_id": context_id
        }
