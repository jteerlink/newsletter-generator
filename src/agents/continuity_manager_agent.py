"""
Continuity Manager Agent for Phase 2 Multi-Agent System

Coordinates cross-section continuity management using the continuity validator.
Ensures narrative coherence, style consistency, and optimal section transitions.
Supports light reordering suggestions when transitions are weak.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base_agent import (
    BaseSpecializedAgent,
    ProcessingContext,
    ProcessingResult,
    AgentConfiguration,
    ProcessingMode,
)
from core.continuity_validator import ContinuityValidator
from core.section_aware_prompts import SectionType

logger = logging.getLogger(__name__)


@dataclass
class ContinuityManagerConfig(AgentConfiguration):
    # Thresholds for suggesting reordering
    min_transition_quality: float = 0.65
    min_overall_continuity: float = 0.7


class ContinuityManagerAgent(BaseSpecializedAgent):
    """Specialized agent for cross-section continuity orchestration."""

    def __init__(self, config: Optional[ContinuityManagerConfig] = None):
        super().__init__("ContinuityManager", config or ContinuityManagerConfig())
        self.validator = ContinuityValidator()
        logger.info("Continuity Manager Agent initialized")

    def _process_internal(self, context: ProcessingContext) -> ProcessingResult:
        # Expect sections in context.metadata: Dict[str, str] mapping section types to content
        raw_sections: Dict[str, str] = context.metadata.get("sections", {})
        # If caller provided a single combined content, try a naive split into pseudo-sections
        if not raw_sections and context.content:
            raw_sections = self._naive_split_into_sections(context.content)
        if not raw_sections:
            return ProcessingResult(
                success=False,
                errors=["No sections provided for continuity analysis"],
            )

        # Convert to SectionType keys when possible
        sections: Dict[SectionType, str] = {}
        for key, value in raw_sections.items():
            try:
                st = SectionType(key)
            except Exception:
                try:
                    st = SectionType(key.lower())
                except Exception:
                    st = SectionType.ANALYSIS
            sections[st] = value

        report = self.validator.validate_newsletter_continuity(sections)

        # Determine if reordering suggestions are needed
        suggestions: List[str] = []
        if report.transition_quality_score < self.config.min_transition_quality:
            suggestions.append("Consider smoothing transitions; add bridging sentences between adjacent sections")
        if report.overall_continuity_score < self.config.min_overall_continuity:
            suggestions.append("Evaluate section order; minor reordering may improve narrative flow")

        # Build optional reordered plan (heuristic: place INTRODUCTION first, CONCLUSION last)
        proposed_order = self._propose_section_order(list(sections.keys()))
        if proposed_order != list(sections.keys()):
            suggestions.append("Proposed section order: " + ", ".join([s.value for s in proposed_order]))

        quality_score = max(0.0, min(1.0, report.overall_continuity_score))

        return ProcessingResult(
            success=True,
            processed_content=context.content,
            quality_score=quality_score,
            confidence_score=quality_score,
            suggestions=suggestions,
            warnings=[i.description for i in report.issues[:8]],
            metadata={
                "continuity_report": {
                    "overall": report.overall_continuity_score,
                    "narrative_flow": report.narrative_flow_score,
                    "style_consistency": report.style_consistency_score,
                    "transition_quality": report.transition_quality_score,
                    "redundancy": report.redundancy_score,
                    "issues": [i.to_dict() if hasattr(i, "to_dict") else str(i) for i in report.issues[:50]],
                },
                "proposed_order": [s.value for s in proposed_order],
            },
        )

    def _propose_section_order(self, current: List[SectionType]) -> List[SectionType]:
        order = current[:]
        # Ensure INTRODUCTION at start if present
        if SectionType.INTRODUCTION in order and order[0] != SectionType.INTRODUCTION:
            order.remove(SectionType.INTRODUCTION)
            order.insert(0, SectionType.INTRODUCTION)
        # Ensure CONCLUSION at end if present
        if SectionType.CONCLUSION in order and order[-1] != SectionType.CONCLUSION:
            order.remove(SectionType.CONCLUSION)
            order.append(SectionType.CONCLUSION)
        return order

    def _naive_split_into_sections(self, content: str) -> Dict[str, str]:
        sections: Dict[SectionType, str] = {}
        # Split by markdown headers as a heuristic
        parts = re.split(r"^##\s+", content, flags=re.MULTILINE)
        if parts:
            intro = parts[0].strip()
            if intro:
                sections[SectionType.INTRODUCTION] = intro
            # Remaining headers considered analysis/news/tutorial chunks
            for chunk in parts[1:]:
                chunk = chunk.strip()
                if not chunk:
                    continue
                # Assign alternating section types to diversify analysis
                target = SectionType.ANALYSIS if SectionType.ANALYSIS not in sections else SectionType.NEWS
                if target in sections:
                    # Append
                    sections[target] = sections[target] + "\n\n" + chunk
                else:
                    sections[target] = chunk
        # Ensure conclusion exists
        if SectionType.CONCLUSION not in sections and content.strip():
            tail = content.split('\n')[-80:]
            sections[SectionType.CONCLUSION] = "\n".join(tail)
        return {k: v for k, v in sections.items() if v and v.strip()}


__all__ = ["ContinuityManagerAgent", "ContinuityManagerConfig"]


