"""
Enhanced Continuity Validation System

This module implements improved cross-section coherence and flow validation
as specified in Phase 1 FR1.4 of the Multi-Agent Enhancement PRD.

Features:
- Validate narrative flow between sections
- Check style consistency across sections
- Ensure proper section transitions
- Detect and flag content redundancy
- Transition quality scoring algorithm
- Style consistency checking
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .section_aware_prompts import SectionType

logger = logging.getLogger(__name__)


class ContinuityIssueType(Enum):
    """Types of continuity issues that can be detected."""
    ABRUPT_TRANSITION = "abrupt_transition"
    STYLE_INCONSISTENCY = "style_inconsistency"
    CONTENT_REDUNDANCY = "content_redundancy"
    LOGICAL_GAP = "logical_gap"
    TONE_SHIFT = "tone_shift"
    REFERENCE_MISMATCH = "reference_mismatch"
    STRUCTURAL_BREAK = "structural_break"


@dataclass
class ContinuityIssue:
    """Represents a continuity issue in the newsletter."""
    issue_type: ContinuityIssueType
    severity: float  # 0.0 (minor) to 1.0 (critical)
    location: str
    description: str
    suggestion: str
    section_1: Optional[SectionType] = None
    section_2: Optional[SectionType] = None
    content_snippet: str = ""


@dataclass
class TransitionAnalysis:
    """Analysis of transition between two sections."""
    from_section: SectionType
    to_section: SectionType
    transition_quality: float  # 0.0 to 1.0
    has_explicit_transition: bool
    transition_text: str
    issues: List[ContinuityIssue] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class StyleAnalysis:
    """Analysis of style consistency across sections."""
    overall_consistency: float  # 0.0 to 1.0
    tone_consistency: float
    vocabulary_consistency: float
    structure_consistency: float
    issues: List[ContinuityIssue] = field(default_factory=list)
    style_profile: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContinuityReport:
    """Comprehensive continuity validation report."""
    overall_continuity_score: float
    narrative_flow_score: float
    style_consistency_score: float
    transition_quality_score: float
    redundancy_score: float
    sections_analyzed: int
    transitions_analyzed: List[TransitionAnalysis] = field(default_factory=list)
    style_analysis: Optional[StyleAnalysis] = None
    issues: List[ContinuityIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class TransitionAnalyzer:
    """Analyzes transitions between newsletter sections."""
    
    def __init__(self):
        """Initialize transition analyzer."""
        # Transition words and phrases
        self.transition_indicators = {
            'continuation': ['furthermore', 'moreover', 'additionally', 'in addition', 'also'],
            'contrast': ['however', 'nevertheless', 'on the other hand', 'conversely', 'in contrast'],
            'conclusion': ['therefore', 'thus', 'consequently', 'as a result', 'in conclusion'],
            'sequence': ['next', 'then', 'following', 'subsequently', 'meanwhile'],
            'emphasis': ['indeed', 'in fact', 'particularly', 'especially', 'notably'],
            'example': ['for example', 'for instance', 'specifically', 'such as', 'including']
        }
        
        # Expected transitions between section types
        self.expected_transitions = {
            (SectionType.INTRODUCTION, SectionType.NEWS): ['recent', 'latest', 'current'],
            (SectionType.INTRODUCTION, SectionType.ANALYSIS): ['examining', 'analyzing', 'exploring'],
            (SectionType.NEWS, SectionType.ANALYSIS): ['these developments', 'this trend', 'analyzing'],
            (SectionType.ANALYSIS, SectionType.TUTORIAL): ['implementing', 'applying', 'practical'],
            (SectionType.TUTORIAL, SectionType.CONCLUSION): ['summary', 'recap', 'takeaways'],
            (SectionType.NEWS, SectionType.CONCLUSION): ['summary', 'overall', 'key points']
        }
        
        logger.info("Transition analyzer initialized")

    def analyze_transition(self, from_section_content: str, from_section_type: SectionType,
                          to_section_content: str, to_section_type: SectionType) -> TransitionAnalysis:
        """
        Analyze transition between two sections.
        
        Args:
            from_section_content: Content of the source section
            from_section_type: Type of the source section
            to_section_content: Content of the target section
            to_section_type: Type of the target section
            
        Returns:
            TransitionAnalysis: Detailed transition analysis
        """
        # Extract transition area (end of first section + start of second)
        from_end = from_section_content[-200:] if len(from_section_content) > 200 else from_section_content
        to_start = to_section_content[:200] if len(to_section_content) > 200 else to_section_content
        
        transition_text = f"{from_end}\n\n{to_start}"
        
        # Check for explicit transition indicators
        has_explicit_transition = self._has_explicit_transition(transition_text)
        
        # Calculate transition quality
        transition_quality = self._calculate_transition_quality(
            from_end, to_start, from_section_type, to_section_type
        )
        
        # Identify issues
        issues = self._identify_transition_issues(
            from_end, to_start, from_section_type, to_section_type
        )
        
        # Generate suggestions
        suggestions = self._generate_transition_suggestions(
            from_section_type, to_section_type, has_explicit_transition, issues
        )
        
        return TransitionAnalysis(
            from_section=from_section_type,
            to_section=to_section_type,
            transition_quality=transition_quality,
            has_explicit_transition=has_explicit_transition,
            transition_text=transition_text,
            issues=issues,
            suggestions=suggestions
        )

    def _has_explicit_transition(self, transition_text: str) -> bool:
        """Check if transition has explicit transitional phrases."""
        transition_text_lower = transition_text.lower()
        
        for category, indicators in self.transition_indicators.items():
            for indicator in indicators:
                if indicator in transition_text_lower:
                    return True
        
        return False

    def _calculate_transition_quality(self, from_end: str, to_start: str,
                                    from_type: SectionType, to_type: SectionType) -> float:
        """Calculate quality score for section transition."""
        score = 0.0
        
        # Check for explicit transition words
        transition_text = f"{from_end} {to_start}".lower()
        has_transition_words = any(
            indicator in transition_text
            for indicators in self.transition_indicators.values()
            for indicator in indicators
        )
        
        if has_transition_words:
            score += 0.3
        
        # Check for context-appropriate transitions
        expected = self.expected_transitions.get((from_type, to_type), [])
        if any(phrase in transition_text for phrase in expected):
            score += 0.4
        
        # Check for logical flow (simplified heuristic)
        # Look for connecting concepts or themes
        from_words = set(re.findall(r'\b\w+\b', from_end.lower()))
        to_words = set(re.findall(r'\b\w+\b', to_start.lower()))
        common_words = from_words.intersection(to_words)
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        meaningful_common = common_words - stop_words
        
        if len(meaningful_common) > 2:
            score += 0.2
        elif len(meaningful_common) > 0:
            score += 0.1
        
        # Penalize abrupt topic changes
        if from_type != to_type and not has_transition_words:
            score -= 0.1
        
        return max(0.0, min(1.0, score))

    def _identify_transition_issues(self, from_end: str, to_start: str,
                                  from_type: SectionType, to_type: SectionType) -> List[ContinuityIssue]:
        """Identify specific transition issues."""
        issues = []
        
        # Check for abrupt transitions
        if not self._has_explicit_transition(f"{from_end} {to_start}"):
            if from_type != to_type:  # Different section types need transitions
                issues.append(ContinuityIssue(
                    issue_type=ContinuityIssueType.ABRUPT_TRANSITION,
                    severity=0.6,
                    location=f"Between {from_type.value} and {to_type.value}",
                    description="Abrupt transition between section types without connecting language",
                    suggestion="Add transitional phrases to connect the sections smoothly",
                    section_1=from_type,
                    section_2=to_type
                ))
        
        # Check for logical gaps
        if self._has_logical_gap(from_end, to_start):
            issues.append(ContinuityIssue(
                issue_type=ContinuityIssueType.LOGICAL_GAP,
                severity=0.5,
                location=f"Between {from_type.value} and {to_type.value}",
                description="Content appears to have logical gap or missing information",
                suggestion="Ensure logical progression of ideas between sections",
                section_1=from_type,
                section_2=to_type
            ))
        
        return issues

    def _generate_transition_suggestions(self, from_type: SectionType, to_type: SectionType,
                                       has_explicit_transition: bool, issues: List[ContinuityIssue]) -> List[str]:
        """Generate suggestions for improving transitions."""
        suggestions = []
        
        if not has_explicit_transition:
            expected = self.expected_transitions.get((from_type, to_type))
            if expected:
                suggestions.append(f"Consider using phrases like: {', '.join(expected[:3])}")
            else:
                suggestions.append("Add transitional phrases to connect sections smoothly")
        
        if any(issue.issue_type == ContinuityIssueType.LOGICAL_GAP for issue in issues):
            suggestions.append("Ensure ideas flow logically from one section to the next")
        
        return suggestions

    def _has_logical_gap(self, from_end: str, to_start: str) -> bool:
        """Detect potential logical gaps (simplified heuristic)."""
        # Very basic heuristic - could be much more sophisticated
        from_sentences = re.findall(r'[^.!?]*[.!?]', from_end)
        to_sentences = re.findall(r'[^.!?]*[.!?]', to_start)
        
        if not from_sentences or not to_sentences:
            return False
        
        last_sentence = from_sentences[-1].lower()
        first_sentence = to_sentences[0].lower()
        
        # Check for unresolved questions or incomplete thoughts
        if '?' in last_sentence and 'answer' not in first_sentence and 'because' not in first_sentence:
            return True
        
        return False


class StyleConsistencyAnalyzer:
    """Analyzes style consistency across newsletter sections."""
    
    def __init__(self):
        """Initialize style consistency analyzer."""
        # Style indicators
        self.formal_indicators = [
            'therefore', 'furthermore', 'consequently', 'nonetheless', 'nevertheless',
            'moreover', 'additionally', 'specifically', 'particularly'
        ]
        
        self.informal_indicators = [
            'so', 'well', 'basically', 'just', 'pretty', 'really', 'quite',
            'kind of', 'sort of', 'you know'
        ]
        
        self.technical_indicators = [
            'algorithm', 'implementation', 'architecture', 'framework', 'methodology',
            'optimization', 'configuration', 'deployment', 'scalability'
        ]
        
        logger.info("Style consistency analyzer initialized")

    def analyze_style_consistency(self, sections: Dict[SectionType, str]) -> StyleAnalysis:
        """
        Analyze style consistency across sections.
        
        Args:
            sections: Dictionary mapping section types to content
            
        Returns:
            StyleAnalysis: Detailed style consistency analysis
        """
        if len(sections) < 2:
            return StyleAnalysis(
                overall_consistency=1.0,
                tone_consistency=1.0,
                vocabulary_consistency=1.0,
                structure_consistency=1.0
            )
        
        # Analyze each section's style profile
        section_profiles = {}
        for section_type, content in sections.items():
            section_profiles[section_type] = self._create_style_profile(content)
        
        # Calculate consistency scores
        tone_consistency = self._calculate_tone_consistency(section_profiles)
        vocabulary_consistency = self._calculate_vocabulary_consistency(section_profiles)
        structure_consistency = self._calculate_structure_consistency(section_profiles)
        
        overall_consistency = (tone_consistency + vocabulary_consistency + structure_consistency) / 3
        
        # Identify style issues
        issues = self._identify_style_issues(section_profiles, sections)
        
        return StyleAnalysis(
            overall_consistency=overall_consistency,
            tone_consistency=tone_consistency,
            vocabulary_consistency=vocabulary_consistency,
            structure_consistency=structure_consistency,
            issues=issues,
            style_profile=self._create_aggregate_profile(section_profiles)
        )

    def _create_style_profile(self, content: str) -> Dict[str, Any]:
        """Create style profile for a section."""
        words = re.findall(r'\b\w+\b', content.lower())
        sentences = re.findall(r'[^.!?]*[.!?]', content)
        
        if not words:
            return {}
        
        # Calculate style metrics
        formal_count = sum(1 for word in words if word in self.formal_indicators)
        informal_count = sum(1 for word in words if word in self.informal_indicators)
        technical_count = sum(1 for word in words if word in self.technical_indicators)
        
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Personal pronouns (indicates informality)
        personal_pronouns = sum(1 for word in words if word in ['you', 'your', 'we', 'our', 'i', 'my'])
        
        # Passive voice indicators (simplified)
        passive_indicators = ['was', 'were', 'been', 'being']
        passive_count = sum(1 for word in words if word in passive_indicators)
        
        return {
            'formality_score': (formal_count - informal_count) / max(len(words), 1),
            'technical_density': technical_count / max(len(words), 1),
            'avg_sentence_length': avg_sentence_length,
            'personal_pronoun_ratio': personal_pronouns / max(len(words), 1),
            'passive_voice_ratio': passive_count / max(len(words), 1),
            'word_count': len(words),
            'sentence_count': len(sentences)
        }

    def _calculate_tone_consistency(self, section_profiles: Dict[SectionType, Dict[str, Any]]) -> float:
        """Calculate tone consistency across sections."""
        if len(section_profiles) < 2:
            return 1.0
        
        formality_scores = [profile.get('formality_score', 0) for profile in section_profiles.values()]
        pronoun_ratios = [profile.get('personal_pronoun_ratio', 0) for profile in section_profiles.values()]
        
        # Calculate variance in formality and pronoun usage
        formality_variance = self._calculate_variance(formality_scores)
        pronoun_variance = self._calculate_variance(pronoun_ratios)
        
        # Convert variance to consistency score (lower variance = higher consistency)
        formality_consistency = max(0.0, 1.0 - formality_variance * 10)
        pronoun_consistency = max(0.0, 1.0 - pronoun_variance * 50)
        
        return (formality_consistency + pronoun_consistency) / 2

    def _calculate_vocabulary_consistency(self, section_profiles: Dict[SectionType, Dict[str, Any]]) -> float:
        """Calculate vocabulary consistency across sections."""
        if len(section_profiles) < 2:
            return 1.0
        
        technical_densities = [profile.get('technical_density', 0) for profile in section_profiles.values()]
        
        # Calculate variance in technical density
        technical_variance = self._calculate_variance(technical_densities)
        
        # Convert to consistency score
        return max(0.0, 1.0 - technical_variance * 20)

    def _calculate_structure_consistency(self, section_profiles: Dict[SectionType, Dict[str, Any]]) -> float:
        """Calculate structural consistency across sections."""
        if len(section_profiles) < 2:
            return 1.0
        
        sentence_lengths = [profile.get('avg_sentence_length', 0) for profile in section_profiles.values()]
        passive_ratios = [profile.get('passive_voice_ratio', 0) for profile in section_profiles.values()]
        
        # Calculate variance
        length_variance = self._calculate_variance(sentence_lengths)
        passive_variance = self._calculate_variance(passive_ratios)
        
        # Convert to consistency scores
        length_consistency = max(0.0, 1.0 - length_variance / 100)
        passive_consistency = max(0.0, 1.0 - passive_variance * 20)
        
        return (length_consistency + passive_consistency) / 2

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

    def _identify_style_issues(self, section_profiles: Dict[SectionType, Dict[str, Any]],
                             sections: Dict[SectionType, str]) -> List[ContinuityIssue]:
        """Identify specific style consistency issues."""
        issues = []
        
        # Check for tone inconsistencies
        formality_scores = [(section_type, profile.get('formality_score', 0)) 
                           for section_type, profile in section_profiles.items()]
        
        # Find sections with significantly different formality
        if len(formality_scores) > 1:
            avg_formality = sum(score for _, score in formality_scores) / len(formality_scores)
            
            for section_type, score in formality_scores:
                if abs(score - avg_formality) > 0.05:  # Threshold for significant difference
                    tone_desc = "more formal" if score > avg_formality else "more informal"
                    issues.append(ContinuityIssue(
                        issue_type=ContinuityIssueType.TONE_SHIFT,
                        severity=0.4,
                        location=f"{section_type.value} section",
                        description=f"Section has noticeably {tone_desc} tone than other sections",
                        suggestion="Adjust tone to match overall newsletter style",
                        section_1=section_type
                    ))
        
        return issues

    def _create_aggregate_profile(self, section_profiles: Dict[SectionType, Dict[str, Any]]) -> Dict[str, Any]:
        """Create aggregate style profile across all sections."""
        if not section_profiles:
            return {}
        
        all_values = {}
        for profile in section_profiles.values():
            for key, value in profile.items():
                if key not in all_values:
                    all_values[key] = []
                all_values[key].append(value)
        
        # Calculate averages
        aggregate = {}
        for key, values in all_values.items():
            if values:
                aggregate[f"avg_{key}"] = sum(values) / len(values)
                aggregate[f"variance_{key}"] = self._calculate_variance(values)
        
        return aggregate


class RedundancyDetector:
    """Detects content redundancy between sections."""
    
    def __init__(self):
        """Initialize redundancy detector."""
        self.similarity_threshold = 0.7
        logger.info("Redundancy detector initialized")

    def detect_redundancy(self, sections: Dict[SectionType, str]) -> List[ContinuityIssue]:
        """
        Detect content redundancy between sections.
        
        Args:
            sections: Dictionary mapping section types to content
            
        Returns:
            List[ContinuityIssue]: Detected redundancy issues
        """
        issues = []
        section_items = list(sections.items())
        
        for i, (section_type_1, content_1) in enumerate(section_items):
            for section_type_2, content_2 in section_items[i+1:]:
                similarity = self._calculate_content_similarity(content_1, content_2)
                
                if similarity > self.similarity_threshold:
                    issues.append(ContinuityIssue(
                        issue_type=ContinuityIssueType.CONTENT_REDUNDANCY,
                        severity=similarity,
                        location=f"Between {section_type_1.value} and {section_type_2.value}",
                        description=f"High content similarity ({similarity:.1%}) detected between sections",
                        suggestion="Remove or consolidate redundant information",
                        section_1=section_type_1,
                        section_2=section_type_2
                    ))
        
        return issues

    def _calculate_content_similarity(self, content_1: str, content_2: str) -> float:
        """Calculate similarity between two content pieces."""
        # Simple word-based similarity using Jaccard index
        words_1 = set(re.findall(r'\b\w+\b', content_1.lower()))
        words_2 = set(re.findall(r'\b\w+\b', content_2.lower()))
        
        if not words_1 and not words_2:
            return 1.0
        
        if not words_1 or not words_2:
            return 0.0
        
        intersection = words_1.intersection(words_2)
        union = words_1.union(words_2)
        
        return len(intersection) / len(union)


class ContinuityValidator:
    """
    Enhanced continuity validation system.
    
    Implements FR1.4 requirements for cross-section coherence checking,
    style consistency validation, and transition quality assessment.
    """
    
    def __init__(self):
        """Initialize continuity validator."""
        self.transition_analyzer = TransitionAnalyzer()
        self.style_analyzer = StyleConsistencyAnalyzer()
        self.redundancy_detector = RedundancyDetector()
        
        logger.info("Continuity validator initialized")

    def validate_newsletter_continuity(self, sections: Dict[SectionType, str],
                                     context: Optional[Dict[str, Any]] = None) -> ContinuityReport:
        """
        Validate continuity across newsletter sections.
        
        Args:
            sections: Dictionary mapping section types to content
            context: Optional context for validation
            
        Returns:
            ContinuityReport: Comprehensive continuity analysis
        """
        context = context or {}
        
        # Analyze transitions between sections
        transitions = self._analyze_all_transitions(sections)
        
        # Analyze style consistency
        style_analysis = self.style_analyzer.analyze_style_consistency(sections)
        
        # Detect redundancy
        redundancy_issues = self.redundancy_detector.detect_redundancy(sections)
        
        # Calculate overall scores
        narrative_flow_score = self._calculate_narrative_flow_score(transitions)
        transition_quality_score = self._calculate_transition_quality_score(transitions)
        redundancy_score = 1.0 - (len(redundancy_issues) / max(len(sections), 1))
        
        overall_continuity_score = (
            narrative_flow_score * 0.3 +
            style_analysis.overall_consistency * 0.3 +
            transition_quality_score * 0.25 +
            redundancy_score * 0.15
        )
        
        # Collect all issues
        all_issues = []
        for transition in transitions:
            all_issues.extend(transition.issues)
        all_issues.extend(style_analysis.issues)
        all_issues.extend(redundancy_issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(transitions, style_analysis, redundancy_issues)
        
        return ContinuityReport(
            overall_continuity_score=overall_continuity_score,
            narrative_flow_score=narrative_flow_score,
            style_consistency_score=style_analysis.overall_consistency,
            transition_quality_score=transition_quality_score,
            redundancy_score=redundancy_score,
            sections_analyzed=len(sections),
            transitions_analyzed=transitions,
            style_analysis=style_analysis,
            issues=all_issues,
            recommendations=recommendations
        )

    def _analyze_all_transitions(self, sections: Dict[SectionType, str]) -> List[TransitionAnalysis]:
        """Analyze all transitions between consecutive sections."""
        transitions = []
        section_items = list(sections.items())
        
        for i in range(len(section_items) - 1):
            from_type, from_content = section_items[i]
            to_type, to_content = section_items[i + 1]
            
            transition = self.transition_analyzer.analyze_transition(
                from_content, from_type, to_content, to_type
            )
            transitions.append(transition)
        
        return transitions

    def _calculate_narrative_flow_score(self, transitions: List[TransitionAnalysis]) -> float:
        """Calculate overall narrative flow score."""
        if not transitions:
            return 1.0
        
        total_quality = sum(transition.transition_quality for transition in transitions)
        return total_quality / len(transitions)

    def _calculate_transition_quality_score(self, transitions: List[TransitionAnalysis]) -> float:
        """Calculate overall transition quality score."""
        if not transitions:
            return 1.0
        
        explicit_transitions = sum(1 for t in transitions if t.has_explicit_transition)
        return explicit_transitions / len(transitions)

    def _generate_recommendations(self, transitions: List[TransitionAnalysis],
                                style_analysis: StyleAnalysis,
                                redundancy_issues: List[ContinuityIssue]) -> List[str]:
        """Generate prioritized recommendations for improving continuity."""
        recommendations = []
        
        # Transition recommendations
        poor_transitions = [t for t in transitions if t.transition_quality < 0.5]
        if poor_transitions:
            recommendations.append(
                f"Improve {len(poor_transitions)} poor section transitions with connecting language"
            )
        
        missing_transitions = [t for t in transitions if not t.has_explicit_transition]
        if len(missing_transitions) > len(transitions) * 0.5:
            recommendations.append("Add explicit transitional phrases between sections")
        
        # Style recommendations
        if style_analysis.overall_consistency < 0.7:
            if style_analysis.tone_consistency < 0.7:
                recommendations.append("Standardize tone across all sections")
            if style_analysis.vocabulary_consistency < 0.7:
                recommendations.append("Maintain consistent technical language level")
            if style_analysis.structure_consistency < 0.7:
                recommendations.append("Use consistent sentence structure and length")
        
        # Redundancy recommendations
        if redundancy_issues:
            high_redundancy = [issue for issue in redundancy_issues if issue.severity > 0.8]
            if high_redundancy:
                recommendations.append("Remove or consolidate highly redundant content")
        
        return recommendations


# Export main classes and functions
__all__ = [
    'ContinuityValidator',
    'ContinuityReport',
    'TransitionAnalysis',
    'StyleAnalysis',
    'ContinuityIssue',
    'ContinuityIssueType'
]