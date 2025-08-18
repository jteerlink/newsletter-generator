"""
Advanced Quality Gates

Multi-dimensional quality assessment for production deployment as specified 
in PRD Phase 4 FR4.1. Provides comprehensive quality validation with 
configurable enforcement levels and production-ready monitoring.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from core.quality_gates import QualityGateViolation, ToolUsageQualityGate
from core.tool_cache import get_tool_cache

logger = logging.getLogger(__name__)


class EnforcementLevel(Enum):
    """Quality gate enforcement levels."""
    ADVISORY = "advisory"          # Log violations only
    WARNING = "warning"            # Log violations + notify users
    ENFORCING = "enforcing"        # Block content + allow override
    STRICT = "strict"              # Block content + no override


class QualityDimension(Enum):
    """Quality assessment dimensions."""
    TOOL_USAGE = "tool_usage"
    CLAIM_VERIFICATION = "claim_verification"
    INFORMATION_FRESHNESS = "information_freshness"
    SOURCE_AUTHORITY = "source_authority"
    CONTENT_COMPLETENESS = "content_completeness"
    TECHNICAL_ACCURACY = "technical_accuracy"
    PERFORMANCE = "performance"
    ACCESSIBILITY = "accessibility"
    TEMPLATE_COMPLIANCE = "template_compliance"
    WORD_COUNT_COMPLIANCE = "word_count_compliance"
    SECTION_COMPLIANCE = "section_compliance"
    # Phase 1: Content Expansion Quality Dimensions
    CONTENT_EXPANSION_QUALITY = "content_expansion_quality"
    EXPANSION_TARGET_ACHIEVEMENT = "expansion_target_achievement"
    EXPANSION_TECHNICAL_ACCURACY = "expansion_technical_accuracy"
    # Phase 2: Mobile Optimization Quality Dimensions
    MOBILE_READABILITY = "mobile_readability"
    MOBILE_TYPOGRAPHY = "mobile_typography"
    MOBILE_STRUCTURE_OPTIMIZATION = "mobile_structure_optimization"
    MOBILE_PERFORMANCE = "mobile_performance"
    MOBILE_ACCESSIBILITY = "mobile_accessibility"


@dataclass
class QualityScore:
    """Quality score for a specific dimension."""
    dimension: QualityDimension
    score: float
    confidence: float
    evidence: List[str] = field(default_factory=list)
    violations: List[QualityGateViolation] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    overall_score: float
    dimension_scores: Dict[QualityDimension, QualityScore]
    total_violations: int
    critical_violations: int
    enforcement_level: EnforcementLevel
    content_blocked: bool
    override_allowed: bool
    assessment_timestamp: datetime
    processing_time_ms: float
    content_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityThreshold:
    """Quality threshold configuration."""
    dimension: QualityDimension
    minimum_score: float
    enforcement_level: EnforcementLevel
    auto_adjust: bool = False
    historical_performance: List[float] = field(default_factory=list)


class ConfigurableQualityGate:
    """Quality gate with multiple enforcement levels."""
    
    ENFORCEMENT_LEVELS = {
        "advisory": {"log_violations": True, "block_content": False, "notify_users": False, "allow_override": True},
        "warning": {"log_violations": True, "block_content": False, "notify_users": True, "allow_override": True},
        "enforcing": {"log_violations": True, "block_content": True, "notify_users": True, "allow_override": True},
        "strict": {"log_violations": True, "block_content": True, "notify_users": True, "allow_override": False}
    }
    
    def __init__(self, enforcement_level: str = "advisory"):
        """Initialize configurable quality gate with adaptive thresholds."""
        self.enforcement_level = EnforcementLevel(enforcement_level)
        self.tool_usage_gate = ToolUsageQualityGate()
        self.cache = get_tool_cache()
        
        # Default quality thresholds (will be adapted based on content complexity)
        self.base_thresholds = {
            QualityDimension.TOOL_USAGE: QualityThreshold(
                QualityDimension.TOOL_USAGE, 0.7, self.enforcement_level),
            QualityDimension.TEMPLATE_COMPLIANCE: QualityThreshold(
                QualityDimension.TEMPLATE_COMPLIANCE, 7.0, self.enforcement_level),
            QualityDimension.WORD_COUNT_COMPLIANCE: QualityThreshold(
                QualityDimension.WORD_COUNT_COMPLIANCE, 0.8, self.enforcement_level),
            QualityDimension.SECTION_COMPLIANCE: QualityThreshold(
                QualityDimension.SECTION_COMPLIANCE, 0.9, self.enforcement_level),
            QualityDimension.CLAIM_VERIFICATION: QualityThreshold(
                QualityDimension.CLAIM_VERIFICATION, 0.6, self.enforcement_level),
            QualityDimension.INFORMATION_FRESHNESS: QualityThreshold(
                QualityDimension.INFORMATION_FRESHNESS, 0.8, self.enforcement_level),
            QualityDimension.SOURCE_AUTHORITY: QualityThreshold(
                QualityDimension.SOURCE_AUTHORITY, 0.7, self.enforcement_level),
            QualityDimension.CONTENT_COMPLETENESS: QualityThreshold(
                QualityDimension.CONTENT_COMPLETENESS, 0.8, self.enforcement_level),
            QualityDimension.TECHNICAL_ACCURACY: QualityThreshold(
                QualityDimension.TECHNICAL_ACCURACY, 0.85, self.enforcement_level),
            QualityDimension.PERFORMANCE: QualityThreshold(
                QualityDimension.PERFORMANCE, 0.7, EnforcementLevel.WARNING),
            QualityDimension.ACCESSIBILITY: QualityThreshold(
                QualityDimension.ACCESSIBILITY, 0.8, EnforcementLevel.ENFORCING),
            # Phase 1: Content Expansion Quality Thresholds
            QualityDimension.CONTENT_EXPANSION_QUALITY: QualityThreshold(
                QualityDimension.CONTENT_EXPANSION_QUALITY, 0.8, EnforcementLevel.WARNING),
            QualityDimension.EXPANSION_TARGET_ACHIEVEMENT: QualityThreshold(
                QualityDimension.EXPANSION_TARGET_ACHIEVEMENT, 0.85, EnforcementLevel.WARNING),
            QualityDimension.EXPANSION_TECHNICAL_ACCURACY: QualityThreshold(
                QualityDimension.EXPANSION_TECHNICAL_ACCURACY, 0.9, EnforcementLevel.ENFORCING),
            # Phase 2: Mobile Optimization Quality Thresholds
            QualityDimension.MOBILE_READABILITY: QualityThreshold(
                QualityDimension.MOBILE_READABILITY, 0.85, EnforcementLevel.WARNING),
            QualityDimension.MOBILE_TYPOGRAPHY: QualityThreshold(
                QualityDimension.MOBILE_TYPOGRAPHY, 0.8, EnforcementLevel.WARNING),
            QualityDimension.MOBILE_STRUCTURE_OPTIMIZATION: QualityThreshold(
                QualityDimension.MOBILE_STRUCTURE_OPTIMIZATION, 0.8, EnforcementLevel.WARNING),
            QualityDimension.MOBILE_PERFORMANCE: QualityThreshold(
                QualityDimension.MOBILE_PERFORMANCE, 0.75, EnforcementLevel.WARNING),
            QualityDimension.MOBILE_ACCESSIBILITY: QualityThreshold(
                QualityDimension.MOBILE_ACCESSIBILITY, 0.85, EnforcementLevel.ENFORCING)
        }
        
        # Enhanced template compliance thresholds
        self.base_thresholds[QualityDimension.TEMPLATE_COMPLIANCE] = QualityThreshold(
            QualityDimension.TEMPLATE_COMPLIANCE, 7.0, EnforcementLevel.ENFORCING)
        self.base_thresholds[QualityDimension.WORD_COUNT_COMPLIANCE] = QualityThreshold(
            QualityDimension.WORD_COUNT_COMPLIANCE, 0.8, EnforcementLevel.WARNING)
        self.base_thresholds[QualityDimension.SECTION_COMPLIANCE] = QualityThreshold(
            QualityDimension.SECTION_COMPLIANCE, 0.9, EnforcementLevel.WARNING)
        
        # Current adaptive thresholds (updated per validation)
        self.thresholds = self.base_thresholds.copy()
        
        logger.info(f"Configurable quality gate initialized with {enforcement_level} enforcement")
    
    def _calculate_content_complexity(self, content: str, metadata: Dict[str, Any]) -> float:
        """Calculate content complexity score (0.0-1.0) for adaptive threshold adjustment."""
        complexity_factors = {}
        
        # Length complexity (longer content is more complex)
        word_count = len(content.split())
        length_complexity = min(1.0, word_count / 5000)  # Normalize to 5000 words = 1.0
        complexity_factors['length'] = length_complexity
        
        # Technical complexity
        technical_terms = ['api', 'algorithm', 'framework', 'implementation', 'architecture', 
                          'deployment', 'integration', 'optimization', 'configuration', 'specification']
        tech_count = sum(1 for term in technical_terms if term.lower() in content.lower())
        tech_complexity = min(1.0, tech_count / 10)  # Normalize to 10 terms = 1.0
        complexity_factors['technical'] = tech_complexity
        
        # Structural complexity (headers, code blocks, lists)
        structure_score = 0.0
        structure_score += min(0.3, content.count('#') / 10)  # Headers
        structure_score += min(0.3, content.count('```') / 6)  # Code blocks
        structure_score += min(0.2, content.count('•') / 10)  # Lists
        structure_score += min(0.2, content.count('1.') / 10)  # Numbered lists
        complexity_factors['structure'] = structure_score
        
        # Sentence complexity (average sentence length)
        sentences = [s for s in content.split('.') if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            sentence_complexity = min(1.0, (avg_sentence_length - 10) / 25)  # 10-35 word range
            complexity_factors['sentence'] = max(0.0, sentence_complexity)
        else:
            complexity_factors['sentence'] = 0.0
        
        # Template type complexity
        template_complexity = {
            'technical_deep_dive': 0.9,
            'tutorial_guide': 0.8,
            'trend_analysis': 0.6,
            'research_summary': 0.7,
            'product_review': 0.5,
            'general': 0.4
        }
        template_type = metadata.get('template_type', 'general')
        complexity_factors['template'] = template_complexity.get(template_type, 0.4)
        
        # Weighted complexity calculation
        weights = {
            'length': 0.15,
            'technical': 0.25, 
            'structure': 0.20,
            'sentence': 0.15,
            'template': 0.25
        }
        
        overall_complexity = sum(complexity_factors[factor] * weights[factor] 
                               for factor in weights)
        
        logger.info(f"Content complexity: {overall_complexity:.3f} (factors: {complexity_factors})")
        return overall_complexity
    
    def _adapt_thresholds_to_complexity(self, complexity: float, metadata: Dict[str, Any]):
        """Adapt quality thresholds based on content complexity."""
        # Reset to base thresholds
        self.thresholds = {}
        
        for dimension, base_threshold in self.base_thresholds.items():
            adapted_threshold = self._calculate_adaptive_threshold(dimension, base_threshold, complexity, metadata)
            self.thresholds[dimension] = adapted_threshold
            
        logger.info(f"Thresholds adapted for complexity {complexity:.3f}")
    
    def _calculate_adaptive_threshold(self, dimension: QualityDimension, 
                                    base_threshold: QualityThreshold, 
                                    complexity: float, 
                                    metadata: Dict[str, Any]) -> QualityThreshold:
        """Calculate adaptive threshold for a specific quality dimension."""
        base_score = base_threshold.minimum_score
        
        # Different adaptation strategies by dimension
        if dimension == QualityDimension.WORD_COUNT_COMPLIANCE:
            # More lenient word count thresholds for highly complex content
            if complexity > 0.8:
                adapted_score = max(0.75, base_score - 0.1)  # Lower threshold for complex content
            elif complexity > 0.6:
                adapted_score = base_score - 0.05
            else:
                adapted_score = base_score
                
        elif dimension == QualityDimension.TECHNICAL_ACCURACY:
            # Higher technical accuracy requirements for technical content
            if complexity > 0.7 and metadata.get('template_type') == 'technical_deep_dive':
                adapted_score = min(0.95, base_score + 0.1)
            elif complexity > 0.5:
                adapted_score = base_score + 0.05
            else:
                adapted_score = base_score
                
        elif dimension == QualityDimension.CONTENT_COMPLETENESS:
            # Adjust completeness thresholds based on content complexity
            if complexity > 0.8:
                adapted_score = max(0.7, base_score - 0.1)  # More lenient for complex content
            elif complexity < 0.3:
                adapted_score = min(0.9, base_score + 0.1)  # Stricter for simple content
            else:
                adapted_score = base_score
                
        elif dimension == QualityDimension.TEMPLATE_COMPLIANCE:
            # Template compliance adjustments based on complexity
            if complexity > 0.8:
                adapted_score = max(6.0, base_score - 1.0)  # More lenient scoring
            elif complexity > 0.6:
                adapted_score = base_score - 0.5
            else:
                adapted_score = base_score
                
        elif dimension == QualityDimension.SECTION_COMPLIANCE:
            # Section compliance adjustments
            if complexity > 0.7:
                adapted_score = max(0.8, base_score - 0.1)
            else:
                adapted_score = base_score
                
        else:
            # Default: small adjustments for other dimensions
            if complexity > 0.8:
                adapted_score = max(0.5, base_score - 0.05)
            elif complexity > 0.6:
                adapted_score = base_score - 0.02
            elif complexity < 0.3:
                adapted_score = min(1.0, base_score + 0.05)
            else:
                adapted_score = base_score
        
        # Create new adaptive threshold
        return QualityThreshold(
            dimension=dimension,
            minimum_score=adapted_score,
            enforcement_level=base_threshold.enforcement_level,
            auto_adjust=True,
            historical_performance=[adapted_score]
        )
    
    def validate_with_level(self, content: str, tool_usage: Dict, 
                          metadata: Dict[str, Any] = None,
                          level: Optional[str] = None) -> QualityReport:
        """Validate content with adaptive thresholds based on complexity."""
        start_time = time.time()
        
        if level:
            current_level = EnforcementLevel(level)
        else:
            current_level = self.enforcement_level
        
        if metadata is None:
            metadata = {}
        
        # Calculate content complexity and adapt thresholds
        complexity = self._calculate_content_complexity(content, metadata)
        self._adapt_thresholds_to_complexity(complexity, metadata)
        
        # Assess all quality dimensions with adaptive thresholds
        dimension_scores = {}
        all_violations = []
        
        # Template compliance assessment
        template_score = self._assess_template_compliance(content, metadata)
        dimension_scores[QualityDimension.TEMPLATE_COMPLIANCE] = template_score
        all_violations.extend(template_score.violations)
        
        # Word count compliance assessment
        word_count_score = self._assess_word_count_compliance(content, metadata)
        dimension_scores[QualityDimension.WORD_COUNT_COMPLIANCE] = word_count_score
        all_violations.extend(word_count_score.violations)
        
        # Section compliance assessment
        section_score = self._assess_section_compliance(content, metadata)
        dimension_scores[QualityDimension.SECTION_COMPLIANCE] = section_score
        all_violations.extend(section_score.violations)
        
        # Tool usage assessment
        tool_score = self._assess_tool_usage(content, tool_usage, metadata)
        dimension_scores[QualityDimension.TOOL_USAGE] = tool_score
        all_violations.extend(tool_score.violations)
        
        # Claim verification assessment
        claim_score = self._assess_claim_verification(content, tool_usage, metadata)
        dimension_scores[QualityDimension.CLAIM_VERIFICATION] = claim_score
        all_violations.extend(claim_score.violations)
        
        # Information freshness assessment
        freshness_score = self._assess_information_freshness(content, metadata)
        dimension_scores[QualityDimension.INFORMATION_FRESHNESS] = freshness_score
        all_violations.extend(freshness_score.violations)
        
        # Source authority assessment
        authority_score = self._assess_source_authority(tool_usage, metadata)
        dimension_scores[QualityDimension.SOURCE_AUTHORITY] = authority_score
        all_violations.extend(authority_score.violations)
        
        # Content completeness assessment
        completeness_score = self._assess_content_completeness(content, metadata)
        dimension_scores[QualityDimension.CONTENT_COMPLETENESS] = completeness_score
        all_violations.extend(completeness_score.violations)
        
        # Technical accuracy assessment
        accuracy_score = self._assess_technical_accuracy(content, metadata)
        dimension_scores[QualityDimension.TECHNICAL_ACCURACY] = accuracy_score
        all_violations.extend(accuracy_score.violations)
        
        # Performance assessment
        performance_score = self._assess_performance(metadata)
        dimension_scores[QualityDimension.PERFORMANCE] = performance_score
        all_violations.extend(performance_score.violations)
        
        # Accessibility assessment
        accessibility_score = self._assess_accessibility(content, metadata)
        dimension_scores[QualityDimension.ACCESSIBILITY] = accessibility_score
        all_violations.extend(accessibility_score.violations)
        
        # Phase 1: Content Expansion Quality Assessments
        if metadata.get('expansion_applied'):
            expansion_quality_score = self._assess_content_expansion_quality(content, metadata)
            dimension_scores[QualityDimension.CONTENT_EXPANSION_QUALITY] = expansion_quality_score
            all_violations.extend(expansion_quality_score.violations)
            
            expansion_target_score = self._assess_expansion_target_achievement(content, metadata)
            dimension_scores[QualityDimension.EXPANSION_TARGET_ACHIEVEMENT] = expansion_target_score
            all_violations.extend(expansion_target_score.violations)
            
            expansion_accuracy_score = self._assess_expansion_technical_accuracy(content, metadata)
            dimension_scores[QualityDimension.EXPANSION_TECHNICAL_ACCURACY] = expansion_accuracy_score
            all_violations.extend(expansion_accuracy_score.violations)
        
        # Phase 2: Mobile Optimization Quality Assessments
        if metadata.get('mobile_optimization_applied'):
            mobile_readability_score = self._assess_mobile_readability(content, metadata)
            dimension_scores[QualityDimension.MOBILE_READABILITY] = mobile_readability_score
            all_violations.extend(mobile_readability_score.violations)
            
            mobile_typography_score = self._assess_mobile_typography(content, metadata)
            dimension_scores[QualityDimension.MOBILE_TYPOGRAPHY] = mobile_typography_score
            all_violations.extend(mobile_typography_score.violations)
            
            mobile_structure_score = self._assess_mobile_structure_optimization(content, metadata)
            dimension_scores[QualityDimension.MOBILE_STRUCTURE_OPTIMIZATION] = mobile_structure_score
            all_violations.extend(mobile_structure_score.violations)
            
            mobile_performance_score = self._assess_mobile_performance(content, metadata)
            dimension_scores[QualityDimension.MOBILE_PERFORMANCE] = mobile_performance_score
            all_violations.extend(mobile_performance_score.violations)
            
            mobile_accessibility_score = self._assess_mobile_accessibility(content, metadata)
            dimension_scores[QualityDimension.MOBILE_ACCESSIBILITY] = mobile_accessibility_score
            all_violations.extend(mobile_accessibility_score.violations)
        
        # Dynamic quality scoring with complexity-based adjustments
        overall_score = self._calculate_dynamic_quality_score(dimension_scores, complexity, metadata)
        
        # Count violations by severity
        critical_violations = sum(1 for v in all_violations if v.severity == "critical")
        total_violations = len(all_violations)
        
        # Determine content blocking
        config = self.ENFORCEMENT_LEVELS[current_level.value]
        content_blocked = config["block_content"] and (critical_violations > 0 or overall_score < 0.6)
        override_allowed = config["allow_override"]
        
        # Log violations if configured
        if config["log_violations"]:
            self._log_violations(all_violations, current_level)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Enhanced metadata with complexity and adaptive threshold info
        enhanced_metadata = metadata.copy()
        enhanced_metadata.update({
            'content_complexity': complexity,
            'adaptive_thresholds_used': True,
            'threshold_adaptations': {
                dim.value: threshold.minimum_score 
                for dim, threshold in self.thresholds.items()
            }
        })
        
        report = QualityReport(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            total_violations=total_violations,
            critical_violations=critical_violations,
            enforcement_level=current_level,
            content_blocked=content_blocked,
            override_allowed=override_allowed,
            assessment_timestamp=datetime.now(),
            processing_time_ms=processing_time,
            content_metadata=enhanced_metadata
        )
        
        # Cache the quality report
        self.cache.cache_analysis_results(
            f"quality_report_{hash(content[:100])}",
            report,
            session_id=metadata.get('session_id'),
            workflow_id=metadata.get('workflow_id')
        )
        
        logger.info(f"Quality assessment completed: overall={overall_score:.3f}, "
                   f"violations={total_violations}, blocked={content_blocked}")
        
        return report
    
    def _assess_tool_usage(self, content: str, tool_usage: Dict, metadata: Dict) -> QualityScore:
        """Assess tool usage quality dimension."""
        try:
            # Use existing tool usage gate
            result = self.tool_usage_gate.validate_content_generation(
                content, tool_usage, metadata.get('agent_type', 'unknown'))
            
            score = 1.0 if result["ok"] else 0.5
            confidence = 0.9
            evidence = [f"Vector queries: {tool_usage.get('vector_queries', 0)}",
                       f"Web searches: {tool_usage.get('web_searches', 0)}"]
            
            violations = []
            for violation_data in result.get("violations", []):
                violation = QualityGateViolation(
                    gate_name=violation_data["gate"],
                    expected=violation_data["expected"],
                    actual=violation_data["actual"],
                    severity=violation_data["severity"],
                    message=violation_data["message"]
                )
                violations.append(violation)
            
            recommendations = []
            if not result["ok"]:
                recommendations.append("Increase tool usage to meet minimum thresholds")
                if tool_usage.get('vector_queries', 0) == 0:
                    recommendations.append("Add vector database queries for context")
                if tool_usage.get('web_searches', 0) == 0:
                    recommendations.append("Add web searches for verification")
            
        except Exception as e:
            logger.error(f"Tool usage assessment failed: {e}")
            score = 0.0
            confidence = 0.1
            evidence = [f"Assessment failed: {e}"]
            violations = []
            recommendations = ["Fix tool usage assessment system"]
        
        return QualityScore(
            dimension=QualityDimension.TOOL_USAGE,
            score=score,
            confidence=confidence,
            evidence=evidence,
            violations=violations,
            recommendations=recommendations
        )
    
    def _assess_claim_verification(self, content: str, tool_usage: Dict, metadata: Dict) -> QualityScore:
        """Assess claim verification quality dimension."""
        verified_claims = tool_usage.get('verified_claims', [])
        
        # Extract claims from content (simple approach)
        import re
        claim_patterns = [r'\d+%', r'\d+\s*million', r'according to', r'study shows']
        potential_claims = []
        
        for pattern in claim_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            potential_claims.extend(matches)
        
        total_claims = len(potential_claims)
        verified_count = len(verified_claims)
        
        if total_claims == 0:
            score = 0.8  # No claims to verify
            evidence = ["No verifiable claims found in content"]
        else:
            verification_rate = verified_count / total_claims
            score = min(1.0, verification_rate + 0.3)  # Base score boost
            evidence = [f"Claims verified: {verified_count}/{total_claims}",
                       f"Verification rate: {verification_rate:.1%}"]
        
        violations = []
        if score < self.thresholds[QualityDimension.CLAIM_VERIFICATION].minimum_score:
            violations.append(QualityGateViolation(
                gate_name="claim_verification",
                expected=f"{self.thresholds[QualityDimension.CLAIM_VERIFICATION].minimum_score:.1%}",
                actual=f"{score:.1%}",
                severity="warning",
                message=f"Claim verification rate below threshold: {score:.1%}"
            ))
        
        recommendations = []
        if score < 0.7:
            recommendations.append("Increase claim verification coverage")
            recommendations.append("Add source validation for statistical claims")
        
        return QualityScore(
            dimension=QualityDimension.CLAIM_VERIFICATION,
            score=score,
            confidence=0.8,
            evidence=evidence,
            violations=violations,
            recommendations=recommendations
        )
    
    def _assess_information_freshness(self, content: str, metadata: Dict) -> QualityScore:
        """Assess information freshness quality dimension."""
        # Check for temporal indicators
        import re
        
        fresh_indicators = ['2024', '2025', 'recently', 'today', 'this week', 'this month', 'latest']
        dated_indicators = ['2022', '2021', '2020', 'last year', 'previously']
        
        fresh_count = sum(1 for indicator in fresh_indicators 
                         if indicator in content.lower())
        dated_count = sum(1 for indicator in dated_indicators 
                         if indicator in content.lower())
        
        # Calculate freshness score
        if fresh_count + dated_count == 0:
            score = 0.6  # Neutral when no temporal indicators
        else:
            freshness_ratio = fresh_count / (fresh_count + dated_count)
            score = 0.4 + (freshness_ratio * 0.6)  # Scale to 0.4-1.0
        
        evidence = [f"Fresh indicators: {fresh_count}",
                   f"Dated indicators: {dated_count}"]
        
        violations = []
        if score < self.thresholds[QualityDimension.INFORMATION_FRESHNESS].minimum_score:
            violations.append(QualityGateViolation(
                gate_name="information_freshness",
                expected=f"{self.thresholds[QualityDimension.INFORMATION_FRESHNESS].minimum_score:.1%}",
                actual=f"{score:.1%}",
                severity="warning",
                message="Content may contain outdated information"
            ))
        
        recommendations = []
        if score < 0.7:
            recommendations.append("Update content with recent developments")
            recommendations.append("Add current year references where appropriate")
        
        return QualityScore(
            dimension=QualityDimension.INFORMATION_FRESHNESS,
            score=score,
            confidence=0.7,
            evidence=evidence,
            violations=violations,
            recommendations=recommendations
        )
    
    def _assess_source_authority(self, tool_usage: Dict, metadata: Dict) -> QualityScore:
        """Assess source authority quality dimension."""
        search_providers = tool_usage.get('search_providers', [])
        verified_claims = tool_usage.get('verified_claims', [])
        
        # Calculate authority based on source diversity and quality
        provider_count = len(set(search_providers))
        
        # Authority scoring
        if provider_count >= 3:
            score = 0.9
        elif provider_count >= 2:
            score = 0.7
        elif provider_count >= 1:
            score = 0.5
        else:
            score = 0.3
        
        # Boost for verified claims with high-authority sources
        high_authority_claims = sum(1 for claim in verified_claims
                                  if hasattr(claim, 'confidence') and claim.confidence > 0.8)
        if high_authority_claims > 0:
            score += min(0.2, high_authority_claims * 0.05)
        
        score = min(1.0, score)
        
        evidence = [f"Search providers used: {provider_count}",
                   f"High-authority verified claims: {high_authority_claims}"]
        
        violations = []
        if score < self.thresholds[QualityDimension.SOURCE_AUTHORITY].minimum_score:
            violations.append(QualityGateViolation(
                gate_name="source_authority",
                expected=f"{self.thresholds[QualityDimension.SOURCE_AUTHORITY].minimum_score:.1%}",
                actual=f"{score:.1%}",
                severity="warning",
                message="Low source authority diversity"
            ))
        
        recommendations = []
        if score < 0.7:
            recommendations.append("Use multiple authoritative search providers")
            recommendations.append("Verify claims against high-authority sources")
        
        return QualityScore(
            dimension=QualityDimension.SOURCE_AUTHORITY,
            score=score,
            confidence=0.8,
            evidence=evidence,
            violations=violations,
            recommendations=recommendations
        )
    
    def _assess_content_completeness(self, content: str, metadata: Dict) -> QualityScore:
        """Assess content completeness quality dimension."""
        word_count = len(content.split())
        
        # Structure indicators
        has_headers = '##' in content or '#' in content
        has_links = '[' in content and '](' in content
        has_lists = '•' in content or content.count('-') > 2
        
        # Calculate completeness score
        structure_score = 0.0
        if has_headers:
            structure_score += 0.3
        if has_links:
            structure_score += 0.2
        if has_lists:
            structure_score += 0.2
        
        # Length score
        target_length = metadata.get('target_length', 1000)
        length_ratio = min(1.0, word_count / target_length)
        length_score = length_ratio * 0.3
        
        score = structure_score + length_score
        
        evidence = [f"Word count: {word_count}",
                   f"Has headers: {has_headers}",
                   f"Has links: {has_links}",
                   f"Has lists: {has_lists}"]
        
        violations = []
        if score < self.thresholds[QualityDimension.CONTENT_COMPLETENESS].minimum_score:
            violations.append(QualityGateViolation(
                gate_name="content_completeness",
                expected=f"{self.thresholds[QualityDimension.CONTENT_COMPLETENESS].minimum_score:.1%}",
                actual=f"{score:.1%}",
                severity="warning",
                message="Content lacks structure or sufficient detail"
            ))
        
        recommendations = []
        if not has_headers:
            recommendations.append("Add section headers for better structure")
        if not has_links:
            recommendations.append("Include relevant links and references")
        if word_count < target_length * 0.8:
            recommendations.append("Expand content to meet target length")
        
        return QualityScore(
            dimension=QualityDimension.CONTENT_COMPLETENESS,
            score=score,
            confidence=0.9,
            evidence=evidence,
            violations=violations,
            recommendations=recommendations
        )
    
    def _assess_technical_accuracy(self, content: str, metadata: Dict) -> QualityScore:
        """Assess technical accuracy quality dimension."""
        # Check for technical indicators
        technical_terms = ['API', 'algorithm', 'framework', 'implementation', 'architecture']
        code_blocks = content.count('```')
        technical_term_count = sum(1 for term in technical_terms 
                                 if term.lower() in content.lower())
        
        # Basic accuracy assessment
        if technical_term_count > 0 or code_blocks > 0:
            # Technical content requires higher standards
            score = 0.7  # Conservative score for technical content
            evidence = [f"Technical terms: {technical_term_count}",
                       f"Code blocks: {code_blocks // 2}"]  # Pair of ``` marks
        else:
            # Non-technical content
            score = 0.85
            evidence = ["Non-technical content - standard accuracy assumed"]
        
        violations = []
        recommendations = []
        
        # Check for potential accuracy issues
        accuracy_issues = []
        if 'TODO' in content or 'FIXME' in content:
            accuracy_issues.append("Content contains placeholder text")
            score -= 0.2
        
        if len(content.split('.')) < 5:  # Very short content
            accuracy_issues.append("Content too brief for accuracy assessment")
            score -= 0.1
        
        if accuracy_issues:
            violations.append(QualityGateViolation(
                gate_name="technical_accuracy",
                expected="High accuracy standards",
                actual="Potential accuracy issues detected",
                severity="warning",
                message="; ".join(accuracy_issues)
            ))
        
        if score < 0.8:
            recommendations.append("Review technical content for accuracy")
            recommendations.append("Validate code examples and technical claims")
        
        return QualityScore(
            dimension=QualityDimension.TECHNICAL_ACCURACY,
            score=max(0.0, score),
            confidence=0.6,  # Lower confidence for automated technical assessment
            evidence=evidence,
            violations=violations,
            recommendations=recommendations
        )
    
    def _assess_performance(self, metadata: Dict) -> QualityScore:
        """Assess performance quality dimension."""
        generation_time = metadata.get('generation_time_ms', 0)
        target_time = metadata.get('target_generation_time_ms', 30000)  # 30 seconds
        
        if generation_time == 0:
            score = 0.8  # No timing data available
            evidence = ["No performance data available"]
        else:
            # Performance score based on generation time
            time_ratio = generation_time / target_time
            if time_ratio <= 0.5:
                score = 1.0  # Excellent performance
            elif time_ratio <= 1.0:
                score = 0.8  # Good performance
            elif time_ratio <= 1.5:
                score = 0.6  # Acceptable performance
            else:
                score = 0.4  # Poor performance
            
            evidence = [f"Generation time: {generation_time:.0f}ms",
                       f"Target time: {target_time:.0f}ms",
                       f"Performance ratio: {time_ratio:.2f}"]
        
        violations = []
        if score < self.thresholds[QualityDimension.PERFORMANCE].minimum_score:
            violations.append(QualityGateViolation(
                gate_name="performance",
                expected=f"<{target_time}ms",
                actual=f"{generation_time:.0f}ms",
                severity="warning",
                message="Generation time exceeds target"
            ))
        
        recommendations = []
        if score < 0.7:
            recommendations.append("Optimize content generation performance")
            recommendations.append("Consider caching or parallel processing")
        
        return QualityScore(
            dimension=QualityDimension.PERFORMANCE,
            score=score,
            confidence=0.9,
            evidence=evidence,
            violations=violations,
            recommendations=recommendations
        )
    
    def _assess_accessibility(self, content: str, metadata: Dict) -> QualityScore:
        """Assess accessibility quality dimension."""
        # Basic accessibility checks
        accessibility_score = 0.0
        evidence = []
        
        # Check for proper heading structure
        import re
        h1_count = len(re.findall(r'^#\s+', content, re.MULTILINE))
        h2_count = len(re.findall(r'^##\s+', content, re.MULTILINE))
        
        if h1_count > 0:
            accessibility_score += 0.3
            evidence.append("Has main headings (H1)")
        
        if h2_count > 0:
            accessibility_score += 0.2
            evidence.append("Has section headings (H2)")
        
        # Check for descriptive link text
        link_matches = re.findall(r'\[([^\]]+)\]\([^)]+\)', content)
        descriptive_links = [link for link in link_matches 
                           if len(link) > 5 and link.lower() not in ['here', 'click', 'link']]
        
        if len(link_matches) > 0:
            link_quality = len(descriptive_links) / len(link_matches)
            accessibility_score += link_quality * 0.2
            evidence.append(f"Descriptive links: {len(descriptive_links)}/{len(link_matches)}")
        
        # Check for alt text indicators (if images referenced)
        if '![' in content:
            accessibility_score += 0.1
            evidence.append("Contains image alt text")
        
        # Check for list structure
        if '•' in content or content.count('-') > 2:
            accessibility_score += 0.1
            evidence.append("Uses list structure")
        
        # Readability (simple check)
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if avg_sentence_length < 20:  # Reasonable sentence length
            accessibility_score += 0.1
            evidence.append(f"Good readability (avg {avg_sentence_length:.1f} words/sentence)")
        
        score = min(1.0, accessibility_score)
        
        violations = []
        if score < self.thresholds[QualityDimension.ACCESSIBILITY].minimum_score:
            violations.append(QualityGateViolation(
                gate_name="accessibility",
                expected=f"{self.thresholds[QualityDimension.ACCESSIBILITY].minimum_score:.1%}",
                actual=f"{score:.1%}",
                severity="warning",
                message="Content accessibility could be improved"
            ))
        
        recommendations = []
        if h1_count == 0:
            recommendations.append("Add main heading (H1) for better structure")
        if len(link_matches) > len(descriptive_links):
            recommendations.append("Use more descriptive link text")
        if avg_sentence_length > 25:
            recommendations.append("Break up long sentences for better readability")
        
        return QualityScore(
            dimension=QualityDimension.ACCESSIBILITY,
            score=score,
            confidence=0.7,
            evidence=evidence,
            violations=violations,
            recommendations=recommendations
        )
    
    def _assess_template_compliance(self, content: str, metadata: Dict[str, Any]) -> QualityScore:
        """Assess template compliance using the template compliance validator."""
        try:
            from .template_compliance import TemplateComplianceValidator, ComplianceLevel
            
            template_type = metadata.get('template_type', 'general')
            validator = TemplateComplianceValidator(ComplianceLevel.STANDARD)
            report = validator.validate_content_compliance(content, template_type)
            
            violations = []
            evidence = []
            recommendations = []
            
            # Convert compliance report to quality score
            score = report.overall_score
            confidence = 0.9  # High confidence in template validation
            
            # Add evidence and violations
            if report.section_compliance:
                evidence.append(f"Section compliance: {report.section_compliance.compliance_percentage:.1%}")
                if report.section_compliance.missing_sections:
                    violations.append(QualityGateViolation(
                        gate_name="template_compliance",
                        expected="All required sections",
                        actual=f"Missing sections: {', '.join(report.section_compliance.missing_sections)}",
                        severity="critical",
                        message=f"Missing sections: {', '.join(report.section_compliance.missing_sections)}"
                    ))
                    recommendations.append("Add missing template sections")
            
            if report.word_count_compliance:
                evidence.append(f"Word count compliance: {report.word_count_compliance.overall_compliance:.1%}")
                if report.word_count_compliance.overall_compliance < 0.8:
                    violations.append(QualityGateViolation(
                        gate_name="template_compliance",
                        expected="80%+ word count compliance",
                        actual=f"{report.word_count_compliance.overall_compliance:.1%}",
                        severity="warning",
                        message=f"Word count below target: {report.word_count_compliance.total_words_actual}/{report.word_count_compliance.total_words_target}"
                    ))
                    recommendations.append("Adjust content length to meet template requirements")
            
            if not report.is_compliant:
                violations.append(QualityGateViolation(
                    gate_name="template_compliance",
                    expected="Template compliant content",
                    actual=f"Score: {score:.1f}/10.0",
                    severity="critical",
                    message=f"Overall template compliance failed: {score:.1f}/10.0"
                ))
                recommendations.append("Review template requirements and restructure content")
            
            return QualityScore(
                dimension=QualityDimension.TEMPLATE_COMPLIANCE,
                score=score / 10.0,  # Normalize to 0-1 scale
                confidence=confidence,
                evidence=evidence,
                violations=violations,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Template compliance assessment failed: {e}")
            return QualityScore(
                dimension=QualityDimension.TEMPLATE_COMPLIANCE,
                score=0.0,
                confidence=0.0,
                evidence=[],
                violations=[QualityGateViolation(
                    gate_name="template_compliance",
                    expected="Successful validation",
                    actual="Validation error",
                    severity="critical",
                    message=f"Template validation error: {e}"
                )],
                recommendations=["Fix template compliance validation system"]
            )
    
    def _assess_word_count_compliance(self, content: str, metadata: Dict[str, Any]) -> QualityScore:
        """Assess word count compliance against template targets."""
        try:
            word_count = len(content.split())
            target_words = metadata.get('target_word_count', 3000)
            template_type = metadata.get('template_type', 'general')
            
            # Calculate compliance based on acceptable range (80%-120% of target)
            min_acceptable = int(target_words * 0.8)
            max_acceptable = int(target_words * 1.2)
            
            if min_acceptable <= word_count <= max_acceptable:
                score = 1.0
                violation_severity = None
            elif word_count < min_acceptable:
                score = max(0.0, word_count / min_acceptable)
                violation_severity = "warning" if score > 0.6 else "critical"
            else:  # word_count > max_acceptable
                score = max(0.0, 1.0 - (word_count - max_acceptable) / target_words)
                violation_severity = "warning"
            
            violations = []
            recommendations = []
            evidence = [f"Word count: {word_count}/{target_words} (target)"]
            
            if violation_severity:
                violations.append(QualityGateViolation(
                    gate_name="word_count_compliance",
                    expected=f"{min_acceptable}-{max_acceptable} words",
                    actual=f"{word_count} words",
                    severity=violation_severity,
                    message=f"Word count {word_count} outside acceptable range {min_acceptable}-{max_acceptable}"
                ))
                if word_count < min_acceptable:
                    recommendations.append("Expand content to meet minimum word count requirements")
                else:
                    recommendations.append("Consider reducing content length or splitting into multiple sections")
            
            return QualityScore(
                dimension=QualityDimension.WORD_COUNT_COMPLIANCE,
                score=score,
                confidence=0.95,
                evidence=evidence,
                violations=violations,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Word count compliance assessment failed: {e}")
            return QualityScore(
                dimension=QualityDimension.WORD_COUNT_COMPLIANCE,
                score=0.0,
                confidence=0.0,
                evidence=[],
                violations=[QualityGateViolation(
                    gate_name="word_count_compliance",
                    expected="Successful validation",
                    actual="Validation error",
                    severity="critical",
                    message=f"Word count validation error: {e}"
                )],
                recommendations=["Fix word count validation system"]
            )
    
    def _assess_section_compliance(self, content: str, metadata: Dict[str, Any]) -> QualityScore:
        """Assess section structure compliance against template requirements."""
        try:
            template_type = metadata.get('template_type', 'general')
            
            # Basic section detection using headers
            import re
            headers = re.findall(r'^#+\s+(.+)$|^\*\*(.+)\*\*$', content, re.MULTILINE)
            section_count = len([h for h in headers if any(h)])
            
            # Expected sections based on template type
            expected_sections = {
                'technical_deep_dive': 6,
                'tutorial_guide': 6,
                'trend_analysis': 6,
                'product_review': 6,
                'research_summary': 5,
                'general': 4
            }
            
            expected_count = expected_sections.get(template_type, 4)
            compliance_ratio = min(1.0, section_count / expected_count)
            
            violations = []
            recommendations = []
            evidence = [f"Sections found: {section_count}/{expected_count} (expected)"]
            
            if compliance_ratio < 0.9:
                severity = "critical" if compliance_ratio < 0.7 else "warning"
                violations.append(QualityGateViolation(
                    gate_name="section_compliance",
                    expected=f"{expected_count} sections",
                    actual=f"{section_count} sections",
                    severity=severity,
                    message=f"Insufficient sections: {section_count}/{expected_count}"
                ))
                recommendations.append("Add missing sections according to template structure")
            
            return QualityScore(
                dimension=QualityDimension.SECTION_COMPLIANCE,
                score=compliance_ratio,
                confidence=0.8,
                evidence=evidence,
                violations=violations,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Section compliance assessment failed: {e}")
            return QualityScore(
                dimension=QualityDimension.SECTION_COMPLIANCE,
                score=0.0,
                confidence=0.0,
                evidence=[],
                violations=[QualityGateViolation(
                    gate_name="section_compliance",
                    expected="Successful validation",
                    actual="Validation error",
                    severity="critical",
                    message=f"Section validation error: {e}"
                )],
                recommendations=["Fix section compliance validation system"]
            )
    
    def _log_violations(self, violations: List[QualityGateViolation], level: EnforcementLevel):
        """Log quality gate violations."""
        if not violations:
            return
        
        for violation in violations:
            if violation.severity == "critical":
                logger.error(f"CRITICAL Quality Violation [{level.value}]: {violation.message}")
            elif violation.severity == "error":
                logger.warning(f"ERROR Quality Violation [{level.value}]: {violation.message}")
            else:
                logger.info(f"WARNING Quality Violation [{level.value}]: {violation.message}")
    
    def _calculate_dynamic_quality_score(self, dimension_scores: Dict[QualityDimension, QualityScore], 
                                       complexity: float, metadata: Dict[str, Any]) -> float:
        """Calculate dynamic overall quality score with complexity-based weighting."""
        # Base dimension weights
        base_weights = {
            QualityDimension.TEMPLATE_COMPLIANCE: 0.20,
            QualityDimension.WORD_COUNT_COMPLIANCE: 0.15,
            QualityDimension.SECTION_COMPLIANCE: 0.15,
            QualityDimension.CONTENT_COMPLETENESS: 0.12,
            QualityDimension.TECHNICAL_ACCURACY: 0.10,
            QualityDimension.TOOL_USAGE: 0.08,
            QualityDimension.CLAIM_VERIFICATION: 0.07,
            QualityDimension.INFORMATION_FRESHNESS: 0.05,
            QualityDimension.SOURCE_AUTHORITY: 0.05,
            QualityDimension.ACCESSIBILITY: 0.02,
            QualityDimension.PERFORMANCE: 0.01
        }
        
        # Complexity-based weight adjustments
        adjusted_weights = self._adjust_weights_for_complexity(base_weights, complexity, metadata)
        
        # Calculate weighted score
        weighted_score = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            weight = adjusted_weights.get(dimension, 0.01)
            
            # Apply confidence weighting
            confidence_factor = score.confidence
            effective_weight = weight * confidence_factor
            
            weighted_score += score.score * effective_weight
            total_weight += effective_weight
        
        # Normalize to 0-1 range
        if total_weight > 0:
            base_score = weighted_score / total_weight
        else:
            base_score = 0.0
        
        # Apply complexity bonus/penalty
        complexity_adjustment = self._calculate_complexity_adjustment(complexity, dimension_scores, metadata)
        final_score = min(1.0, max(0.0, base_score + complexity_adjustment))
        
        logger.info(f"Dynamic quality score: base={base_score:.3f}, complexity_adj={complexity_adjustment:.3f}, final={final_score:.3f}")
        return final_score
    
    def _adjust_weights_for_complexity(self, base_weights: dict, complexity: float, metadata: dict) -> dict:
        """Adjust quality dimension weights based on content complexity."""
        adjusted_weights = base_weights.copy()
        template_type = metadata.get('template_type', 'general')
        
        # High complexity content adjustments
        if complexity > 0.8:
            # Prioritize structural compliance for complex content
            adjusted_weights[QualityDimension.TEMPLATE_COMPLIANCE] *= 1.2
            adjusted_weights[QualityDimension.SECTION_COMPLIANCE] *= 1.15
            # Reduce emphasis on strict word count compliance
            adjusted_weights[QualityDimension.WORD_COUNT_COMPLIANCE] *= 0.85
            
        # Technical content adjustments
        if template_type == 'technical_deep_dive' and complexity > 0.6:
            adjusted_weights[QualityDimension.TECHNICAL_ACCURACY] *= 1.5
            adjusted_weights[QualityDimension.CLAIM_VERIFICATION] *= 1.3
            adjusted_weights[QualityDimension.SOURCE_AUTHORITY] *= 1.2
            
        # Educational content adjustments
        elif template_type in ['tutorial_guide', 'research_summary']:
            adjusted_weights[QualityDimension.CONTENT_COMPLETENESS] *= 1.3
            adjusted_weights[QualityDimension.ACCESSIBILITY] *= 1.5
            
        # Simple content adjustments
        elif complexity < 0.4:
            # Higher standards for simple content
            adjusted_weights[QualityDimension.WORD_COUNT_COMPLIANCE] *= 1.2
            adjusted_weights[QualityDimension.CONTENT_COMPLETENESS] *= 1.1
        
        # Normalize weights to sum to 1.0
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _calculate_complexity_adjustment(self, complexity: float, 
                                       dimension_scores: Dict[QualityDimension, QualityScore], 
                                       metadata: dict) -> float:
        """Calculate quality score adjustment based on complexity and achievement."""
        adjustment = 0.0
        
        # Complexity achievement bonus/penalty
        template_compliance_score = dimension_scores.get(QualityDimension.TEMPLATE_COMPLIANCE)
        word_count_score = dimension_scores.get(QualityDimension.WORD_COUNT_COMPLIANCE)
        
        if template_compliance_score and word_count_score:
            # Bonus for achieving good compliance on complex content
            if complexity > 0.8:
                if template_compliance_score.score > 0.8 and word_count_score.score > 0.85:
                    adjustment += 0.1  # Significant bonus for complex content compliance
                elif template_compliance_score.score > 0.6 and word_count_score.score > 0.75:
                    adjustment += 0.05  # Moderate bonus
                    
            # Penalty for poor performance on simple content
            elif complexity < 0.4:
                if template_compliance_score.score < 0.7 or word_count_score.score < 0.8:
                    adjustment -= 0.05  # Penalty for underperforming on simple content
        
        # Template-specific adjustments
        template_type = metadata.get('template_type', 'general')
        if template_type == 'technical_deep_dive':
            technical_score = dimension_scores.get(QualityDimension.TECHNICAL_ACCURACY)
            if technical_score and technical_score.score > 0.9:
                adjustment += 0.03  # Bonus for high technical accuracy
                
        # Overall performance consistency bonus
        scores = [score.score for score in dimension_scores.values()]
        if scores:
            score_std = (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5
            if score_std < 0.15:  # Low variance = consistent quality
                adjustment += 0.02
        
        return min(0.15, max(-0.1, adjustment))  # Cap adjustment at ±15%/10%


class AdvancedQualityGate:
    """Advanced quality validation with multiple dimensions."""
    
    def __init__(self, enforcement_level: str = "advisory"):
        """Initialize advanced quality gate."""
        self.configurable_gate = ConfigurableQualityGate(enforcement_level)
        logger.info("Advanced quality gate initialized")
    
    def validate_comprehensive_quality(self, content: str, metadata: Dict[str, Any]) -> QualityReport:
        """Comprehensive quality assessment."""
        return self.configurable_gate.validate_with_level(content, metadata.get('tool_usage', {}), metadata)
    
    def get_quality_summary(self, report: QualityReport) -> str:
        """Generate human-readable quality summary."""
        summary_lines = [
            f"Quality Assessment Summary",
            f"========================",
            f"Overall Score: {report.overall_score:.1%}",
            f"Total Violations: {report.total_violations}",
            f"Critical Violations: {report.critical_violations}",
            f"Enforcement Level: {report.enforcement_level.value.title()}",
            f"Content Blocked: {'Yes' if report.content_blocked else 'No'}",
            f"Processing Time: {report.processing_time_ms:.1f}ms",
            "",
            "Dimension Scores:",
        ]
        
        for dimension, score in report.dimension_scores.items():
            status = "✓" if score.score >= 0.7 else "⚠" if score.score >= 0.5 else "✗"
            summary_lines.append(f"  {status} {dimension.value.replace('_', ' ').title()}: {score.score:.1%}")
        
        if report.total_violations > 0:
            summary_lines.extend(["", "Top Recommendations:"])
            all_recommendations = []
            for score in report.dimension_scores.values():
                all_recommendations.extend(score.recommendations)
            
            for i, rec in enumerate(all_recommendations[:5], 1):
                summary_lines.append(f"  {i}. {rec}")
        
        return "\n".join(summary_lines)
    
    # Phase 1: Content Expansion Quality Assessment Methods
    
    def _assess_content_expansion_quality(self, content: str, metadata: Dict[str, Any]) -> QualityScore:
        """Assess the quality of content expansion operations."""
        try:
            expansion_metrics = metadata.get('expansion_metrics', {})
            
            # Base score from expansion quality metrics
            quality_score = expansion_metrics.get('quality_score', 0.0)
            
            # Assess technical accuracy preservation
            technical_terms_preserved = self._check_technical_accuracy_preservation(content)
            
            # Assess content coherence after expansion
            coherence_score = self._assess_content_coherence(content)
            
            # Calculate overall expansion quality
            overall_score = (quality_score * 0.5 + technical_terms_preserved * 0.3 + coherence_score * 0.2)
            
            evidence = [
                f"Expansion quality score: {quality_score:.2f}",
                f"Technical accuracy preserved: {technical_terms_preserved:.2f}",
                f"Content coherence: {coherence_score:.2f}"
            ]
            
            violations = []
            recommendations = []
            
            threshold = self.thresholds[QualityDimension.CONTENT_EXPANSION_QUALITY].minimum_score
            if overall_score < threshold:
                violations.append(QualityGateViolation(
                    gate_name="content_expansion_quality",
                    expected=f">= {threshold}",
                    actual=f"{overall_score:.3f}",
                    severity="warning",
                    description="Content expansion quality below threshold"
                ))
                recommendations.append("Review expansion quality and ensure technical accuracy is maintained")
            
            return QualityScore(
                dimension=QualityDimension.CONTENT_EXPANSION_QUALITY,
                score=overall_score,
                confidence=0.8,
                evidence=evidence,
                violations=violations,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Content expansion quality assessment failed: {e}")
            return self._create_fallback_quality_score(QualityDimension.CONTENT_EXPANSION_QUALITY)
    
    def _assess_expansion_target_achievement(self, content: str, metadata: Dict[str, Any]) -> QualityScore:
        """Assess achievement of expansion word count targets."""
        try:
            expansion_metrics = metadata.get('expansion_metrics', {})
            target_achievement = expansion_metrics.get('target_achievement', 0.0)
            
            # Get actual word counts
            original_words = expansion_metrics.get('original_words', 0)
            final_words = expansion_metrics.get('final_words', len(content.split()))
            target_words = metadata.get('target_word_count', 3000)
            
            # Calculate achievement ratio
            if target_words > 0:
                achievement_ratio = final_words / target_words
            else:
                achievement_ratio = target_achievement
            
            # Score based on how close to target (optimal: 85-100%)
            if 0.85 <= achievement_ratio <= 1.0:
                score = 1.0
            elif 0.75 <= achievement_ratio < 0.85:
                score = 0.8
            elif 0.65 <= achievement_ratio < 0.75:
                score = 0.6
            else:
                score = max(0.0, achievement_ratio)
            
            evidence = [
                f"Target achievement: {achievement_ratio:.1%}",
                f"Word count: {final_words}/{target_words}",
                f"Expansion achieved: {final_words - original_words} words"
            ]
            
            violations = []
            recommendations = []
            
            threshold = self.thresholds[QualityDimension.EXPANSION_TARGET_ACHIEVEMENT].minimum_score
            if score < threshold:
                violations.append(QualityGateViolation(
                    gate_name="expansion_target_achievement",
                    expected=f">= {threshold:.1%}",
                    actual=f"{achievement_ratio:.1%}",
                    severity="warning",
                    description="Word count target achievement below threshold"
                ))
                recommendations.append(f"Increase content to reach {threshold:.1%} of target word count")
            
            return QualityScore(
                dimension=QualityDimension.EXPANSION_TARGET_ACHIEVEMENT,
                score=score,
                confidence=0.9,
                evidence=evidence,
                violations=violations,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Expansion target achievement assessment failed: {e}")
            return self._create_fallback_quality_score(QualityDimension.EXPANSION_TARGET_ACHIEVEMENT)
    
    def _assess_expansion_technical_accuracy(self, content: str, metadata: Dict[str, Any]) -> QualityScore:
        """Assess technical accuracy preservation during expansion."""
        try:
            # Check for technical accuracy indicators
            technical_indicators = [
                'implementation', 'architecture', 'algorithm', 'framework',
                'API', 'library', 'function', 'method', 'class', 'interface'
            ]
            
            technical_density = sum(
                content.lower().count(indicator) for indicator in technical_indicators
            ) / max(1, len(content.split()) / 100)  # Per 100 words
            
            # Check for code examples and technical explanations
            code_blocks = content.count('```')
            technical_explanations = len([
                sent for sent in content.split('.')
                if any(indicator in sent.lower() for indicator in technical_indicators)
            ])
            
            # Calculate technical accuracy score
            if technical_density >= 3.0:  # High technical density
                density_score = 1.0
            elif technical_density >= 2.0:
                density_score = 0.8
            elif technical_density >= 1.0:
                density_score = 0.6
            else:
                density_score = 0.4
            
            code_score = min(1.0, code_blocks / 5) if code_blocks > 0 else 0.0
            explanation_score = min(1.0, technical_explanations / 10)
            
            overall_score = (density_score * 0.5 + code_score * 0.25 + explanation_score * 0.25)
            
            evidence = [
                f"Technical density: {technical_density:.1f} terms per 100 words",
                f"Code blocks: {code_blocks}",
                f"Technical explanations: {technical_explanations}"
            ]
            
            violations = []
            recommendations = []
            
            threshold = self.thresholds[QualityDimension.EXPANSION_TECHNICAL_ACCURACY].minimum_score
            if overall_score < threshold:
                violations.append(QualityGateViolation(
                    gate_name="expansion_technical_accuracy",
                    expected=f">= {threshold}",
                    actual=f"{overall_score:.3f}",
                    severity="warning",
                    description="Technical accuracy below threshold in expanded content"
                ))
                recommendations.append("Ensure expanded content maintains technical depth and accuracy")
            
            return QualityScore(
                dimension=QualityDimension.EXPANSION_TECHNICAL_ACCURACY,
                score=overall_score,
                confidence=0.8,
                evidence=evidence,
                violations=violations,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Expansion technical accuracy assessment failed: {e}")
            return self._create_fallback_quality_score(QualityDimension.EXPANSION_TECHNICAL_ACCURACY)
    
    # Phase 2: Mobile Optimization Quality Assessment Methods
    
    def _assess_mobile_readability(self, content: str, metadata: Dict[str, Any]) -> QualityScore:
        """Assess mobile readability optimization quality."""
        try:
            mobile_metrics = metadata.get('mobile_metrics', {})
            readability_score = mobile_metrics.get('readability_score', 0.0)
            
            # If no mobile metrics, perform basic readability assessment
            if not mobile_metrics:
                readability_score = self._basic_mobile_readability_assessment(content)
            
            evidence = [
                f"Mobile readability score: {readability_score:.2f}",
                f"Average sentence length: {self._calculate_avg_sentence_length(content):.1f}",
                f"Paragraph count: {len([p for p in content.split('\\n\\n') if p.strip()])}"
            ]
            
            violations = []
            recommendations = []
            
            threshold = self.thresholds[QualityDimension.MOBILE_READABILITY].minimum_score
            if readability_score < threshold:
                violations.append(QualityGateViolation(
                    gate_name="mobile_readability",
                    expected=f">= {threshold}",
                    actual=f"{readability_score:.3f}",
                    severity="warning",
                    description="Mobile readability below optimization threshold"
                ))
                recommendations.append("Optimize content structure and language for mobile reading")
            
            return QualityScore(
                dimension=QualityDimension.MOBILE_READABILITY,
                score=readability_score,
                confidence=0.8,
                evidence=evidence,
                violations=violations,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Mobile readability assessment failed: {e}")
            return self._create_fallback_quality_score(QualityDimension.MOBILE_READABILITY)
    
    def _assess_mobile_typography(self, content: str, metadata: Dict[str, Any]) -> QualityScore:
        """Assess mobile typography optimization quality."""
        try:
            mobile_metrics = metadata.get('mobile_metrics', {})
            typography_score = mobile_metrics.get('typography_compliance', 0.0)
            
            # Basic typography assessment if no metrics
            if not mobile_metrics:
                typography_score = self._basic_typography_assessment(content)
            
            evidence = [
                f"Typography compliance: {typography_score:.2f}",
                f"Heading structure present: {'Yes' if content.count('#') > 0 else 'No'}",
                f"Code blocks formatted: {content.count('```') // 2}"
            ]
            
            violations = []
            recommendations = []
            
            threshold = self.thresholds[QualityDimension.MOBILE_TYPOGRAPHY].minimum_score
            if typography_score < threshold:
                violations.append(QualityGateViolation(
                    gate_name="mobile_typography",
                    expected=f">= {threshold}",
                    actual=f"{typography_score:.3f}",
                    severity="warning",
                    description="Mobile typography optimization below threshold"
                ))
                recommendations.append("Improve heading hierarchy and mobile-friendly formatting")
            
            return QualityScore(
                dimension=QualityDimension.MOBILE_TYPOGRAPHY,
                score=typography_score,
                confidence=0.7,
                evidence=evidence,
                violations=violations,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Mobile typography assessment failed: {e}")
            return self._create_fallback_quality_score(QualityDimension.MOBILE_TYPOGRAPHY)
    
    def _assess_mobile_structure_optimization(self, content: str, metadata: Dict[str, Any]) -> QualityScore:
        """Assess mobile content structure optimization."""
        try:
            mobile_metrics = metadata.get('mobile_metrics', {})
            structure_score = mobile_metrics.get('structure_optimization', 0.0)
            
            # Basic structure assessment if no metrics
            if not mobile_metrics:
                structure_score = self._basic_structure_assessment(content)
            
            evidence = [
                f"Structure optimization: {structure_score:.2f}",
                f"Lists present: {content.count('- ') + content.count('* ')}",
                f"Visual breaks: {content.count('\\n\\n')}"
            ]
            
            violations = []
            recommendations = []
            
            threshold = self.thresholds[QualityDimension.MOBILE_STRUCTURE_OPTIMIZATION].minimum_score
            if structure_score < threshold:
                violations.append(QualityGateViolation(
                    gate_name="mobile_structure_optimization",
                    expected=f">= {threshold}",
                    actual=f"{structure_score:.3f}",
                    severity="warning",
                    description="Mobile structure optimization below threshold"
                ))
                recommendations.append("Improve content structure with better use of lists and visual breaks")
            
            return QualityScore(
                dimension=QualityDimension.MOBILE_STRUCTURE_OPTIMIZATION,
                score=structure_score,
                confidence=0.7,
                evidence=evidence,
                violations=violations,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Mobile structure optimization assessment failed: {e}")
            return self._create_fallback_quality_score(QualityDimension.MOBILE_STRUCTURE_OPTIMIZATION)
    
    def _assess_mobile_performance(self, content: str, metadata: Dict[str, Any]) -> QualityScore:
        """Assess mobile performance optimization."""
        try:
            mobile_metrics = metadata.get('mobile_metrics', {})
            performance_score = mobile_metrics.get('performance_score', 0.0)
            load_time = mobile_metrics.get('load_time_estimate', 0.0)
            
            # Basic performance assessment if no metrics
            if not mobile_metrics:
                performance_score = self._basic_performance_assessment(content)
                load_time = self._estimate_load_time(content)
            
            evidence = [
                f"Performance score: {performance_score:.2f}",
                f"Estimated load time: {load_time:.1f}s",
                f"Content length: {len(content.split())} words"
            ]
            
            violations = []
            recommendations = []
            
            threshold = self.thresholds[QualityDimension.MOBILE_PERFORMANCE].minimum_score
            if performance_score < threshold:
                violations.append(QualityGateViolation(
                    gate_name="mobile_performance",
                    expected=f">= {threshold}",
                    actual=f"{performance_score:.3f}",
                    severity="warning",
                    description="Mobile performance optimization below threshold"
                ))
                recommendations.append("Optimize content for faster mobile loading")
            
            return QualityScore(
                dimension=QualityDimension.MOBILE_PERFORMANCE,
                score=performance_score,
                confidence=0.6,
                evidence=evidence,
                violations=violations,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Mobile performance assessment failed: {e}")
            return self._create_fallback_quality_score(QualityDimension.MOBILE_PERFORMANCE)
    
    def _assess_mobile_accessibility(self, content: str, metadata: Dict[str, Any]) -> QualityScore:
        """Assess mobile accessibility optimization."""
        try:
            mobile_metrics = metadata.get('mobile_metrics', {})
            accessibility_score = mobile_metrics.get('accessibility_score', 0.0)
            
            # Basic accessibility assessment if no metrics
            if not mobile_metrics:
                accessibility_score = self._basic_accessibility_assessment(content)
            
            evidence = [
                f"Accessibility score: {accessibility_score:.2f}",
                f"Images with alt text: {self._count_images_with_alt(content)}",
                f"Heading structure: {'Proper' if self._check_heading_structure(content) else 'Needs improvement'}"
            ]
            
            violations = []
            recommendations = []
            
            threshold = self.thresholds[QualityDimension.MOBILE_ACCESSIBILITY].minimum_score
            if accessibility_score < threshold:
                violations.append(QualityGateViolation(
                    gate_name="mobile_accessibility",
                    expected=f">= {threshold}",
                    actual=f"{accessibility_score:.3f}",
                    severity="warning",
                    description="Mobile accessibility below threshold"
                ))
                recommendations.append("Improve accessibility with better alt text and heading structure")
            
            return QualityScore(
                dimension=QualityDimension.MOBILE_ACCESSIBILITY,
                score=accessibility_score,
                confidence=0.8,
                evidence=evidence,
                violations=violations,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Mobile accessibility assessment failed: {e}")
            return self._create_fallback_quality_score(QualityDimension.MOBILE_ACCESSIBILITY)
    
    # Helper methods for assessments
    
    def _check_technical_accuracy_preservation(self, content: str) -> float:
        """Check if technical accuracy is preserved in content."""
        technical_patterns = [
            r'\\b(?:API|SDK|framework|library|algorithm|implementation)\\b',
            r'\\b(?:function|method|class|interface|module)\\b',
            r'\\b(?:performance|optimization|scalability|architecture)\\b'
        ]
        
        total_matches = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in technical_patterns)
        words = len(content.split())
        
        # Score based on technical term density
        density = total_matches / max(1, words / 100)
        return min(1.0, density / 3.0)  # Normalize to 3 terms per 100 words = 1.0
    
    def _assess_content_coherence(self, content: str) -> float:
        """Assess content coherence and flow."""
        paragraphs = [p.strip() for p in content.split('\\n\\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return 0.8  # Single paragraph gets decent score
        
        # Check for transition words and coherence indicators
        transition_words = ['however', 'therefore', 'furthermore', 'additionally', 'moreover', 'consequently']
        coherence_score = 0.0
        
        for paragraph in paragraphs:
            if any(word in paragraph.lower() for word in transition_words):
                coherence_score += 0.2
        
        # Normalize and ensure minimum baseline
        return max(0.6, min(1.0, coherence_score + 0.6))
    
    def _basic_mobile_readability_assessment(self, content: str) -> float:
        """Basic mobile readability assessment."""
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if not sentences:
            return 0.5
        
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Score based on sentence length (optimal: 10-20 words)
        if 10 <= avg_sentence_length <= 20:
            return 0.9
        elif 8 <= avg_sentence_length <= 25:
            return 0.7
        else:
            return max(0.3, 1.0 - abs(avg_sentence_length - 15) / 20)
    
    def _calculate_avg_sentence_length(self, content: str) -> float:
        """Calculate average sentence length."""
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if not sentences:
            return 0.0
        return sum(len(s.split()) for s in sentences) / len(sentences)
    
    def _basic_typography_assessment(self, content: str) -> float:
        """Basic typography assessment."""
        score = 0.0
        
        # Check for heading structure
        if content.count('#') > 0:
            score += 0.3
        
        # Check for proper spacing
        if content.count('\\n\\n') > 2:
            score += 0.3
        
        # Check for formatted elements
        if content.count('```') > 0 or content.count('**') > 0:
            score += 0.2
        
        # Check for lists
        if content.count('- ') > 0 or content.count('* ') > 0:
            score += 0.2
        
        return min(1.0, score + 0.3)  # Baseline + enhancements
    
    def _basic_structure_assessment(self, content: str) -> float:
        """Basic structure assessment."""
        paragraphs = [p.strip() for p in content.split('\\n\\n') if p.strip()]
        if not paragraphs:
            return 0.0
        
        # Check paragraph length distribution
        long_paragraphs = [p for p in paragraphs if len(p.split()) > 100]
        structure_score = max(0.0, 1.0 - len(long_paragraphs) / len(paragraphs))
        
        # Boost for good use of lists and formatting
        if content.count('- ') + content.count('* ') > 2:
            structure_score = min(1.0, structure_score + 0.2)
        
        return structure_score
    
    def _basic_performance_assessment(self, content: str) -> float:
        """Basic performance assessment."""
        word_count = len(content.split())
        
        # Score based on content length (optimal: <5000 words)
        if word_count <= 3000:
            return 0.9
        elif word_count <= 5000:
            return 0.7
        else:
            return max(0.3, 1.0 - (word_count - 5000) / 5000)
    
    def _estimate_load_time(self, content: str) -> float:
        """Estimate content load time."""
        base_time = 1.0  # 1 second base
        word_time = len(content.split()) / 1000 * 0.5  # 0.5s per 1000 words
        return base_time + word_time
    
    def _basic_accessibility_assessment(self, content: str) -> float:
        """Basic accessibility assessment."""
        score = 0.0
        
        # Check heading structure
        h1_count = content.count('# ')
        if h1_count == 1:
            score += 0.3
        
        # Check for alt text in images
        images = re.findall(r'!\\[.*?\\]\\(.*?\\)', content)
        images_with_alt = re.findall(r'!\\[.+\\]\\(.*?\\)', content)
        if images:
            score += (len(images_with_alt) / len(images)) * 0.4
        else:
            score += 0.4  # No images = no accessibility issues
        
        # Check for descriptive links
        links = re.findall(r'\\[(.+?)\\]\\(.*?\\)', content)
        poor_links = [link for link in links if link.lower() in ['here', 'click here', 'read more']]
        if links:
            score += max(0.0, (1.0 - len(poor_links) / len(links)) * 0.3)
        else:
            score += 0.3
        
        return min(1.0, score)
    
    def _count_images_with_alt(self, content: str) -> int:
        """Count images with alt text."""
        return len(re.findall(r'!\\[.+\\]\\(.*?\\)', content))
    
    def _check_heading_structure(self, content: str) -> bool:
        """Check if heading structure is proper."""
        h1_count = content.count('# ')
        h2_count = content.count('## ')
        return h1_count == 1 and h2_count >= 1
    
    def _create_fallback_quality_score(self, dimension: QualityDimension) -> QualityScore:
        """Create fallback quality score for error cases."""
        return QualityScore(
            dimension=dimension,
            score=0.5,
            confidence=0.1,
            evidence=["Assessment failed - using fallback score"],
            violations=[],
            recommendations=["Manual review recommended due to assessment failure"]
        )


# Export main classes
__all__ = [
    'AdvancedQualityGate',
    'ConfigurableQualityGate', 
    'QualityReport',
    'QualityScore',
    'QualityDimension',
    'EnforcementLevel',
    'QualityThreshold'
]