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

from src.core.quality_gates import QualityGateViolation, ToolUsageQualityGate
from src.core.tool_cache import get_tool_cache

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
        """Initialize configurable quality gate."""
        self.enforcement_level = EnforcementLevel(enforcement_level)
        self.tool_usage_gate = ToolUsageQualityGate()
        self.cache = get_tool_cache()
        
        # Default quality thresholds
        self.thresholds = {
            QualityDimension.TOOL_USAGE: QualityThreshold(
                QualityDimension.TOOL_USAGE, 0.7, self.enforcement_level),
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
                QualityDimension.ACCESSIBILITY, 0.8, EnforcementLevel.ENFORCING)
        }
        
        logger.info(f"Configurable quality gate initialized with {enforcement_level} enforcement")
    
    def validate_with_level(self, content: str, tool_usage: Dict, 
                          metadata: Dict[str, Any] = None,
                          level: Optional[str] = None) -> QualityReport:
        """Validate content with specified enforcement level."""
        start_time = time.time()
        
        if level:
            current_level = EnforcementLevel(level)
        else:
            current_level = self.enforcement_level
        
        if metadata is None:
            metadata = {}
        
        # Assess all quality dimensions
        dimension_scores = {}
        all_violations = []
        
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
        
        # Calculate overall score
        overall_score = sum(score.score for score in dimension_scores.values()) / len(dimension_scores)
        
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
            content_metadata=metadata
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