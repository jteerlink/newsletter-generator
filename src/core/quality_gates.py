"""
Quality Gates for Tool Usage Enforcement

Implements quality gates as specified in PRD FR4.1 to ensure minimum tool usage
thresholds are met during content generation. Phase 1: Advisory mode with
configurable enforcement.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.constants import (
    MIN_VECTOR_QUERIES,
    MIN_WEB_SEARCHES,
    TOOL_ENFORCEMENT_ENABLED
)

logger = logging.getLogger(__name__)


@dataclass
class QualityGateViolation:
    """Represents a quality gate violation."""
    gate_name: str
    expected: Any
    actual: Any
    severity: str  # "warning", "error", "critical"
    message: str


class QualityGateError(Exception):
    """Exception raised when quality gates fail."""
    
    def __init__(self, message: str, violations: List[QualityGateViolation] = None):
        super().__init__(message)
        self.violations = violations or []


class ToolUsageQualityGate:
    """Quality gate that validates tool usage metrics against thresholds."""
    
    def __init__(self):
        self.enabled = TOOL_ENFORCEMENT_ENABLED
        self.min_vector_queries = MIN_VECTOR_QUERIES
        self.min_web_searches = MIN_WEB_SEARCHES
    
    def validate_content_generation(self, 
                                   content: str, 
                                   tool_usage: Dict[str, Any],
                                   agent_type: str = "unknown") -> Dict[str, Any]:
        """Validate content meets tool usage requirements.
        
        Args:
            content: Generated content to validate
            tool_usage: Dictionary containing tool usage metrics
            agent_type: Type of agent for context-specific validation
            
        Returns:
            Dictionary with validation results, issues, and warnings
            
        Raises:
            QualityGateError: If validation fails and enforcement is enabled
        """
        violations = []
        vector_queries = int(tool_usage.get('vector_queries', 0))
        web_searches = int(tool_usage.get('web_searches', 0))
        
        # Check minimum tool consultations
        if vector_queries < self.min_vector_queries:
            violations.append(QualityGateViolation(
                gate_name="vector_queries",
                expected=self.min_vector_queries,
                actual=vector_queries,
                severity="error" if self.enabled else "warning",
                message=f"Insufficient vector database consultation: {vector_queries} < {self.min_vector_queries}"
            ))
        
        if web_searches < self.min_web_searches:
            violations.append(QualityGateViolation(
                gate_name="web_searches",
                expected=self.min_web_searches,
                actual=web_searches,
                severity="error" if self.enabled else "warning",
                message=f"Insufficient web search validation: {web_searches} < {self.min_web_searches}"
            ))
        
        # Validate claim verification (if content contains claims)
        claims = self.extract_claims(content)
        if claims:
            verified_claims = tool_usage.get('verified_claims', [])
            verification_rate = len(verified_claims) / len(claims) if claims else 0
            min_verification_rate = 0.7  # 70% minimum from PRD
            
            if verification_rate < min_verification_rate:
                violations.append(QualityGateViolation(
                    gate_name="claim_verification",
                    expected=f"{min_verification_rate:.1%}",
                    actual=f"{verification_rate:.1%}",
                    severity="warning",
                    message=f"Low claim verification rate: {verification_rate:.1%} < {min_verification_rate:.1%}"
                ))
        
        # Log violations
        for violation in violations:
            if violation.severity == "critical":
                logger.error(f"Quality Gate CRITICAL: {violation.message}")
            elif violation.severity == "error":
                logger.warning(f"Quality Gate ERROR: {violation.message}")
            else:
                logger.info(f"Quality Gate WARNING: {violation.message}")
        
        # Prepare results
        issues = [v.message for v in violations if v.severity in ["error", "critical"]]
        warnings = [v.message for v in violations if v.severity == "warning"]
        
        # Determine if we should block or warn
        if violations and self.enabled:
            error_violations = [v for v in violations if v.severity in ["error", "critical"]]
            if error_violations:
                raise QualityGateError(
                    f"Quality gate validation failed with {len(error_violations)} errors",
                    violations=error_violations
                )
        
        return {
            "ok": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "enforcement_enabled": self.enabled,
            "metrics": {
                "vector_queries": vector_queries,
                "web_searches": web_searches,
                "claims_found": len(claims),
                "claims_verified": len(tool_usage.get('verified_claims', []))
            },
            "violations": [
                {
                    "gate": v.gate_name,
                    "expected": v.expected,
                    "actual": v.actual,
                    "severity": v.severity,
                    "message": v.message
                }
                for v in violations
            ]
        }
    
    def extract_claims(self, content: str) -> List[str]:
        """Extract potential claims from content for verification.
        
        Simple implementation that identifies sentences with statistics,
        percentages, or definitive statements.
        """
        claims = []
        
        # Patterns that indicate claims
        claim_patterns = [
            r'[0-9]+%',  # Percentages
            r'[0-9]+\s*(?:million|billion|thousand)',  # Large numbers
            r'according to',  # Attribution
            r'research shows',  # Research claims
            r'study found',  # Study claims
            r'data indicates',  # Data claims
            r'statistics show',  # Statistical claims
        ]
        
        sentences = content.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Minimum length for meaningful claim
                for pattern in claim_patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        claims.append(sentence)
                        break
        
        return claims


def validate_tool_usage_quality(content: str, 
                               tool_usage: Dict[str, Any],
                               agent_type: str = "unknown") -> Dict[str, Any]:
    """Convenience function to run tool usage quality gates and return results."""
    
    gate = ToolUsageQualityGate()
    try:
        return gate.validate_content_generation(content, tool_usage, agent_type)
    except QualityGateError as e:
        logger.warning(f"Quality gate validation failed: {e}")
        # Return failed result instead of raising
        return {
            "ok": False,
            "issues": [str(e)],
            "warnings": [],
            "enforcement_enabled": gate.enabled,
            "violations": [
                {
                    "gate": v.gate_name,
                    "expected": v.expected,
                    "actual": v.actual,
                    "severity": v.severity,
                    "message": v.message
                }
                for v in e.violations
            ]
        }


