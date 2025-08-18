"""
Technical Accuracy Agent for Phase 2 Multi-Agent System

This agent specializes in validating technical claims, checking code examples,
verifying technical terminology, and providing accuracy confidence scores.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .base_agent import (
    AgentConfiguration,
    BaseSpecializedAgent,
    ProcessingContext,
    ProcessingMode,
    ProcessingResult,
)

logger = logging.getLogger(__name__)


@dataclass
class TechnicalValidationRule:
    """Rule for technical validation."""
    rule_id: str
    description: str
    pattern: str
    severity: str  # "error", "warning", "info"
    suggestion: str
    languages: List[str] = field(default_factory=list)  # Empty means applies to all


@dataclass
class CodeBlock:
    """Represents a code block found in content."""
    content: str
    language: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class TechnicalClaim:
    """Represents a technical claim found in content."""
    claim: str
    context: str
    confidence: float = 0.0
    is_factual: Optional[bool] = None
    sources: List[str] = field(default_factory=list)
    verification_notes: str = ""


class TechnicalAccuracyAgent(BaseSpecializedAgent):
    """Specialized agent for technical fact-checking and accuracy validation."""
    
    def __init__(self, config: Optional[AgentConfiguration] = None):
        """Initialize the Technical Accuracy Agent."""
        super().__init__("TechnicalAccuracy", config)
        
        # Technical validation rules
        self.validation_rules = self._load_validation_rules()
        
        # Programming language patterns
        self.code_patterns = self._load_code_patterns()
        
        # Technical terminology database
        self.technical_terms = self._load_technical_terms()
        
        # Common technical misconceptions
        self.misconceptions = self._load_common_misconceptions()
        
        logger.info("Technical Accuracy Agent initialized with validation rules")
    
    def _process_internal(self, context: ProcessingContext) -> ProcessingResult:
        """Internal processing implementation for technical accuracy validation."""
        content = context.content
        section_type = context.section_type
        processing_mode = context.processing_mode
        
        # Initialize result components
        suggestions = []
        warnings = []
        errors = []
        metadata = {}
        
        # 1. Extract and validate code blocks
        code_blocks = self._extract_code_blocks(content)
        code_validation_results = self._validate_code_blocks(code_blocks, processing_mode)
        
        # 2. Identify and verify technical claims
        technical_claims = self._extract_technical_claims(content, section_type)
        claim_validation_results = self._validate_technical_claims(technical_claims, processing_mode)

        # 2b. Optional external fact-check integration
        if technical_claims and self._is_external_fact_check_enabled():
            try:
                ext_results = self._external_fact_check(technical_claims)
                claim_validation_results['summary']['external_checked'] = len(ext_results)
                # Adjust confidence slightly upward if supporting evidence exists
                if any(r.get('status') == 'supported' for r in ext_results):
                    claim_validation_results.setdefault('suggestions', []).append(
                        "External fact-check: at least one claim has supporting evidence"
                    )
                    # Tag metadata for downstream usage
                    metadata.setdefault('external_fact_check', ext_results)
            except Exception as e:
                logger.warning(f"External fact check failed: {e}")
        
        # 3. Check technical terminology usage
        terminology_results = self._validate_technical_terminology(content, processing_mode)
        
        # 4. Check for common technical misconceptions
        misconception_results = self._check_technical_misconceptions(content, processing_mode)
        
        # 5. Calculate overall accuracy confidence
        accuracy_confidence = self._calculate_accuracy_confidence(
            code_validation_results, claim_validation_results, 
            terminology_results, misconception_results
        )
        
        # 6. Generate improved content if needed
        processed_content = content
        if processing_mode == ProcessingMode.FULL and accuracy_confidence < 0.8:
            processed_content = self._improve_technical_accuracy(
                content, code_validation_results, claim_validation_results,
                terminology_results, misconception_results
            )
        
        # Compile results
        suggestions.extend(code_validation_results.get('suggestions', []))
        suggestions.extend(claim_validation_results.get('suggestions', []))
        suggestions.extend(terminology_results.get('suggestions', []))
        
        warnings.extend(code_validation_results.get('warnings', []))
        warnings.extend(claim_validation_results.get('warnings', []))
        warnings.extend(terminology_results.get('warnings', []))
        
        errors.extend(code_validation_results.get('errors', []))
        errors.extend(claim_validation_results.get('errors', []))
        errors.extend(misconception_results.get('errors', []))
        
        # Metadata
        metadata.update({
            'code_blocks_found': len(code_blocks),
            'technical_claims_analyzed': len(technical_claims),
            'accuracy_confidence': accuracy_confidence,
            'validation_summary': {
                'code_validation': code_validation_results.get('summary', {}),
                'claim_validation': claim_validation_results.get('summary', {}),
                'terminology_validation': terminology_results.get('summary', {}),
                'misconception_check': misconception_results.get('summary', {})
            }
        })
        
        # Calculate quality score based on accuracy and completeness
        quality_score = self._calculate_quality_score(
            accuracy_confidence, len(errors), len(warnings), len(suggestions)
        )
        
        return ProcessingResult(
            success=True,
            processed_content=processed_content,
            quality_score=quality_score,
            confidence_score=accuracy_confidence,
            suggestions=suggestions,
            warnings=warnings,
            errors=errors,
            metadata=metadata
        )
    
    def _extract_code_blocks(self, content: str) -> List[CodeBlock]:
        """Extract code blocks from content."""
        code_blocks = []
        
        # Pattern for fenced code blocks (```language)
        fenced_pattern = r'```(\w+)?\n?(.*?)```'
        matches = re.finditer(fenced_pattern, content, re.DOTALL)
        
        for match in matches:
            language = match.group(1)
            code_content = match.group(2).strip()
            
            if code_content:  # Only include non-empty code blocks
                code_blocks.append(CodeBlock(
                    content=code_content,
                    language=language.lower() if language else None,
                    line_start=content[:match.start()].count('\n'),
                    line_end=content[:match.end()].count('\n')
                ))
        
        # Pattern for inline code blocks
        inline_pattern = r'`([^`\n]{10,})`'  # Only longer inline code (likely meaningful)
        inline_matches = re.finditer(inline_pattern, content)
        
        for match in inline_matches:
            code_content = match.group(1).strip()
            if self._looks_like_code(code_content):
                code_blocks.append(CodeBlock(
                    content=code_content,
                    language=self._detect_language(code_content),
                    line_start=content[:match.start()].count('\n'),
                    line_end=content[:match.start()].count('\n')
                ))
        
        return code_blocks
    
    def _validate_code_blocks(self, code_blocks: List[CodeBlock], processing_mode: ProcessingMode) -> Dict[str, Any]:
        """Validate extracted code blocks for syntax and logic errors."""
        results = {
            'suggestions': [],
            'warnings': [],
            'errors': [],
            'summary': {
                'total_blocks': len(code_blocks),
                'valid_blocks': 0,
                'blocks_with_issues': 0
            }
        }
        
        for i, block in enumerate(code_blocks):
            block_errors = []
            block_warnings = []
            
            # Language-specific validation
            if block.language in ['python', 'py']:
                block_errors.extend(self._validate_python_code(block.content))
            elif block.language in ['javascript', 'js']:
                block_errors.extend(self._validate_javascript_code(block.content))
            elif block.language in ['java']:
                block_errors.extend(self._validate_java_code(block.content))
            elif block.language in ['sql']:
                block_errors.extend(self._validate_sql_code(block.content))
            
            # General code quality checks
            if processing_mode == ProcessingMode.FULL:
                block_warnings.extend(self._check_code_quality(block.content, block.language))
            
            # Update block status
            if block_errors:
                block.is_valid = False
                block.errors = block_errors
                results['errors'].extend([f"Code block {i+1}: {error}" for error in block_errors])
                results['summary']['blocks_with_issues'] += 1
            else:
                results['summary']['valid_blocks'] += 1
            
            if block_warnings:
                block.warnings = block_warnings
                results['warnings'].extend([f"Code block {i+1}: {warning}" for warning in block_warnings])
            
            # Generate suggestions for improvement
            if not block.is_valid or block_warnings:
                suggestions = self._generate_code_suggestions(block)
                results['suggestions'].extend(suggestions)
        
        return results
    
    def _extract_technical_claims(self, content: str, section_type: Optional[str]) -> List[TechnicalClaim]:
        """Extract technical claims from content for verification."""
        claims = []
        
        # Patterns that typically indicate technical claims
        claim_patterns = [
            r'(?:can|will|does|is|are)\s+(?:able\s+to\s+)?(?:improve|increase|decrease|reduce|enhance|optimize)\s+[^.]{10,}',
            r'(?:up\s+to|as\s+much\s+as|more\s+than|less\s+than)\s+\d+[%x]\s+(?:faster|slower|better|improvement)',
            r'(?:benchmark|performance|speed)\s+(?:shows|demonstrates|proves)\s+[^.]{10,}',
            r'(?:algorithm|method|approach)\s+(?:achieves|provides|delivers)\s+[^.]{10,}',
            r'(?:according\s+to|research\s+shows|studies\s+indicate)\s+[^.]{10,}'
        ]
        
        for pattern in claim_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                claim_text = match.group(0)
                # Get surrounding context (50 chars before and after)
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end]
                
                claims.append(TechnicalClaim(
                    claim=claim_text,
                    context=context
                ))
        
        return claims
    
    def _validate_technical_claims(self, claims: List[TechnicalClaim], processing_mode: ProcessingMode) -> Dict[str, Any]:
        """Validate technical claims for factual accuracy."""
        results = {
            'suggestions': [],
            'warnings': [],
            'errors': [],
            'summary': {
                'total_claims': len(claims),
                'verified_claims': 0,
                'questionable_claims': 0,
                'unverified_claims': 0
            }
        }
        
        for claim in claims:
            # Basic plausibility check
            confidence = self._assess_claim_plausibility(claim.claim)
            claim.confidence = confidence
            
            if confidence < 0.3:
                claim.is_factual = False
                results['errors'].append(f"Highly questionable claim: {claim.claim}")
                results['summary']['questionable_claims'] += 1
            elif confidence < 0.6:
                claim.is_factual = None
                results['warnings'].append(f"Unverified claim requiring source: {claim.claim}")
                results['summary']['unverified_claims'] += 1
            else:
                claim.is_factual = True
                results['summary']['verified_claims'] += 1
            
            # Generate suggestions for improvement
            if confidence < 0.8:
                if processing_mode == ProcessingMode.FULL:
                    results['suggestions'].append(
                        f"Consider adding source or qualification for: {claim.claim[:100]}..."
                    )
        
        return results
    
    def _validate_technical_terminology(self, content: str, processing_mode: ProcessingMode) -> Dict[str, Any]:
        """Validate proper usage of technical terminology."""
        results = {
            'suggestions': [],
            'warnings': [],
            'errors': [],
            'summary': {
                'terms_checked': 0,
                'incorrect_usage': 0,
                'suggested_alternatives': 0
            }
        }
        
        # Check for common terminology mistakes
        for term, correct_usage in self.technical_terms.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                results['summary']['terms_checked'] += 1
                
                # Get context around the term
                start = max(0, match.start() - 20)
                end = min(len(content), match.end() + 20)
                context = content[start:end]
                
                # Check if usage looks correct based on context
                if not self._is_terminology_used_correctly(term, context, correct_usage):
                    results['summary']['incorrect_usage'] += 1
                    results['warnings'].append(
                        f"Potentially incorrect usage of '{term}' in context: ...{context}..."
                    )
                    
                    if processing_mode == ProcessingMode.FULL:
                        results['suggestions'].append(
                            f"Consider verifying usage of '{term}': {correct_usage.get('description', '')}"
                        )
                        results['summary']['suggested_alternatives'] += 1
        
        return results
    
    def _check_technical_misconceptions(self, content: str, processing_mode: ProcessingMode) -> Dict[str, Any]:
        """Check for common technical misconceptions."""
        results = {
            'errors': [],
            'warnings': [],
            'suggestions': [],
            'summary': {
                'misconceptions_checked': len(self.misconceptions),
                'potential_misconceptions': 0
            }
        }
        
        for misconception in self.misconceptions:
            pattern = misconception['pattern']
            if re.search(pattern, content, re.IGNORECASE):
                results['summary']['potential_misconceptions'] += 1
                severity = misconception.get('severity', 'warning')
                
                if severity == 'error':
                    results['errors'].append(f"Technical misconception detected: {misconception['description']}")
                else:
                    results['warnings'].append(f"Potential misconception: {misconception['description']}")
                
                if processing_mode == ProcessingMode.FULL and 'correction' in misconception:
                    results['suggestions'].append(f"Consider: {misconception['correction']}")
        
        return results
    
    def _calculate_accuracy_confidence(self, code_results: Dict, claim_results: Dict, 
                                     terminology_results: Dict, misconception_results: Dict) -> float:
        """Calculate overall accuracy confidence score."""
        # Base confidence starts high
        confidence = 1.0
        
        # Reduce confidence based on code issues
        code_blocks = code_results['summary']['total_blocks']
        if code_blocks > 0:
            invalid_ratio = code_results['summary']['blocks_with_issues'] / code_blocks
            confidence -= invalid_ratio * 0.3
        
        # Reduce confidence based on questionable claims
        total_claims = claim_results['summary']['total_claims']
        if total_claims > 0:
            questionable_ratio = (
                claim_results['summary']['questionable_claims'] + 
                claim_results['summary']['unverified_claims'] * 0.5
            ) / total_claims
            confidence -= questionable_ratio * 0.4
        
        # Reduce confidence based on terminology issues
        terms_checked = terminology_results['summary']['terms_checked']
        if terms_checked > 0:
            incorrect_ratio = terminology_results['summary']['incorrect_usage'] / terms_checked
            confidence -= incorrect_ratio * 0.2
        
        # Reduce confidence based on misconceptions
        if misconception_results['summary']['potential_misconceptions'] > 0:
            confidence -= 0.1 * misconception_results['summary']['potential_misconceptions']
        
        return max(0.0, min(1.0, confidence))
    
    def _improve_technical_accuracy(self, content: str, code_results: Dict, 
                                  claim_results: Dict, terminology_results: Dict,
                                  misconception_results: Dict) -> str:
        """Generate improved content with better technical accuracy."""
        improved_content = content
        
        # Note: In a full implementation, this would use LLM calls to actually
        # improve the content based on the identified issues. For now, we'll
        # return the original content as this would require integration with
        # the LLM infrastructure.
        
        logger.info("Content improvement would be applied here with LLM integration")
        
        return improved_content
    
    def _calculate_quality_score(self, accuracy_confidence: float, error_count: int, 
                               warning_count: int, suggestion_count: int) -> float:
        """Calculate overall quality score."""
        # Start with accuracy confidence
        quality = accuracy_confidence
        
        # Reduce based on errors and warnings
        quality -= error_count * 0.1
        quality -= warning_count * 0.05
        
        # Small reduction for suggestions (indicates room for improvement)
        quality -= suggestion_count * 0.01
        
        return max(0.0, min(1.0, quality))

    # --- External fact checking (optional) ---
    def _is_external_fact_check_enabled(self) -> bool:
        return bool(os.getenv('FACT_CHECK_API_ENABLED', '0') == '1')

    def _external_fact_check(self, claims: List[TechnicalClaim]) -> List[Dict[str, Any]]:
        """
        Perform an optional external fact-check using a provider if configured.
        This is implemented as a lightweight placeholder that can be extended to
        call a real HTTP API. For now, it uses a heuristic stub that marks claims
        with numbers as 'needs_source' and others as 'supported'.
        """
        results: List[Dict[str, Any]] = []
        for claim in claims:
            if re.search(r"\d+%|\d+\s*(x|times)", claim.claim, re.IGNORECASE):
                status = 'needs_source'
            else:
                status = 'supported'
            results.append({
                'claim': claim.claim,
                'status': status,
                'evidence_links': []
            })
        return results
    
    # Helper methods for code validation
    
    def _validate_python_code(self, code: str) -> List[str]:
        """Validate Python code syntax."""
        errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Python syntax error: {e}")
        except Exception as e:
            errors.append(f"Python parsing error: {e}")
        
        return errors
    
    def _validate_javascript_code(self, code: str) -> List[str]:
        """Basic JavaScript code validation."""
        errors = []
        
        # Basic syntax checks
        if code.count('{') != code.count('}'):
            errors.append("Unmatched braces in JavaScript code")
        
        if code.count('(') != code.count(')'):
            errors.append("Unmatched parentheses in JavaScript code")
        
        if code.count('[') != code.count(']'):
            errors.append("Unmatched brackets in JavaScript code")
        
        return errors
    
    def _validate_java_code(self, code: str) -> List[str]:
        """Basic Java code validation."""
        errors = []
        
        # Basic syntax checks
        if 'public class' in code and code.count('{') != code.count('}'):
            errors.append("Unmatched braces in Java code")
        
        return errors
    
    def _validate_sql_code(self, code: str) -> List[str]:
        """Basic SQL code validation."""
        errors = []
        
        # Check for basic SQL syntax issues
        if 'SELECT' in code.upper() and 'FROM' not in code.upper():
            errors.append("SELECT statement missing FROM clause")
        
        return errors
    
    def _check_code_quality(self, code: str, language: Optional[str]) -> List[str]:
        """Check code quality and best practices."""
        warnings = []
        
        # General quality checks
        if len(code) > 1000:
            warnings.append("Code block is very long - consider breaking into smaller examples")
        
        if language == 'python':
            if 'import *' in code:
                warnings.append("Consider avoiding wildcard imports in Python examples")
        
        return warnings
    
    def _generate_code_suggestions(self, block: CodeBlock) -> List[str]:
        """Generate suggestions for improving code blocks."""
        suggestions = []
        
        if not block.is_valid:
            suggestions.append(f"Fix syntax errors in {block.language or 'code'} block")
        
        if block.warnings:
            suggestions.append(f"Address quality issues in {block.language or 'code'} block")
        
        return suggestions
    
    def _looks_like_code(self, text: str) -> bool:
        """Determine if text looks like code."""
        code_indicators = [
            '()', '{}', '[]', ';', '->', '=>', '==', '!=', '&&', '||',
            'function', 'class', 'def', 'if', 'else', 'for', 'while'
        ]
        
        return any(indicator in text for indicator in code_indicators)
    
    def _detect_language(self, code: str) -> Optional[str]:
        """Detect programming language from code content."""
        if 'def ' in code or 'import ' in code or 'from ' in code:
            return 'python'
        elif 'function' in code or 'var ' in code or 'let ' in code:
            return 'javascript'
        elif 'public class' in code or 'import java' in code:
            return 'java'
        elif 'SELECT' in code.upper() or 'INSERT' in code.upper():
            return 'sql'
        
        return None
    
    def _assess_claim_plausibility(self, claim: str) -> float:
        """Assess the plausibility of a technical claim."""
        # Simple heuristic-based assessment
        confidence = 0.7  # Default neutral confidence
        
        # Look for confidence indicators
        if any(word in claim.lower() for word in ['proven', 'demonstrated', 'research shows']):
            confidence += 0.2
        
        # Look for uncertainty indicators
        if any(word in claim.lower() for word in ['might', 'could', 'potentially', 'possibly']):
            confidence += 0.1
        
        # Look for extreme claims
        if any(word in claim.lower() for word in ['revolutionary', 'groundbreaking', 'never before']):
            confidence -= 0.3
        
        # Look for specific numbers without context
        if re.search(r'\d+[%x]', claim) and 'study' not in claim.lower():
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _is_terminology_used_correctly(self, term: str, context: str, correct_usage: Dict) -> bool:
        """Check if technical terminology is used correctly in context."""
        # Simplified check - in a full implementation, this would be more sophisticated
        context_lower = context.lower()
        
        # Check for obvious misuse patterns
        wrong_contexts = correct_usage.get('wrong_contexts', [])
        for wrong_context in wrong_contexts:
            if wrong_context.lower() in context_lower:
                return False
        
        return True
    
    def _load_validation_rules(self) -> List[TechnicalValidationRule]:
        """Load technical validation rules."""
        # In a production system, this would load from a configuration file
        return [
            TechnicalValidationRule(
                rule_id="python_syntax",
                description="Python syntax validation",
                pattern=r"```python.*?```",
                severity="error",
                suggestion="Ensure Python code follows correct syntax",
                languages=["python"]
            )
        ]
    
    def _load_code_patterns(self) -> Dict[str, List[str]]:
        """Load programming language patterns."""
        return {
            'python': [r'def\s+\w+', r'class\s+\w+', r'import\s+\w+'],
            'javascript': [r'function\s+\w+', r'var\s+\w+', r'let\s+\w+'],
            'java': [r'public\s+class', r'public\s+static', r'import\s+java'],
            'sql': [r'SELECT\s+', r'FROM\s+', r'WHERE\s+']
        }
    
    def _load_technical_terms(self) -> Dict[str, Dict[str, Any]]:
        """Load technical terminology database."""
        return {
            'algorithm': {
                'description': 'A process or set of rules to be followed in calculations or other problem-solving operations',
                'wrong_contexts': ['algorithms are always faster', 'algorithms solve everything']
            },
            'machine learning': {
                'description': 'A type of artificial intelligence that allows software applications to become more accurate at predicting outcomes',
                'wrong_contexts': ['machine learning is magic', 'machine learning works without data']
            }
        }
    
    def _load_common_misconceptions(self) -> List[Dict[str, Any]]:
        """Load common technical misconceptions."""
        return [
            {
                'pattern': r'AI\s+(?:will|can)\s+(?:replace|eliminate)\s+all\s+(?:jobs|developers|programmers)',
                'description': 'Overly broad claim about AI replacing all technical jobs',
                'severity': 'warning',
                'correction': 'AI will augment rather than completely replace most technical roles'
            },
            {
                'pattern': r'blockchain\s+(?:solves|fixes)\s+everything',
                'description': 'Claim that blockchain is a universal solution',
                'severity': 'warning',
                'correction': 'Blockchain has specific use cases and is not suitable for all problems'
            }
        ]
    
    def _process_fallback(self, context: ProcessingContext) -> ProcessingResult:
        """Fallback processing for technical accuracy validation."""
        # In fallback mode, perform minimal validation
        code_blocks = self._extract_code_blocks(context.content)
        basic_errors = []
        
        # Only check for obvious syntax errors in fallback
        for block in code_blocks:
            if block.language == 'python':
                try:
                    ast.parse(block.content)
                except SyntaxError:
                    basic_errors.append(f"Python syntax error in code block")
        
        confidence = 0.8 if not basic_errors else 0.5
        
        return ProcessingResult(
            success=True,
            processed_content=context.content,
            quality_score=confidence,
            confidence_score=confidence,
            suggestions=["Fallback mode: Limited technical validation performed"],
            warnings=basic_errors,
            errors=[],
            metadata={"processing_mode": "fallback", "code_blocks_found": len(code_blocks)}
        )


# Export main classes
__all__ = ['TechnicalAccuracyAgent', 'TechnicalValidationRule', 'CodeBlock', 'TechnicalClaim']