"""
Technical Quality Validator

This module provides technical validation functionality for newsletter content,
including code validation, technical accuracy assessment, and mobile readability.
"""

from __future__ import annotations

import re
import ast
import logging
import time
from typing import Dict, List, Any, Optional, Union

from .base import QualityValidator, QualityMetrics, QualityReport, QualityStatus

logger = logging.getLogger(__name__)


class TechnicalQualityValidator(QualityValidator):
    """Technical quality validator for newsletter content."""
    
    def __init__(self):
        super().__init__("TechnicalQualityValidator")
        
        # Technical terms database for accuracy validation
        self.technical_terms_db = self._load_technical_terms_database()
        
        # Code validation patterns
        self.code_patterns = {
            'python': r'```python\s*\n(.*?)```',
            'javascript': r'```javascript\s*\n(.*?)```',
            'bash': r'```bash\s*\n(.*?)```',
            'json': r'```json\s*\n(.*?)```',
            'yaml': r'```yaml\s*\n(.*?)```'
        }
        
        # Mobile readability thresholds
        self.mobile_thresholds = {
            'max_subject_length': 50,
            'max_preview_length': 150,
            'max_paragraph_length': 100,
            'max_headline_length': 60,
            'min_white_space_ratio': 0.1
        }
    
    def validate(self, content: Union[str, Dict[str, Any]], **kwargs) -> QualityReport:
        """Validate technical aspects of content and return comprehensive report."""
        start_time = time.time()
        
        # Extract text content
        text_content = self._extract_text_content(content)
        
        # Perform technical validations
        technical_accuracy = self._validate_technical_accuracy(text_content)
        code_validation = self._validate_code_examples(text_content)
        mobile_readability = self._validate_mobile_readability(content)
        
        # Calculate overall metrics
        metrics = self._calculate_technical_metrics(
            technical_accuracy, code_validation, mobile_readability
        )
        
        # Generate issues and recommendations
        issues, warnings, recommendations, blocking_issues = self._generate_technical_issues(
            technical_accuracy, code_validation, mobile_readability
        )
        
        # Determine status
        status = self._determine_technical_status(issues, warnings, blocking_issues, metrics)
        
        # Create quality report
        report = QualityReport(
            status=status,
            metrics=metrics,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            blocking_issues=blocking_issues,
            strengths=self._identify_technical_strengths(technical_accuracy, code_validation, mobile_readability),
            detailed_analysis={
                'technical_accuracy': technical_accuracy,
                'code_validation': code_validation,
                'mobile_readability': mobile_readability,
                'content_length': len(text_content),
                'processing_time': time.time() - start_time
            }
        )
        
        return report
    
    def get_metrics(self, content: Union[str, Dict[str, Any]], **kwargs) -> QualityMetrics:
        """Extract technical quality metrics from content."""
        text_content = self._extract_text_content(content)
        
        technical_accuracy = self._validate_technical_accuracy(text_content)
        code_validation = self._validate_code_examples(text_content)
        mobile_readability = self._validate_mobile_readability(content)
        
        return self._calculate_technical_metrics(technical_accuracy, code_validation, mobile_readability)
    
    def _extract_text_content(self, content: Union[str, Dict[str, Any]]) -> str:
        """Extract text content from various input formats."""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Extract text from common newsletter content structure
            if 'content' in content:
                return str(content['content'])
            elif 'body' in content:
                return str(content['body'])
            elif 'text' in content:
                return str(content['text'])
            else:
                # Try to find any text-like field
                for key, value in content.items():
                    if isinstance(value, str) and len(value) > 50:
                        return value
                return str(content)
        else:
            return str(content)
    
    def _load_technical_terms_database(self) -> Dict[str, List[str]]:
        """Load technical terms database for accuracy validation."""
        return {
            'ai_ml_terms': [
                'neural network', 'transformer', 'attention mechanism', 'gradient descent',
                'backpropagation', 'overfitting', 'regularization', 'cross-validation',
                'reinforcement learning', 'supervised learning', 'unsupervised learning',
                'deep learning', 'machine learning', 'artificial intelligence', 'AI',
                'natural language processing', 'NLP', 'computer vision', 'CV'
            ],
            'programming_terms': [
                'API', 'REST', 'GraphQL', 'microservices', 'containerization',
                'kubernetes', 'docker', 'CI/CD', 'deployment', 'scalability',
                'object-oriented programming', 'OOP', 'functional programming',
                'design patterns', 'algorithm', 'data structure', 'database',
                'SQL', 'NoSQL', 'caching', 'load balancing'
            ],
            'cloud_terms': [
                'AWS', 'Azure', 'GCP', 'serverless', 'lambda', 'edge computing',
                'distributed systems', 'auto-scaling', 'cloud computing',
                'infrastructure as code', 'IaC', 'container orchestration',
                'service mesh', 'API gateway', 'CDN'
            ],
            'data_terms': [
                'data science', 'big data', 'data analytics', 'data mining',
                'data visualization', 'ETL', 'data pipeline', 'data warehouse',
                'data lake', 'business intelligence', 'BI', 'data governance'
            ]
        }
    
    def _validate_technical_accuracy(self, content: str) -> Dict[str, Any]:
        """Validate technical accuracy of content for professional audience."""
        start_time = time.time()
        
        # Extract technical claims and statements
        technical_claims = self._extract_technical_claims(content)
        
        # Validate each technical claim
        validation_results = []
        for claim in technical_claims:
            validation = self._validate_single_claim(claim)
            validation_results.append(validation)
        
        # Calculate overall accuracy score
        accuracy_score = self._calculate_accuracy_score(validation_results)
        
        processing_time = time.time() - start_time
        
        return {
            'accuracy_score': accuracy_score,
            'technical_claims_count': len(technical_claims),
            'validated_claims': validation_results,
            'processing_time_seconds': processing_time,
            'passes_threshold': accuracy_score >= 0.8,
            'issues_found': [v for v in validation_results if not v['is_accurate']],
            'validation_timestamp': time.time()
        }
    
    def _extract_technical_claims(self, content: str) -> List[str]:
        """Extract technical claims from content."""
        # Extract sentences containing technical terms
        sentences = re.split(r'[.!?]+', content)
        technical_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Check if sentence contains technical terms
            for category, terms in self.technical_terms_db.items():
                for term in terms:
                    if term.lower() in sentence.lower():
                        technical_sentences.append(sentence)
                        break
                if sentence in technical_sentences:
                    break
        
        return technical_sentences
    
    def _validate_single_claim(self, claim: str) -> Dict[str, Any]:
        """Validate a single technical claim."""
        # Simple validation logic - in practice, this could use LLM validation
        validation_score = 0.8  # Default score
        
        # Check for common technical inaccuracies
        inaccuracies = [
            'AI can think like humans',
            'machine learning is always accurate',
            'neural networks are infallible',
            'data science is just statistics'
        ]
        
        is_accurate = True
        issues = []
        
        for inaccuracy in inaccuracies:
            if inaccuracy.lower() in claim.lower():
                is_accurate = False
                issues.append(f"Contains inaccurate statement: {inaccuracy}")
                validation_score = 0.3
        
        # Check for proper technical terminology
        technical_terms_found = 0
        for category, terms in self.technical_terms_db.items():
            for term in terms:
                if term.lower() in claim.lower():
                    technical_terms_found += 1
        
        if technical_terms_found == 0:
            issues.append("No technical terminology found")
            validation_score = 0.5
        
        return {
            'claim': claim,
            'is_accurate': is_accurate,
            'validation_score': validation_score,
            'issues': issues,
            'technical_terms_found': technical_terms_found
        }
    
    def _calculate_accuracy_score(self, validation_results: List[Dict[str, Any]]) -> float:
        """Calculate overall technical accuracy score."""
        if not validation_results:
            return 0.0
        
        total_score = sum(result['validation_score'] for result in validation_results)
        return total_score / len(validation_results)
    
    def _validate_code_examples(self, content: str) -> Dict[str, Any]:
        """Validate code examples in content."""
        validation_results = []
        
        for language, pattern in self.code_patterns.items():
            code_blocks = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            
            for code_block in code_blocks:
                validation = self._validate_code_block(code_block, language)
                validation_results.append(validation)
        
        # Calculate overall code quality score
        code_score = self._calculate_code_score(validation_results)
        
        return {
            'code_score': code_score,
            'code_blocks_count': len(validation_results),
            'validation_results': validation_results,
            'languages_found': list(set(r['language'] for r in validation_results)),
            'passes_threshold': code_score >= 0.7
        }
    
    def _validate_code_block(self, code: str, language: str) -> Dict[str, Any]:
        """Validate a single code block."""
        validation_score = 0.8  # Default score
        issues = []
        
        # Clean up the code by removing leading/trailing whitespace and fixing indentation
        code = code.strip()
        
        # For Python, try to fix common indentation issues
        if language == 'python':
            lines = code.split('\n')
            # Find the minimum indentation
            min_indent = float('inf')
            for line in lines:
                if line.strip():  # Skip empty lines
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
            
            # Remove the minimum indentation from all lines
            if min_indent != float('inf'):
                cleaned_lines = []
                for line in lines:
                    if line.strip():
                        cleaned_lines.append(line[min_indent:])
                    else:
                        cleaned_lines.append('')
                code = '\n'.join(cleaned_lines)
        
        try:
            if language == 'python':
                # Basic Python syntax validation
                ast.parse(code)
                validation_score = 0.9
            elif language == 'javascript':
                # Basic JavaScript validation (simplified)
                if 'function' in code or 'const' in code or 'let' in code:
                    validation_score = 0.8
                else:
                    issues.append("No clear JavaScript structure")
                    validation_score = 0.5
            elif language == 'bash':
                # Basic bash validation
                if code.strip().startswith('#') or 'echo' in code or 'ls' in code:
                    validation_score = 0.8
                else:
                    issues.append("No clear bash commands")
                    validation_score = 0.5
            elif language in ['json', 'yaml']:
                # Basic structure validation
                if '{' in code and '}' in code:
                    validation_score = 0.8
                else:
                    issues.append("Invalid JSON/YAML structure")
                    validation_score = 0.4
        except SyntaxError as e:
            issues.append(f"Syntax error: {str(e)}")
            validation_score = 0.2
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
            validation_score = 0.3
        
        return {
            'code': code[:100] + '...' if len(code) > 100 else code,
            'language': language,
            'validation_score': validation_score,
            'issues': issues,
            'is_valid': validation_score >= 0.6
        }
    
    def _calculate_code_score(self, validation_results: List[Dict[str, Any]]) -> float:
        """Calculate overall code quality score."""
        if not validation_results:
            return 0.0
        
        total_score = sum(result['validation_score'] for result in validation_results)
        return total_score / len(validation_results)
    
    def _validate_mobile_readability(self, content: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Validate mobile readability of newsletter content."""
        # Handle None content
        if content is None:
            content = ""
        
        # Convert non-string content to string
        if not isinstance(content, (str, dict)):
            content = str(content)
        
        # Extract newsletter components
        if isinstance(content, dict):
            subject_line = content.get('subject', '')
            preview_text = content.get('preview', '')
            body_content = self._extract_text_content(content)
        else:
            # Try to extract components from text
            lines = content.split('\n')
            subject_line = lines[0] if lines else ''
            preview_text = lines[1] if len(lines) > 1 else ''
            body_content = content
        
        # Validate individual components
        subject_validation = self._validate_subject_line(subject_line)
        preview_validation = self._validate_preview_text(preview_text)
        paragraph_validation = self._validate_paragraph_lengths(body_content)
        headline_validation = self._validate_headline_readability(body_content)
        structure_validation = self._validate_content_structure(body_content)
        whitespace_validation = self._validate_white_space_usage(body_content)
        
        # Calculate overall mobile score
        mobile_score = self._calculate_mobile_score({
            'subject': subject_validation,
            'preview': preview_validation,
            'paragraphs': paragraph_validation,
            'headlines': headline_validation,
            'structure': structure_validation,
            'whitespace': whitespace_validation
        })
        
        return {
            'mobile_score': mobile_score,
            'subject_validation': subject_validation,
            'preview_validation': preview_validation,
            'paragraph_validation': paragraph_validation,
            'headline_validation': headline_validation,
            'structure_validation': structure_validation,
            'whitespace_validation': whitespace_validation,
            'passes_threshold': mobile_score >= 0.7
        }
    
    def _validate_subject_line(self, subject_line: str) -> Dict[str, Any]:
        """Validate subject line for mobile readability."""
        length = len(subject_line)
        is_optimal = length <= self.mobile_thresholds['max_subject_length']
        
        return {
            'length': length,
            'is_optimal': is_optimal,
            'score': 1.0 if is_optimal else max(0.0, 1.0 - (length - self.mobile_thresholds['max_subject_length']) / 20),
            'recommendation': f"Keep subject line under {self.mobile_thresholds['max_subject_length']} characters" if not is_optimal else "Subject line length is optimal"
        }
    
    def _validate_preview_text(self, preview_text: str) -> Dict[str, Any]:
        """Validate preview text for mobile readability."""
        length = len(preview_text)
        is_optimal = length <= self.mobile_thresholds['max_preview_length']
        
        return {
            'length': length,
            'is_optimal': is_optimal,
            'score': 1.0 if is_optimal else max(0.0, 1.0 - (length - self.mobile_thresholds['max_preview_length']) / 50),
            'recommendation': f"Keep preview text under {self.mobile_thresholds['max_preview_length']} characters" if not is_optimal else "Preview text length is optimal"
        }
    
    def _validate_paragraph_lengths(self, content: str) -> Dict[str, Any]:
        """Validate paragraph lengths for mobile readability."""
        paragraphs = content.split('\n\n')
        long_paragraphs = []
        optimal_paragraphs = 0
        
        for i, paragraph in enumerate(paragraphs):
            word_count = len(paragraph.split())
            if word_count > self.mobile_thresholds['max_paragraph_length']:
                long_paragraphs.append({
                    'index': i,
                    'word_count': word_count,
                    'text': paragraph[:50] + '...'
                })
            else:
                optimal_paragraphs += 1
        
        total_paragraphs = len(paragraphs)
        score = optimal_paragraphs / total_paragraphs if total_paragraphs > 0 else 1.0
        
        return {
            'total_paragraphs': total_paragraphs,
            'optimal_paragraphs': optimal_paragraphs,
            'long_paragraphs': long_paragraphs,
            'score': score,
            'recommendation': f"Break {len(long_paragraphs)} long paragraphs into shorter ones" if long_paragraphs else "Paragraph lengths are optimal"
        }
    
    def _validate_headline_readability(self, content: str) -> Dict[str, Any]:
        """Validate headline readability for mobile."""
        headlines = re.findall(r'^#{1,3}\s+(.+)$', content, re.MULTILINE)
        long_headlines = []
        optimal_headlines = 0
        
        for headline in headlines:
            if len(headline) > self.mobile_thresholds['max_headline_length']:
                long_headlines.append(headline)
            else:
                optimal_headlines += 1
        
        total_headlines = len(headlines)
        score = optimal_headlines / total_headlines if total_headlines > 0 else 1.0
        
        return {
            'total_headlines': total_headlines,
            'optimal_headlines': optimal_headlines,
            'long_headlines': long_headlines,
            'score': score,
            'recommendation': f"Shorten {len(long_headlines)} long headlines" if long_headlines else "Headline lengths are optimal"
        }
    
    def _validate_content_structure(self, content: str) -> Dict[str, Any]:
        """Validate content structure for mobile readability."""
        structure_elements = {
            'has_headings': bool(re.search(r'^#{1,3}\s+', content, re.MULTILINE)),
            'has_lists': bool(re.search(r'^[\s]*[-*+]\s+', content, re.MULTILINE)),
            'has_paragraphs': '\n\n' in content,
            'has_code_blocks': bool(re.search(r'```', content)),
            'has_links': bool(re.search(r'\[.*?\]\(.*?\)', content))
        }
        
        score = sum(structure_elements.values()) / len(structure_elements)
        
        return {
            'structure_elements': structure_elements,
            'score': score,
            'recommendation': "Content structure is well-organized" if score >= 0.6 else "Improve content structure with headings, lists, and paragraphs"
        }
    
    def _validate_white_space_usage(self, content: str) -> Dict[str, Any]:
        """Validate white space usage for mobile readability."""
        total_chars = len(content)
        whitespace_chars = len(re.findall(r'\s', content))
        whitespace_ratio = whitespace_chars / total_chars if total_chars > 0 else 0
        
        is_optimal = whitespace_ratio >= self.mobile_thresholds['min_white_space_ratio']
        
        return {
            'whitespace_ratio': whitespace_ratio,
            'is_optimal': is_optimal,
            'score': 1.0 if is_optimal else whitespace_ratio / self.mobile_thresholds['min_white_space_ratio'],
            'recommendation': "Add more white space for better readability" if not is_optimal else "White space usage is optimal"
        }
    
    def _calculate_mobile_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall mobile readability score."""
        scores = [
            validation_results['subject']['score'],
            validation_results['preview']['score'],
            validation_results['paragraphs']['score'],
            validation_results['headlines']['score'],
            validation_results['structure']['score'],
            validation_results['whitespace']['score']
        ]
        
        return sum(scores) / len(scores)
    
    def _calculate_technical_metrics(self, technical_accuracy: Dict[str, Any],
                                   code_validation: Dict[str, Any],
                                   mobile_readability: Dict[str, Any]) -> QualityMetrics:
        """Calculate comprehensive technical quality metrics."""
        technical_accuracy_score = technical_accuracy.get('accuracy_score', 0.0) * 10
        code_quality_score = code_validation.get('code_score', 0.0) * 10
        mobile_readability_score = mobile_readability.get('mobile_score', 0.0) * 10
        
        # Calculate other metrics based on technical validation
        content_quality_score = technical_accuracy_score * 0.8  # Technical accuracy affects content quality
        readability_score = mobile_readability_score * 0.9  # Mobile readability affects general readability
        structure_score = mobile_readability.get('structure_validation', {}).get('score', 0.0) * 10
        
        return QualityMetrics(
            overall_score=0.0,  # Will be calculated by post_init
            technical_accuracy_score=technical_accuracy_score,
            content_quality_score=content_quality_score,
            readability_score=readability_score,
            engagement_score=8.0,  # Placeholder
            structure_score=structure_score,
            code_quality_score=code_quality_score,
            mobile_readability_score=mobile_readability_score,
            source_credibility_score=9.0,  # Placeholder
            content_balance_score=7.0,  # Placeholder
            performance_score=10.0  # Placeholder
        )
    
    def _generate_technical_issues(self, technical_accuracy: Dict[str, Any],
                                 code_validation: Dict[str, Any],
                                 mobile_readability: Dict[str, Any]) -> tuple:
        """Generate technical issues, warnings, recommendations, and blocking issues."""
        issues = []
        warnings = []
        recommendations = []
        blocking_issues = []
        
        # Technical accuracy issues
        accuracy_score = technical_accuracy.get('accuracy_score', 0.0)
        if accuracy_score < 0.7:
            issues.append(f"Low technical accuracy (score: {accuracy_score:.2f})")
            recommendations.append("Review technical claims for accuracy")
        
        if accuracy_score < 0.5:
            blocking_issues.append("Critical technical inaccuracies detected")
        
        # Code validation issues
        code_score = code_validation.get('code_score', 0.0)
        if code_score < 0.6:
            issues.append(f"Poor code quality (score: {code_score:.2f})")
            recommendations.append("Review and fix code examples")
        
        # Mobile readability issues
        mobile_score = mobile_readability.get('mobile_score', 0.0)
        if mobile_score < 0.6:
            issues.append(f"Poor mobile readability (score: {mobile_score:.2f})")
            recommendations.append("Optimize content for mobile devices")
        
        # Specific mobile issues
        mobile_validation = mobile_readability
        for component, validation in mobile_validation.items():
            if component.endswith('_validation') and validation.get('score', 1.0) < 0.7:
                component_name = component.replace('_validation', '').replace('_', ' ')
                warnings.append(f"Suboptimal {component_name}")
                recommendations.append(validation.get('recommendation', f"Improve {component_name}"))
        
        return issues, warnings, recommendations, blocking_issues
    
    def _determine_technical_status(self, issues: List[str], warnings: List[str],
                                  blocking_issues: List[str], metrics: QualityMetrics) -> QualityStatus:
        """Determine overall technical quality status."""
        if blocking_issues:
            return QualityStatus.FAILED
        
        if metrics.technical_accuracy_score < 5.0:
            return QualityStatus.FAILED
        
        if len(issues) > 2 or metrics.technical_accuracy_score < 7.0:
            return QualityStatus.NEEDS_REVIEW
        
        if warnings or len(issues) > 0:
            return QualityStatus.WARNING
        
        return QualityStatus.PASSED
    
    def _identify_technical_strengths(self, technical_accuracy: Dict[str, Any],
                                    code_validation: Dict[str, Any],
                                    mobile_readability: Dict[str, Any]) -> List[str]:
        """Identify technical strengths based on validation results."""
        strengths = []
        
        # Technical accuracy strengths
        accuracy_score = technical_accuracy.get('accuracy_score', 0.0)
        if accuracy_score >= 0.8:
            strengths.append("Excellent technical accuracy")
        elif accuracy_score >= 0.6:
            strengths.append("Good technical accuracy")
        
        # Code quality strengths
        code_score = code_validation.get('code_score', 0.0)
        if code_score >= 0.8:
            strengths.append("High-quality code examples")
        elif code_score >= 0.6:
            strengths.append("Good code examples")
        
        # Mobile readability strengths
        mobile_score = mobile_readability.get('mobile_score', 0.0)
        if mobile_score >= 0.8:
            strengths.append("Excellent mobile readability")
        elif mobile_score >= 0.6:
            strengths.append("Good mobile readability")
        
        return strengths