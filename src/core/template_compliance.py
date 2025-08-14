"""
Template Compliance Validation System

This module provides comprehensive template compliance validation to ensure
newsletters meet structural and content requirements before publication.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Compliance validation levels."""
    STRICT = "strict"          # 100% compliance required
    STANDARD = "standard"      # 90% compliance required  
    RELAXED = "relaxed"       # 80% compliance required
    WARNING_ONLY = "warning_only"  # Log warnings but don't fail


@dataclass
class SectionCompliance:
    """Compliance data for newsletter sections."""
    required_count: int
    found_count: int
    missing_sections: List[str]
    compliance_percentage: float
    section_details: Dict[str, Dict[str, any]] = None
    
    @property
    def is_compliant(self) -> bool:
        """Check if section compliance meets minimum standards."""
        return self.compliance_percentage >= 0.9  # 90% of sections required


@dataclass
class WordCountData:
    """Word count compliance data for a section."""
    target: int
    actual: int
    min_acceptable: int
    max_acceptable: int
    is_compliant: bool
    variance_percentage: float = 0.0
    
    def __post_init__(self):
        """Calculate variance percentage."""
        if self.target > 0:
            self.variance_percentage = (self.actual - self.target) / self.target * 100


@dataclass
class WordCountCompliance:
    """Word count compliance for all sections."""
    section_data: Dict[str, WordCountData]
    overall_compliance: float = 0.0
    total_words_actual: int = 0
    total_words_target: int = 0
    
    def __post_init__(self):
        """Calculate overall compliance metrics."""
        compliant_sections = sum(1 for data in self.section_data.values() if data.is_compliant)
        total_sections = len(self.section_data)
        
        if total_sections > 0:
            self.overall_compliance = compliant_sections / total_sections
        
        self.total_words_actual = sum(data.actual for data in self.section_data.values())
        self.total_words_target = sum(data.target for data in self.section_data.values())
    
    @property
    def is_compliant(self) -> bool:
        """Check if overall word count compliance meets standards."""
        return self.overall_compliance >= 0.8  # 80% of sections must meet word count


@dataclass
class ElementsCompliance:
    """Compliance data for required elements."""
    required_elements: List[str]
    found_elements: List[str]
    missing_elements: List[str]
    compliance_score: float
    element_locations: Dict[str, List[int]] = None  # Element -> line numbers
    
    @property
    def is_compliant(self) -> bool:
        """Check if required elements compliance meets standards."""
        return self.compliance_score >= 0.85  # 85% of required elements must be present


@dataclass
class StructuralCompliance:
    """Compliance data for newsletter structure."""
    has_title: bool
    has_introduction: bool
    has_conclusion: bool
    has_sections: bool
    section_count: int
    heading_structure_valid: bool
    compliance_score: float
    
    @property
    def is_compliant(self) -> bool:
        """Check if structural compliance meets standards."""
        return self.compliance_score >= 0.8  # 80% structural compliance required


@dataclass
class CompletionCompliance:
    """Compliance data for content completion."""
    ends_properly: bool
    has_complete_sentences: bool
    minimum_length_met: bool
    no_abrupt_cutoff: bool
    completion_score: float
    last_content_line: str = ""
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
    
    @property
    def is_compliant(self) -> bool:
        """Check if completion compliance meets standards."""
        return self.completion_score >= 0.95  # 95% completion required


@dataclass
class ComplianceReport:
    """Comprehensive template compliance report."""
    template_name: str
    content_length: int
    
    # Compliance components
    section_compliance: Optional[SectionCompliance] = None
    word_count_compliance: Optional[WordCountCompliance] = None  
    elements_compliance: Optional[ElementsCompliance] = None
    structural_compliance: Optional[StructuralCompliance] = None
    completion_compliance: Optional[CompletionCompliance] = None
    
    # Overall metrics
    overall_score: float = 0.0
    is_compliant: bool = False
    compliance_level: ComplianceLevel = ComplianceLevel.STANDARD
    
    # Issues and recommendations
    critical_issues: List[str] = None
    warnings: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        """Initialize lists and calculate overall compliance."""
        if self.critical_issues is None:
            self.critical_issues = []
        if self.warnings is None:
            self.warnings = []
        if self.recommendations is None:
            self.recommendations = []
        
        self._calculate_overall_compliance()
    
    def _calculate_overall_compliance(self):
        """Calculate overall compliance score and status."""
        scores = []
        weights = []
        
        # Section compliance (25% weight)
        if self.section_compliance:
            scores.append(self.section_compliance.compliance_percentage)
            weights.append(0.25)
        
        # Word count compliance (20% weight)
        if self.word_count_compliance:
            scores.append(self.word_count_compliance.overall_compliance)
            weights.append(0.20)
        
        # Elements compliance (20% weight)
        if self.elements_compliance:
            scores.append(self.elements_compliance.compliance_score)
            weights.append(0.20)
        
        # Structural compliance (15% weight)
        if self.structural_compliance:
            scores.append(self.structural_compliance.compliance_score)
            weights.append(0.15)
        
        # Completion compliance (20% weight) - Critical
        if self.completion_compliance:
            scores.append(self.completion_compliance.completion_score)
            weights.append(0.20)
        
        # Calculate weighted average
        if scores and weights:
            self.overall_score = sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)
        
        # Determine compliance based on level thresholds
        thresholds = {
            ComplianceLevel.STRICT: 0.95,
            ComplianceLevel.STANDARD: 0.90,
            ComplianceLevel.RELAXED: 0.80,
            ComplianceLevel.WARNING_ONLY: 0.0
        }
        
        self.is_compliant = self.overall_score >= thresholds.get(self.compliance_level, 0.90)
    
    def add_critical_issue(self, issue: str):
        """Add a critical compliance issue."""
        self.critical_issues.append(issue)
        logger.error(f"Critical compliance issue: {issue}")
    
    def add_warning(self, warning: str):
        """Add a compliance warning."""
        self.warnings.append(warning)
        logger.warning(f"Compliance warning: {warning}")
    
    def add_recommendation(self, recommendation: str):
        """Add a compliance recommendation."""
        self.recommendations.append(recommendation)
        logger.info(f"Compliance recommendation: {recommendation}")


class TemplateComplianceValidator:
    """Validates newsletter content against template requirements."""
    
    def __init__(self, compliance_level: ComplianceLevel = ComplianceLevel.STANDARD):
        """
        Initialize template compliance validator.
        
        Args:
            compliance_level: Level of compliance validation to apply
        """
        self.compliance_level = compliance_level
        
        # Standard newsletter structure patterns
        self.section_patterns = {
            'title': [r'^#\s+.+$', r'^\*.*\*$'],  # H1 or bold title
            'introduction': [r'introduction', r'brief', r'overview', r'executive summary'],
            'main_content': [r'##\s+', r'###\s+'],  # H2/H3 headers
            'technical_insights': [r'technical', r'implementation', r'architecture', r'deep.*dive'],
            'practical_implications': [r'practical', r'implications', r'applications', r'real.*world'],
            'future_outlook': [r'future', r'outlook', r'trends', r'predictions', r'directions'],
            'conclusion': [r'conclusion', r'summary', r'call.*action', r'next.*steps']
        }
        
        # Required elements for different newsletter types
        self.required_elements = {
            'technical_deep_dive': [
                'code examples', 'technical concepts', 'implementation details',
                'architecture', 'best practices'
            ],
            'trend_analysis': [
                'market data', 'trends', 'analysis', 'implications', 'predictions'
            ],
            'product_review': [
                'features', 'evaluation', 'pros and cons', 'recommendation'
            ]
        }
    
    def validate_content_compliance(self, content: str, template_type: str = "general") -> ComplianceReport:
        """
        Perform comprehensive template compliance validation.
        
        Args:
            content: Newsletter content to validate
            template_type: Type of newsletter template to validate against
            
        Returns:
            ComplianceReport with detailed compliance analysis
        """
        logger.info(f"Starting template compliance validation ({self.compliance_level.value})")
        
        report = ComplianceReport(
            template_name=template_type,
            content_length=len(content),
            compliance_level=self.compliance_level
        )
        
        # Validate sections
        report.section_compliance = self._validate_sections(content, template_type)
        
        # Validate word counts
        report.word_count_compliance = self._validate_word_counts(content, template_type)
        
        # Validate required elements
        report.elements_compliance = self._validate_required_elements(content, template_type)
        
        # Validate structure
        report.structural_compliance = self._validate_structure(content)
        
        # Validate completion (CRITICAL)
        report.completion_compliance = self._validate_completion(content)
        
        # Generate issues and recommendations
        self._generate_compliance_issues(report)
        
        logger.info(f"Template compliance validation complete: {report.overall_score:.2f} "
                   f"({'PASS' if report.is_compliant else 'FAIL'})")
        
        return report
    
    def _validate_sections(self, content: str, template_type: str) -> SectionCompliance:
        """Validate presence and quality of required sections."""
        content_lower = content.lower()
        lines = content.split('\n')
        
        # Find sections using pattern matching
        found_sections = []
        section_details = {}
        
        for section_name, patterns in self.section_patterns.items():
            section_found = False
            for pattern in patterns:
                if re.search(pattern, content_lower, re.MULTILINE | re.IGNORECASE):
                    section_found = True
                    found_sections.append(section_name)
                    
                    # Get section details
                    matches = list(re.finditer(pattern, content_lower, re.MULTILINE | re.IGNORECASE))
                    section_details[section_name] = {
                        'pattern_matched': pattern,
                        'occurrence_count': len(matches),
                        'locations': [match.start() for match in matches]
                    }
                    break
        
        # Determine missing sections
        required_sections = list(self.section_patterns.keys())
        missing_sections = [section for section in required_sections if section not in found_sections]
        
        compliance_percentage = len(found_sections) / len(required_sections)
        
        return SectionCompliance(
            required_count=len(required_sections),
            found_count=len(found_sections),
            missing_sections=missing_sections,
            compliance_percentage=compliance_percentage,
            section_details=section_details
        )
    
    def _validate_word_counts(self, content: str, template_type: str) -> WordCountCompliance:
        """Validate word counts for different sections."""
        # Extract sections and count words
        sections = self._extract_sections_with_content(content)
        
        # Target word counts by template type (updated for balanced length)
        word_count_targets = {
            'technical_deep_dive': {
                'introduction': 350,                    # 300-400 words
                'technical_foundation': 600,            # 500-700 words  
                'implementation': 650,                  # 500-800 words (Deep Technical Analysis)
                'practical_applications': 450,          # 400-500 words
                'future_outlook': 200,                  # 150-250 words
                'conclusion': 200                       # 150-250 words
            },
            'trend_analysis': {
                'overview': 350,
                'market_analysis': 600,
                'implications': 450,
                'recommendations': 350
            },
            'general': {
                'introduction': 300,
                'main_content': 800,
                'conclusion': 200
            }
        }
        
        targets = word_count_targets.get(template_type, word_count_targets['general'])
        section_data = {}
        
        for section_name, target_words in targets.items():
            actual_words = len(sections.get(section_name, "").split())
            min_acceptable = int(target_words * 0.7)  # 70% of target
            max_acceptable = int(target_words * 1.5)  # 150% of target
            
            is_compliant = min_acceptable <= actual_words <= max_acceptable
            
            section_data[section_name] = WordCountData(
                target=target_words,
                actual=actual_words,
                min_acceptable=min_acceptable,
                max_acceptable=max_acceptable,
                is_compliant=is_compliant
            )
        
        return WordCountCompliance(section_data=section_data)
    
    def _validate_required_elements(self, content: str, template_type: str) -> ElementsCompliance:
        """Validate presence of required elements."""
        content_lower = content.lower()
        required = self.required_elements.get(template_type, [])
        
        found_elements = []
        element_locations = {}
        
        for element in required:
            # Create flexible pattern for element detection
            element_pattern = element.replace(' ', r'\s+')
            if re.search(element_pattern, content_lower, re.IGNORECASE):
                found_elements.append(element)
                
                # Find locations
                matches = list(re.finditer(element_pattern, content_lower, re.IGNORECASE))
                element_locations[element] = [match.start() for match in matches]
        
        missing_elements = [elem for elem in required if elem not in found_elements]
        compliance_score = len(found_elements) / max(1, len(required))
        
        return ElementsCompliance(
            required_elements=required,
            found_elements=found_elements,
            missing_elements=missing_elements,
            compliance_score=compliance_score,
            element_locations=element_locations
        )
    
    def _validate_structure(self, content: str) -> StructuralCompliance:
        """Validate overall newsletter structure with balanced formatting allowance."""
        lines = content.strip().split('\n')
        
        # Check for title (first line should be H1 or bold)
        has_title = bool(lines and (lines[0].startswith('#') or 
                                  (lines[0].startswith('*') and lines[0].endswith('*'))))
        
        # Check for introduction (first few paragraphs)
        has_introduction = bool(re.search(r'introduction|brief|overview', content[:500], re.IGNORECASE))
        
        # Check for conclusion (last few paragraphs)
        has_conclusion = bool(re.search(r'conclusion|summary|call.*action', content[-500:], re.IGNORECASE))
        
        # Check for section headers
        section_headers = re.findall(r'^#{2,6}\s+.+$', content, re.MULTILINE)
        has_sections = len(section_headers) >= 3
        section_count = len(section_headers)
        
        # Check heading hierarchy (should be properly nested)
        heading_levels = [len(re.match(r'^#+', header).group()) for header in section_headers]
        heading_structure_valid = self._validate_heading_hierarchy(heading_levels)
        
        # Check for balanced formatting (not excessive structured content)
        bulleted_lists = re.findall(r'^[\s]*[-*•]\s+.+$', content, re.MULTILINE)
        tables = re.findall(r'^\s*\|.+\|\s*$', content, re.MULTILINE)
        
        # Allow reasonable amount of structured content
        reasonable_formatting = (
            len(bulleted_lists) <= 20 and  # Max ~20 bullet points total
            len(tables) <= 15               # Max ~15 table rows total
        )
        
        # Calculate structural compliance score
        structure_checks = [has_title, has_introduction, has_conclusion, has_sections, heading_structure_valid, reasonable_formatting]
        compliance_score = sum(structure_checks) / len(structure_checks)
        
        return StructuralCompliance(
            has_title=has_title,
            has_introduction=has_introduction,
            has_conclusion=has_conclusion,
            has_sections=has_sections,
            section_count=section_count,
            heading_structure_valid=heading_structure_valid,
            compliance_score=compliance_score
        )
    
    def _validate_completion(self, content: str) -> CompletionCompliance:
        """Validate content completion and detect abrupt cutoffs."""
        content = content.strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        if not lines:
            return CompletionCompliance(
                ends_properly=False,
                has_complete_sentences=False,
                minimum_length_met=False,
                no_abrupt_cutoff=False,
                completion_score=0.0,
                issues=["Content is empty"]
            )
        
        # Find last substantial content line
        last_content_line = ""
        for line in reversed(lines):
            if not line.startswith('#') and not line.startswith('-') and len(line) > 20:
                last_content_line = line
                break
        
        issues = []
        
        # Check for proper ending
        proper_endings = ['.', '!', '?', ':', '---', ')', ']']
        ends_properly = any(last_content_line.endswith(ending) for ending in proper_endings)
        
        if not ends_properly:
            issues.append("Content does not end with proper punctuation")
        
        # Check for incomplete sentences
        incomplete_indicators = [',', 'and', 'or', 'the', 'a ', 'an ', 'in ', 'on ', 'at ', 'to ', 'for ']
        has_incomplete = any(last_content_line.rstrip().endswith(indicator) for indicator in incomplete_indicators)
        
        if has_incomplete:
            issues.append("Last sentence appears incomplete")
        
        # Check for abrupt cutoff patterns
        cutoff_patterns = [
            r'-$',  # Ends with dash
            r'\s+$',  # Ends with whitespace
            r'[,;]$',  # Ends with comma or semicolon
            r'\b(and|or|but|however|therefore|thus|also|additionally|furthermore|moreover|meanwhile|subsequently|consequently|nevertheless|nonetheless|furthermore|additionally)\s*$'
        ]
        
        abrupt_cutoff = any(re.search(pattern, last_content_line, re.IGNORECASE) for pattern in cutoff_patterns)
        
        if abrupt_cutoff:
            issues.append("Content appears to end abruptly")
        
        # Check minimum length
        word_count = len(content.split())
        minimum_length_met = word_count >= 500  # Minimum 500 words
        
        if not minimum_length_met:
            issues.append(f"Content too short: {word_count} words (minimum 500)")
        
        # Check for complete sentences throughout
        sentences = re.split(r'[.!?]+', content)
        complete_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        has_complete_sentences = len(complete_sentences) >= 5
        
        if not has_complete_sentences:
            issues.append("Insufficient complete sentences")
        
        # Calculate completion score
        completion_checks = [
            ends_properly,
            not has_incomplete,
            not abrupt_cutoff,
            minimum_length_met,
            has_complete_sentences
        ]
        
        completion_score = sum(completion_checks) / len(completion_checks)
        
        return CompletionCompliance(
            ends_properly=ends_properly,
            has_complete_sentences=has_complete_sentences,
            minimum_length_met=minimum_length_met,
            no_abrupt_cutoff=not abrupt_cutoff,
            completion_score=completion_score,
            last_content_line=last_content_line,
            issues=issues
        )
    
    def _extract_sections_with_content(self, content: str) -> Dict[str, str]:
        """Extract sections with their content for word count analysis."""
        sections = {}
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            # Check if line is a section header
            header_match = re.match(r'^#{1,6}\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = header_match.group(1).lower().strip()
                current_content = []
            else:
                # Add to current section content
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _validate_heading_hierarchy(self, heading_levels: List[int]) -> bool:
        """Validate that heading levels follow proper hierarchy."""
        if not heading_levels:
            return False
        
        # Check for logical progression (no skipping levels)
        for i in range(1, len(heading_levels)):
            level_diff = heading_levels[i] - heading_levels[i-1]
            if level_diff > 1:  # Skipped a level
                return False
        
        return True
    
    def _generate_compliance_issues(self, report: ComplianceReport):
        """Generate issues and recommendations based on compliance results."""
        
        # Critical issues
        if report.completion_compliance and not report.completion_compliance.is_compliant:
            report.add_critical_issue("Content appears incomplete or cut off")
            
            for issue in report.completion_compliance.issues:
                report.add_critical_issue(f"Completion issue: {issue}")
        
        if report.section_compliance and not report.section_compliance.is_compliant:
            missing = ', '.join(report.section_compliance.missing_sections)
            report.add_critical_issue(f"Missing required sections: {missing}")
        
        # Warnings
        if report.word_count_compliance and not report.word_count_compliance.is_compliant:
            report.add_warning("Word count targets not met for some sections")
        
        if report.structural_compliance and not report.structural_compliance.is_compliant:
            report.add_warning("Structural compliance issues detected")
        
        # Recommendations
        if report.elements_compliance and not report.elements_compliance.is_compliant:
            missing = ', '.join(report.elements_compliance.missing_elements)
            report.add_recommendation(f"Consider adding missing elements: {missing}")
        
        if report.overall_score < 0.95:
            report.add_recommendation("Consider reviewing content against template guidelines")


def validate_newsletter_compliance(content: str, template_type: str = "general", 
                                 compliance_level: ComplianceLevel = ComplianceLevel.STANDARD) -> ComplianceReport:
    """
    Convenience function to validate newsletter template compliance.
    
    Args:
        content: Newsletter content to validate
        template_type: Type of newsletter template
        compliance_level: Level of compliance validation
        
    Returns:
        ComplianceReport with validation results
    """
    validator = TemplateComplianceValidator(compliance_level)
    return validator.validate_content_compliance(content, template_type)