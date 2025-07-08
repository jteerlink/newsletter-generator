"""
Quality Gate System for Newsletter Content
Integrates template system, content validation, and code generation
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .content_validator import ContentValidator
from .template_manager import AIMLTemplateManager, NewsletterType
from .code_generator import AIMLCodeGenerator, CodeType

logger = logging.getLogger(__name__)

class QualityGateStatus(Enum):
    """Quality gate evaluation status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NEEDS_REVIEW = "needs_review"

@dataclass
class QualityGateResult:
    """Result of quality gate evaluation"""
    status: QualityGateStatus
    overall_score: float
    grade: str
    blocking_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    detailed_metrics: Dict[str, Any]
    template_compliance: Dict[str, Any]
    code_quality: Dict[str, Any]

class NewsletterQualityGate:
    """Comprehensive quality gate for newsletter content"""
    
    def __init__(self):
        self.content_validator = ContentValidator()
        self.template_manager = AIMLTemplateManager()
        self.code_generator = AIMLCodeGenerator()
        
        # Default quality thresholds
        self.default_thresholds = {
            "minimum_overall_score": 7.0,
            "minimum_technical_accuracy": 6.0,
            "minimum_ai_ml_relevance": 7.0,
            "minimum_practical_value": 6.0,
            "minimum_code_quality": 6.0,
            "maximum_repetition_score": 0.6,
            "maximum_unverifiable_claims": 5,
            "maximum_suspicious_quotes": 3
        }
    
    def evaluate_content(self, content: str, template_type: NewsletterType = None,
                        custom_thresholds: Dict[str, float] = None) -> QualityGateResult:
        """Comprehensive content evaluation through quality gate"""
        
        logger.info("Starting comprehensive quality gate evaluation")
        
        # Use custom thresholds if provided
        thresholds = {**self.default_thresholds, **(custom_thresholds or {})}
        
        # Content validation
        content_assessment = self.content_validator.assess_content_quality(content)
        repetition_analysis = self.content_validator.detect_repetition(content)
        expert_analysis = self.content_validator.analyze_expert_quotes(content)
        factual_analysis = self.content_validator.analyze_factual_claims(content)
        
        # Template compliance (if template specified)
        template_compliance = self._evaluate_template_compliance(content, template_type)
        
        # Code quality assessment
        code_quality = self._evaluate_code_quality(content)
        
        # Overall evaluation
        gate_result = self._compile_gate_result(
            content_assessment, repetition_analysis, expert_analysis, 
            factual_analysis, template_compliance, code_quality, thresholds
        )
        
        logger.info(f"Quality gate evaluation completed: {gate_result.status.value}")
        return gate_result
    
    def _evaluate_template_compliance(self, content: str, 
                                    template_type: NewsletterType = None) -> Dict[str, Any]:
        """Evaluate content compliance with template requirements"""
        
        if template_type is None:
            return {"template_used": None, "compliance_score": 0.0, "issues": []}
        
        template = self.template_manager.get_template(template_type)
        if not template:
            return {"template_used": None, "compliance_score": 0.0, 
                   "issues": ["Template not found"]}
        
        compliance_score = 8.0  # Base score
        issues = []
        
        # Check section presence
        missing_sections = []
        for section in template.sections:
            # Simple check for section keywords in content
            section_keywords = section.name.lower().split()
            if not any(keyword in content.lower() for keyword in section_keywords):
                missing_sections.append(section.name)
        
        if missing_sections:
            compliance_score -= len(missing_sections) * 0.5
            issues.append(f"Missing sections: {', '.join(missing_sections)}")
        
        # Check target audience alignment
        if template.target_audience:
            audience_keywords = template.target_audience.lower().split()
            if not any(keyword in content.lower() for keyword in audience_keywords):
                compliance_score -= 1.0
                issues.append("Content may not align with target audience")
        
        # Check special instructions compliance
        if template.special_instructions:
            for instruction in template.special_instructions:
                if "code" in instruction.lower() and "```" not in content:
                    compliance_score -= 1.0
                    issues.append("Template requires code examples but none found")
        
        return {
            "template_used": template.name,
            "compliance_score": max(0.0, compliance_score),
            "issues": issues,
            "missing_sections": missing_sections
        }
    
    def _evaluate_code_quality(self, content: str) -> Dict[str, Any]:
        """Evaluate quality of code examples in content"""
        
        import re
        
        # Extract code blocks
        code_blocks = re.findall(r'```(?:python)?\n(.*?)```', content, re.DOTALL)
        
        if not code_blocks:
            return {
                "code_blocks_found": 0,
                "average_quality": 0.0,
                "issues": ["No code blocks found"],
                "total_score": 5.0  # Neutral score for no code
            }
        
        total_score = 0.0
        all_issues = []
        
        for i, code_block in enumerate(code_blocks):
            validation_result = self.code_generator.validate_code(code_block)
            total_score += validation_result["score"]
            
            if validation_result["issues"]:
                all_issues.extend([f"Code block {i+1}: {issue}" 
                                  for issue in validation_result["issues"]])
        
        average_quality = total_score / len(code_blocks)
        
        return {
            "code_blocks_found": len(code_blocks),
            "average_quality": average_quality,
            "issues": all_issues,
            "total_score": average_quality
        }
    
    def _compile_gate_result(self, content_assessment: Dict[str, Any], 
                           repetition_analysis: Dict[str, Any],
                           expert_analysis: Dict[str, Any], 
                           factual_analysis: Dict[str, Any],
                           template_compliance: Dict[str, Any],
                           code_quality: Dict[str, Any],
                           thresholds: Dict[str, float]) -> QualityGateResult:
        """Compile all evaluation results into final quality gate result"""
        
        blocking_issues = []
        warnings = []
        recommendations = []
        
        # Check blocking conditions
        overall_score = content_assessment["overall_score"]
        if overall_score < thresholds["minimum_overall_score"]:
            blocking_issues.append(
                f"Overall score ({overall_score:.2f}) below minimum ({thresholds['minimum_overall_score']:.2f})"
            )
        
        # Check technical accuracy
        tech_accuracy = content_assessment["quality_metrics"]["technical_accuracy"]
        if tech_accuracy < thresholds["minimum_technical_accuracy"]:
            blocking_issues.append(
                f"Technical accuracy ({tech_accuracy:.2f}) below minimum ({thresholds['minimum_technical_accuracy']:.2f})"
            )
        
        # Check AI/ML relevance
        ai_ml_relevance = content_assessment["quality_metrics"]["ai_ml_relevance"]
        if ai_ml_relevance < thresholds["minimum_ai_ml_relevance"]:
            blocking_issues.append(
                f"AI/ML relevance ({ai_ml_relevance:.2f}) below minimum ({thresholds['minimum_ai_ml_relevance']:.2f})"
            )
        
        # Check repetition
        repetition_score = repetition_analysis["repetition_score"]
        if repetition_score > thresholds["maximum_repetition_score"]:
            blocking_issues.append(
                f"Repetition score ({repetition_score:.2f}) above maximum ({thresholds['maximum_repetition_score']:.2f})"
            )
        
        # Check factual claims
        unverifiable_claims = len(factual_analysis["claims"])
        if unverifiable_claims > thresholds["maximum_unverifiable_claims"]:
            blocking_issues.append(
                f"Too many unverifiable claims ({unverifiable_claims} > {thresholds['maximum_unverifiable_claims']})"
            )
        
        # Check suspicious quotes
        suspicious_quotes = expert_analysis["suspicious_quotes"]
        suspicious_quotes_count = len(suspicious_quotes) if isinstance(suspicious_quotes, list) else suspicious_quotes
        if suspicious_quotes_count > thresholds["maximum_suspicious_quotes"]:
            blocking_issues.append(
                f"Too many suspicious quotes ({suspicious_quotes_count} > {thresholds['maximum_suspicious_quotes']})"
            )
        
        # Check code quality
        code_score = code_quality["total_score"]
        if code_score < thresholds["minimum_code_quality"]:
            warnings.append(
                f"Code quality ({code_score:.2f}) below recommended minimum ({thresholds['minimum_code_quality']:.2f})"
            )
        
        # Add template compliance warnings
        if template_compliance["issues"]:
            warnings.extend(template_compliance["issues"])
        
        # Add code quality issues as warnings
        if code_quality["issues"]:
            warnings.extend(code_quality["issues"])
        
        # Add improvement recommendations
        recommendations.extend(content_assessment["improvement_recommendations"])
        
        # Determine overall status
        if blocking_issues:
            status = QualityGateStatus.FAILED
        elif warnings:
            status = QualityGateStatus.WARNING
        elif overall_score < 8.0:
            status = QualityGateStatus.NEEDS_REVIEW
        else:
            status = QualityGateStatus.PASSED
        
        return QualityGateResult(
            status=status,
            overall_score=overall_score,
            grade=content_assessment["grade"],
            blocking_issues=blocking_issues,
            warnings=warnings,
            recommendations=recommendations,
            detailed_metrics=content_assessment["quality_metrics"],
            template_compliance=template_compliance,
            code_quality=code_quality
        )
    
    def generate_enhancement_suggestions(self, content: str, 
                                       template_type: NewsletterType = None) -> Dict[str, Any]:
        """Generate specific suggestions for content enhancement"""
        
        suggestions = {
            "code_examples": [],
            "structural_improvements": [],
            "content_enhancements": [],
            "technical_additions": []
        }
        
        # Suggest code examples based on content
        if template_type:
            template = self.template_manager.get_template(template_type)
            if template and any("code" in instruction.lower() for instruction in template.special_instructions):
                # Extract topics that could benefit from code examples
                ai_ml_topics = [
                    "neural network", "machine learning", "deep learning", 
                    "transformer", "classification", "regression"
                ]
                
                for topic in ai_ml_topics:
                    if topic in content.lower():
                        suggested_framework = self.code_generator.suggest_framework(topic)
                        suggestions["code_examples"].append({
                            "topic": topic,
                            "framework": suggested_framework,
                            "reason": f"Content mentions {topic} but lacks code examples"
                        })
        
        # Suggest structural improvements
        if "```" not in content:
            suggestions["structural_improvements"].append(
                "Add code examples to illustrate technical concepts"
            )
        
        # Content quality assessment
        quality_assessment = self.content_validator.assess_content_quality(content)
        
        # Add specific technical suggestions
        if quality_assessment["quality_metrics"]["technical_accuracy"] < 7.0:
            suggestions["technical_additions"].append(
                "Include more specific technical details and accurate terminology"
            )
        
        if quality_assessment["quality_metrics"]["practical_value"] < 7.0:
            suggestions["content_enhancements"].append(
                "Add more practical examples and real-world applications"
            )
        
        return suggestions
    
    def create_quality_report(self, gate_result: QualityGateResult) -> str:
        """Generate a comprehensive quality report"""
        
        report = f"""
# Newsletter Quality Gate Report

## Overall Assessment
- **Status**: {gate_result.status.value.upper()}
- **Overall Score**: {gate_result.overall_score:.2f}/10.0
- **Grade**: {gate_result.grade}

## Detailed Metrics
"""
        
        for metric, score in gate_result.detailed_metrics.items():
            report += f"- **{metric.replace('_', ' ').title()}**: {score:.2f}/10.0\n"
        
        if gate_result.blocking_issues:
            report += f"\n## 🚫 Blocking Issues\n"
            for issue in gate_result.blocking_issues:
                report += f"- {issue}\n"
        
        if gate_result.warnings:
            report += f"\n## ⚠️ Warnings\n"
            for warning in gate_result.warnings:
                report += f"- {warning}\n"
        
        if gate_result.recommendations:
            report += f"\n## 💡 Recommendations\n"
            for rec in gate_result.recommendations:
                report += f"- {rec}\n"
        
        # Template compliance
        if gate_result.template_compliance["template_used"]:
            report += f"\n## 📋 Template Compliance\n"
            report += f"- **Template**: {gate_result.template_compliance['template_used']}\n"
            report += f"- **Compliance Score**: {gate_result.template_compliance['compliance_score']:.2f}/10.0\n"
        
        # Code quality
        if gate_result.code_quality["code_blocks_found"] > 0:
            report += f"\n## 💻 Code Quality\n"
            report += f"- **Code Blocks Found**: {gate_result.code_quality['code_blocks_found']}\n"
            report += f"- **Average Quality**: {gate_result.code_quality['average_quality']:.2f}/10.0\n"
        
        return report 