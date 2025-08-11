"""
Template Manager for AI/ML Newsletter Generation
Provides specialized templates for different newsletter types and content frameworks
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NewsletterType(Enum):
    """Types of newsletters supported by the template system"""
    TECHNICAL_DEEP_DIVE = "technical_deep_dive"
    TREND_ANALYSIS = "trend_analysis"
    PRODUCT_REVIEW = "product_review"
    RESEARCH_SUMMARY = "research_summary"
    TUTORIAL_GUIDE = "tutorial_guide"


@dataclass
class TemplateSection:
    """Represents a section within a newsletter template"""
    name: str
    description: str
    content_guidelines: List[str]
    word_count_target: int
    required_elements: List[str]
    optional_elements: List[str]


@dataclass
class NewsletterTemplate:
    """Complete newsletter template with all sections and metadata"""
    name: str
    type: NewsletterType
    description: str
    target_audience: str
    sections: List[TemplateSection]
    total_word_target: int
    special_instructions: List[str]


class AIMLTemplateManager:
    """Manages AI/ML focused newsletter templates and content frameworks"""

    def __init__(self):
        self.templates = self._initialize_templates()
        self.content_frameworks = self._initialize_content_frameworks()

    def _initialize_templates(
            self) -> Dict[NewsletterType, NewsletterTemplate]:
        """Initialize all AI/ML newsletter templates"""
        templates = {}

        # Technical Deep-Dive Template
        templates[NewsletterType.TECHNICAL_DEEP_DIVE] = NewsletterTemplate(
            name="AI/ML Technical Deep-Dive",
            type=NewsletterType.TECHNICAL_DEEP_DIVE,
            description="In-depth technical analysis of AI/ML concepts, architectures, and implementations",
            target_audience="AI/ML engineers, researchers, and technical professionals",
            sections=[
                TemplateSection(
                    name="Executive Summary",
                    description="High-level overview of the technical topic and its significance",
                    content_guidelines=[
                        "Start with the core problem or opportunity being addressed",
                        "Explain why this topic matters to the AI/ML community",
                        "Provide a clear roadmap of what readers will learn",
                        "Include practical applications and real-world relevance"
                    ],
                    word_count_target=400,
                    required_elements=["problem statement", "significance", "key takeaways"],
                    optional_elements=["industry context", "historical perspective"]
                ),
                TemplateSection(
                    name="Technical Foundation",
                    description="Core concepts, mathematical foundations, and theoretical background",
                    content_guidelines=[
                        "Explain fundamental concepts clearly and accurately",
                        "Include mathematical formulations where relevant",
                        "Provide intuitive explanations alongside technical details",
                        "Use analogies to make complex concepts accessible"
                    ],
                    word_count_target=800,
                    required_elements=["core concepts", "mathematical foundations"],
                    optional_elements=["historical development", "alternative approaches"]
                ),
                TemplateSection(
                    name="Architecture Deep-Dive",
                    description="Detailed exploration of system architecture, model design, and implementation details",
                    content_guidelines=[
                        "Provide detailed architectural diagrams and explanations",
                        "Explain design decisions and trade-offs",
                        "Include code examples and implementation details",
                        "Discuss scalability and performance considerations"
                    ],
                    word_count_target=1000,
                    required_elements=["architecture overview", "implementation details", "code examples"],
                    optional_elements=["performance benchmarks", "comparison with alternatives"]
                ),
                TemplateSection(
                    name="Practical Implementation",
                    description="Hands-on implementation guide with code examples and best practices",
                    content_guidelines=[
                        "Provide working code examples with explanations",
                        "Include step-by-step implementation guides",
                        "Discuss common pitfalls and how to avoid them",
                        "Share optimization techniques and best practices",
                        "Generate validated, executable code examples",
                        "Include multiple complexity levels (beginner to advanced)",
                        "Use appropriate frameworks (PyTorch, TensorFlow, etc.)"
                    ],
                    word_count_target=800,
                    required_elements=["code examples", "implementation steps", "best practices", "working code"],
                    optional_elements=["troubleshooting guide", "performance tips", "code validation results"]
                ),
                TemplateSection(
                    name="Real-World Applications",
                    description="Case studies and practical applications in industry",
                    content_guidelines=[
                        "Present concrete case studies from industry",
                        "Explain how the technology solves real problems",
                        "Include performance metrics and results",
                        "Discuss lessons learned and insights gained"
                    ],
                    word_count_target=600,
                    required_elements=["case studies", "practical applications"],
                    optional_elements=["industry partnerships", "success metrics"]
                ),
                TemplateSection(
                    name="Future Directions",
                    description="Emerging trends, research directions, and future possibilities",
                    content_guidelines=[
                        "Identify current research trends and developments",
                        "Discuss potential future applications and improvements",
                        "Highlight ongoing challenges and opportunities",
                        "Provide actionable insights for readers"
                    ],
                    word_count_target=400,
                    required_elements=["research trends", "future possibilities"],
                    optional_elements=["research roadmap", "collaboration opportunities"]
                )
            ],
            total_word_target=4000,
            special_instructions=[
                "Focus on technical accuracy and depth",
                "Include working code examples in Python",
                "Provide mathematical formulations where appropriate",
                "Balance technical detail with accessibility",
                "Include references to papers and resources"
            ]
        )

        # Trend Analysis Template
        templates[NewsletterType.TREND_ANALYSIS] = NewsletterTemplate(
            name="AI/ML Trend Analysis",
            type=NewsletterType.TREND_ANALYSIS,
            description="Analysis of emerging trends, market developments, and technological shifts in AI/ML",
            target_audience="AI/ML professionals, business leaders, and technology strategists",
            sections=[
                TemplateSection(
                    name="Trend Overview",
                    description="Introduction to the key trends and their significance",
                    content_guidelines=[
                        "Identify and categorize major trends in AI/ML",
                        "Explain the driving forces behind these trends",
                        "Provide context for why these trends matter now",
                        "Set up the analytical framework for the newsletter"
                    ],
                    word_count_target=400,
                    required_elements=["trend identification", "significance analysis"],
                    optional_elements=["historical context", "global perspective"]
                ),
                TemplateSection(
                    name="Market Analysis",
                    description="Analysis of market forces, adoption patterns, and business implications",
                    content_guidelines=[
                        "Analyze market size, growth rates, and investment trends",
                        "Identify key players and competitive dynamics",
                        "Discuss adoption patterns across industries",
                        "Examine business model innovations and disruptions"
                    ],
                    word_count_target=800,
                    required_elements=["market data", "competitive analysis", "adoption patterns"],
                    optional_elements=["investment flows", "regulatory factors"]
                ),
                TemplateSection(
                    name="Technology Evolution",
                    description="Deep dive into technological developments and innovations",
                    content_guidelines=[
                        "Analyze technological breakthroughs and improvements",
                        "Discuss convergence of different technologies",
                        "Explain implications for existing systems and workflows",
                        "Identify emerging technology combinations"
                    ],
                    word_count_target=700,
                    required_elements=["technology analysis", "innovation assessment"],
                    optional_elements=["convergence opportunities", "disruption potential"]
                ),
                TemplateSection(
                    name="Industry Impact",
                    description="Sector-specific analysis of how trends affect different industries",
                    content_guidelines=[
                        "Analyze impact across key industries (healthcare, finance, retail, etc.)",
                        "Identify winners and losers in the shifting landscape",
                        "Discuss transformation timelines and implementation challenges",
                        "Highlight innovative use cases and success stories"
                    ],
                    word_count_target=800,
                    required_elements=["industry analysis", "transformation examples"],
                    optional_elements=["implementation timelines", "success metrics"]
                ),
                TemplateSection(
                    name="Strategic Implications",
                    description="Strategic recommendations and action items for different stakeholders",
                    content_guidelines=[
                        "Provide strategic recommendations for businesses",
                        "Identify opportunities for innovation and growth",
                        "Discuss risk mitigation strategies",
                        "Offer actionable insights for decision-makers"
                    ],
                    word_count_target=600,
                    required_elements=["strategic recommendations", "action items"],
                    optional_elements=["risk assessment", "opportunity mapping"]
                ),
                TemplateSection(
                    name="Future Outlook",
                    description="Predictions and scenarios for future developments",
                    content_guidelines=[
                        "Develop scenarios for trend evolution",
                        "Identify potential inflection points and catalysts",
                        "Discuss uncertainty factors and wild cards",
                        "Provide timeline estimates for key developments"
                    ],
                    word_count_target=500,
                    required_elements=["future scenarios", "timeline predictions"],
                    optional_elements=["uncertainty analysis", "catalyst identification"]
                )
            ],
            total_word_target=3800,
            special_instructions=[
                "Focus on data-driven analysis with supporting evidence",
                "Include market data, research findings, and documented insights",
                "Balance optimism with realistic assessment of challenges",
                "Provide actionable insights for different audience segments",
                "Include references to credible sources and research"
            ]
        )

        # Product Review Template
        templates[NewsletterType.PRODUCT_REVIEW] = NewsletterTemplate(
            name="AI/ML Product Review",
            type=NewsletterType.PRODUCT_REVIEW,
            description="Comprehensive review of AI/ML tools, frameworks, and platforms",
            target_audience="AI/ML practitioners, developers, and technology evaluators",
            sections=[
                TemplateSection(
                    name="Product Overview",
                    description="Introduction to the product and its positioning in the market",
                    content_guidelines=[
                        "Provide clear product description and purpose",
                        "Explain the problem the product solves",
                        "Identify target users and use cases",
                        "Position the product in the competitive landscape"
                    ],
                    word_count_target=400,
                    required_elements=["product description", "target users", "problem statement"],
                    optional_elements=["company background", "market positioning"]
                ),
                TemplateSection(
                    name="Technical Evaluation",
                    description="In-depth technical analysis of features, capabilities, and architecture",
                    content_guidelines=[
                        "Evaluate core features and functionality",
                        "Analyze technical architecture and design decisions",
                        "Assess performance, scalability, and reliability",
                        "Compare with existing solutions and alternatives"
                    ],
                    word_count_target=800,
                    required_elements=["feature analysis", "performance evaluation", "architecture review"],
                    optional_elements=["benchmark comparisons", "scalability testing"]
                ),
                TemplateSection(
                    name="Hands-On Testing",
                    description="Practical testing results with code examples and usage scenarios",
                    content_guidelines=[
                        "Provide detailed testing methodology and setup",
                        "Include code examples and implementation details",
                        "Document testing results and performance metrics",
                        "Share practical insights and observations"
                    ],
                    word_count_target=700,
                    required_elements=["testing methodology", "code examples", "results analysis"],
                    optional_elements=["performance benchmarks", "comparison tests"]
                ),
                TemplateSection(
                    name="User Experience",
                    description="Evaluation of usability, documentation, and developer experience",
                    content_guidelines=[
                        "Assess ease of use and learning curve",
                        "Evaluate documentation quality and completeness",
                        "Analyze developer tools and workflow integration",
                        "Consider community support and ecosystem"
                    ],
                    word_count_target=500,
                    required_elements=["usability assessment", "documentation review"],
                    optional_elements=["community analysis", "ecosystem evaluation"]
                ),
                TemplateSection(
                    name="Pros and Cons",
                    description="Balanced analysis of strengths and weaknesses",
                    content_guidelines=[
                        "Identify key strengths and unique advantages",
                        "Highlight limitations and potential drawbacks",
                        "Provide context for when to use or avoid the product",
                        "Consider different user personas and use cases"
                    ],
                    word_count_target=400,
                    required_elements=["strengths analysis", "limitations assessment"],
                    optional_elements=["use case recommendations", "persona-specific insights"]
                ),
                TemplateSection(
                    name="Recommendation",
                    description="Final verdict and recommendations for different user types",
                    content_guidelines=[
                        "Provide clear recommendation based on evaluation",
                        "Identify ideal use cases and user types",
                        "Compare with alternatives and provide selection guidance",
                        "Include implementation tips and best practices"
                    ],
                    word_count_target=400,
                    required_elements=["final verdict", "use case recommendations"],
                    optional_elements=["implementation tips", "alternative suggestions"]
                )
            ],
            total_word_target=3200,
            special_instructions=[
                "Maintain objectivity and balance in evaluation",
                "Include hands-on testing with real code examples",
                "Provide practical insights for decision-making",
                "Consider different user perspectives and needs",
                "Include relevant benchmarks and performance data"
            ]
        )

        return templates

    def _initialize_content_frameworks(self) -> Dict[str, Dict[str, Any]]:
        """Initialize content frameworks for common AI/ML newsletter sections"""
        return {
            "technical_explanation": {
                "structure": [
                    "Concept introduction with clear definition",
                    "Mathematical or algorithmic foundation",
                    "Intuitive explanation with analogies",
                    "Code implementation example",
                    "Practical applications and use cases"
                ],
                "guidelines": [
                    "Start with the big picture before diving into details",
                    "Use progressive disclosure of complexity",
                    "Include both theoretical and practical perspectives",
                    "Provide working code examples in Python",
                    "Connect to real-world applications"
                ]
            },
            "research_summary": {
                "structure": [
                    "Research context and motivation",
                    "Key methodology and approach",
                    "Primary findings and results",
                    "Implications and significance",
                    "Future research directions"
                ],
                "guidelines": [
                    "Explain research in accessible terms",
                    "Highlight practical implications",
                    "Include key metrics and results",
                    "Discuss limitations and future work",
                    "Connect to broader research trends"
                ]
            },
            "tool_comparison": {
                "structure": [
                    "Comparison criteria and methodology",
                    "Feature-by-feature analysis",
                    "Performance and benchmark results",
                    "Use case suitability assessment",
                    "Final recommendations"
                ],
                "guidelines": [
                    "Use objective criteria for comparison",
                    "Include hands-on testing results",
                    "Consider different user perspectives",
                    "Provide clear decision frameworks",
                    "Include cost and resource considerations"
                ]
            },
            "case_study": {
                "structure": [
                    "Problem definition and context",
                    "Solution approach and methodology",
                    "Implementation details and challenges",
                    "Results and impact measurement",
                    "Lessons learned and recommendations"
                ],
                "guidelines": [
                    "Tell a complete story with clear narrative",
                    "Include specific metrics and outcomes",
                    "Discuss both successes and failures",
                    "Extract actionable insights",
                    "Connect to broader industry patterns"
                ]
            }
        }

    def get_template(
            self,
            template_type: NewsletterType) -> NewsletterTemplate:
        """Get a specific newsletter template"""
        return self.templates.get(template_type)

    def get_available_templates(self) -> List[NewsletterTemplate]:
        """Get all available templates"""
        return list(self.templates.values())

    def suggest_template(self, topic: str) -> NewsletterType:
        """Suggest the most appropriate template for a given topic"""
        topic_lower = topic.lower()

        # AI/ML pattern matching for template suggestion
        if any(
            keyword in topic_lower for keyword in [
                "deep learning",
                "neural network",
                "machine learning",
                "ai architecture",
                "transformer",
                "model training",
                "technical analysis",
                "algorithm",
                "implementation",
                "architecture",
                "system design"]):
            return NewsletterType.TECHNICAL_DEEP_DIVE

        elif any(keyword in topic_lower for keyword in [
            "trend", "future", "emerging", "prediction", "market", "industry",
            "analysis", "forecast", "outlook", "development"
        ]):
            return NewsletterType.TREND_ANALYSIS

        elif any(keyword in topic_lower for keyword in [
            "tool", "framework", "library", "review", "comparison", "evaluation",
            "product", "platform", "service", "solution"
        ]):
            return NewsletterType.PRODUCT_REVIEW

        elif any(keyword in topic_lower for keyword in [
            "research", "study", "paper", "academic", "findings", "breakthrough",
            "discovery", "investigation", "analysis", "results"
        ]):
            return NewsletterType.RESEARCH_SUMMARY

        elif any(keyword in topic_lower for keyword in [
            "tutorial", "guide", "how-to", "implementation", "step-by-step",
            "walkthrough", "example", "demonstration", "hands-on"
        ]):
            return NewsletterType.TUTORIAL_GUIDE

        # Default to technical deep dive for AI/ML topics
        return NewsletterType.TECHNICAL_DEEP_DIVE

    def get_content_framework(self, framework_name: str) -> Dict[str, Any]:
        """Get a specific content framework"""
        return self.content_frameworks.get(framework_name)

    def generate_template_prompt(
            self,
            template: NewsletterTemplate,
            topic: str) -> str:
        """Generate a detailed prompt based on a template"""
        prompt_parts = [
            f"Create a comprehensive {
                template.name} newsletter about '{topic}' for {
                template.target_audience}.",
            f"\nNewsletter Description: {
                template.description}",
            f"\nTarget Word Count: {
                template.total_word_target} words",
            "\nSTRUCTURE YOUR NEWSLETTER WITH THE FOLLOWING SECTIONS:"]

        for i, section in enumerate(template.sections, 1):
            prompt_parts.append(
                f"\n{i}. **{section.name}** ({section.word_count_target} words)")
            prompt_parts.append(f"   {section.description}")
            prompt_parts.append("   Content Guidelines:")
            for guideline in section.content_guidelines:
                prompt_parts.append(f"   • {guideline}")
            prompt_parts.append("   Required Elements:")
            for element in section.required_elements:
                prompt_parts.append(f"   • {element}")
            if section.optional_elements:
                prompt_parts.append("   Optional Elements:")
                for element in section.optional_elements:
                    prompt_parts.append(f"   • {element}")

        prompt_parts.append("\nSPECIAL INSTRUCTIONS:")
        for instruction in template.special_instructions:
            prompt_parts.append(f"• {instruction}")

        prompt_parts.append(
            "\nWrite in flowing narrative prose, avoiding bullet points in the final content.")
        prompt_parts.append(
            "Include specific examples, code snippets, and practical insights throughout.")
        prompt_parts.append(
            "Ensure technical accuracy while maintaining accessibility for the target audience.")

        return "\n".join(prompt_parts)

    def validate_template_content(
            self, content: str, template: NewsletterTemplate) -> Dict[str, Any]:
        """Validate content against template requirements"""
        validation_results = {
            "template_compliance": True,
            "issues": [],
            "recommendations": [],
            "section_analysis": {}
        }

        # Basic word count validation
        word_count = len(content.split())
        target_range = (
            template.total_word_target * 0.8,
            template.total_word_target * 1.2)

        if word_count < target_range[0]:
            validation_results["issues"].append(
                f"Content too short: {word_count} words (target: {
                    template.total_word_target})")
            validation_results["template_compliance"] = False
        elif word_count > target_range[1]:
            validation_results["issues"].append(
                f"Content too long: {word_count} words (target: {
                    template.total_word_target})")
            validation_results["template_compliance"] = False

        # Check for required elements in each section
        for section in template.sections:
            section_analysis = {
                "found_elements": [],
                "missing_elements": [],
                "suggestions": []
            }

            for element in section.required_elements:
                # Simple keyword-based checking - could be enhanced with NLP
                if element.lower() in content.lower():
                    section_analysis["found_elements"].append(element)
                else:
                    section_analysis["missing_elements"].append(element)
                    validation_results["template_compliance"] = False

            if section_analysis["missing_elements"]:
                validation_results["issues"].append(
                    f"Missing required elements in {
                        section.name}: {
                        section_analysis['missing_elements']}")

            validation_results["section_analysis"][section.name] = section_analysis

        return validation_results
    
    def enhance_template_with_code_examples(self, template: NewsletterTemplate, 
                                          topic: str) -> NewsletterTemplate:
        """Enhance a template with code example generation capabilities"""
        enhanced_sections = []
        
        for section in template.sections:
            enhanced_section = section
            
            # Add code generation guidelines for technical sections
            if ("implementation" in section.name.lower() or 
                "technical" in section.name.lower() or
                "architecture" in section.name.lower()):
                
                enhanced_guidelines = section.content_guidelines + [
                    "Include 2-3 working code examples with different complexity levels",
                    "Validate all code examples for syntax and executability", 
                    "Provide clear explanations for each code block",
                    "Use framework-appropriate best practices",
                    "Include expected outputs and results"
                ]
                
                enhanced_elements = section.required_elements + ["validated code examples"]
                
                enhanced_section = TemplateSection(
                    name=section.name,
                    description=section.description,
                    content_guidelines=enhanced_guidelines,
                    word_count_target=section.word_count_target,
                    required_elements=enhanced_elements,
                    optional_elements=section.optional_elements
                )
            
            enhanced_sections.append(enhanced_section)
        
        # Update special instructions to include code generation
        enhanced_instructions = template.special_instructions + [
            "Generate working code examples using the integrated code generation system",
            "Validate all code for syntax correctness and executability",
            "Include code examples that demonstrate key concepts practically",
            "Use appropriate AI/ML frameworks (PyTorch, TensorFlow, scikit-learn, etc.)",
            "Provide multiple complexity levels to serve different audiences"
        ]
        
        enhanced_template = NewsletterTemplate(
            name=f"Enhanced {template.name}",
            type=template.type,
            description=f"{template.description} (with integrated code generation)",
            target_audience=template.target_audience,
            sections=enhanced_sections,
            total_word_target=template.total_word_target,
            special_instructions=enhanced_instructions
        )
        
        return enhanced_template
    
    def generate_code_enhanced_prompt(self, template: NewsletterTemplate, 
                                    topic: str, enable_code_generation: bool = True) -> str:
        """Generate a template prompt with code generation capabilities"""
        base_prompt = self.generate_template_prompt(template, topic)
        
        if not enable_code_generation:
            return base_prompt
        
        # Add code generation instructions
        code_instructions = """

## CODE GENERATION REQUIREMENTS

For technical sections, you MUST include working code examples:

1. **Code Quality Standards:**
   - All code must be syntactically correct and executable
   - Include appropriate imports and dependencies
   - Use proper variable names and follow PEP 8 style
   - Add comments to explain complex logic

2. **Code Example Structure:**
   - Provide 2-3 examples per technical section
   - Start with basic examples, progress to more complex
   - Include expected output or results
   - Explain what each code block demonstrates

3. **Framework Selection:**
   - Choose the most appropriate framework for the topic
   - PyTorch for deep learning concepts
   - TensorFlow for production-ready models  
   - scikit-learn for traditional ML
   - pandas for data manipulation
   - NumPy for numerical computing

4. **Code Documentation:**
   - Include docstrings for functions and classes
   - Add inline comments for complex operations
   - Provide usage examples and expected inputs/outputs
   - Explain parameter choices and design decisions

5. **Error Handling:**
   - Include basic error handling where appropriate
   - Show common pitfalls and how to avoid them
   - Provide debugging tips and troubleshooting guidance

IMPORTANT: All code examples will be automatically validated for syntax and tested for execution. Ensure code quality meets professional standards.
"""
        
        return base_prompt + code_instructions
    
    def suggest_code_frameworks_for_topic(self, topic: str) -> List[str]:
        """Suggest appropriate frameworks for a given topic"""
        topic_lower = topic.lower()
        suggested_frameworks = []
        
        # Deep learning frameworks
        if any(keyword in topic_lower for keyword in [
            "neural network", "deep learning", "cnn", "rnn", "transformer"
        ]):
            suggested_frameworks.extend(["pytorch", "tensorflow"])
        
        # NLP frameworks
        if any(keyword in topic_lower for keyword in [
            "nlp", "text", "language", "bert", "gpt"
        ]):
            suggested_frameworks.extend(["huggingface", "pytorch"])
        
        # Traditional ML
        if any(keyword in topic_lower for keyword in [
            "classification", "regression", "clustering", "ensemble"
        ]):
            suggested_frameworks.append("sklearn")
        
        # Data analysis
        if any(keyword in topic_lower for keyword in [
            "data", "analysis", "preprocessing", "visualization"
        ]):
            suggested_frameworks.extend(["pandas", "numpy"])
        
        # Computer vision
        if any(keyword in topic_lower for keyword in [
            "vision", "image", "object detection", "segmentation"
        ]):
            suggested_frameworks.extend(["opencv", "pytorch", "tensorflow"])
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(suggested_frameworks)) or ["python"]
    
    def validate_code_enhanced_content(self, content: str, template: NewsletterTemplate,
                                     require_code_examples: bool = True) -> Dict[str, Any]:
        """Validate content with code example requirements"""
        base_validation = self.validate_template_content(content, template)
        
        if not require_code_examples:
            return base_validation
        
        # Additional validation for code examples
        code_validation = {
            "has_code_blocks": "```python" in content or "```" in content,
            "code_block_count": content.count("```python"),
            "has_code_explanations": False,
            "framework_coverage": [],
            "code_quality_issues": []
        }
        
        # Check for code explanations
        if ("explanation" in content.lower() or 
            "demonstrates" in content.lower() or
            "example shows" in content.lower()):
            code_validation["has_code_explanations"] = True
        
        # Check framework coverage
        frameworks = ["pytorch", "tensorflow", "sklearn", "pandas", "numpy"]
        for framework in frameworks:
            if framework in content.lower():
                code_validation["framework_coverage"].append(framework)
        
        # Validate minimum code requirements for technical templates
        if template.type == NewsletterType.TECHNICAL_DEEP_DIVE:
            if code_validation["code_block_count"] < 2:
                base_validation["issues"].append(
                    "Technical deep-dive requires at least 2 code examples"
                )
                base_validation["template_compliance"] = False
            
            if not code_validation["has_code_explanations"]:
                base_validation["issues"].append(
                    "Code examples need clear explanations"
                )
        
        # Add code-specific validation results
        base_validation["code_validation"] = code_validation
        
        return base_validation
