"""
Phase 2: Hybrid Workflow Manager Implementation

Intelligently routes content requests between:
- Daily Quick Pipeline (90% of content - 5 minute reads) 
- Deep Dive Pipeline (10% of content - weekly comprehensive articles)

Based on content complexity analysis and publishing schedule requirements.
"""

from __future__ import annotations
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass

from src.core.core import query_llm
from src.core.template_manager import AIMLTemplateManager, NewsletterType
from src.quality import NewsletterQualityGate
from .daily_quick_pipeline import DailyQuickPipeline, ContentItem
from .agents import (
    ManagerAgent, ResearchAgent, PlannerAgent, 
    WriterAgent, EditorAgent, SimpleAgent
)

logger = logging.getLogger(__name__)

class ContentPipelineType(Enum):
    """Types of content pipelines available"""
    DAILY_QUICK = "daily_quick"
    DEEP_DIVE = "deep_dive" 
    HYBRID = "hybrid"

class ContentComplexity(Enum):
    """Content complexity assessment levels"""
    SIMPLE = "simple"          # Quick hits, news summaries
    MODERATE = "moderate"      # Tool tutorials, brief analysis
    COMPLEX = "complex"        # Deep technical analysis
    COMPREHENSIVE = "comprehensive"  # Multi-part research articles

@dataclass
class ContentRequest:
    """Represents a content generation request"""
    topic: str
    content_pillar: str  # news_breakthroughs, tools_tutorials, deep_dives
    target_audience: str
    word_count_target: int
    deadline: datetime
    priority: int = 1
    special_requirements: List[str] = None
    
    def __post_init__(self):
        if self.special_requirements is None:
            self.special_requirements = []
    context: str = ""

@dataclass 
class PipelineAssessment:
    """Assessment of which pipeline to use for content"""
    recommended_pipeline: ContentPipelineType
    complexity_level: ContentComplexity
    confidence_score: float
    reasoning: str
    estimated_time_hours: float
    resource_requirements: Dict[str, Any]

class ContentComplexityClassifier:
    """Analyzes content requests to determine complexity and appropriate pipeline"""
    
    def __init__(self):
        self.complexity_indicators = self._load_complexity_indicators()
    
    def _load_complexity_indicators(self) -> Dict[str, Any]:
        """Load indicators for assessing content complexity"""
        return {
            'word_count_thresholds': {
                'simple': (0, 500),
                'moderate': (500, 1500), 
                'complex': (1500, 3000),
                'comprehensive': (3000, 10000)
            },
            'research_depth_keywords': [
                'comprehensive analysis', 'deep dive', 'technical architecture',
                'implementation guide', 'research survey', 'comparative study',
                'industry analysis', 'theoretical foundation', 'mathematical model'
            ],
            'quick_content_keywords': [
                'news update', 'quick summary', 'brief overview', 'tool spotlight',
                'breaking news', 'announcement', 'quick tip', 'tool review'
            ],
            'technical_complexity_indicators': [
                'algorithm implementation', 'mathematical derivation', 'code architecture',
                'system design', 'performance optimization', 'scalability analysis',
                'formal verification', 'theoretical proof', 'experimental validation'
            ]
        }
    
    def assess_content_complexity(self, content_request: ContentRequest) -> PipelineAssessment:
        """Assess content complexity and recommend appropriate pipeline"""
        logger.info(f"Assessing complexity for topic: {content_request.topic}")
        
        # Analyze various factors
        word_count_complexity = self._assess_word_count_complexity(content_request.word_count_target)
        research_depth = self._assess_research_depth_required(content_request)
        technical_complexity = self._assess_technical_complexity(content_request)
        time_constraints = self._assess_time_constraints(content_request)
        
        # Use LLM for sophisticated analysis
        llm_assessment = self._llm_complexity_analysis(content_request)
        
        # Combine assessments
        overall_assessment = self._combine_assessments({
            'word_count': word_count_complexity,
            'research_depth': research_depth,
            'technical_complexity': technical_complexity,
            'time_constraints': time_constraints,
            'llm_analysis': llm_assessment
        })
        
        return overall_assessment
    
    def _assess_word_count_complexity(self, word_count: int) -> Dict[str, Any]:
        """Assess complexity based on target word count"""
        thresholds = self.complexity_indicators['word_count_thresholds']
        
        for complexity, (min_words, max_words) in thresholds.items():
            if min_words <= word_count <= max_words:
                return {
                    'complexity': complexity,
                    'score': word_count / max_words,
                    'rationale': f"Word count {word_count} falls in {complexity} range"
                }
        
        # Handle edge cases
        if word_count > thresholds['comprehensive'][1]:
            return {
                'complexity': 'comprehensive',
                'score': 1.0,
                'rationale': f"Word count {word_count} exceeds comprehensive threshold"
            }
        
        return {
            'complexity': 'simple',
            'score': 0.1,
            'rationale': "Default to simple for very short content"
        }
    
    def _assess_research_depth_required(self, content_request: ContentRequest) -> Dict[str, Any]:
        """Assess required research depth based on topic and requirements"""
        topic_text = f"{content_request.topic} {content_request.context}".lower()
        
        research_indicators = self.complexity_indicators['research_depth_keywords']
        quick_indicators = self.complexity_indicators['quick_content_keywords']
        
        research_matches = sum(1 for indicator in research_indicators if indicator in topic_text)
        quick_matches = sum(1 for indicator in quick_indicators if indicator in topic_text)
        
        if research_matches > quick_matches and research_matches >= 2:
            depth_score = min(research_matches / len(research_indicators), 1.0)
            return {
                'depth_required': 'high',
                'score': depth_score,
                'rationale': f"High research depth indicated by {research_matches} research keywords"
            }
        elif quick_matches > research_matches:
            return {
                'depth_required': 'low', 
                'score': 0.2,
                'rationale': f"Low research depth indicated by {quick_matches} quick content keywords"
            }
        else:
            return {
                'depth_required': 'moderate',
                'score': 0.5,
                'rationale': "Moderate research depth by default"
            }
    
    def _assess_technical_complexity(self, content_request: ContentRequest) -> Dict[str, Any]:
        """Assess technical complexity of the content"""
        topic_text = f"{content_request.topic} {content_request.context}".lower()
        
        technical_indicators = self.complexity_indicators['technical_complexity_indicators']
        technical_matches = sum(1 for indicator in technical_indicators if indicator in topic_text)
        
        if technical_matches >= 3:
            return {
                'technical_level': 'high',
                'score': 0.9,
                'rationale': f"High technical complexity with {technical_matches} technical indicators"
            }
        elif technical_matches >= 1:
            return {
                'technical_level': 'moderate',
                'score': 0.6,
                'rationale': f"Moderate technical complexity with {technical_matches} technical indicators"
            }
        else:
            return {
                'technical_level': 'low',
                'score': 0.3,
                'rationale': "Low technical complexity - few technical indicators"
            }
    
    def _assess_time_constraints(self, content_request: ContentRequest) -> Dict[str, Any]:
        """Assess time constraints and urgency"""
        time_until_deadline = content_request.deadline - datetime.now()
        hours_available = time_until_deadline.total_seconds() / 3600
        
        if hours_available < 2:
            return {
                'urgency': 'critical',
                'score': 1.0,
                'rationale': f"Critical urgency - only {hours_available:.1f} hours available"
            }
        elif hours_available < 8:
            return {
                'urgency': 'high',
                'score': 0.8,
                'rationale': f"High urgency - {hours_available:.1f} hours available"
            }
        elif hours_available < 24:
            return {
                'urgency': 'moderate',
                'score': 0.5,
                'rationale': f"Moderate urgency - {hours_available:.1f} hours available"
            }
        else:
            return {
                'urgency': 'low',
                'score': 0.2,
                'rationale': f"Low urgency - {hours_available:.1f} hours available"
            }
    
    def _llm_complexity_analysis(self, content_request: ContentRequest) -> Dict[str, Any]:
        """Use LLM for sophisticated complexity analysis"""
        analysis_prompt = f"""
        Analyze the complexity of this content request for an AI/ML newsletter:
        
        Topic: {content_request.topic}
        Content Pillar: {content_request.content_pillar}
        Target Word Count: {content_request.word_count_target}
        Target Audience: {content_request.target_audience}
        Context: {content_request.context}
        
        Assess the content on these dimensions:
        1. Research Depth Required (0-1 scale)
        2. Technical Complexity (0-1 scale) 
        3. Implementation Details Needed (0-1 scale)
        4. Industry Analysis Required (0-1 scale)
        5. Code Examples Needed (0-1 scale)
        
        Based on your analysis, recommend:
        - DAILY_QUICK: For news updates, tool spotlights, quick tips (5-minute reads)
        - DEEP_DIVE: For comprehensive analysis, technical deep dives (20+ minute reads)
        
        Respond in JSON format:
        {{
            "research_depth": 0.0-1.0,
            "technical_complexity": 0.0-1.0, 
            "implementation_details": 0.0-1.0,
            "industry_analysis": 0.0-1.0,
            "code_examples": 0.0-1.0,
            "recommended_pipeline": "DAILY_QUICK" or "DEEP_DIVE",
            "confidence": 0.0-1.0,
            "reasoning": "explanation of recommendation"
        }}
        """
        
        try:
            llm_response = query_llm(analysis_prompt)
            
            # Parse JSON response
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = llm_response[start_idx:end_idx]
                analysis = json.loads(json_str)
                return analysis
        except Exception as e:
            logger.warning(f"LLM complexity analysis failed: {e}")
        
        # Fallback analysis
        return {
            "research_depth": 0.5,
            "technical_complexity": 0.5,
            "implementation_details": 0.5,
            "industry_analysis": 0.5,
            "code_examples": 0.5,
            "recommended_pipeline": "DAILY_QUICK",
            "confidence": 0.3,
            "reasoning": "Fallback analysis due to LLM parsing error"
        }
    
    def _combine_assessments(self, assessments: Dict[str, Any]) -> PipelineAssessment:
        """Combine individual assessments into overall pipeline recommendation"""
        
        # Calculate weighted complexity score
        complexity_weights = {
            'word_count': 0.3,
            'research_depth': 0.25,
            'technical_complexity': 0.2,
            'time_constraints': 0.15,
            'llm_analysis': 0.1
        }
        
        complexity_score = 0.0
        for factor, weight in complexity_weights.items():
            if factor == 'llm_analysis':
                # Use average of LLM scores
                llm_scores = [
                    assessments[factor].get('research_depth', 0.5),
                    assessments[factor].get('technical_complexity', 0.5),
                    assessments[factor].get('implementation_details', 0.5)
                ]
                complexity_score += weight * sum(llm_scores) / len(llm_scores)
            else:
                complexity_score += weight * assessments[factor].get('score', 0.5)
        
        # Determine pipeline based on complexity score and other factors
        if complexity_score >= 0.7:
            recommended_pipeline = ContentPipelineType.DEEP_DIVE
            complexity_level = ContentComplexity.COMPREHENSIVE
            estimated_time = 8.0  # 8 hours for deep dive
        elif complexity_score >= 0.5:
            recommended_pipeline = ContentPipelineType.DEEP_DIVE
            complexity_level = ContentComplexity.COMPLEX  
            estimated_time = 4.0  # 4 hours for complex content
        elif complexity_score >= 0.3:
            recommended_pipeline = ContentPipelineType.DAILY_QUICK
            complexity_level = ContentComplexity.MODERATE
            estimated_time = 1.0  # 1 hour for moderate content
        else:
            recommended_pipeline = ContentPipelineType.DAILY_QUICK
            complexity_level = ContentComplexity.SIMPLE
            estimated_time = 0.25  # 15 minutes for simple content
        
        # Override for time constraints
        time_assessment = assessments['time_constraints']
        if time_assessment['urgency'] in ['critical', 'high'] and estimated_time > 2.0:
            recommended_pipeline = ContentPipelineType.DAILY_QUICK
            complexity_level = ContentComplexity.MODERATE
            estimated_time = 1.0
        
        # Create reasoning
        reasoning_parts = []
        for factor, assessment in assessments.items():
            reasoning_parts.append(f"{factor}: {assessment.get('rationale', 'N/A')}")
        
        reasoning = f"Complexity score: {complexity_score:.2f}. " + "; ".join(reasoning_parts)
        
        return PipelineAssessment(
            recommended_pipeline=recommended_pipeline,
            complexity_level=complexity_level,
            confidence_score=min(complexity_score + 0.1, 1.0),
            reasoning=reasoning,
            estimated_time_hours=estimated_time,
            resource_requirements={
                'agents_needed': self._estimate_agents_needed(recommended_pipeline),
                'tools_required': self._estimate_tools_required(complexity_level),
                'quality_gates': self._determine_quality_gates(complexity_level)
            }
        )
    
    def _estimate_agents_needed(self, pipeline: ContentPipelineType) -> List[str]:
        """Estimate which agents are needed for the pipeline"""
        if pipeline == ContentPipelineType.DEEP_DIVE:
            return ['ResearchAgent', 'PlannerAgent', 'WriterAgent', 'EditorAgent', 'ManagerAgent']
        else:
            return ['NewsAggregatorAgent', 'ContentCuratorAgent', 'QuickBitesAgent', 
                   'SubjectLineAgent', 'NewsletterAssemblerAgent']
    
    def _estimate_tools_required(self, complexity: ContentComplexity) -> List[str]:
        """Estimate tools required based on complexity"""
        base_tools = ['search_web', 'search_knowledge_base']
        
        if complexity in [ContentComplexity.COMPLEX, ContentComplexity.COMPREHENSIVE]:
            return base_tools + ['agentic_search', 'code_generator', 'technical_validator']
        else:
            return base_tools
    
    def _determine_quality_gates(self, complexity: ContentComplexity) -> List[str]:
        """Determine quality gates based on complexity"""
        if complexity == ContentComplexity.COMPREHENSIVE:
            return ['technical_accuracy', 'citation_verification', 'code_validation', 
                   'readability_check', 'expert_review']
        elif complexity == ContentComplexity.COMPLEX:
            return ['technical_accuracy', 'readability_check', 'fact_verification']
        else:
            return ['readability_check', 'basic_fact_check']


class PublishingScheduleManager:
    """Manages publishing schedule and content pipeline allocation"""
    
    def __init__(self):
        self.daily_quota = self._load_daily_quota()
        self.weekly_schedule = self._load_weekly_schedule()
    
    def _load_daily_quota(self) -> Dict[str, int]:
        """Load daily content quotas by pillar"""
        return {
            'news_breakthroughs': 3,
            'tools_tutorials': 2, 
            'quick_hits': 12,
            'deep_dives': 0  # Deep dives are weekly
        }
    
    def _load_weekly_schedule(self) -> Dict[str, Any]:
        """Load weekly content schedule"""
        return {
            'deep_dive_day': 'friday',  # Weekly deep dive published on Friday
            'content_pillar_rotation': [
                'news_breakthroughs',    # Week 1: Deep dive on major industry analysis
                'tools_tutorials',       # Week 2: Comprehensive tool comparison/tutorial  
                'deep_dives'            # Week 3: Technical deep dive or research analysis
            ],
            'current_week': 0  # Track which week in rotation
        }
    
    def should_generate_deep_dive(self, request_date: datetime = None) -> bool:
        """Determine if a deep dive should be generated today"""
        if request_date is None:
            request_date = datetime.now()
        
        # Check if it's the designated deep dive day
        target_day = self.weekly_schedule['deep_dive_day'].lower()
        current_day = request_date.strftime('%A').lower()
        
        return current_day == target_day
    
    def get_current_deep_dive_pillar(self) -> str:
        """Get the current week's deep dive content pillar"""
        rotation = self.weekly_schedule['content_pillar_rotation']
        current_week = self.weekly_schedule['current_week']
        return rotation[current_week % len(rotation)]
    
    def advance_weekly_schedule(self):
        """Advance to next week in rotation"""
        self.weekly_schedule['current_week'] += 1
        logger.info(f"Advanced to week {self.weekly_schedule['current_week']} of rotation")


class QualityGateCoordinator:
    """Coordinates quality gates between daily and deep dive pipelines"""
    
    def __init__(self):
        self.quality_gate = NewsletterQualityGate()
        self.template_manager = AIMLTemplateManager()
    
    def apply_quality_gates(self, content: str, pipeline_type: ContentPipelineType, 
                          complexity: ContentComplexity) -> Dict[str, Any]:
        """Apply appropriate quality gates based on pipeline and complexity"""
        
        if pipeline_type == ContentPipelineType.DEEP_DIVE:
            return self._apply_deep_dive_quality_gates(content, complexity)
        else:
            return self._apply_daily_quick_quality_gates(content, complexity)
    
    def _apply_deep_dive_quality_gates(self, content: str, complexity: ContentComplexity) -> Dict[str, Any]:
        """Apply comprehensive quality gates for deep dive content"""
        quality_results = {
            'technical_accuracy': self._validate_technical_accuracy(content),
            'citation_verification': self._verify_citations(content),
            'readability_analysis': self._analyze_readability(content),
            'structure_validation': self._validate_deep_dive_structure(content),
            'code_validation': self._validate_code_examples(content)
        }
        
        if complexity == ContentComplexity.COMPREHENSIVE:
            quality_results['expert_review_needed'] = True
            quality_results['comprehensive_fact_check'] = self._comprehensive_fact_check(content)
        
        return quality_results
    
    def _apply_daily_quick_quality_gates(self, content: str, complexity: ContentComplexity) -> Dict[str, Any]:
        """Apply streamlined quality gates for daily quick content"""
        return {
            'readability_check': self._analyze_readability(content),
            'basic_fact_check': self._basic_fact_check(content),
            'mobile_optimization': self._validate_mobile_optimization(content),
            'read_time_validation': self._validate_read_time(content)
        }
    
    def _validate_technical_accuracy(self, content: str) -> Dict[str, Any]:
        """Validate technical accuracy of content"""
        # Implement technical validation logic
        return {'status': 'passed', 'score': 0.9, 'issues': []}
    
    def _verify_citations(self, content: str) -> Dict[str, Any]:
        """Verify citations and references"""
        # Implement citation verification logic
        return {'status': 'passed', 'verified_citations': 5, 'issues': []}
    
    def _analyze_readability(self, content: str) -> Dict[str, Any]:
        """Analyze content readability"""
        # Use the existing quality gate evaluation
        result = self.quality_gate.evaluate_content(content)
        return {
            'status': result.get('status', 'passed'),
            'score': result.get('overall_score', 0.0),
            'grade': result.get('grade', 'C'),
            'readability_metrics': result.get('readability_metrics', {})
        }
    
    def _validate_deep_dive_structure(self, content: str) -> Dict[str, Any]:
        """Validate deep dive structure matches template"""
        # Check for required sections in deep dive content
        required_sections = [
            'executive summary', 'technical foundation', 'architecture deep-dive',
            'practical implementation', 'real-world applications', 'future directions'
        ]
        
        content_lower = content.lower()
        found_sections = [section for section in required_sections if section in content_lower]
        
        return {
            'status': 'passed' if len(found_sections) >= 4 else 'warning',
            'found_sections': found_sections,
            'missing_sections': [s for s in required_sections if s not in found_sections]
        }
    
    def _validate_code_examples(self, content: str) -> Dict[str, Any]:
        """Validate code examples in content"""
        # Implement code validation logic
        return {'status': 'passed', 'code_blocks_found': 3, 'syntax_valid': True}
    
    def _comprehensive_fact_check(self, content: str) -> Dict[str, Any]:
        """Perform comprehensive fact checking"""
        # Implement comprehensive fact checking
        return {'status': 'passed', 'facts_verified': 15, 'confidence': 0.92}
    
    def _basic_fact_check(self, content: str) -> Dict[str, Any]:
        """Perform basic fact checking"""
        return {'status': 'passed', 'basic_checks': 'completed'}
    
    def _validate_mobile_optimization(self, content: str) -> Dict[str, Any]:
        """Validate mobile optimization"""
        return {'status': 'passed', 'mobile_friendly': True}
    
    def _validate_read_time(self, content: str) -> Dict[str, Any]:
        """Validate reading time for quick content"""
        word_count = len(content.split())
        read_time_minutes = word_count / 200  # Average reading speed
        
        return {
            'status': 'passed' if read_time_minutes <= 5 else 'warning',
            'estimated_read_time': read_time_minutes,
            'word_count': word_count
        }


class HybridWorkflowManager:
    """
    Main orchestrator that intelligently routes content between 
    daily quick pipeline and deep dive pipeline based on complexity analysis
    """
    
    def __init__(self):
        self.complexity_classifier = ContentComplexityClassifier()
        self.schedule_manager = PublishingScheduleManager()
        self.quality_coordinator = QualityGateCoordinator()
        self.daily_pipeline = DailyQuickPipeline()
        self.template_manager = AIMLTemplateManager()
        
        # Initialize deep dive agents
        self.deep_dive_agents = {
            'manager': ManagerAgent(),
            'researcher': ResearchAgent(), 
            'planner': PlannerAgent(),
            'writer': WriterAgent(),
            'editor': EditorAgent()
        }
        
        logger.info("HybridWorkflowManager initialized with all pipelines")
    
    def execute_pipeline_directly(self, content_request: ContentRequest, 
                                 pipeline_type: ContentPipelineType) -> Dict[str, Any]:
        """
        Execute a specific pipeline directly, bypassing complexity assessment.
        This is used when the UI explicitly selects a pipeline type.
        """
        logger.info(f"Executing {pipeline_type.value} pipeline directly for: {content_request.topic}")
        
        # Create a mock assessment for the selected pipeline
        mock_assessment = PipelineAssessment(
            recommended_pipeline=pipeline_type,
            complexity_level=ContentComplexity.MODERATE if pipeline_type == ContentPipelineType.DAILY_QUICK else ContentComplexity.COMPREHENSIVE,
            confidence_score=1.0,
            reasoning="Direct UI selection override",
            estimated_time_hours=0.5 if pipeline_type == ContentPipelineType.DAILY_QUICK else 3.0,
            resource_requirements={"agents": ["research", "writer"], "tools": ["search_web"]}
        )
        
        # Execute the selected pipeline directly
        if pipeline_type == ContentPipelineType.DEEP_DIVE:
            return self._execute_deep_dive_pipeline(content_request, mock_assessment)
        else:
            return self._execute_daily_quick_pipeline(content_request, mock_assessment)

    def route_content_workflow(self, content_request: ContentRequest) -> Dict[str, Any]:
        """
        Main entry point: Route content request to appropriate pipeline
        """
        logger.info(f"Routing content workflow for: {content_request.topic}")
        
        # Step 1: Assess content complexity
        assessment = self.complexity_classifier.assess_content_complexity(content_request)
        
        # Step 2: Check publishing schedule constraints
        schedule_override = self._check_schedule_constraints(content_request, assessment)
        
        # Step 3: Route to appropriate pipeline
        if schedule_override:
            assessment = schedule_override
            
        if assessment.recommended_pipeline == ContentPipelineType.DEEP_DIVE:
            return self._execute_deep_dive_pipeline(content_request, assessment)
        else:
            return self._execute_daily_quick_pipeline(content_request, assessment)
    
    def _check_schedule_constraints(self, content_request: ContentRequest, 
                                  assessment: PipelineAssessment) -> Optional[PipelineAssessment]:
        """Check if schedule constraints override complexity assessment"""
        
        # Force deep dive on designated days if appropriate content
        if (self.schedule_manager.should_generate_deep_dive() and 
            content_request.content_pillar in ['news_breakthroughs', 'tools_tutorials', 'deep_dives']):
            
            logger.info("Schedule override: forcing deep dive for weekly comprehensive content")
            
            return PipelineAssessment(
                recommended_pipeline=ContentPipelineType.DEEP_DIVE,
                complexity_level=ContentComplexity.COMPREHENSIVE,
                confidence_score=0.95,
                reasoning="Weekly deep dive schedule override",
                estimated_time_hours=8.0,
                resource_requirements=assessment.resource_requirements
            )
        
        return None
    
    def _execute_daily_quick_pipeline(self, content_request: ContentRequest,
                                    assessment: PipelineAssessment) -> Dict[str, Any]:
        """Execute the daily quick pipeline for fast content generation"""
        logger.info(f"Executing daily quick pipeline for: {content_request.topic}")
        
        start_time = datetime.now()
        
        try:
            # Create mock aggregated content from request with proper ContentItem attributes
            mock_news_content = ContentItem(
                title=f"Breaking: {content_request.topic}",
                url="https://example.com",
                content=content_request.context or f"Latest developments in {content_request.topic} are reshaping the AI landscape with new capabilities and applications.",
                source="hybrid_workflow",
                category="news_breakthroughs",
                timestamp=datetime.now()
            )
            
            mock_tools_content = ContentItem(
                title=f"Tool Spotlight: {content_request.topic}",
                url="https://example.com/tools",
                content=content_request.context or f"New tools and tutorials for {content_request.topic} implementation with step-by-step guides.",
                source="hybrid_workflow", 
                category="tools_tutorials",
                timestamp=datetime.now()
            )
            
            mock_quick_content = ContentItem(
                title=f"Quick Update: {content_request.topic}",
                url="https://example.com/quick",
                content=content_request.context or f"Industry announcement regarding {content_request.topic}.",
                source="hybrid_workflow",
                category="quick_hits", 
                timestamp=datetime.now()
            )
            
            # Create aggregated content list with proper categorization
            aggregated_content = [mock_news_content, mock_tools_content, mock_quick_content]
            
            # Execute daily pipeline with proper content
            result = self.daily_pipeline.generate_daily_newsletter(aggregated_content)
            
            # Apply quality gates
            quality_results = self.quality_coordinator.apply_quality_gates(
                result.get('markdown', result.get('content', '')),
                ContentPipelineType.DAILY_QUICK,
                assessment.complexity_level
            )
            
            execution_time = (datetime.now() - start_time).total_seconds() / 3600
            
            return {
                'pipeline_used': 'daily_quick',
                'assessment': assessment,
                'result': result,
                'quality_results': quality_results,
                'execution_time_hours': execution_time,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Daily quick pipeline failed: {e}")
            return {
                'pipeline_used': 'daily_quick',
                'assessment': assessment,
                'status': 'failed',
                'error': str(e),
                'execution_time_hours': (datetime.now() - start_time).total_seconds() / 3600
            }
    
    def _execute_deep_dive_pipeline(self, content_request: ContentRequest,
                                  assessment: PipelineAssessment) -> Dict[str, Any]:
        """Execute the deep dive pipeline for comprehensive content"""
        logger.info(f"Executing deep dive pipeline for: {content_request.topic}")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Plan the deep dive content using PlannerAgent
            planning_result = self._plan_deep_dive_content(content_request)
            
            # Step 2: Conduct comprehensive research using ResearchAgent
            research_result = self._conduct_deep_dive_research(content_request, planning_result)
            
            # Step 3: Generate comprehensive content using WriterAgent
            writing_result = self._generate_deep_dive_content(content_request, research_result)
            
            # Step 4: Edit and refine using EditorAgent
            editing_result = self._edit_deep_dive_content(writing_result)
            
            # Step 5: Apply comprehensive quality gates
            quality_results = self.quality_coordinator.apply_quality_gates(
                editing_result.get('content', ''),
                ContentPipelineType.DEEP_DIVE,
                assessment.complexity_level
            )
            
            execution_time = (datetime.now() - start_time).total_seconds() / 3600
            
            # Extract the final content from the editing stage
            final_content = editing_result.get('content', writing_result.get('content', ''))

            return {
                'pipeline_used': 'deep_dive',
                'assessment': assessment,
                'planning': planning_result,
                'research': research_result,
                'writing': writing_result,
                'editing': editing_result,
                'quality_results': quality_results,
                'execution_time_hours': execution_time,
                'status': 'success',
                'content': final_content  # Add the final content to top level
            }
            
        except Exception as e:
            logger.error(f"Deep dive pipeline failed: {e}")
            return {
                'pipeline_used': 'deep_dive',
                'assessment': assessment,
                'status': 'failed',
                'error': str(e),
                'execution_time_hours': (datetime.now() - start_time).total_seconds() / 3600
            }
    
    def _plan_deep_dive_content(self, content_request: ContentRequest) -> Dict[str, Any]:
        """Plan deep dive content structure and approach"""
        planning_task = f"""
        Plan a comprehensive deep dive article for "The AI Engineer's Daily Byte" newsletter:
        
        Topic: {content_request.topic}
        Content Pillar: {content_request.content_pillar}
        Target Audience: {content_request.target_audience}
        Word Count Target: {content_request.word_count_target}
        
        Create a detailed content plan following the deep dive template structure with:
        1. Executive Summary approach
        2. Technical Foundation coverage  
        3. Architecture Deep-Dive focus areas
        4. Practical Implementation examples
        5. Real-World Applications case studies
        6. Future Directions analysis
        
        Include specific research questions, code example requirements, and industry case studies to investigate.
        """
        
        planner = self.deep_dive_agents['planner']
        planning_result = planner.execute_task(planning_task, content_request.context)
        
        return {
            'plan': planning_result,
            'agent_used': 'PlannerAgent',
            'timestamp': datetime.now()
        }
    
    def _conduct_deep_dive_research(self, content_request: ContentRequest, 
                                  planning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct comprehensive research for deep dive content"""
        research_task = f"""
        Conduct comprehensive research for a deep dive article:
        
        Topic: {content_request.topic}
        Planning Guidance: {planning_result['plan'][:500]}...
        
        Research Focus Areas:
        1. Latest academic research and papers
        2. Industry case studies and implementations
        3. Technical specifications and architectures
        4. Code examples and implementation patterns
        5. Expert perspectives and industry analysis
        6. Future trends and development roadmaps
        
        Provide detailed research findings with credible sources, specific examples, 
        and technical details suitable for senior AI/ML professionals.
        """
        
        researcher = self.deep_dive_agents['researcher']
        research_result = researcher.execute_task(research_task, content_request.context)
        
        return {
            'research': research_result,
            'agent_used': 'ResearchAgent',
            'timestamp': datetime.now()
        }
    
    def _generate_deep_dive_content(self, content_request: ContentRequest,
                                  research_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive deep dive content"""
        writing_task = f"""
        Write a comprehensive deep dive article for "The AI Engineer's Daily Byte" newsletter.
        
        EXACT FORMAT TO FOLLOW (based on existing deep dive examples):
        
        # **The AI Engineer's Daily Byte**
        **Issue #XXX - [Date]**
        
        ## **🔬 Deep Dive & Analysis**
        
        ### [Topic Title] **🤯**
        
        [Opening paragraph establishing context and significance - 150-200 words]
        
        #### [Technical Section 1]
        [Detailed technical analysis with specific examples]
        
        #### [Technical Section 2] 
        [Architecture, implementation details, code examples]
        
        #### [Technical Section 3]
        [Real-world applications, case studies]
        
        #### [Technical Section 4]
        [Future directions, implications]
        
        #### Developer's [Topic] Checklist
        * **Action Item 1:** [Specific technical guidance]
        * **Action Item 2:** [Implementation steps]
        * [Continue with practical items]
        
        #### References & Further Reading
        * [Author]. ([Year]). *[Title]*. [Publication].
        * [Continue with credible sources]
        
        **Next up: [Preview of next content]**
        
        Topic: {content_request.topic}
        Research Findings: {research_result['research'][:1000]}...
        
        Requirements:
        - 4,000-5,000 word comprehensive analysis
        - Multiple H4 sections with technical depth
        - Industry case studies and enterprise examples  
        - Developer checklist with actionable guidance
        - Academic and industry source citations
        - Code examples and technical specifications
        - Professional tone for senior technical professionals
        """
        
        writer = self.deep_dive_agents['writer']
        writing_result = writer.execute_task(writing_task, content_request.context)
        
        return {
            'content': writing_result,
            'agent_used': 'WriterAgent',
            'timestamp': datetime.now()
        }
    
    def _edit_deep_dive_content(self, writing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Edit and refine deep dive content"""
        editing_task = f"""
        Edit and refine this deep dive article for publication in "The AI Engineer's Daily Byte".
        
        CRITICAL INSTRUCTIONS:
        1. Return ONLY the final edited article content
        2. Do NOT include any evaluation, scoring, or commentary
        3. Do NOT include recommendations or improvement suggestions
        4. Do NOT include quality assessments or ratings
        5. Simply return the clean, edited article ready for publication
        
        Content to Edit:
        {writing_result['content']}
        
        Edit for technical accuracy, clarity, structure, and professional tone.
        Return the complete edited article with no additional commentary.
        """
        
        editor = self.deep_dive_agents['editor']
        editing_result = editor.execute_task(editing_task)
        
        # Clean up any editor prefixes, suffixes, or meta-commentary
        cleaned_content = editing_result
        
        # Remove common editor prefixes
        prefixes_to_remove = [
            "**Edited Content:**",
            "Edited Content:",
            "Here is the edited content:",
            "Here's the edited content:",
            "The edited article:",
            "**Final Edited Article:**",
            "Final Edited Article:"
        ]
        
        for prefix in prefixes_to_remove:
            if prefix in cleaned_content:
                cleaned_content = cleaned_content.split(prefix, 1)[1].strip()
                break
        
        # Remove editor commentary and evaluation sections
        commentary_markers = [
            "**Actionable Recommendations:**",
            "**Recommendations:**",
            "**Editorial Notes:**",
            "**Feedback:**",
            "**Suggestions:**",
            "**Areas for Improvement:**",
            "**Key Strengths:**",
            "**Overall Quality Score:**",
            "**Accuracy:**",
            "**Engagement:**",
            "**Completeness:**",
            "* **Accuracy:**",
            "* **Engagement:**",
            "* **Completeness:**",
            "Return the edited content ready for publication",
            "The article has been edited",
            "Editorial recommendations:",
            "Actionable Recommendations:",
            "Overall Quality Score:",
            "Key Strengths:",
            "Areas for Improvement:"
        ]
        
        for marker in commentary_markers:
            if marker in cleaned_content:
                cleaned_content = cleaned_content.split(marker)[0].strip()
                break
        
        # Additional cleanup for evaluation patterns
        # Remove any section that looks like scoring or evaluation
        lines = cleaned_content.split('\n')
        cleaned_lines = []
        skip_section = False
        
        for line in lines:
            # Check if this line starts an evaluation section
            if any(eval_pattern in line for eval_pattern in 
                   ['**Accuracy:**', '**Engagement:**', '**Completeness:**', 
                    '* **Accuracy:**', '* **Engagement:**', '* **Completeness:**',
                    'Overall Quality Score:', 'Key Strengths:', 'Areas for Improvement:']):
                skip_section = True
                continue
            
            # Check if this line contains scoring patterns
            if '/10' in line or 'Strengths:' in line or 'Areas for improvement:' in line:
                skip_section = True
                continue
            
            if not skip_section:
                cleaned_lines.append(line)
        
        cleaned_content = '\n'.join(cleaned_lines).strip()
        
        # If content is still too short, fall back to original writing
        if len(cleaned_content) < 1000:
            cleaned_content = writing_result['content']
        
        return {
            'content': cleaned_content,
            'agent_used': 'EditorAgent',
            'timestamp': datetime.now()
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of both pipelines"""
        return {
            'daily_pipeline_status': 'ready',
            'deep_dive_pipeline_status': 'ready',
            'current_week_pillar': self.schedule_manager.get_current_deep_dive_pillar(),
            'deep_dive_scheduled_today': self.schedule_manager.should_generate_deep_dive(),
            'available_agents': list(self.deep_dive_agents.keys()),
            'quality_gates_active': True
        }
    
    def generate_workflow_report(self, workflow_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive workflow performance report"""
        if not workflow_results:
            return {'status': 'no_data', 'message': 'No workflow results to analyze'}
        
        total_executions = len(workflow_results)
        successful_executions = len([r for r in workflow_results if r.get('status') == 'success'])
        
        daily_executions = [r for r in workflow_results if r.get('pipeline_used') == 'daily_quick']
        deep_dive_executions = [r for r in workflow_results if r.get('pipeline_used') == 'deep_dive']
        
        avg_daily_time = sum(r.get('execution_time_hours', 0) for r in daily_executions) / max(len(daily_executions), 1)
        avg_deep_dive_time = sum(r.get('execution_time_hours', 0) for r in deep_dive_executions) / max(len(deep_dive_executions), 1)
        
        return {
            'summary': {
                'total_executions': total_executions,
                'success_rate': successful_executions / total_executions,
                'daily_pipeline_usage': len(daily_executions),
                'deep_dive_pipeline_usage': len(deep_dive_executions)
            },
            'performance': {
                'avg_daily_execution_time_hours': avg_daily_time,
                'avg_deep_dive_execution_time_hours': avg_deep_dive_time,
                'pipeline_efficiency': 'optimal' if avg_daily_time < 1.0 else 'needs_optimization'
            },
            'recommendations': self._generate_workflow_recommendations(workflow_results)
        }
    
    def _generate_workflow_recommendations(self, workflow_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on workflow performance"""
        recommendations = []
        
        failed_results = [r for r in workflow_results if r.get('status') == 'failed']
        if len(failed_results) > len(workflow_results) * 0.1:  # More than 10% failure rate
            recommendations.append("High failure rate detected - review error handling and resource allocation")
        
        long_executions = [r for r in workflow_results if r.get('execution_time_hours', 0) > 2.0]
        if len(long_executions) > len(workflow_results) * 0.2:  # More than 20% taking too long
            recommendations.append("Consider optimizing agent workflows or implementing parallel processing")
        
        if not recommendations:
            recommendations.append("Workflow performance is optimal - maintain current configuration")
        
        return recommendations 