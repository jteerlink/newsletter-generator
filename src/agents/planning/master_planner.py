"""
Master Planning Agent for Newsletter Creation

This agent orchestrates the entire newsletter creation process from content analysis
to task assignment and quality control.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .content_analyzer import ContentAnalyzer
from .outline_generator import OutlineGenerator
from ..tasks.task_assignment import TaskAssignment
from ..base.agent_base import AgentBase
from ..base.communication import Message, MessageType

logger = logging.getLogger(__name__)


@dataclass
class ContentItem:
    """Represents a piece of content with metadata and analysis."""
    id: str
    title: str
    content: str
    source: str
    url: str
    published_date: datetime
    novelty_score: float = 0.0
    importance_score: float = 0.0
    audience_relevance: float = 0.0
    source_reliability: float = 0.0
    temporal_relevance: float = 0.0
    overall_score: float = 0.0
    themes: List[str] = None
    entities: Dict[str, List[str]] = None
    sentiment: Dict[str, Any] = None


@dataclass
class NewsletterPlan:
    """Complete newsletter plan with outline, tasks, and metadata."""
    id: str
    title: str
    target_audience: str
    estimated_length: int  # pages
    outline: Dict[str, Any]
    content_items: List[ContentItem]
    assigned_tasks: Dict[str, List[Dict]]
    deadlines: Dict[str, datetime]
    quality_requirements: Dict[str, Any]
    created_at: datetime
    status: str = "planned"


class MasterPlanner(AgentBase):
    """
    Master Planning Agent that orchestrates newsletter creation.
    
    Responsibilities:
    - Analyze available content and identify key themes
    - Create newsletter outline and structure
    - Define target audience and tone
    - Break down work into manageable sections
    - Assign tasks to specialized sub-agents
    - Set deadlines and coordinate workflow
    """
    
    def __init__(self, agent_id: str = "master_planner"):
        super().__init__(agent_id)
        self.content_analyzer = ContentAnalyzer()
        self.outline_generator = OutlineGenerator()
        self.task_assignment = TaskAssignment()
        
        # Configuration
        self.max_content_items = 50
        self.min_content_items = 10
        self.target_newsletter_length = 2  # pages
        self.quality_threshold = 0.7
        
        # Source reliability weights
        self.source_weights = {
            'arxiv': 0.9,
            'nature': 0.95,
            'science': 0.95,
            'techcrunch': 0.8,
            'venturebeat': 0.8,
            'wired': 0.85,
            'mit_tech_review': 0.9,
            'default': 0.7
        }
        
        logger.info(f"MasterPlanner {agent_id} initialized")
    
    def plan_newsletter(self, content_sources: List[Dict], 
                       target_audience: str = "mixed",
                       newsletter_title: str = None) -> NewsletterPlan:
        """
        Create a complete newsletter plan from raw content sources.
        
        Args:
            content_sources: List of raw content dictionaries
            target_audience: Target audience type ('business', 'technical', 'mixed')
            newsletter_title: Optional custom title
            
        Returns:
            NewsletterPlan: Complete plan with outline, tasks, and assignments
        """
        logger.info(f"Starting newsletter planning for {len(content_sources)} content sources")
        
        # Step 1: Analyze and score content
        content_items = self._analyze_and_score_content(content_sources)
        
        # Step 2: Filter and prioritize content
        selected_content = self._filter_and_prioritize_content(content_items)
        
        # Step 3: Generate newsletter outline
        outline = self.outline_generator.generate_outline(
            selected_content, target_audience, self.target_newsletter_length
        )
        
        # Step 4: Create tasks and assignments
        tasks = self._create_tasks(selected_content, outline)
        assignments = self.task_assignment.assign_tasks(tasks, self._get_available_agents())
        
        # Step 5: Set deadlines
        deadlines = self._set_deadlines(assignments)
        
        # Step 6: Create quality requirements
        quality_requirements = self._create_quality_requirements(target_audience)
        
        # Step 7: Generate plan
        plan = NewsletterPlan(
            id=f"newsletter_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=newsletter_title or self._generate_title(selected_content),
            target_audience=target_audience,
            estimated_length=self._estimate_length(selected_content),
            outline=outline,
            content_items=selected_content,
            assigned_tasks=assignments,
            deadlines=deadlines,
            quality_requirements=quality_requirements,
            created_at=datetime.now()
        )
        
        logger.info(f"Newsletter plan created: {plan.id}")
        return plan
    
    def _analyze_and_score_content(self, content_sources: List[Dict]) -> List[ContentItem]:
        """Analyze content and assign priority scores."""
        content_items = []
        
        for source in content_sources:
            # Create content item
            item = ContentItem(
                id=source.get('id', f"item_{len(content_items)}"),
                title=source.get('title', ''),
                content=source.get('content', ''),
                source=source.get('source', ''),
                url=source.get('url', ''),
                published_date=source.get('published_date', datetime.now()),
                themes=[],
                entities={},
                sentiment={}
            )
            
            # Analyze content
            analysis = self.content_analyzer.analyze_content(item.content)
            item.themes = analysis.get('themes', [])
            item.entities = analysis.get('entities', {})
            item.sentiment = analysis.get('sentiment', {})
            
            # Calculate scores
            item.novelty_score = self._calculate_novelty_score(item)
            item.importance_score = self._calculate_importance_score(item)
            item.audience_relevance = self._calculate_audience_relevance(item)
            item.source_reliability = self._calculate_source_reliability(item)
            item.temporal_relevance = self._calculate_temporal_relevance(item)
            
            # Overall score (weighted average)
            item.overall_score = (
                item.novelty_score * 0.25 +
                item.importance_score * 0.3 +
                item.audience_relevance * 0.2 +
                item.source_reliability * 0.15 +
                item.temporal_relevance * 0.1
            )
            
            content_items.append(item)
        
        return content_items
    
    def _calculate_novelty_score(self, item: ContentItem) -> float:
        """Calculate how novel/unique the content is."""
        # Check for emerging technologies, new models, breakthrough announcements
        emerging_keywords = [
            'breakthrough', 'revolutionary', 'first', 'new', 'unprecedented',
            'groundbreaking', 'innovative', 'cutting-edge', 'state-of-the-art'
        ]
        
        novelty_indicators = sum(1 for keyword in emerging_keywords 
                               if keyword.lower() in item.content.lower())
        
        # Check for new entities (companies, models, technologies)
        new_entities = len(item.entities.get('models', [])) + len(item.entities.get('technologies', []))
        
        # Normalize to 0-1 scale
        return min(1.0, (novelty_indicators * 0.1 + new_entities * 0.05))
    
    def _calculate_importance_score(self, item: ContentItem) -> float:
        """Calculate the importance/impact of the content."""
        # Check for major companies and models
        major_companies = ['openai', 'google', 'microsoft', 'nvidia', 'anthropic', 'meta']
        major_models = ['gpt-4', 'claude', 'gemini', 'llama', 'bert', 'transformer']
        
        company_importance = sum(1 for company in major_companies 
                               if company in item.content.lower())
        model_importance = sum(1 for model in major_models 
                             if model in item.content.lower())
        
        # Check for breakthrough indicators
        breakthrough_indicators = [
            'accuracy', 'performance', 'efficiency', 'speed', 'quality',
            'benchmark', 'record', 'improvement', 'advancement'
        ]
        
        breakthrough_score = sum(1 for indicator in breakthrough_indicators 
                               if indicator in item.content.lower())
        
        return min(1.0, (company_importance * 0.2 + model_importance * 0.2 + breakthrough_score * 0.05))
    
    def _calculate_audience_relevance(self, item: ContentItem) -> float:
        """Calculate relevance to target audience."""
        # Business-focused keywords
        business_keywords = [
            'market', 'revenue', 'investment', 'funding', 'startup', 'enterprise',
            'business', 'commercial', 'product', 'launch', 'acquisition', 'ipo'
        ]
        
        # Technical-focused keywords
        technical_keywords = [
            'algorithm', 'architecture', 'implementation', 'code', 'api',
            'framework', 'library', 'model', 'training', 'inference', 'optimization'
        ]
        
        business_score = sum(1 for keyword in business_keywords 
                           if keyword in item.content.lower())
        technical_score = sum(1 for keyword in technical_keywords 
                            if keyword in item.content.lower())
        
        # For mixed audience, balance is good
        total_score = business_score + technical_score
        balance_score = 1.0 - abs(business_score - technical_score) / max(total_score, 1)
        
        return min(1.0, (total_score * 0.1 + balance_score * 0.5))
    
    def _calculate_source_reliability(self, item: ContentItem) -> float:
        """Calculate source reliability score."""
        source_lower = item.source.lower()
        
        # Check known source weights
        for source_pattern, weight in self.source_weights.items():
            if source_pattern in source_lower:
                return weight
        
        return self.source_weights['default']
    
    def _calculate_temporal_relevance(self, item: ContentItem) -> float:
        """Calculate temporal relevance based on publication date."""
        now = datetime.now()
        age_days = (now - item.published_date).days
        
        # Exponential decay: newer content is more relevant
        if age_days <= 1:
            return 1.0
        elif age_days <= 7:
            return 0.9
        elif age_days <= 30:
            return 0.7
        elif age_days <= 90:
            return 0.5
        else:
            return 0.3
    
    def _filter_and_prioritize_content(self, content_items: List[ContentItem]) -> List[ContentItem]:
        """Filter and prioritize content based on scores."""
        # Sort by overall score
        sorted_items = sorted(content_items, key=lambda x: x.overall_score, reverse=True)
        
        # Filter by quality threshold
        quality_items = [item for item in sorted_items if item.overall_score >= self.quality_threshold]
        
        # Limit to max content items
        if len(quality_items) > self.max_content_items:
            quality_items = quality_items[:self.max_content_items]
        
        # Ensure minimum content
        if len(quality_items) < self.min_content_items:
            # Add lower quality items to meet minimum
            remaining_items = [item for item in sorted_items if item not in quality_items]
            quality_items.extend(remaining_items[:self.min_content_items - len(quality_items)])
        
        logger.info(f"Selected {len(quality_items)} content items from {len(content_items)} total")
        return quality_items
    
    def _create_tasks(self, content_items: List[ContentItem], outline: Dict) -> List[Dict]:
        """Create tasks based on content and outline."""
        tasks = []
        
        for section in outline.get('sections', []):
            section_tasks = self._create_section_tasks(section, content_items)
            tasks.extend(section_tasks)
        
        return tasks
    
    def _create_section_tasks(self, section: Dict, content_items: List[ContentItem]) -> List[Dict]:
        """Create tasks for a specific section."""
        tasks = []
        section_type = section.get('type', 'general')
        
        # Filter content relevant to this section
        relevant_content = self._filter_content_for_section(content_items, section)
        
        if section_type == 'research_summary':
            tasks.append({
                'id': f"research_{section['id']}",
                'type': 'research_summary',
                'content_items': relevant_content,
                'requirements': {
                    'length': section.get('target_length', 300),
                    'technical_depth': 'high',
                    'include_citations': True
                }
            })
        elif section_type == 'industry_news':
            tasks.append({
                'id': f"industry_{section['id']}",
                'type': 'industry_news',
                'content_items': relevant_content,
                'requirements': {
                    'length': section.get('target_length', 250),
                    'business_focus': True,
                    'include_quotes': True
                }
            })
        elif section_type == 'technical_deep_dive':
            tasks.append({
                'id': f"technical_{section['id']}",
                'type': 'technical_deep_dive',
                'content_items': relevant_content,
                'requirements': {
                    'length': section.get('target_length', 400),
                    'include_code_examples': True,
                    'technical_depth': 'expert'
                }
            })
        elif section_type == 'trend_analysis':
            tasks.append({
                'id': f"trend_{section['id']}",
                'type': 'trend_analysis',
                'content_items': relevant_content,
                'requirements': {
                    'length': section.get('target_length', 350),
                    'trend_focus': True,
                    'include_predictions': True
                }
            })
        elif section_type == 'interview_profile':
            tasks.append({
                'id': f"interview_{section['id']}",
                'type': 'interview_profile',
                'content_items': relevant_content,
                'requirements': {
                    'length': section.get('target_length', 300),
                    'include_key_figures': True,
                    'conference_coverage': True
                }
            })
        else:  # General content
            tasks.append({
                'id': f"general_{section['id']}",
                'type': 'general_content',
                'content_items': relevant_content,
                'requirements': {
                    'length': section.get('target_length', 200),
                    'style': 'informative'
                }
            })
        # NOTE: Add test coverage for each section type and agent assignment
        return tasks
    
    def _filter_content_for_section(self, content_items: List[ContentItem], section: Dict) -> List[ContentItem]:
        """Filter content items relevant to a specific section."""
        section_themes = section.get('themes', [])
        section_keywords = section.get('keywords', [])
        
        relevant_items = []
        
        for item in content_items:
            # Check theme overlap
            theme_overlap = any(theme in item.themes for theme in section_themes)
            
            # Check keyword presence
            keyword_presence = any(keyword.lower() in item.content.lower() 
                                 for keyword in section_keywords)
            
            if theme_overlap or keyword_presence:
                relevant_items.append(item)
        
        return relevant_items[:5]  # Limit to top 5 relevant items per section
    
    def _get_available_agents(self) -> List[str]:
        """Get list of available specialized agents."""
        # This would be populated from the agent registry
        return [
            'research_summary_agent',
            'industry_news_agent',
            'technical_deep_dive_agent',
            'trend_analysis_agent',
            'interview_profile_agent'
        ]
    
    def _set_deadlines(self, assignments: Dict) -> Dict[str, datetime]:
        """Set deadlines for task completion."""
        deadlines = {}
        base_time = datetime.now()
        
        # Set staggered deadlines based on task complexity
        for agent_id, tasks in assignments.items():
            for i, task in enumerate(tasks):
                task_id = task['id']
                
                # Base deadline: 2 hours from now
                base_deadline = base_time + timedelta(hours=2)
                
                # Adjust based on task type and complexity
                if task['type'] == 'research_summary':
                    deadline = base_deadline + timedelta(hours=i * 0.5)
                elif task['type'] == 'technical_deep_dive':
                    deadline = base_deadline + timedelta(hours=i * 1.0)
                else:
                    deadline = base_deadline + timedelta(hours=i * 0.3)
                
                deadlines[task_id] = deadline
        
        return deadlines
    
    def _create_quality_requirements(self, target_audience: str) -> Dict[str, Any]:
        """Create quality requirements based on target audience."""
        requirements = {
            'min_sources_per_claim': 3,
            'fact_checking_required': True,
            'citation_style': 'academic' if target_audience == 'technical' else 'journalistic',
            'technical_depth': 'expert' if target_audience == 'technical' else 'mixed',
            'business_focus': target_audience == 'business',
            'readability_score': 0.8,
            'grammar_check': True,
            'plagiarism_check': True
        }
        
        return requirements
    
    def _generate_title(self, content_items: List[ContentItem]) -> str:
        """Generate newsletter title based on content themes."""
        # Extract common themes
        all_themes = []
        for item in content_items:
            all_themes.extend(item.themes)
        
        # Find most common themes
        theme_counts = {}
        for theme in all_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        if theme_counts:
            top_theme = max(theme_counts, key=theme_counts.get)
            return f"AI Newsletter: {top_theme.replace('_', ' ').title()} Focus"
        
        return "AI Newsletter: Latest Developments"
    
    def _estimate_length(self, content_items: List[ContentItem]) -> int:
        """Estimate newsletter length in pages."""
        total_words = sum(len(item.content.split()) for item in content_items)
        # Assume 500 words per page
        estimated_pages = max(1, total_words // 500)
        return min(estimated_pages, 5)  # Cap at 5 pages
    
    def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming messages."""
        if message.type == MessageType.REQUEST:
            if message.content.get('action') == 'plan_newsletter':
                return self._handle_plan_request(message)
        
        return None
    
    def _handle_plan_request(self, message: Message) -> Message:
        """Handle newsletter planning request."""
        try:
            content_sources = message.content.get('content_sources', [])
            target_audience = message.content.get('target_audience', 'mixed')
            newsletter_title = message.content.get('title')
            
            plan = self.plan_newsletter(content_sources, target_audience, newsletter_title)
            
            return Message(
                sender=self.agent_id,
                recipient=message.sender,
                type=MessageType.RESPONSE,
                content={'plan': plan}
            )
        
        except Exception as e:
            logger.error(f"Error planning newsletter: {e}")
            return Message(
                sender=self.agent_id,
                recipient=message.sender,
                type=MessageType.ERROR,
                content={'error': str(e)}
            )

    def run(self):
        """Main execution loop for the agent."""
        # MasterPlanner doesn't have a continuous run loop
        # It processes requests on-demand
        pass

    def receive_message(self, message: dict):
        """Handle incoming messages."""
        # Convert dict to Message object if needed
        if isinstance(message, dict):
            # Handle dict format
            return self.process_dict_message(message)
        elif hasattr(message, 'type'):
            # Handle Message object
            return self.process_message(message)
        else:
            logger.warning(f"Unknown message format: {type(message)}")
            return None

    def send_message(self, recipient_id: str, message: dict):
        """Send a message to another agent."""
        # This would typically use a message bus
        # For now, just log the message
        logger.info(f"Sending message to {recipient_id}: {message}")
        return True

    def process_dict_message(self, message_dict: dict):
        """Process a message in dictionary format."""
        # Convert dict to Message object for processing
        message = Message(
            sender=message_dict.get('sender', 'unknown'),
            recipient=message_dict.get('recipient', self.agent_id),
            type=MessageType(message_dict.get('type', 'request')),
            content=message_dict.get('content', {})
        )
        return self.process_message(message) 