from typing import List, Dict, Any, Optional
from datetime import datetime
from .content_analyzer import ContentAnalyzer
from .outline_generator import OutlineGenerator

class MasterPlanningAgent:
    """
    Master planning agent that orchestrates newsletter creation.
    """
    def __init__(self):
        self.content_analyzer = ContentAnalyzer()
        self.outline_generator = OutlineGenerator()
        self.planning_history = []
        
    def create_newsletter_plan(self, content_list: List[str], 
                             target_audience: str = "mixed",
                             tone: str = "professional",
                             max_length: str = "medium") -> Dict[str, Any]:
        """
        Create a comprehensive newsletter plan from content list.
        """
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Step 1: Analyze all content
        content_analysis = self._analyze_content_batch(content_list)
        
        # Step 2: Generate newsletter outline
        outline = self.outline_generator.generate_outline(content_list)
        
        # Step 3: Create task breakdown
        tasks = self._create_task_breakdown(outline, content_analysis)
        
        # Step 4: Define audience and tone guidelines
        guidelines = self._create_content_guidelines(target_audience, tone, max_length)
        
        plan = {
            'plan_id': plan_id,
            'created_at': datetime.now().isoformat(),
            'target_audience': target_audience,
            'tone': tone,
            'max_length': max_length,
            'content_analysis': content_analysis,
            'outline': outline,
            'tasks': tasks,
            'guidelines': guidelines,
            'estimated_completion_time': self._estimate_completion_time(tasks),
            'status': 'planned'
        }
        
        self.planning_history.append(plan)
        return plan
    
    def _analyze_content_batch(self, content_list: List[str]) -> Dict[str, Any]:
        """
        Analyze a batch of content items.
        """
        batch_analysis = {
            'total_items': len(content_list),
            'themes_distribution': {},
            'top_topics': [],
            'content_quality_scores': [],
            'word_count_distribution': []
        }
        
        all_themes = {}
        all_topics = {}
        total_words = 0
        
        for content in content_list:
            analysis = self.content_analyzer.analyze_content(content)
            
            # Aggregate themes
            for theme in analysis['themes']:
                all_themes[theme] = all_themes.get(theme, 0) + 1
            
            # Aggregate topics
            for topic in analysis['key_topics']:
                all_topics[topic] = all_topics.get(topic, 0) + 1
            
            total_words += analysis['word_count']
            
            # Calculate simple quality score based on word count and unique words
            quality_score = min(1.0, (analysis['word_count'] / 100) * (analysis['unique_words'] / analysis['word_count']))
            batch_analysis['content_quality_scores'].append(quality_score)
            batch_analysis['word_count_distribution'].append(analysis['word_count'])
        
        # Sort and get top themes and topics
        batch_analysis['themes_distribution'] = dict(
            sorted(all_themes.items(), key=lambda x: x[1], reverse=True)
        )
        batch_analysis['top_topics'] = [
            topic for topic, count in sorted(all_topics.items(), key=lambda x: x[1], reverse=True)[:20]
        ]
        batch_analysis['average_word_count'] = total_words / len(content_list) if content_list else 0
        
        return batch_analysis
    
    def _create_task_breakdown(self, outline: Dict[str, Any], 
                              content_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create a breakdown of tasks for newsletter creation.
        """
        tasks = []
        task_id = 1
        
        # Task 1: Content research and validation
        tasks.append({
            'task_id': f"task_{task_id:03d}",
            'type': 'research',
            'title': 'Content Research and Validation',
            'description': 'Research and validate key claims from content sources',
            'assigned_agent': 'research_agent',
            'estimated_duration': '30 minutes',
            'priority': 'high',
            'dependencies': [],
            'status': 'pending'
        })
        task_id += 1
        
        # Task 2: Generate section content
        for section in outline['sections']:
            tasks.append({
                'task_id': f"task_{task_id:03d}",
                'type': 'writing',
                'title': f"Write {section['title']} Section",
                'description': f"Create content for {section['title']} section",
                'assigned_agent': self._get_agent_for_theme(section['theme']),
                'estimated_duration': '45 minutes',
                'priority': 'high',
                'dependencies': ['task_001'],
                'status': 'pending',
                'section_data': section
            })
            task_id += 1
        
        # Task 3: Content review and fact-checking
        tasks.append({
            'task_id': f"task_{task_id:03d}",
            'type': 'quality',
            'title': 'Content Review and Fact-Checking',
            'description': 'Review all content for accuracy and quality',
            'assigned_agent': 'quality_agent',
            'estimated_duration': '30 minutes',
            'priority': 'high',
            'dependencies': [f"task_{i:03d}" for i in range(2, task_id)],
            'status': 'pending'
        })
        task_id += 1
        
        # Task 4: Final assembly and formatting
        tasks.append({
            'task_id': f"task_{task_id:03d}",
            'type': 'assembly',
            'title': 'Final Assembly and Formatting',
            'description': 'Assemble all sections into final newsletter format',
            'assigned_agent': 'assembly_agent',
            'estimated_duration': '15 minutes',
            'priority': 'medium',
            'dependencies': [f"task_{task_id-1:03d}"],
            'status': 'pending'
        })
        
        return tasks
    
    def _get_agent_for_theme(self, theme: str) -> str:
        """
        Determine which agent should handle a specific theme.
        """
        agent_mapping = {
            'machine_learning': 'research_summary_agent',
            'deep_learning': 'research_summary_agent',
            'nlp': 'research_summary_agent',
            'computer_vision': 'technical_deep_dive_agent',
            'robotics': 'technical_deep_dive_agent',
            'ethics': 'trend_analysis_agent',
            'general': 'industry_news_agent'
        }
        return agent_mapping.get(theme, 'industry_news_agent')
    
    def _create_content_guidelines(self, target_audience: str, tone: str, 
                                  max_length: str) -> Dict[str, Any]:
        """
        Create content guidelines based on audience and tone.
        """
        guidelines = {
            'target_audience': target_audience,
            'tone': tone,
            'max_length': max_length,
            'writing_style': self._get_writing_style(target_audience, tone),
            'technical_depth': self._get_technical_depth(target_audience),
            'section_guidelines': self._get_section_guidelines(max_length)
        }
        return guidelines
    
    def _get_writing_style(self, audience: str, tone: str) -> Dict[str, str]:
        """
        Define writing style based on audience and tone.
        """
        styles = {
            'mixed': {
                'professional': {
                    'style': 'Clear, professional, accessible to both technical and business readers',
                    'avoid': 'Jargon without explanation, overly casual language',
                    'include': 'Clear explanations, business implications, technical context'
                }
            },
            'technical': {
                'professional': {
                    'style': 'Detailed technical explanations with code examples',
                    'avoid': 'Oversimplification, lack of technical depth',
                    'include': 'Technical details, implementation considerations, performance metrics'
                }
            },
            'business': {
                'professional': {
                    'style': 'Business-focused with clear value propositions',
                    'avoid': 'Excessive technical jargon, lack of business context',
                    'include': 'Market implications, ROI considerations, competitive analysis'
                }
            }
        }
        return styles.get(audience, {}).get(tone, styles['mixed']['professional'])
    
    def _get_technical_depth(self, audience: str) -> str:
        """
        Determine appropriate technical depth for audience.
        """
        depth_mapping = {
            'mixed': 'moderate',
            'technical': 'high',
            'business': 'low'
        }
        return depth_mapping.get(audience, 'moderate')
    
    def _get_section_guidelines(self, max_length: str) -> Dict[str, int]:
        """
        Define section length guidelines based on max length.
        """
        guidelines = {
            'short': {
                'max_sections': 3,
                'max_words_per_section': 300,
                'total_max_words': 1000
            },
            'medium': {
                'max_sections': 5,
                'max_words_per_section': 500,
                'total_max_words': 2500
            },
            'long': {
                'max_sections': 7,
                'max_words_per_section': 800,
                'total_max_words': 5000
            }
        }
        return guidelines.get(max_length, guidelines['medium'])
    
    def _estimate_completion_time(self, tasks: List[Dict[str, Any]]) -> str:
        """
        Estimate total completion time for all tasks.
        """
        total_minutes = 0
        for task in tasks:
            duration = task['estimated_duration']
            if 'minutes' in duration:
                minutes = int(duration.split()[0])
                total_minutes += minutes
        
        if total_minutes < 60:
            return f"{total_minutes} minutes"
        else:
            hours = total_minutes // 60
            remaining_minutes = total_minutes % 60
            if remaining_minutes == 0:
                return f"{hours} hours"
            else:
                return f"{hours} hours {remaining_minutes} minutes"
    
    def get_planning_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all created plans.
        """
        return self.planning_history
    
    def get_plan_by_id(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific plan by ID.
        """
        for plan in self.planning_history:
            if plan['plan_id'] == plan_id:
                return plan
        return None 