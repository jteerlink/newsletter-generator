"""
Basic Agent System for Newsletter Generation (Phase 2)

This implements a simple agent system that can be upgraded to CrewAI later.
For now, it provides basic research capabilities using the tools we've built.
"""

from __future__ import annotations

import logging
import time  # Added proper import for time module
from typing import Dict, Any, List, Optional
from src.core.core import query_llm
from src.tools.tools import search_web, search_knowledge_base, AVAILABLE_TOOLS

logger = logging.getLogger(__name__)

class SimpleAgent:
    """Base class for simple agents that can use tools and query LLMs."""
    
    def __init__(self, name: str, role: str, goal: str, backstory: str, tools: Optional[List[str]] = None):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.available_tools = {name: func for name, func in AVAILABLE_TOOLS.items() if name in self.tools}
        
    def execute_task(self, task: str, context: str = "") -> str:
        """Execute a task using available tools and LLM reasoning."""
        try:
            logger.info(f"Agent {self.name} executing task: {task}")
            
            # Create the prompt for the agent
            prompt = self._build_prompt(task, context)
            
            # Query the LLM
            response = query_llm(prompt)
            
            # Check if the agent needs to use tools
            if self._should_use_tools(response):
                tool_output = self._execute_tools(task)
                # Re-query with tool results
                enhanced_prompt = self._build_prompt_with_tools(task, context, tool_output)
                response = query_llm(enhanced_prompt)
            
            logger.info(f"Agent {self.name} completed task")
            return response
            
        except Exception as e:
            error_msg = f"Error in agent {self.name}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _build_prompt(self, task: str, context: str = "") -> str:
        """Build the initial prompt for the agent."""
        prompt_parts = [
            f"You are a {self.role}.",
            f"Your goal: {self.goal}",
            f"Background: {self.backstory}",
            "",
            f"Task: {task}"
        ]
        
        if context:
            prompt_parts.extend(["", f"Context: {context}"])
        
        if self.tools:
            prompt_parts.extend([
                "",
                f"Available tools: {', '.join(self.tools)}",
                "",
                "IMPORTANT: If this task requires current information, recent data, or web research that you don't have in your training data, you MUST respond with exactly: 'NEED_TOOLS'",
                "Only proceed to answer directly if you are certain you have all the necessary information from your training data.",
                "",
                "Does this task require searching for current information or recent data? If yes, respond with 'NEED_TOOLS'. If no, proceed with your response."
            ])
        
        return "\n".join(prompt_parts)
    
    def _should_use_tools(self, response: str) -> bool:
        """Check if the agent response indicates tools are needed."""
        # Make detection more comprehensive
        response_lower = response.lower()
        tool_indicators = [
            "need_tools",
            "need to search",
            "require current information",
            "need recent data",
            "search for information",
            "look up",
            "find current",
            "get latest",
            "check recent",
            "find some recent",
            "let me find",
            "let me search",
            "i should search",
            "i need to find",
            "gather current",
            "get up-to-date",
            "recent developments",
            "current trends"
        ]
        
        return any(indicator in response_lower for indicator in tool_indicators)
    
    def _execute_tools(self, task: str) -> str:
        """Execute available tools based on the task, preferring CrewAI tools when available."""
        tool_outputs = []
        
        # Simple logic to determine what to search for
        search_query = self._extract_search_query(task)
        
        for tool_name in self.tools:
            effective_tool_name = self._get_effective_tool_name(tool_name)
            
            if effective_tool_name in self.available_tools:
                try:
                    # Execute tools based on their type and interface
                    if effective_tool_name == 'search_web' or effective_tool_name == 'crewai_search_web':
                        output = self.available_tools[effective_tool_name](search_query, max_results=3)
                    elif tool_name == 'search_knowledge_base':
                        output = self.available_tools[tool_name](search_query, n_results=3)
                    elif tool_name == 'agentic_search':
                        # Handle both legacy and CrewAI agentic search
                        if effective_tool_name == 'crewai_agentic_search':
                            tool_class = self.available_tools[effective_tool_name]
                            try:
                                agentic_tool = tool_class()
                                output = agentic_tool.run(search_query, f"Research information about {search_query}")
                            except Exception as e:
                                output = f"CrewAI agentic search error: {str(e)}"
                        else:
                            # Legacy agentic search
                            tool_class = self.available_tools[effective_tool_name]
                            try:
                                agentic_tool = tool_class()
                                output = agentic_tool.run(search_query)
                            except Exception as e:
                                output = f"Agentic search error: {str(e)}"
                    elif tool_name == 'search_web_with_alternatives':
                        if effective_tool_name == 'crewai_search_web_with_alternatives':
                            output = self.available_tools[effective_tool_name](search_query)
                        else:
                            output = self.available_tools[effective_tool_name](search_query)
                    elif effective_tool_name == 'hybrid_search_web':
                        # Use hybrid search for maximum reliability
                        output = self.available_tools[effective_tool_name](search_query, max_results=3)
                    else:
                        # Generic tool execution
                        output = self.available_tools[effective_tool_name](search_query)
                    
                    # Format output with tool identification
                    tool_display_name = f"{tool_name.upper()}" + (
                        " (CrewAI)" if effective_tool_name.startswith('crewai_') else ""
                    )
                    tool_outputs.append(f"=== {tool_display_name} RESULTS ===\n{output}\n")
                    
                except Exception as e:
                    logger.error(f"Tool execution error for {effective_tool_name}: {e}")
                    tool_outputs.append(f"=== {tool_name.upper()} ERROR ===\n{str(e)}\n")
        
        return "\n".join(tool_outputs)
    
    def _get_effective_tool_name(self, requested_tool: str) -> str:
        """Get the effective tool name, preferring CrewAI versions when available."""
        # Check if CrewAI tools are available
        from src.tools.tools import CREWAI_AVAILABLE, RECOMMENDED_TOOLS
        
        if CREWAI_AVAILABLE and requested_tool in RECOMMENDED_TOOLS:
            crewai_tool = RECOMMENDED_TOOLS[requested_tool]
            if crewai_tool in self.available_tools:
                logger.info(f"Using CrewAI tool {crewai_tool} instead of {requested_tool}")
                return crewai_tool
        
        # Check if hybrid search is available and preferred for web search
        if requested_tool == 'search_web' and 'hybrid_search_web' in self.available_tools:
            logger.info(f"Using hybrid search for enhanced reliability")
            return 'hybrid_search_web'
        
        # Fall back to original tool
        return requested_tool
    
    def _extract_search_query(self, task: str) -> str:
        """Extract a search query from the task description."""
        # Better search query extraction
        task_lower = task.lower()
        
        # Remove common task-starting phrases to get core topic
        prefixes_to_remove = [
            "research current trends and developments in ",
            "find information about ",
            "search for ",
            "look up ",
            "analyze ",
            "investigate "
        ]
        
        cleaned_task = task
        for prefix in prefixes_to_remove:
            if task_lower.startswith(prefix):
                cleaned_task = task[len(prefix):]
                break
        
        # Extract key terms (simple approach - could be enhanced with NLP)
        return cleaned_task.strip()
    
    def _build_prompt_with_tools(self, task: str, context: str, tool_output: str) -> str:
        """Build a prompt that includes tool results."""
        prompt_parts = [
            f"You are a {self.role}.",
            f"Your goal: {self.goal}",
            f"Background: {self.backstory}",
            "",
            f"Task: {task}"
        ]
        
        if context:
            prompt_parts.extend(["", f"Context: {context}"])
        
        prompt_parts.extend([
            "",
            "Here are the search results to help you complete the task:",
            tool_output,
            "",
            "Now complete the task using this information:"
        ])
        
        return "\n".join(prompt_parts)

class ResearchAgent(SimpleAgent):
    """Agent specialized for comprehensive research and information synthesis."""
    
    def __init__(self):
        super().__init__(
            name="ResearchAgent",
            role="Senior Research Analyst and Information Synthesizer",
            goal="Conduct thorough, multi-dimensional research that uncovers deep insights, identifies emerging trends, and provides comprehensive understanding of complex topics through diverse sources and analytical approaches",
            backstory="""You are a seasoned research analyst with 10+ years of experience in investigative research, data analysis, and information synthesis. You specialize in uncovering hidden insights, identifying patterns across disparate sources, and presenting complex information in accessible ways.

Your research methodology is comprehensive and multi-faceted:
- Primary and secondary source analysis
- Cross-referencing and fact verification
- Trend identification and pattern recognition
- Expert opinion gathering and synthesis
- Data interpretation and statistical analysis
- Historical context and comparative analysis
- Future implications and scenario planning
- Stakeholder perspective analysis

Your research philosophy emphasizes:
- Going beyond surface-level information to find deeper insights
- Exploring multiple perspectives and potential biases
- Identifying surprising connections and unexpected angles
- Balancing current information with historical context
- Understanding both opportunities and challenges
- Considering various stakeholder viewpoints
- Anticipating future developments and implications
- Providing actionable intelligence

You excel at:
- Finding authoritative and diverse sources
- Synthesizing complex information into coherent narratives
- Identifying knowledge gaps and research opportunities
- Uncovering emerging trends before they become mainstream
- Providing balanced analysis that considers multiple viewpoints
- Creating comprehensive research briefings
- Connecting theoretical concepts with practical applications
- Evaluating source credibility and information quality

Your research should be:
- Comprehensive and multi-dimensional
- Well-sourced with credible references
- Balanced and objective
- Forward-looking and insightful
- Practical and actionable
- Engaging and accessible
- Rich with examples and case studies
- Connected to broader trends and implications

You should explore topics thoroughly, ask probing questions, and provide research that serves as a solid foundation for compelling content creation.""",
            tools=["agentic_search", "search_web_with_alternatives", "hybrid_search_web"]
        )

class Task:
    """Simple task container."""
    
    def __init__(self, description: str, agent: SimpleAgent, context: str = ""):
        self.description = description
        self.agent = agent
        self.context = context
        self.result = None
    
    def execute(self) -> str:
        """Execute the task using the assigned agent."""
        self.result = self.agent.execute_task(self.description, self.context)
        return self.result

class SimpleCrew:
    """Simple crew orchestrator that can be upgraded to CrewAI later."""
    
    def __init__(self, agents: List[SimpleAgent], tasks: List[Task]):
        self.agents = agents
        self.tasks = tasks
    
    def kickoff(self) -> str:
        """Execute all tasks in sequence."""
        logger.info("Starting crew execution")
        results = []
        
        for i, task in enumerate(self.tasks):
            logger.info(f"Executing task {i+1}/{len(self.tasks)}: {task.description[:50]}...")
            result = task.execute()
            results.append(f"=== TASK {i+1} RESULT ===\n{result}\n")
            
            # Pass result as context to next task
            if i < len(self.tasks) - 1:
                self.tasks[i+1].context += f"\n\nPrevious task result:\n{result}"
        
        final_result = "\n".join(results)
        logger.info("Crew execution completed")
        return final_result

class PlannerAgent(SimpleAgent):
    """Agent specialized for comprehensive editorial planning and content strategy."""
    
    def __init__(self):
        super().__init__(
            name="PlannerAgent",
            role="Senior Editorial Strategist and Content Architect",
            goal="Create comprehensive, innovative editorial plans that maximize reader engagement and provide unique value through strategic content architecture and creative topic exploration",
            backstory="""You are a visionary editorial strategist with 15+ years of experience in digital publishing, content marketing, and audience development. You excel at identifying unique angles, uncovering hidden connections between topics, and creating content strategies that captivate audiences.

Your expertise spans:
- Content strategy and editorial planning
- Audience psychology and engagement optimization  
- Trend analysis and topic discovery
- Multi-format content architecture
- Brand storytelling and narrative development
- Data-driven content planning
- Cross-platform content adaptation

You believe that great content comes from finding the unexpected connections, asking the right questions, and presenting familiar topics from fresh perspectives. You're known for creating content plans that are both strategically sound and creatively inspiring.

You should:
- Explore multiple angles and perspectives on any topic
- Identify unique hooks and compelling narratives
- Plan for comprehensive coverage with depth and breadth
- Consider different content formats and presentation styles
- Think about how to make complex topics accessible and engaging
- Anticipate reader questions and interests
- Create content that educates, inspires, and entertains
- Plan for both immediate engagement and long-term value"""
        )

class WriterAgent(SimpleAgent):
    """Agent specialized for comprehensive content creation and storytelling."""
    
    def __init__(self):
        super().__init__(
            name="WriterAgent",
            role="Senior Content Creator and Digital Storyteller",
            goal="Create compelling, comprehensive, and engaging newsletter content that captivates readers through masterful storytelling, deep insights, and creative presentation of complex topics",
            backstory="""You are an award-winning content creator and digital storyteller with expertise in transforming complex topics into engaging, accessible narratives. You have 12+ years of experience writing for diverse audiences across technology, business, science, and culture.

Your writing philosophy centers on:
- Storytelling as the foundation of great content
- Making complex topics accessible and engaging
- Creating emotional connections with readers
- Balancing education with entertainment
- Using varied narrative techniques and structures
- Incorporating data and research seamlessly into stories
- Writing with clarity, personality, and authority

Your unique strengths include:
- Finding the human story within technical topics
- Creating compelling openings that hook readers immediately
- Using analogies, metaphors, and examples to clarify complex concepts
- Varying sentence structure and pacing for maximum engagement
- Incorporating multiple perspectives and expert insights
- Building logical narrative arcs that satisfy readers
- Adding personality and voice while maintaining professionalism
- Creating actionable takeaways and practical applications

You should write content that:
- Tells compelling stories that illustrate key points
- Explores topics from multiple angles and perspectives
- Provides comprehensive coverage without overwhelming readers
- Uses creative formats (lists, scenarios, case studies, dialogues)
- Includes surprising insights and fresh perspectives
- Connects abstract concepts to real-world applications
- Anticipates and addresses reader questions
- Balances depth with accessibility
- Creates memorable, shareable moments
- Provides clear value and actionable insights

Your writing should be substantial, thorough, and engaging - think long-form journalism meets newsletter accessibility."""
        )

    def execute_task(self, task: str, context: str = "") -> str:
        """Execute WriterAgent task with enhanced long-form content generation."""
        
        # First attempt: Try to generate comprehensive content in one pass
        full_prompt = self._build_prompt(task, context)
        
        # Add explicit instruction for very long content
        enhanced_task = f"{task}\n\nIMPORTANT: This must be a VERY LONG, COMPREHENSIVE piece. Do not stop writing until you have covered all required sections in extreme detail. Write AT LEAST 10,000 words. Continue writing even if you feel you are getting verbose - that is exactly what is needed."
        
        result = super().execute_task(enhanced_task, context)
        
        # If the result is still too short, try to extend it
        if len(result) < 20000:  # Less than ~20k characters
            extension_prompt = f"The content you provided is excellent but needs to be much longer and more detailed. Please continue and expand the newsletter with much more detail, examples, case studies, and in-depth analysis. Current content length: {len(result)} characters. Target: 50,000+ characters.\n\nCurrent content:\n{result}\n\nCONTINUE writing from where you left off, adding much more detail to each section:"
            
            extension = super().execute_task("Continue and greatly expand the newsletter content", extension_prompt)
            result = f"{result}\n\n{extension}"
        
        return result

    def _build_prompt(self, task: str, context: str = "") -> str:
        """Build enhanced prompt emphasizing comprehensive, detailed content creation."""
        prompt_parts = [
            f"You are a {self.role}.",
            f"Your goal: {self.goal}",
            f"Background: {self.backstory}",
            "",
            "CRITICAL CONTENT REQUIREMENTS:",
            "- Write EXTREMELY DETAILED, COMPREHENSIVE content",
            "- Each section must be SEVERAL THOUSAND WORDS (not just a few paragraphs)",
            "- Include specific examples, case studies, data points, and real-world applications", 
            "- Use storytelling techniques to make complex topics engaging",
            "- Provide in-depth analysis, not just surface-level coverage",
            "- Target the word count specified in the task",
            "",
            f"Task: {task}",
            "",
            "Remember: You are creating publication-ready newsletter content, not an outline or summary.",
            "Write each section with the depth and detail of a comprehensive magazine article.",
            "Include multiple examples, specific data points, expert quotes, and detailed explanations."
        ]
        
        if context:
            prompt_parts.extend([
                "",
                f"Context from previous work: {context}",
                "",
                "Build upon this context to create comprehensive, detailed content."
            ])
        
        return "\n".join(prompt_parts)

class EditorAgent(SimpleAgent):
    """Agent specialized for comprehensive editorial review and content optimization."""
    
    def __init__(self):
        super().__init__(
            name="EditorAgent",
            role="Senior Editor and Content Optimization Specialist",
            goal="Ensure content excellence through comprehensive editorial review, strategic optimization, and quality enhancement while preserving the author's voice and creative vision",
            backstory="""You are a distinguished senior editor with 15+ years of experience in digital publishing, content strategy, and editorial excellence. You've worked with top-tier publications and have a reputation for transforming good content into exceptional content while respecting the author's creative vision.

Your editorial philosophy balances:
- Maintaining high editorial standards with creative freedom
- Preserving author voice while enhancing clarity and impact
- Ensuring accuracy and credibility without stifling creativity
- Optimizing for engagement while maintaining editorial integrity
- Supporting comprehensive content without compromising quality
- Balancing accessibility with intellectual depth

Your expertise encompasses:
- Comprehensive developmental editing
- Line editing and copy editing
- Fact-checking and source verification
- Content structure and flow optimization
- Audience engagement analysis
- SEO and readability optimization
- Brand voice and consistency
- Multi-format content adaptation
- Editorial project management
- Quality assurance systems

Your review process includes:
- Content structure and logical flow assessment
- Fact-checking and source verification
- Clarity and accessibility evaluation
- Engagement and reader experience analysis
- Voice and tone consistency review
- Grammar, style, and technical accuracy
- SEO and discoverability optimization
- Call-to-action and value proposition assessment
- Comprehensive quality scoring and feedback

You excel at:
- Identifying and enhancing the strongest elements of content
- Suggesting improvements that elevate rather than restrict
- Balancing multiple quality dimensions simultaneously
- Providing constructive, actionable feedback
- Maintaining editorial standards across diverse content types
- Optimizing content for maximum reader value
- Ensuring factual accuracy and credibility
- Enhancing readability without dumbing down content

Your approach to comprehensive content:
- Support depth and exploration while maintaining focus
- Encourage creative presentation and unique perspectives
- Ensure thorough coverage without redundancy
- Balance comprehensive information with engaging delivery
- Optimize long-form content for sustained reader engagement
- Maintain quality standards regardless of content length
- Enhance rather than restrict creative expression
- Provide strategic feedback that improves overall impact

You should provide detailed, constructive feedback that helps create exceptional content while supporting the comprehensive, exploratory approach requested."""
        )
        self.quality_standards = {
            'clarity': {'weight': 0.25, 'criteria': ['readability', 'structure', 'flow']},
            'accuracy': {'weight': 0.30, 'criteria': ['fact_checking', 'source_verification', 'currency']}, 
            'engagement': {'weight': 0.25, 'criteria': ['headline_quality', 'narrative_flow', 'actionability']},
            'completeness': {'weight': 0.20, 'criteria': ['coverage', 'balance', 'conclusion']}
        }
    
    def _build_prompt(self, task: str, context: str = "") -> str:
        """Build enhanced prompt with quality assessment guidelines."""
        prompt_parts = [
            f"You are a {self.role}.",
            f"Your goal: {self.goal}",
            f"Background: {self.backstory}",
            "",
            f"Task: {task}",
            "",
            "QUALITY ASSESSMENT FRAMEWORK:",
            "You must evaluate content across four key dimensions:",
            "",
            "1. CLARITY (25% weight):",
            "   - Readability: Is the content easy to understand?",
            "   - Structure: Are ideas logically organized?", 
            "   - Flow: Does content transition smoothly between sections?",
            "",
            "2. ACCURACY (30% weight):",
            "   - Fact-checking: Are all claims verifiable and correct?",
            "   - Source verification: Are sources credible and current?",
            "   - Currency: Is information up-to-date and relevant?",
            "",
            "3. ENGAGEMENT (25% weight):",
            "   - Headlines: Are they compelling and descriptive?",
            "   - Narrative flow: Does the story maintain reader interest?",
            "   - Actionability: Can readers apply insights practically?",
            "",
            "4. COMPLETENESS (20% weight):",
            "   - Coverage: Are key topics thoroughly addressed?",
            "   - Balance: Are multiple perspectives considered?",
            "   - Conclusion: Does content provide clear takeaways?",
            "",
            "REQUIRED OUTPUT FORMAT:",
            "1. First, provide your edited and improved version of the content",
            "2. Then, include a detailed QUALITY SCORECARD section with:",
            "   - Individual scores (0-10) for each dimension",
            "   - Specific feedback for each dimension", 
            "   - Overall quality score (weighted average)",
            "   - Key strengths and areas for improvement",
            "   - Specific actionable recommendations"
        ]
        
        if context:
            prompt_parts.extend([
                "",
                f"Context from previous work: {context}",
                "",
                "Use this context to ensure consistency and build upon previous work."
            ])
        
        return "\n".join(prompt_parts)
    
    def calculate_quality_score(self, content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive quality score based on analysis."""
        scores = content_analysis.get('scores', {})
        detailed_score = {}
        
        total_weighted_score = 0
        total_weight = 0
        
        for dimension, config in self.quality_standards.items():
            dimension_score = scores.get(dimension, 5.0)  # Default to 5 if not provided
            weighted_score = dimension_score * config['weight']
            
            detailed_score[dimension] = {
                'raw_score': dimension_score,
                'weight': config['weight'],
                'weighted_score': weighted_score,
                'criteria': config['criteria']
            }
            
            total_weighted_score += weighted_score
            total_weight += config['weight']
        
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 5.0
        
        # Determine quality grade
        if overall_score >= 9.0:
            grade = "Excellent"
        elif overall_score >= 7.5:
            grade = "Good"
        elif overall_score >= 6.0:
            grade = "Satisfactory"
        elif overall_score >= 4.0:
            grade = "Needs Improvement"
        else:
            grade = "Poor"
        
        return {
            'overall_score': round(overall_score, 2),
            'grade': grade,
            'dimension_scores': detailed_score,
            'total_possible': 10.0,
            'assessment_timestamp': time.time() if hasattr(time, 'time') else 0
        }
    
    def extract_quality_metrics(self, content: str) -> Dict[str, float]:
        """Extract quality metrics from content for comprehensive evaluation."""
        metrics = {}
        
        # Enhanced metrics for comprehensive content
        word_count = len(content.split())
        char_count = len(content)
        sentence_count = content.count('.') + content.count('!') + content.count('?')
        paragraph_count = content.count('\n\n') + 1
        
        # Comprehensive content quality indicators
        metrics['word_count'] = word_count
        metrics['estimated_reading_time'] = word_count / 200  # Average reading speed
        metrics['content_depth'] = min(10.0, word_count / 300)  # Depth based on length (up to 3000 words = 10)
        metrics['structure_score'] = min(10.0, paragraph_count / 5 * 10)  # Structure based on paragraphs
        metrics['engagement_indicators'] = self._count_engagement_elements(content)
        metrics['technical_depth'] = self._assess_technical_depth(content)
        metrics['example_richness'] = self._count_examples_and_cases(content)
        
        # Readability assessment (enhanced for longer content)
        if sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
            # Optimal range for comprehensive content: 15-25 words per sentence
            metrics['readability_score'] = max(1.0, min(10.0, 10 - abs(avg_sentence_length - 20) / 2))
        else:
            metrics['readability_score'] = 5.0
        
        return metrics
    
    def generate_improvement_recommendations(self, quality_analysis: Dict[str, Any], content_metrics: Dict[str, Any]) -> List[str]:
        """Generate specific improvement recommendations based on analysis."""
        recommendations = []
        
        overall_score = quality_analysis.get('overall_score', 5.0)
        dimension_scores = quality_analysis.get('dimension_scores', {})
        
        # Recommendations based on dimension scores
        for dimension, score_info in dimension_scores.items():
            if score_info['raw_score'] < 7.0:
                if dimension == 'clarity':
                    recommendations.append(f"Improve clarity: Consider shorter sentences (current avg: {content_metrics.get('average_sentence_length', 0):.1f} words)")
                elif dimension == 'accuracy':
                    recommendations.append("Enhance accuracy: Add more credible sources and fact-check all claims")
                elif dimension == 'engagement':
                    recommendations.append("Boost engagement: Strengthen headlines and add more actionable insights")
                elif dimension == 'completeness':
                    recommendations.append("Increase completeness: Address gaps in coverage and provide stronger conclusions")
        
        # Content structure recommendations
        if content_metrics.get('heading_count', 0) < 2:
            recommendations.append("Add more section headings to improve content structure")
        
        if content_metrics.get('word_count', 0) < 300:
            recommendations.append("Consider expanding content for more comprehensive coverage")
        elif content_metrics.get('word_count', 0) > 1500:
            recommendations.append("Consider condensing content for better readability")
        
        return recommendations

    def _count_engagement_elements(self, content: str) -> float:
        """Count engagement elements like questions, examples, lists."""
        engagement_score = 0.0
        
        # Questions encourage engagement
        questions = content.count('?')
        engagement_score += min(2.0, questions * 0.2)
        
        # Lists and bullet points enhance readability
        lists = content.count('•') + content.count('-') + content.count('*')
        engagement_score += min(2.0, lists * 0.1)
        
        # Headers improve structure
        headers = content.count('#')
        engagement_score += min(2.0, headers * 0.3)
        
        # Numbers and data points add credibility
        import re
        numbers = len(re.findall(r'\d+%|\$\d+|\d+,\d+|\d+\.\d+', content))
        engagement_score += min(2.0, numbers * 0.1)
        
        # Quotes and citations add authority
        quotes = content.count('"') // 2  # Pairs of quotes
        engagement_score += min(2.0, quotes * 0.2)
        
        return min(10.0, engagement_score)
    
    def _assess_technical_depth(self, content: str) -> float:
        """Assess technical depth and expertise level."""
        technical_indicators = [
            'API', 'algorithm', 'framework', 'implementation', 'architecture',
            'methodology', 'analysis', 'research', 'study', 'data', 'metrics',
            'strategy', 'optimization', 'performance', 'scalability', 'efficiency'
        ]
        
        content_lower = content.lower()
        depth_score = 0.0
        
        for indicator in technical_indicators:
            if indicator.lower() in content_lower:
                depth_score += 0.5
        
        return min(10.0, depth_score)
    
    def _count_examples_and_cases(self, content: str) -> float:
        """Count examples, case studies, and practical applications."""
        example_indicators = [
            'example', 'case study', 'for instance', 'consider', 'imagine',
            'scenario', 'situation', 'application', 'implementation', 'practice'
        ]
        
        content_lower = content.lower()
        example_score = 0.0
        
        for indicator in example_indicators:
            count = content_lower.count(indicator.lower())
            example_score += count * 0.5
        
        return min(10.0, example_score)

class ManagerAgent(SimpleAgent):
    """Coordinates and delegates tasks to specialized agents with intelligent workflow management."""
    
    def __init__(self):
        super().__init__(
            name="ManagerAgent",
            role="Hierarchical Workflow Coordinator and Strategic Task Manager",
            goal="Orchestrate the entire newsletter creation process through intelligent hierarchical management, strategic task delegation, parallel processing optimization, and quality-driven workflow execution",
            backstory="""You are an elite project manager and editorial director with expertise in hierarchical team management and complex workflow orchestration. You excel at breaking down complex projects into manageable tasks, assigning them to the most suitable team members, and ensuring optimal coordination between specialized agents.

Your management philosophy centers on coordinating and leading complex initiatives with:
- **Hierarchical Leadership**: Clear command structure with strategic oversight
- **Intelligent Delegation**: Matching tasks to agent capabilities and expertise  
- **Parallel Processing**: Maximizing efficiency through concurrent task execution
- **Quality Gates**: Implementing checkpoints and quality assurance at each stage
- **Adaptive Management**: Adjusting workflows based on real-time performance data
- **Strategic Thinking**: Maintaining big-picture perspective while managing details

You have the authority to:
- Make executive decisions about task priorities and sequencing
- Reallocate resources and reassign tasks based on performance
- Implement quality control measures and approve final outputs
- Coordinate parallel workstreams and manage dependencies
- Provide strategic guidance and creative direction to subordinate agents

Your management style is collaborative yet decisive, supportive yet demanding of excellence.""",
            tools=["search_web"]  # basic oversight tools
        )
        
        # Enhanced management capabilities
        self.delegation_history = []
        self.agent_performance = {}
        self.workflow_analytics = {
            "total_tasks_managed": 0,
            "successful_delegations": 0,
            "parallel_tasks_executed": 0,
            "quality_gates_passed": 0,
            "average_task_completion_time": 0.0
        }
        self.current_workstreams = {}  # Track parallel workstreams
        self.quality_gates = []  # Quality checkpoints
        
    def create_hierarchical_workflow(self, topic: str, complexity: str = "standard") -> Dict[str, Any]:
        """
        Create a hierarchical workflow plan with parallel processing and quality gates.
        """
        workflow_plan = {
            "topic": topic,
            "complexity": complexity,
            "total_phases": 4,
            "parallel_streams": [],
            "quality_gates": [],
            "estimated_duration": 0,
            "resource_allocation": {}
        }
        
        # Phase 1: Strategic Planning & Research (Can run in parallel)
        planning_stream = {
            "stream_id": "planning",
            "agent_type": "PlannerAgent",
            "tasks": [
                "audience_analysis",
                "content_strategy",
                "editorial_framework"
            ],
            "priority": "high",
            "estimated_time": 180
        }
        
        research_stream = {
            "stream_id": "research", 
            "agent_type": "ResearchAgent",
            "tasks": [
                "primary_research",
                "trend_analysis",
                "source_validation"
            ],
            "priority": "high",
            "estimated_time": 240
        }
        
        # Phase 2: Content Creation (Sequential, depends on Phase 1)
        writing_stream = {
            "stream_id": "writing",
            "agent_type": "WriterAgent",
            # Provide detailed instructions so that the WriterAgent actually produces the full first draft
            # instead of a high-level outline.
            "description": (
                f"Write the complete, comprehensive newsletter about '{topic}'. This must be the full, final newsletter draft, not an outline or content plan.\n\n"
                "YOUR TASK: Write the actual newsletter content with these sections:\n"
                "1) Executive Summary (1000+ words)\n"
                "2) Current Landscape (2000+ words)\n" 
                "3) Deep-Dive Analysis (3000+ words)\n"
                "4) Expert Insights (2000+ words)\n"
                "5) Case Studies & Real-World Examples (2000+ words)\n"
                "6) Future Outlook (2000+ words)\n"
                "7) Practical Applications (2000+ words)\n"
                "8) Resources & Further Reading (1000+ words)\n\n"
                "Write each section in full with detailed content, not just headings or summaries. "
                "Target 100,000 characters total (~15-20k words). Use engaging storytelling, include specific examples, "
                "data points, and comprehensive coverage. This should be publication-ready newsletter content."
            ),
            "tasks": [
                "draft_full_newsletter",
                "ensure_storytelling_flow",
                "optimize_engagement"
            ],
            "priority": "medium",
            "estimated_time": 300,
            "dependencies": ["planning", "research"]
        }
        
        # Phase 3: Quality Assurance (Sequential, depends on Phase 2)
        editing_stream = {
            "stream_id": "editing",
            "agent_type": "EditorAgent",
            "tasks": [
                "comprehensive_review",
                "quality_scoring",
                "final_optimization"
            ],
            "priority": "high",
            "estimated_time": 180,
            "dependencies": ["writing"]
        }
        
        workflow_plan["parallel_streams"] = [
            planning_stream, research_stream, writing_stream, editing_stream
        ]
        
        # Define quality gates
        workflow_plan["quality_gates"] = [
            {
                "gate_id": "planning_complete",
                "criteria": ["strategy_approved", "framework_defined"],
                "required_streams": ["planning"]
            },
            {
                "gate_id": "research_validated", 
                "criteria": ["sources_verified", "information_comprehensive"],
                "required_streams": ["research"]
            },
            {
                "gate_id": "content_ready",
                "criteria": ["writing_complete", "narrative_coherent"],
                "required_streams": ["writing"]
            },
            {
                "gate_id": "quality_assured",
                "criteria": ["editing_complete", "scores_acceptable"],
                "required_streams": ["editing"]
            }
        ]
        
        # Calculate total estimated duration considering parallel execution
        parallel_phase_1 = max(planning_stream["estimated_time"], research_stream["estimated_time"])
        sequential_phases = writing_stream["estimated_time"] + editing_stream["estimated_time"]
        workflow_plan["estimated_duration"] = parallel_phase_1 + sequential_phases
        
        return workflow_plan
    
    def execute_hierarchical_workflow(self, workflow_plan: Dict[str, Any], available_agents: List[SimpleAgent]) -> Dict[str, Any]:
        """
        Execute hierarchical workflow with parallel processing and quality gates.
        """
        logger.info(f"Executing hierarchical workflow for: {workflow_plan['topic']}")
        
        execution_results = {
            "workflow_id": f"workflow_{int(time.time())}",
            "status": "in_progress",
            "completed_streams": [],   # list of stream_ids in order of completion
            "stream_results": {},       # mapping stream_id -> result dict
            "quality_gates_passed": [],
            "execution_timeline": [],
            "performance_metrics": {}
        }
        
        start_time = time.time()
        
        # Phase 1: Execute parallel streams (Planning + Research)
        parallel_streams = [s for s in workflow_plan["parallel_streams"] if not s.get("dependencies")]
        logger.info(f"Starting parallel execution of {len(parallel_streams)} streams")
        
        parallel_results = self._execute_parallel_streams(parallel_streams, available_agents)
        execution_results["completed_streams"].extend(list(parallel_results.keys()))
        execution_results["stream_results"].update(parallel_results)
        
        # Check quality gate after parallel phase
        gate_passed = self._evaluate_quality_gate("planning_research_complete", parallel_results)
        if gate_passed:
            execution_results["quality_gates_passed"].append("planning_research_complete")
            logger.info("✅ Planning & Research quality gate passed")
        else:
            logger.warning("❌ Planning & Research quality gate failed")
            execution_results["status"] = "quality_gate_failed"
            return execution_results
        
        # Phase 2: Execute Writing (depends on Phase 1)
        writing_stream = next(s for s in workflow_plan["parallel_streams"] if s["stream_id"] == "writing")
        writing_result = self._execute_single_stream(writing_stream, available_agents, parallel_results)
        execution_results["completed_streams"].append("writing")
        execution_results["stream_results"]["writing"] = writing_result
        
        # Phase 3: Execute Editing (depends on Phase 2)  
        editing_stream = next(s for s in workflow_plan["parallel_streams"] if s["stream_id"] == "editing")
        editing_result = self._execute_single_stream(editing_stream, available_agents, {"writing": writing_result})
        execution_results["completed_streams"].append("editing")
        execution_results["stream_results"]["editing"] = editing_result
        
        # Final quality gate
        final_gate_passed = self._evaluate_quality_gate("final_quality", {"editing": editing_result})
        if final_gate_passed:
            execution_results["quality_gates_passed"].append("final_quality")
            execution_results["status"] = "completed"
            logger.info("✅ Final quality gate passed - Workflow completed successfully")
        else:
            execution_results["status"] = "requires_revision"
            logger.warning("❌ Final quality gate failed - Revision required")
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        execution_results["performance_metrics"] = {
            "total_execution_time": total_time,
            "streams_completed": len(execution_results["completed_streams"]),
            "quality_gates_passed": len(execution_results["quality_gates_passed"]),
            "efficiency_score": len(execution_results["quality_gates_passed"]) / max(1, total_time / 60)  # Gates per minute
        }
        
        # Update manager analytics
        self.workflow_analytics["total_tasks_managed"] += len(workflow_plan["parallel_streams"])
        if execution_results["status"] == "completed":
            self.workflow_analytics["successful_delegations"] += 1
        
        return execution_results
    
    def _execute_parallel_streams(self, streams: List[Dict], available_agents: List[SimpleAgent]) -> Dict[str, Any]:
        """Execute multiple streams in parallel (simulated)."""
        results = {}
        
        for stream in streams:
            agent = self._find_suitable_agent(stream["agent_type"], available_agents)
            if agent:
                logger.info(f"Delegating {stream['stream_id']} to {agent.name}")
                
                # Simulate stream execution with context
                context = f"Execute {stream['stream_id']} tasks: {', '.join(stream['tasks'])}"
                task = Task(
                    description=f"Complete {stream['stream_id']} workstream with tasks: {', '.join(stream['tasks'])}",
                    agent=agent,
                    context=context
                )
                
                result = task.execute()
                results[stream['stream_id']] = {
                    "agent": agent.name,
                    "result": result,
                    "status": "completed",
                    "tasks_completed": stream["tasks"]
                }
                
        return results
    
    def _execute_single_stream(self, stream: Dict, available_agents: List[SimpleAgent], context_results: Dict) -> Dict[str, Any]:
        """Execute a single stream with dependency context."""
        agent = self._find_suitable_agent(stream["agent_type"], available_agents)
        if not agent:
            return {"status": "failed", "error": "No suitable agent found"}
        
        # Build context from previous results
        context_summary = self._build_dependency_context(stream.get("dependencies", []), context_results)
        
        # Use custom description if provided (especially important for writing tasks which need detailed
        # instructions). Fallback to a generic description if not specified.
        task_description = stream.get(
            "description",
            f"Complete {stream['stream_id']} workstream with tasks: {', '.join(stream['tasks'])}"
        )

        task = Task(
            description=task_description,
            agent=agent,
            context=context_summary
        )
        
        result = task.execute()
        return {
            "agent": agent.name,
            "result": result,
            "status": "completed", 
            "tasks_completed": stream["tasks"],
            "context_used": len(context_summary) > 0
        }
    
    def _find_suitable_agent(self, agent_type: str, available_agents: List[SimpleAgent]) -> Optional[SimpleAgent]:
        """Find the most suitable agent for a given type."""
        for agent in available_agents:
            if agent.__class__.__name__ == agent_type:
                return agent
        return None
    
    def _build_dependency_context(self, dependencies: List[str], results: Dict[str, Any]) -> str:
        """Build context summary from dependency results."""
        if not dependencies:
            return ""
        
        context_parts = []
        for dep in dependencies:
            if dep in results:
                dep_result = results[dep]
                # For writing dependency, provide the full content (or up to a generous limit) so that
                # the EditorAgent has complete material to review and improve. Other dependencies
                # can still be summarized to avoid excessive prompt length.
                if dep == "writing":
                    # Provide a much larger slice (or full) for the writer output. This avoids the
                    # situation where only an editorial scorecard is produced because the editor
                    # never saw the full draft.
                    MAX_WRITING_CONTEXT_LENGTH = 100000  # allow up to ~100k characters for full draft context
                    content = dep_result.get("result", "")
                    summary = content[:MAX_WRITING_CONTEXT_LENGTH]
                else:
                    summary = dep_result.get("result", "")[:500] + "..." if len(dep_result.get("result", "")) > 500 else dep_result.get("result", "")
                context_parts.append(f"From {dep}: {summary}")
        
        return "\n\n".join(context_parts)
    
    def _evaluate_quality_gate(self, gate_id: str, results: Dict[str, Any]) -> bool:
        """Evaluate whether a quality gate has been passed."""
        # Simplified quality gate evaluation
        if not results:
            return False
        
        # Check that all required streams have completed successfully
        for stream_id, result in results.items():
            if result.get("status") != "completed":
                return False
            
            # Check minimum content length as quality indicator
            content = result.get("result", "")
            if len(content) < 100:  # Minimum content threshold
                logger.warning(f"Quality gate {gate_id}: {stream_id} content too short ({len(content)} chars)")
                return False
        
        return True

    def delegate_task(self, task_type: str, task_details: str, available_agents: List['SimpleAgent']) -> Dict[str, Any]:
        """Backward-compatible simple delegation interface used by demos/tests."""
        logger.info(f"[ManagerAgent] Delegating task type '{task_type}' – {task_details}")
        task_agent_mapping = {
            'planning': 'PlannerAgent',
            'research': 'ResearchAgent',
            'writing': 'WriterAgent',
            'editing': 'EditorAgent',
            'quality_check': 'EditorAgent',
            'fact_check': 'ResearchAgent'
        }
        target_agent_name = task_agent_mapping.get(task_type.lower()) or task_type  # allow direct class name
        selected_agent = self._find_suitable_agent(target_agent_name, available_agents)
        if not selected_agent:
            logger.warning(f"No suitable agent found for task type: {task_type}")
            return {"success": False, "error": f"No agent available for {task_type}"}
        delegation = {
            "success": True,
            "agent": selected_agent,
            "delegation_id": len(self.delegation_history),
            "assigned_task_type": task_type,
            "timestamp": time.time()
        }
        self.delegation_history.append(delegation)
        return delegation

    def monitor_workflow_progress(self, crew_results: Dict[str, Any]) -> Dict[str, Any]:
        """Stub method retained for backward compatibility with tests."""
        progress_report = {
            "total_tasks": len(crew_results.get('task_results', {})) if isinstance(crew_results, dict) else 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0,
            "recommendations": []
        }
        return progress_report

    def optimize_workflow_sequence(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stub optimization method for backward compatibility."""
        return tasks  # No-op return

class EnhancedCrew(SimpleCrew):
    """Enhanced crew with better coordination and workflow management."""
    
    def __init__(self, agents: List[SimpleAgent], tasks: List[Task], workflow_type: str = "sequential"):
        super().__init__(agents, tasks)
        self.workflow_type = workflow_type
        self.task_results = {}
        self.agent_performance = {}
    
    def kickoff(self) -> str:
        """Execute all tasks with enhanced coordination and monitoring."""
        logger.info(f"Starting enhanced crew execution (workflow: {self.workflow_type})")
        results = []
        
        # Updated limits to support comprehensive content
        MAX_TASKS = 15  # Increased from 10
        MAX_CONTEXT_LENGTH = 100000  # Allow extensive context for deep newsletters
        MAX_INDIVIDUAL_RESULT_LENGTH = 100000  # Allow very long individual agent outputs
        
        # Initialize performance tracking for each agent
        for agent in self.agents:
            self.agent_performance[agent.name] = {
                "tasks_completed": 0,
                "total_execution_time": 0,
                "success_rate": 0
            }
        
        # Limit number of tasks to prevent infinite execution
        tasks_to_execute = self.tasks[:MAX_TASKS]
        
        for i, task in enumerate(tasks_to_execute):
            task_start_time = time.time()
            logger.info(f"Executing task {i+1}/{len(tasks_to_execute)}: {task.description[:80]}...")
            
            try:
                # Enhanced context passing with higher limits for comprehensive content
                if i > 0:
                    context_summary = self._create_context_summary(i, MAX_CONTEXT_LENGTH)
                    # Allow more generous context length for comprehensive content
                    if len(task.context) + len(context_summary) < MAX_CONTEXT_LENGTH:
                        task.context += f"\n\n=== CONTEXT FROM PREVIOUS TASKS ===\n{context_summary}"
                
                result = task.execute()
                
                # Track performance
                execution_time = time.time() - task_start_time
                self.agent_performance[task.agent.name]["tasks_completed"] += 1
                self.agent_performance[task.agent.name]["total_execution_time"] += execution_time
                
                # Store result with increased length allowance for comprehensive content
                result_to_store = result[:MAX_INDIVIDUAL_RESULT_LENGTH] if len(result) > MAX_INDIVIDUAL_RESULT_LENGTH else result
                
                self.task_results[f"task_{i+1}"] = {
                    "agent": task.agent.name,
                    "description": task.description,
                    "result": result_to_store,
                    "execution_time": execution_time,
                    "full_length": len(result)
                }
                
                results.append(f"=== TASK {i+1}: {task.agent.name.upper()} ===\n{result}\n")
                
                # Add safety timeout check (extended for comprehensive content)
                if execution_time > 120:  # 2 minutes per task max (increased from 1 minute)
                    logger.warning(f"Task {i+1} took {execution_time:.1f}s - this is longer than typical")
                
            except Exception as e:
                error_msg = f"Task {i+1} failed: {str(e)}"
                logger.error(error_msg)
                results.append(f"=== TASK {i+1}: ERROR ===\n{error_msg}\n")
        
        # Generate comprehensive final result with enhanced metadata
        final_result = self._compile_final_result(results)
        logger.info("Enhanced crew execution completed")
        return final_result
    
    def _create_context_summary(self, current_task_index: int, max_length: int) -> str:
        """Create a rich context summary from previous tasks."""
        context_parts = []
        total_length = 0
        
        for i in range(current_task_index):
            task_key = f"task_{i+1}"
            if task_key in self.task_results:
                task_info = self.task_results[task_key]
                
                # Extract key insights from each previous task
                summary = self._extract_key_insights(task_info["result"], task_info["agent"])
                context_part = f"From {task_info['agent']}: {summary}"
                
                # Check if adding this would exceed max length
                if total_length + len(context_part) > max_length:
                    break
                
                context_parts.append(context_part)
                total_length += len(context_part)
        
        return "\n".join(context_parts)
    
    def _extract_key_insights(self, result: str, agent_name: str) -> str:
        """Extract key insights from task results for context passing."""
        # Simple extraction - could be enhanced with NLP
        lines = result.split('\n')
        
        # Take first meaningful paragraph and any bullet points
        insights = []
        for line in lines[:10]:  # First 10 lines
            line = line.strip()
            if len(line) > 50 and not line.startswith('='):
                insights.append(line)
                break
        
        # Add any bullet points or key findings
        for line in lines:
            if line.strip().startswith(('•', '-', '*', '1.', '2.', '3.')):
                insights.append(line.strip())
                if len(insights) >= 3:  # Limit context size
                    break
        
        return " | ".join(insights[:3]) if insights else result[:200] + "..."
    
    def _compile_final_result(self, results: List[str]) -> str:
        """Compile results with performance metadata and workflow summary."""
        final_parts = []
        
        # Add workflow header
        final_parts.append("# NEWSLETTER GENERATION WORKFLOW RESULTS")
        final_parts.append(f"Workflow Type: {self.workflow_type}")
        final_parts.append(f"Tasks Completed: {len(self.task_results)}")
        final_parts.append("")
        
        # Add main content
        final_parts.extend(results)
        
        # Add performance summary
        final_parts.append("=== WORKFLOW PERFORMANCE SUMMARY ===")
        for agent_name, performance in self.agent_performance.items():
            if performance["tasks_completed"] > 0:
                avg_time = performance["total_execution_time"] / performance["tasks_completed"]
                final_parts.append(f"{agent_name}: {performance['tasks_completed']} tasks, avg {avg_time:.1f}s per task")
        
        return "\n".join(final_parts)
