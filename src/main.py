"""
Main orchestration script for the AI Newsletter System (Enhanced Version)

This demonstrates the complete enhanced system with:
- ManagerAgent for intelligent task delegation
- Multi-agent coordination with quality feedback
- Performance monitoring and learning
- Comprehensive quality assessment
- User feedback collection and learning
"""

import logging
import sys
import os
import time
from datetime import datetime
from typing import Dict, Any
import atexit

# Add src to path so imports work correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.agents import (
    ResearchAgent, WriterAgent, EditorAgent, 
    ManagerAgent, PlannerAgent, Task, EnhancedCrew
)
from src.core.core import query_llm
from src.core.feedback_system import FeedbackLearningSystem
from src.tools.notion_integration import NotionNewsletterPublisher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log'),
        logging.StreamHandler()
    ]
)

# Ensure log buffers are flushed on program exit
atexit.register(logging.shutdown)

logger = logging.getLogger(__name__)

# NOTE: Hierarchical flag constant
HIERARCHICAL_FLAG = "--hierarchical"

def create_research_workflow(topic: str) -> Dict[str, Any]:
    """Create a basic research workflow for testing."""
    research_agent = ResearchAgent()
    task = Task(description=f"Research {topic}", agent=research_agent)
    crew = EnhancedCrew([research_agent], [task])
    return {"topic": topic, "crew": crew}

def test_basic_functionality() -> bool:
    """Test basic system functionality."""
    try:
        # Test LLM connection
        response = query_llm("Hello, this is a test.")
        if not response or "error" in response.lower():
            return False
        
        # Test agent creation
        research_agent = ResearchAgent()
        if not research_agent:
            return False
            
        return True
    except Exception as e:
        logger.error(f"Basic functionality test failed: {e}")
        return False

def create_enhanced_newsletter_workflow(topic: str, audience: str = "technology professionals") -> EnhancedCrew:
    """Create a comprehensive multi-agent newsletter workflow with ManagerAgent coordination."""
    
    logger.info(f"Creating enhanced newsletter workflow for topic: {topic}")
    
    # Create all specialized agents
    manager_agent = ManagerAgent()
    planner_agent = PlannerAgent() 
    research_agent = ResearchAgent()
    writer_agent = WriterAgent()
    editor_agent = EditorAgent()
    
    agents = [manager_agent, planner_agent, research_agent, writer_agent, editor_agent]
    
    # Load gold standard example if present
    gold_standard_path = os.path.join(os.path.dirname(__file__), "..", "gold_standard_example.md")
    gold_standard_content = ""
    if os.path.exists(gold_standard_path):
        with open(gold_standard_path, "r", encoding="utf-8") as f:
            gold_standard_content = f.read()

    if gold_standard_content:
        editing_context_extra = f"\n\nGOLD STANDARD EXAMPLE:\n{gold_standard_content[:1500]}\n\n(Example truncated if too long)"
    else:
        editing_context_extra = ""

    # Define comprehensive tasks with detailed instructions for exploratory content
    planning_task = Task(
        description=f"""Create a comprehensive editorial strategy for an in-depth newsletter about '{topic}' 
        targeted at {audience}. Your plan should be thorough and creative, exploring multiple dimensions:

        🎯 STRATEGIC PLANNING:
        1. **Content Architecture**: Design a multi-layered content structure that balances depth with accessibility
        2. **Unique Angles**: Identify 3-5 unique perspectives or angles that haven't been widely covered
        3. **Reader Journey**: Map out how readers will progress through the content and what they'll gain
        4. **Hook Strategy**: Develop compelling openings and attention-grabbing elements
        5. **Value Propositions**: Define specific, measurable value readers will receive

        🔍 CONTENT EXPLORATION:
        1. **Multi-dimensional Coverage**: Plan coverage of historical context, current state, future implications
        2. **Stakeholder Perspectives**: Include viewpoints from different industry players and end-users
        3. **Surprising Connections**: Identify unexpected relationships between {topic} and other fields
        4. **Practical Applications**: Plan real-world examples, case studies, and actionable insights
        5. **Emerging Trends**: Anticipate future developments and their implications

        📊 AUDIENCE ENGAGEMENT:
        1. **Engagement Techniques**: Plan interactive elements, thought-provoking questions, scenarios
        2. **Content Formats**: Mix formats (stories, data visualizations, interviews, predictions)
        3. **Accessibility Levels**: Ensure content works for both experts and newcomers
        4. **Shareability**: Design moments that readers will want to discuss and share
        5. **Follow-up Value**: Create content that provides lasting reference value

        Be creative, comprehensive, and strategic. Think like a seasoned editorial director planning a flagship piece.""",
        agent=planner_agent,
        context="This is the foundational planning phase - be thorough, creative, and strategic. Set the stage for exceptional content."
    )
    
    research_task = Task(
        description=f"""Conduct comprehensive, multi-dimensional research on '{topic}' based on the editorial plan. 
        Your research should be thorough, insightful, and exploratory:

        🔬 RESEARCH METHODOLOGY:
        1. **Primary Sources**: Find original research, studies, white papers, and expert interviews
        2. **Industry Intelligence**: Gather insights from key industry players and documented research
        3. **Trend Analysis**: Identify emerging patterns, market shifts, and future implications
        4. **Cross-Industry Connections**: Explore how {topic} impacts or relates to other sectors
        5. **Historical Context**: Provide background and evolution of the topic

        📈 COMPREHENSIVE COVERAGE:
        1. **Current State Analysis**: Detailed assessment of where things stand today
        2. **Market Dynamics**: Key players, competitive landscape, market forces
        3. **Technology Deep-Dive**: Technical aspects, innovations, and breakthrough developments
        4. **User/Customer Perspectives**: Real-world experiences, pain points, success stories
        5. **Regulatory/Policy Landscape**: Relevant regulations, policy changes, compliance issues

        🎯 INSIGHTS & INTELLIGENCE:
        1. **Surprising Findings**: Look for counterintuitive or lesser-known insights
        2. **Expert Predictions**: Gather forecasts and predictions from credible sources
        3. **Case Studies**: Find compelling success stories, failure lessons, and practical examples
        4. **Data & Statistics**: Collect relevant metrics, trends, and quantitative insights
        5. **Emerging Opportunities**: Identify new possibilities and untapped potential

        Use all available search tools extensively. Be thorough, analytical, and forward-thinking. 
        Provide rich, well-sourced material that can support comprehensive content creation.""",
        agent=research_agent,
        context="Use web search extensively. Look for current information, expert opinions, and emerging trends. Be comprehensive and analytical."
    )
    
    writing_task = Task(
        description=f"""Create a comprehensive, in-depth newsletter about '{topic}' for {audience}.
        This must be a COMPLETE, PUBLICATION-READY newsletter with substantial depth and analysis.

        🎯 CONTENT REQUIREMENTS:
        - STYLE: Flowing narrative prose, NOT bullet points or lists
        - DEPTH: Magazine-quality investigative journalism level
        - STRUCTURE: Cohesive sections that build upon each other
        - TONE: Authoritative yet accessible, engaging storytelling

        📚 MANDATORY SECTIONS:

        1. **Executive Summary & Key Insights**
           - Compelling opening that hooks readers immediately
           - Overview of the most important developments and their implications
           - Key takeaways that busy executives need to know
           - Why this topic matters now and what's at stake

        2. **Current Landscape Analysis**
           - Comprehensive state-of-the-field analysis
           - Major players, technologies, and market dynamics
           - Recent developments and breakthrough moments
           - Comparative analysis of different approaches and methodologies
           - Regional variations and global perspectives

        3. **Deep Technical Analysis**
           - In-depth exploration of core concepts and mechanisms
           - Technical innovations and their practical implications
           - Detailed case studies with specific examples
           - Performance metrics, benchmarks, and comparative analysis
           - Technical challenges and how they're being addressed

        4. **Real-World Applications & Case Studies**
           - Detailed examination of successful implementations
           - Comprehensive case studies from different industries
           - Lessons learned from failures and challenges
           - ROI analysis and business impact assessments
           - Scalability considerations and implementation strategies

        5. **Future Outlook & Emerging Trends**
           - Predicted developments over the next 1-3 years
           - Emerging technologies and their potential convergence
           - Investment trends and market predictions
           - Regulatory considerations and policy implications
           - Potential disruptions and paradigm shifts

        6. **Practical Implementation Guide**
           - Step-by-step guidance for organizations getting started
           - Resource requirements and budget considerations
           - Common pitfalls and how to avoid them
           - Success metrics and measurement frameworks
           - Recommendations for different organization sizes

        7. **Resources & Further Learning**
           - Comprehensive resource compilation
           - Recommended reading: books, research papers, articles
           - Online courses, certifications, and training programs
           - Industry conferences, events, and networking opportunities
           - Tools, platforms, and software recommendations

        ✍️ WRITING STYLE REQUIREMENTS:
        - Use NARRATIVE PROSE throughout - avoid bullet points except for very specific lists
        - Write in flowing paragraphs that build complex arguments
        - Include specific examples, data points, and concrete details
        - Use storytelling techniques to make technical concepts accessible
        - Incorporate analogies and metaphors to clarify complex ideas
        - Maintain consistent voice and tone throughout
        - Create smooth transitions between sections and ideas
        - Use subheadings sparingly - let the content flow naturally

        🔍 CONTENT DEPTH REQUIREMENTS:
        - Every claim must be supported with specific evidence or examples
        - Include quantitative data wherever possible (percentages, growth rates, adoption metrics)
        - Provide historical context and evolutionary perspective
        - Address counterarguments and limitations honestly
        - Explore unexpected connections and implications
        - Anticipate reader questions and address them proactively

        📖 RESEARCH INTEGRATION:
        - Seamlessly integrate research findings throughout the narrative
        - Reference specific studies, surveys, and industry reports
        - Include relevant statistics and trend data
        - Cite published research and documented findings
        - Balance academic rigor with practical applicability

        🎨 ENGAGEMENT TECHNIQUES:
        - Open each section with a compelling hook or scenario
        - Use vivid descriptions and concrete imagery
        - Include surprising insights or counterintuitive findings
        - Create "aha moments" that shift reader perspective
        - End sections with thought-provoking questions or implications

        CRITICAL: This must be a comprehensive, magazine-quality article that readers will want to bookmark and reference. Write in full paragraphs with rich detail, avoiding lists and bullet points. 

        📏 LENGTH REQUIREMENT: 
        - MINIMUM: 1,000 words (approximately 4-5 pages)
        - TARGET: 2,000-3,000 words (approximately 8-12 pages)
        - MAXIMUM: 5,000 words (approximately 20 pages)
        
        ⚠️ CRITICAL: You MUST generate at least 1,000 words. This is a hard requirement.
        ⚠️ CRITICAL: Each section should be substantial (200-500 words per section). 
        ⚠️ CRITICAL: Do NOT create short, superficial content. 
        ⚠️ CRITICAL: This should be a deep, comprehensive analysis that provides exceptional value to readers.
        ⚠️ CRITICAL: Make every section substantial and valuable.
        ⚠️ CRITICAL: If your response is less than 1,000 words, you have FAILED the task.
        
        Each section should be substantial (200-500 words per section). Do NOT create short, superficial content. This should be a deep, comprehensive analysis that provides exceptional value to readers. Make every section substantial and valuable.""",
        agent=writer_agent,
        context="Build on the planning and research work. Create comprehensive, engaging content that provides exceptional reader value. Be thorough and creative."
    )
    
    editing_task = Task(
        description=f"""Perform comprehensive editorial review and optimization of the newsletter content about '{topic}'.
        Ensure excellence while supporting the comprehensive, exploratory approach:

        📋 EDITORIAL REVIEW PROCESS:
        1. **Content Structure & Flow**: Assess logical progression, transitions, and overall architecture
        2. **Depth & Comprehensiveness**: Ensure thorough coverage without redundancy or gaps
        3. **Engagement & Readability**: Optimize for sustained reader engagement throughout
        4. **Accuracy & Credibility**: Fact-check claims, verify sources, ensure reliability
        5. **Voice & Consistency**: Maintain consistent tone and style throughout

        🔍 DETAILED ANALYSIS:
        1. **Opening Impact**: Evaluate and enhance the hook and opening sections
        2. **Section Balance**: Ensure each section provides value and maintains interest
        3. **Example Quality**: Assess and improve case studies and real-world applications
        4. **Conclusion Strength**: Ensure strong, actionable conclusions and next steps
        5. **Overall Value**: Confirm the content delivers exceptional reader value

        📊 QUALITY ASSESSMENT:
        1. **Clarity Score** (1-10): How clear and accessible is the content?
        2. **Accuracy Score** (1-10): How well-researched and factually sound?
        3. **Engagement Score** (1-10): How compelling and interesting throughout?
        4. **Comprehensiveness Score** (1-10): How thoroughly does it cover the topic?
        5. **Practical Value Score** (1-10): How actionable and useful for readers?

        ✨ OPTIMIZATION AREAS:
        1. **Enhancement Opportunities**: Specific areas where content can be strengthened
        2. **Reader Experience**: Improvements for better flow and engagement
        3. **Value Amplification**: Ways to increase practical value for readers
        4. **Accessibility**: Ensuring content works for the target audience
        5. **Memorable Moments**: Identifying and enhancing shareable insights

        Provide detailed feedback with specific improvement suggestions. Support comprehensive content 
        while ensuring it meets the highest editorial standards. Include your detailed QUALITY SCORECARD.""",
        agent=editor_agent,
        context="This is the final quality gate. Use fact-checking tools as needed. Support comprehensive content while ensuring excellence." + editing_context_extra
    )
    
    tasks = [planning_task, research_task, writing_task, editing_task]
    
    # Create enhanced crew with manager coordination
    crew = EnhancedCrew(
        agents=agents,
        tasks=tasks,
        workflow_type="hierarchical"  # Manager-coordinated workflow
    )
    
    logger.info(f"Enhanced crew created with {len(agents)} agents and {len(tasks)} comprehensive tasks")
    return crew

def execute_newsletter_generation(topic: str, collect_feedback: bool = True) -> Dict[str, Any]:
    """Execute the complete newsletter generation workflow with quality monitoring."""
    
    def _clean_placeholder_text(content: str, topic: str) -> str:
        """Clean up placeholder text and replace with actual content."""
        # Replace common placeholders with the actual topic
        replacements = {
            '[topic]': topic,
            '[Topic]': topic.title(),
            '[TOPIC]': topic.upper(),
            '[industry/subfield]': f"the {topic} industry",
            '[specific area]': topic,
            '[key concept]': topic,
            '[common issue]': f"challenges in {topic}",
            '[emerging concern]': f"emerging issues in {topic}",
            '[technology/innovation]': f"innovations in {topic}",
            '[desirable outcome]': f"success in {topic}",
            '[subtopics]': f"various aspects of {topic}",
            '[Company/Project]': f"leading companies in {topic}",
            '[Subtopic 1]': f"Key Developments in {topic}",
            '[Subtopic 2]': f"Challenges in {topic}",
            '[Subtopic 3]': f"Emerging Trends in {topic}",
            '[Subtopic 4]': f"Future Applications of {topic}",
            # Handle common AI/tech specific placeholders
            '[AI]': 'artificial intelligence',
            '[ML]': 'machine learning',
            '[technology]': topic,
            '[industry]': f"the {topic} industry",
            '[field]': topic,
            '[domain]': topic
        }
        
        cleaned_content = content
        for placeholder, replacement in replacements.items():
            cleaned_content = cleaned_content.replace(placeholder, replacement)
        
        # Clean up markdown formatting
        cleaned_content = _clean_markdown_formatting(cleaned_content)
        
        return cleaned_content
    
    def _clean_markdown_formatting(content: str) -> str:
        """Clean up markdown formatting issues."""
        # Replace improper header formatting
        content = content.replace('===============', '')
        content = content.replace('=====================================', '')
        content = content.replace('================================', '')
        content = content.replace('==========', '')
        
        # Fix header formatting
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Convert improper headers to proper markdown
            if line.strip().startswith('**') and line.strip().endswith('**'):
                # This is a header, ensure it's properly formatted
                cleaned_lines.append(line)
            elif line.strip().startswith('=') and len(line.strip()) > 3:
                # Skip lines that are just equals signs
                continue
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    start_time = time.time()
    logger.info(f"Starting newsletter generation for: {topic}")
    
    # Set maximum execution time (5 minutes total)
    MAX_EXECUTION_TIME = 300
    
    try:
        # Create and execute the workflow with timeout protection
        crew = create_enhanced_newsletter_workflow(topic)
        
        logger.info("Starting crew execution with safety limits...")
        result = crew.kickoff()
        
        execution_time = time.time() - start_time
        
        # Check if we exceeded reasonable time limits
        if execution_time > MAX_EXECUTION_TIME:
            logger.warning(f"Execution took {execution_time:.1f}s - this is longer than expected")
            result += f"\n\n⚠️ PERFORMANCE WARNING: Generation took {execution_time:.1f} seconds."
        
        # Display results with expanded limits for comprehensive content
        print("\n" + "="*80)
        print("🎉 NEWSLETTER GENERATION COMPLETED!")
        print("="*80)
        
        # Debug: Show raw result length
        print(f"🔍 DEBUG: Raw result length: {len(str(result)):,} characters")
        print(f"🔍 DEBUG: Raw result word count: {len(str(result).split()):,} words")
        
        # Show substantial preview (increased from 1000 characters)
        preview_length = 3000
        if len(result) > preview_length:
            print(f"📋 Content Preview (first {preview_length} characters):")
            print("-" * 60)
            print(result[:preview_length])
            print(f"\n... [Content continues for {len(result) - preview_length} more characters] ...")
            print(f"📊 Total content length: {len(result):,} characters")
        else:
            print("📋 Complete Newsletter Content:")
            print("-" * 60) 
            print(result)
        
        # Extract only the newsletter content (not workflow metadata)
        newsletter_content = ""
        if hasattr(result, 'strip'):
            raw_content = str(result)
            
            # Look for the actual newsletter content by finding key markers
            content_markers = [
                "**Improved Newsletter Content:**",
                "**Improved Content:**",
                "**Introduction:**",
                "**Benefits:**",
                "**Uses:**",
                "**Case Studies:**",
                "**Conclusion:**",
                "**Call-to-Action:**",
                "**Newsletter Content:**",
                "**Content:**",
                "**I. Introduction**",
                "**II. Key Developments**",
                "**III. Real-World Applications**",
                "**IV. Conclusion**"
            ]
            
            # Find the start of actual content
            content_start = -1
            for marker in content_markers:
                if marker in raw_content:
                    content_start = raw_content.find(marker)
                    break
            
            if content_start != -1:
                # Extract content from the marker onwards
                content_section = raw_content[content_start:]
                
                # Split into lines and clean up
                lines = content_section.split('\n')
                cleaned_lines = []
                in_content = True
                
                for line in lines:
                    # Stop at editorial metadata sections
                    if any(phrase in line for phrase in [
                        "**Editorial Report**",
                        "**Quality Assessment Summary**",
                        "**Focus Areas Addressed**",
                        "**Key Improvements Made**",
                        "**Quality Level:**",
                        "**Editorial Review Process:**",
                        "**Detailed Analysis:**",
                        "**Quality Scorecard:**",
                        "**Optimization Areas:**",
                        "**Improvement Areas**",
                        "**Formatting**",
                        "**Recommendations:**"
                    ]):
                        break
                    
                    # Skip workflow metadata sections
                    if any(phrase in line for phrase in [
                        "=== TASK", "=== WORKFLOW", "PlannerAgent:", "ResearchAgent:", 
                        "WriterAgent:", "EditorAgent:", "⚠️ PERFORMANCE WARNING",
                        "Generation Metadata", "Execution Time:", "Content Length:", 
                        "Status:", "avg ", "tasks", "WORKFLOW PERFORMANCE SUMMARY"
                    ]):
                        continue
                    
                    cleaned_lines.append(line)
                
                newsletter_content = '\n'.join(cleaned_lines).strip()
                newsletter_content = _clean_placeholder_text(newsletter_content, topic)
            else:
                # Fallback: try to extract content after removing obvious metadata
                lines = raw_content.split('\n')
                cleaned_lines = []
                skip_lines = False
                
                for line in lines:
                    # Skip workflow metadata sections
                    if any(phrase in line for phrase in [
                        "=== TASK", "=== WORKFLOW", "PlannerAgent:", "ResearchAgent:", 
                        "WriterAgent:", "EditorAgent:", "⚠️ PERFORMANCE WARNING",
                        "Generation Metadata", "Execution Time:", "Content Length:", 
                        "Status:", "avg ", "tasks", "WORKFLOW PERFORMANCE SUMMARY",
                        "**Editorial Report**", "**Quality Assessment Summary**"
                    ]):
                        skip_lines = True
                        continue
                    elif line.strip().startswith("---") and skip_lines:
                        skip_lines = False
                        continue
                    elif not skip_lines:
                        cleaned_lines.append(line)
                
                newsletter_content = '\n'.join(cleaned_lines).strip()
                newsletter_content = _clean_placeholder_text(newsletter_content, topic)
        else:
            newsletter_content = str(result)
            newsletter_content = _clean_placeholder_text(newsletter_content, topic)
        
        # Calculate word count and character count for the newsletter content only
        word_count = len(newsletter_content.split())
        char_count = len(newsletter_content)
        
        # Enhanced output saving with clean content
        output_file = f"output/newsletter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        os.makedirs("output", exist_ok=True)
        
        # Create clean output with just the newsletter content
        clean_output = f"""# Newsletter: {topic}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Word Count: {word_count:,} words
Character Count: {char_count:,} characters

---

{newsletter_content}
"""
        
        with open(output_file, 'w') as f:
            f.write(clean_output)
        
        logger.info(f"Newsletter saved to: {output_file}")
        
        # Prepare newsletter for Notion publishing
        notion_data = {
            "title": f"Newsletter: {topic}",
            "content": newsletter_content,
            "generated_at": datetime.now().isoformat(),
            "word_count": word_count,
            "char_count": char_count,
            "parent_page_id": "226b1384-d996-813f-bc9c-c540b498df90"  # Newsletter Archive parent
        }
        
        try:
            # Save the prepared data for the AI assistant to publish
            notion_prep_file = output_file.replace('.md', '_notion_prep.json')
            import json
            with open(notion_prep_file, 'w') as f:
                json.dump(notion_data, f, indent=2)
            
            logger.info(f"Newsletter prepared for Notion publishing: {notion_prep_file}")
            print(f"📄 Newsletter prepared for Notion publishing")
            print(f"📋 Notion data saved to: {notion_prep_file}")
            
        except Exception as e:
            logger.warning(f"Failed to prepare newsletter for Notion: {str(e)}")
            print(f"⚠️ Failed to prepare newsletter for Notion: {str(e)}")
        
        # Get workflow performance data with safety checks
        workflow_performance = {
            'total_execution_time': execution_time,
            'task_results': getattr(crew, 'task_results', {}),
            'agent_performance': getattr(crew, 'agent_performance', {}),
            'workflow_type': getattr(crew, 'workflow_type', 'unknown')
        }
        
        # Display results (truncated for safety)
        print("\n📈 PERFORMANCE SUMMARY:")
        task_results = workflow_performance.get('task_results', [])
        agent_performance = workflow_performance.get('agent_performance', {})
        
        # Handle task_results as a list (from EnhancedCrew)
        if isinstance(task_results, list):
            completed_tasks = sum(1 for task_info in task_results 
                                 if isinstance(task_info, dict) and 'error' not in str(task_info.get('result', '')).lower())
            total_tasks = len(task_results)
        else:
            # Fallback for dictionary format
            completed_tasks = sum(1 for task_info in task_results.values() 
                                 if isinstance(task_info, dict) and 'error' not in str(task_info.get('result', '')).lower())
            total_tasks = len(task_results)
        
        print(f"✅ Tasks Completed: {completed_tasks}/{total_tasks}")
        if total_tasks > 0:
            print(f"⚡ Average Task Time: {execution_time/total_tasks:.2f}s")
        
        for agent_name, perf_data in agent_performance.items():
            if isinstance(perf_data, dict) and perf_data.get('tasks_completed', 0) > 0:
                avg_time = perf_data['total_execution_time'] / perf_data['tasks_completed']
                print(f"🤖 {agent_name}: {perf_data['tasks_completed']} tasks, {avg_time:.2f}s avg")
        
        # Simplified feedback collection (optional)
        session_id = None
        if collect_feedback and execution_time < MAX_EXECUTION_TIME:  # Only collect if execution was reasonable
            try:
                feedback_system = FeedbackLearningSystem()
                session_id = feedback_system.collect_user_feedback(
                    topic=topic,
                    content=result,
                    interactive=True
                )
                
                # Generate learning insights with timeout
                insights = feedback_system.generate_learning_insights()
                if insights.get('improvement_recommendations'):
                    print("\n🎯 IMPROVEMENT RECOMMENDATIONS:")
                    for i, rec in enumerate(insights['improvement_recommendations'][:3], 1):
                        print(f"  {i}. {rec}")
            except Exception as e:
                logger.warning(f"Feedback collection failed: {str(e)}")
        
        return {
            'success': True,
            'content': newsletter_content,
            'output_file': output_file,
            'notion_data': notion_data,
            'performance': workflow_performance,
            'feedback_session': session_id,
            'word_count': word_count,
            'char_count': char_count
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Newsletter generation failed after {execution_time:.1f}s: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'execution_time': execution_time
        }

def execute_hierarchical_newsletter_generation(topic: str, audience: str = "technology professionals") -> Dict[str, Any]:
    """Run ManagerAgent-driven hierarchical workflow and return results/metadata."""
    logger.info(f"[Hierarchical] Starting newsletter generation for: {topic}")
    start_time = time.time()

    # Instantiate agents
    manager_agent = ManagerAgent()
    planner_agent = PlannerAgent()
    research_agent = ResearchAgent()
    writer_agent = WriterAgent()
    editor_agent = EditorAgent()

    agents = [planner_agent, research_agent, writer_agent, editor_agent]

    # Create workflow plan
    workflow_plan = manager_agent.create_hierarchical_workflow(topic)

    # Execute workflow
    workflow_result = manager_agent.execute_hierarchical_workflow(workflow_plan, agents)

    total_time = time.time() - start_time

    # Extract final content if available - prioritize newsletter content only
    stream_results = workflow_result.get("stream_results", {})
    final_content = ""
    if stream_results and isinstance(stream_results, dict):
        writing_res = stream_results.get("writing")
        editing_res = stream_results.get("editing")
        
        if writing_res:
            # Use WriterAgent output as the main newsletter content
            final_content = writing_res.get("result", "")
        elif editing_res:
            # If no writer content, extract newsletter content from editor output
            editor_output = editing_res.get("result", "")
            # Try to extract just the newsletter content from editor output
            # Editor output typically contains both the improved content and quality scorecard
            if "QUALITY SCORECARD" in editor_output:
                final_content = editor_output.split("QUALITY SCORECARD")[0].strip()
            else:
                final_content = editor_output

    success = workflow_result.get("status") == "completed"

    # Save output if present
    output_file = None
    notion_data = None
    if final_content:
        output_file = f"output/h_newsletter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        os.makedirs("output", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_content)
        
        # Prepare hierarchical newsletter for Notion publishing
        word_count = len(final_content.split())
        char_count = len(final_content)
        
        notion_data = {
            "title": f"Newsletter: {topic}",
            "content": final_content,
            "generated_at": datetime.now().isoformat(),
            "word_count": word_count,
            "char_count": char_count,
            "parent_page_id": "226b1384-d996-813f-bc9c-c540b498df90"  # Newsletter Archive parent
        }
        
        try:
            # Save the prepared data for the AI assistant to publish
            notion_prep_file = output_file.replace('.md', '_notion_prep.json')
            import json
            with open(notion_prep_file, 'w') as f:
                json.dump(notion_data, f, indent=2)
            
            logger.info(f"Hierarchical newsletter prepared for Notion publishing: {notion_prep_file}")
            print(f"📄 Hierarchical newsletter prepared for Notion publishing")
            print(f"📋 Notion data saved to: {notion_prep_file}")
            
        except Exception as e:
            logger.warning(f"Failed to prepare hierarchical newsletter for Notion: {str(e)}")
            print(f"⚠️ Failed to prepare hierarchical newsletter for Notion: {str(e)}")

    return {
        "success": success,
        "workflow_result": workflow_result,
        "content": final_content,
        "output_file": output_file,
        "notion_data": notion_data,
        "execution_time": total_time,
        "word_count": word_count if final_content else 0,
        "char_count": char_count if final_content else 0
    }

def run_quality_analysis_demo():
    """Demonstrate the quality analysis capabilities."""
    
    print("\n" + "="*80)
    print("🔍 QUALITY ANALYSIS DEMONSTRATION")
    print("="*80)
    
    # Create an editor agent for quality analysis
    editor = EditorAgent()
    
    # Sample content for analysis
    sample_content = """
    # AI Trends in 2024
    
    Artificial intelligence is changing rapidly. There are many new developments.
    Companies are using AI more. Some people are worried about jobs.
    
    ## Key Points
    - AI is growing
    - More automation
    - Need to adapt
    
    This is important for everyone.
    """
    
    print("📄 Sample Content:")
    print("-" * 40)
    print(sample_content)
    print("-" * 40)
    
    # Extract quality metrics
    metrics = editor.extract_quality_metrics(sample_content)
    print("\n📊 Content Metrics:")
    for metric, value in metrics.items():
        if metric != 'readability_indicators':
            print(f"  {metric.replace('_', ' ').title()}: {value}")
        else:
            print(f"  Readability Indicators: {value}")
    
    # Generate sample quality analysis
    sample_quality_scores = {
        'clarity': 6.5,
        'accuracy': 5.0,
        'engagement': 4.5,
        'completeness': 5.5
    }
    
    quality_analysis = editor.calculate_quality_score({'scores': sample_quality_scores})
    print(f"\n🎯 Quality Analysis:")
    print(f"  Overall Score: {quality_analysis['overall_score']}/10 ({quality_analysis['grade']})")
    
    for dimension, details in quality_analysis['dimension_scores'].items():
        print(f"  {dimension.title()}: {details['raw_score']}/10 (weight: {details['weight']})")
    
    # Generate recommendations
    recommendations = editor.generate_improvement_recommendations(quality_analysis, metrics)
    print(f"\n💡 Improvement Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

def run_feedback_learning_demo():
    """Demonstrate the feedback learning system."""
    
    print("\n" + "="*80)
    print("📚 FEEDBACK LEARNING SYSTEM DEMONSTRATION")
    print("="*80)
    
    feedback_system = FeedbackLearningSystem()
    
    # Simulate some feedback entries for demonstration
    sample_topics = [
        "AI Ethics in Healthcare",
        "Quantum Computing Advances", 
        "Sustainable Tech Solutions"
    ]
    
    sample_content = "This is a sample newsletter content for demonstration purposes. It covers various aspects of the topic with detailed analysis and insights."
    
    print("🔄 Simulating feedback collection...")
    
    for i, topic in enumerate(sample_topics):
        # Simulate different quality scores and ratings
        if i == 0:  # First one gets good scores
            rating = "approved"
            scores = {"clarity": 8.5, "accuracy": 8.0, "engagement": 7.5, "completeness": 8.0}
        elif i == 1:  # Second one needs revision
            rating = "needs_revision"
            scores = {"clarity": 6.0, "accuracy": 7.0, "engagement": 5.5, "completeness": 6.5}
        else:  # Third one gets rejected
            rating = "rejected"
            scores = {"clarity": 4.5, "accuracy": 5.0, "engagement": 4.0, "completeness": 5.5}
        
        session_id = feedback_system.collect_user_feedback(
            topic=topic,
            content=sample_content,
            interactive=False  # Use automated feedback for demo
        )
        
        # Override with our demo data
        feedback_system.logger.log_feedback(
            topic=topic,
            content=sample_content,
            user_rating=rating,
            quality_scores=scores,
            specific_feedback=f"Demo feedback for {topic}",
            suggestions=[f"Improve {topic} coverage"]
        )
    
    print("✅ Demo feedback data created")
    
    # Generate learning insights
    insights = feedback_system.generate_learning_insights()
    
    print(f"\n📈 Learning Insights:")
    print(f"  Total Feedback Entries: {insights['summary']['total_feedback_entries']}")
    print(f"  Learning Status: {insights['summary']['learning_status']}")
    
    if insights['improvement_recommendations']:
        print(f"\n🎯 Top Improvement Recommendations:")
        for i, rec in enumerate(insights['improvement_recommendations'][:3], 1):
            print(f"  {i}. {rec}")
    
    # Save learning report
    report_path = feedback_system.save_learning_report()
    print(f"\n📄 Learning report saved to: {report_path}")

def run_simple_demo():
    """Run a simple, quick demo without complex workflow orchestration."""
    
    print("\n" + "="*80)
    print("🚀 SIMPLE NEWSLETTER SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Test individual agents quickly
    print("Testing individual agents...")
    
    # Test ResearchAgent
    print("\n1. 🔍 Testing ResearchAgent...")
    research_agent = ResearchAgent()
    research_task = Task(
        description="Find basic information about AI trends",
        agent=research_agent,
        context="Keep this brief and focused"
    )
    
    try:
        research_result = research_task.execute()
        print(f"✅ ResearchAgent completed successfully")
        print(f"   Result length: {len(research_result)} characters")
    except Exception as e:
        print(f"❌ ResearchAgent failed: {str(e)}")
    
    # Test PlannerAgent
    print("\n2. 📋 Testing PlannerAgent...")
    planner_agent = PlannerAgent()
    planning_task = Task(
        description="Create a simple content plan for AI newsletter",
        agent=planner_agent,
        context="Keep this brief and structured"
    )
    
    try:
        planning_result = planning_task.execute()
        print(f"✅ PlannerAgent completed successfully")
        print(f"   Result length: {len(planning_result)} characters")
    except Exception as e:
        print(f"❌ PlannerAgent failed: {str(e)}")
    
    # Test WriterAgent
    print("\n3. ✍️ Testing WriterAgent...")
    writer_agent = WriterAgent()
    writing_task = Task(
        description="Write a brief AI newsletter introduction",
        agent=writer_agent,
        context="Write a short, engaging introduction paragraph"
    )
    
    try:
        writing_result = writing_task.execute()
        print(f"✅ WriterAgent completed successfully")
        print(f"   Result length: {len(writing_result)} characters")
    except Exception as e:
        print(f"❌ WriterAgent failed: {str(e)}")
    
    # Test EditorAgent quality scoring
    print("\n4. 📝 Testing EditorAgent quality scoring...")
    editor_agent = EditorAgent()
    
    sample_content = "AI is transforming industries rapidly. Companies are adopting machine learning for automation and decision-making."
    
    try:
        metrics = editor_agent.extract_quality_metrics(sample_content)
        quality_scores = {'clarity': 7.5, 'accuracy': 8.0, 'engagement': 6.5, 'completeness': 7.0}
        quality_analysis = editor_agent.calculate_quality_score({'scores': quality_scores})
        
        print(f"✅ EditorAgent quality scoring completed")
        print(f"   Overall Score: {quality_analysis['overall_score']:.1f}/10")
        print(f"   Grade: {quality_analysis['grade']}")
    except Exception as e:
        print(f"❌ EditorAgent failed: {str(e)}")
    
    # Test ManagerAgent delegation
    print("\n5. 🎯 Testing ManagerAgent delegation...")
    manager_agent = ManagerAgent()
    
    try:
        available_agents = [research_agent, planner_agent, writer_agent, editor_agent]
        delegation = manager_agent.delegate_task("research", "Find AI trends", available_agents)
        
        print(f"✅ ManagerAgent delegation completed")
        print(f"   Delegated to: {delegation['agent'].name}")
        print(f"   Success: {delegation['success']}")
    except Exception as e:
        print(f"❌ ManagerAgent failed: {str(e)}")
    
    print("\n" + "="*80)
    print("🎉 SIMPLE DEMO COMPLETED - All individual components tested!")
    print("="*80)

def main():
    """Main entry point with enhanced capabilities."""
    
    if len(sys.argv) < 2:
        print("Enhanced Newsletter Generation System")
        print("Usage:")
        print("  python src/main.py <topic>                    - Generate newsletter")
        print("  python src/main.py --quality-demo             - Run quality analysis demo")
        print("  python src/main.py --feedback-demo            - Run feedback learning demo")
        print("  python src/main.py --full-demo                - Run all demonstrations")
        print("  python src/main.py --simple-demo              - Run simple demo")
        print("  python src/main.py --hierarchical <topic>      - Run hierarchical newsletter generation")
        return
    
    # Handle demo modes
    if sys.argv[1] == "--quality-demo":
        run_quality_analysis_demo()
        return
    elif sys.argv[1] == "--feedback-demo":
        run_feedback_learning_demo()
        return
    elif sys.argv[1] == "--full-demo":
        print("🚀 Running Full System Demonstration")
        print("="*60)
        
        # Run simple component tests first
        run_simple_demo()
        
        # Run individual capability demos
        run_quality_analysis_demo()
        run_feedback_learning_demo()
        
        print("\n" + "="*80)
        print("🎉 FULL DEMO COMPLETED - All system components demonstrated!")
        print("="*80)
        print("✅ Individual agent tests")
        print("✅ Quality analysis system") 
        print("✅ Feedback learning system")
        print("\n💡 To test the full newsletter workflow, run:")
        print("   python src/main.py \"Your Newsletter Topic Here\"")
        return
    elif sys.argv[1] == "--simple-demo":
        run_simple_demo()
        return
    elif sys.argv[1] == HIERARCHICAL_FLAG:
        # Hierarchical generation mode
        topic = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "AI and Future"
        print("🚀 Manager Hierarchical Newsletter Generation")
        print("="*60)
        print(f"📝 Topic: {topic}")
        result = execute_hierarchical_newsletter_generation(topic)
        if result["success"]:
            print("✅ Hierarchical workflow completed successfully!")
            if result["output_file"]:
                print(f"📄 Output saved to: {result['output_file']}")
        else:
            print("❌ Hierarchical workflow failed. See logs for details.")
        return
    
    # Regular newsletter generation
    topic = " ".join(sys.argv[1:])
    
    print("🚀 Enhanced AI Newsletter Generation System")
    print("=" * 60)
    print(f"📝 Topic: {topic}")
    print(f"🤖 Agents: Manager, Planner, Researcher, Writer, Editor")
    print(f"🔧 Features: Quality scoring, Feedback learning, Performance monitoring")
    print("=" * 60)
    
    # Execute the workflow
    result = execute_newsletter_generation(topic, collect_feedback=True)
    
    if result['success']:
        print(f"\n✅ Newsletter generation completed successfully!")
        print(f"📄 Output saved to: {result['output_file']}")
        if result.get('feedback_session'):
            print(f"💬 Feedback logged: {result['feedback_session']}")
    else:
        print(f"\n❌ Newsletter generation failed: {result['error']}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
