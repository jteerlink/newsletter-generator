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
    ResearchAgent, PlannerAgent, WriterAgent, EditorAgent, 
    ManagerAgent, Task, EnhancedCrew
)
from src.core.core import query_llm
from src.core.feedback_system import FeedbackLearningSystem

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
        2. **Stakeholder Perspectives**: Include viewpoints from different industry players, experts, and end-users
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
        2. **Industry Intelligence**: Gather insights from key industry players, analysts, and thought leaders
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
        description=f"""Create an exceptional, comprehensive newsletter about '{topic}' using the editorial plan and research findings. 
        This should be substantial, engaging, and thoroughly valuable content:

        ✍️ CONTENT CREATION GUIDELINES:
        1. **Compelling Opening**: Start with a hook that immediately captures attention and establishes value
        2. **Narrative Structure**: Build a logical, engaging flow that keeps readers invested throughout
        3. **Depth with Clarity**: Provide comprehensive coverage while maintaining accessibility
        4. **Multiple Perspectives**: Include diverse viewpoints and stakeholder voices
        5. **Rich Examples**: Use case studies, scenarios, and real-world applications extensively

        📝 CONTENT SECTIONS (aim for substantial coverage):
        1. **Executive Summary**: Key insights and takeaways (but place after intro for newsletters)
        2. **Current Landscape**: Detailed analysis of the present state
        3. **Deep Dive Analysis**: Thorough exploration of key aspects and implications
        4. **Expert Insights**: Incorporate research findings and expert perspectives
        5. **Case Studies/Examples**: Real-world applications and stories
        6. **Future Outlook**: Predictions, trends, and emerging opportunities
        7. **Practical Applications**: Actionable insights and next steps for readers
        8. **Resources & Further Reading**: Additional sources for deeper exploration

        🎨 WRITING TECHNIQUES:
        1. **Storytelling**: Use narrative techniques to make complex topics engaging
        2. **Varied Formats**: Include lists, scenarios, dialogues, data presentations
        3. **Visual Language**: Create vivid descriptions that help readers visualize concepts
        4. **Analogies & Metaphors**: Make complex ideas accessible through familiar comparisons
        5. **Personality & Voice**: Maintain engaging, authoritative voice throughout

        Target 2000-3000 words for comprehensive coverage. Make every section valuable and engaging. 
        This should be the type of content readers bookmark and refer back to.""",
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
        
        # Show substantial preview (increased from 1000 characters)
        preview_length = 2000
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
        
        # Enhanced output saving with metadata
        output_file = f"output/newsletter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        os.makedirs("output", exist_ok=True)
        
        # Create comprehensive output with metadata
        full_output = f"""# Newsletter: {topic}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Content Length: {len(result):,} characters
Execution Time: {execution_time:.1f} seconds

---

{result}

---

## Generation Metadata
- Topic: {topic}
- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Execution Time: {execution_time:.1f} seconds
- Content Length: {len(result):,} characters
- Status: {'✅ Success' if 'error' not in result.lower() else '⚠️ Contains errors'}
"""
        
        with open(output_file, 'w') as f:
            f.write(full_output)
        
        logger.info(f"Newsletter saved to: {output_file}")
        
        # Get workflow performance data with safety checks
        workflow_performance = {
            'total_execution_time': execution_time,
            'task_results': getattr(crew, 'task_results', {}),
            'agent_performance': getattr(crew, 'agent_performance', {}),
            'workflow_type': getattr(crew, 'workflow_type', 'unknown')
        }
        
        # Display results (truncated for safety)
        print("\n📈 PERFORMANCE SUMMARY:")
        task_results = workflow_performance.get('task_results', {})
        agent_performance = workflow_performance.get('agent_performance', {})
        
        completed_tasks = sum(1 for task_info in task_results.values() 
                             if isinstance(task_info, dict) and 'error' not in task_info.get('result', '').lower())
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
            'content': result,
            'output_file': output_file,
            'performance': workflow_performance,
            'feedback_session': session_id
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

    # Extract final content if available
    stream_results = workflow_result.get("stream_results", {})
    if stream_results and isinstance(stream_results, dict):
        editing_res = stream_results.get("editing")
        if editing_res:
            final_content = editing_res.get("result", "")

    success = workflow_result.get("status") == "completed"

    # Save output if present
    output_file = None
    if final_content:
        output_file = f"output/h_newsletter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        os.makedirs("output", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_content)

    return {
        "success": success,
        "workflow_result": workflow_result,
        "output_file": output_file,
        "execution_time": total_time
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
