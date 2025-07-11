"""
Enhanced Tool Usage Tracking System Demo

This demo script showcases the comprehensive tool usage tracking system
including real-time monitoring, analytics, feedback correlation, and insights.

Run this script to see the enhanced tracking capabilities in action.
"""

import os
import sys
import time
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, 'src')

from core.tool_usage_tracker import (
    ToolUsageTracker, ToolExecutionStatus, track_tool_call, get_tool_tracker
)
from core.feedback_system import FeedbackLearningSystem
from core.tool_usage_analytics import ToolUsageAnalyticsDashboard
from agents.agents import SimpleAgent
from tools.tools import AVAILABLE_TOOLS

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"🔧 {title}")
    print("="*80)

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n📊 {title}")
    print("-"*60)

def print_json(data: dict, title: str = ""):
    """Pretty print JSON data."""
    if title:
        print(f"\n{title}:")
    print(json.dumps(data, indent=2))

def simulate_tool_usage(tracker: ToolUsageTracker):
    """Simulate realistic tool usage patterns."""
    print_section("Simulating Tool Usage Patterns")
    
    # Define realistic scenarios
    scenarios = [
        {
            "tool_name": "search_web",
            "agent_name": "researcher",
            "session_id": "newsletter_session_1",
            "workflow_id": "newsletter_generation",
            "execution_time": 2.3,
            "status": ToolExecutionStatus.SUCCESS,
            "input_data": {"query": "latest AI developments", "max_results": 10},
            "output_data": {"articles_found": 15, "relevant_articles": 8}
        },
        {
            "tool_name": "search_knowledge_base",
            "agent_name": "researcher", 
            "session_id": "newsletter_session_1",
            "workflow_id": "newsletter_generation",
            "execution_time": 1.8,
            "status": ToolExecutionStatus.SUCCESS,
            "input_data": {"query": "machine learning trends", "similarity_threshold": 0.8},
            "output_data": {"documents_found": 12, "relevance_score": 0.92}
        },
        {
            "tool_name": "generate_content",
            "agent_name": "writer",
            "session_id": "newsletter_session_1", 
            "workflow_id": "newsletter_generation",
            "execution_time": 5.7,
            "status": ToolExecutionStatus.SUCCESS,
            "input_data": {"template": "newsletter", "content_length": "medium"},
            "output_data": {"content_generated": True, "word_count": 1200}
        },
        {
            "tool_name": "quality_check",
            "agent_name": "editor",
            "session_id": "newsletter_session_1",
            "workflow_id": "newsletter_generation", 
            "execution_time": 1.2,
            "status": ToolExecutionStatus.SUCCESS,
            "input_data": {"content": "newsletter_content", "check_grammar": True},
            "output_data": {"quality_score": 8.7, "grammar_errors": 2}
        },
        {
            "tool_name": "search_web",
            "agent_name": "researcher",
            "session_id": "newsletter_session_2",
            "workflow_id": "newsletter_generation",
            "execution_time": 15.2,  # Slow execution
            "status": ToolExecutionStatus.TIMEOUT,
            "input_data": {"query": "breaking tech news", "timeout": 10},
            "error_details": {
                "error_type": "TimeoutError",
                "error_message": "Request timed out after 10 seconds",
                "retry_attempts": 3
            }
        },
        {
            "tool_name": "search_knowledge_base",
            "agent_name": "researcher",
            "session_id": "newsletter_session_2", 
            "workflow_id": "newsletter_generation",
            "execution_time": 1.9,
            "status": ToolExecutionStatus.SUCCESS,
            "input_data": {"query": "backup search for tech news"},
            "output_data": {"documents_found": 8, "used_as_fallback": True}
        }
    ]
    
    # Log all scenarios
    entries = []
    for i, scenario in enumerate(scenarios):
        print(f"  📝 Logging {scenario['tool_name']} usage by {scenario['agent_name']}...")
        time.sleep(0.1)  # Small delay for realism
        
        entry = tracker.log_tool_usage(**scenario)
        entries.append(entry)
        
        # Show immediate feedback
        status_icon = "✅" if scenario["status"] == ToolExecutionStatus.SUCCESS else "❌" if scenario["status"] == ToolExecutionStatus.FAILURE else "⏱️"
        print(f"    {status_icon} {scenario['tool_name']} - {scenario['execution_time']:.1f}s - {scenario['status'].value}")
    
    print(f"\n✅ Logged {len(entries)} tool usage entries")
    return entries

def demonstrate_decorator_tracking():
    """Demonstrate the tool call tracking decorator."""
    print_section("Demonstrating Decorator-Based Tracking")
    
    @track_tool_call(tool_name="demo_analysis_tool", agent_name="analyst")
    def analyze_data(data_size: int, complexity: str = "medium") -> dict:
        """Simulate data analysis with variable execution time."""
        # Simulate work based on complexity
        complexity_times = {"simple": 0.5, "medium": 1.2, "complex": 2.8}
        time.sleep(complexity_times.get(complexity, 1.0))
        
        return {
            "analysis_complete": True,
            "data_points_processed": data_size,
            "complexity_level": complexity,
            "insights_found": data_size // 100
        }
    
    @track_tool_call(tool_name="demo_validation_tool", agent_name="validator")
    def validate_results(results: dict) -> bool:
        """Simulate validation with possible failure."""
        time.sleep(0.3)
        # Simulate occasional validation failure
        if results.get("data_points_processed", 0) < 50:
            raise ValueError("Insufficient data points for validation")
        return True
    
    # Demonstrate successful tracking
    print("  🔄 Running successful analysis...")
    result1 = analyze_data(data_size=1000, complexity="medium")
    print(f"  ✅ Analysis completed: {result1['insights_found']} insights found")
    
    validation1 = validate_results(result1)
    print(f"  ✅ Validation passed: {validation1}")
    
    # Demonstrate error tracking
    print("\n  🔄 Running analysis with validation error...")
    try:
        result2 = analyze_data(data_size=25, complexity="simple")  # Will cause validation error
        validate_results(result2)
    except ValueError as e:
        print(f"  ❌ Validation failed: {e}")
    
    print("\n  📊 Recent decorator-tracked calls:")
    recent_entries = get_tool_tracker().get_tool_usage_history(hours_back=0.1)
    decorator_entries = [e for e in recent_entries if e.tool_name.startswith("demo_")]
    
    for entry in decorator_entries:
        status_icon = "✅" if entry.status == ToolExecutionStatus.SUCCESS else "❌"
        print(f"    {status_icon} {entry.tool_name} - {entry.execution_time:.2f}s - {entry.status.value}")

def demonstrate_analytics_dashboard(tracker: ToolUsageTracker, feedback_system: FeedbackLearningSystem):
    """Demonstrate the analytics dashboard capabilities."""
    print_section("Analytics Dashboard Demonstration")
    
    # Create dashboard
    dashboard = ToolUsageAnalyticsDashboard(
        tool_tracker=tracker,
        feedback_system=feedback_system
    )
    
    # Generate comprehensive analytics
    print("  🔄 Generating dashboard analytics...")
    dashboard_data = dashboard.refresh_dashboard_data(hours_back=1)
    
    # Show key metrics
    summary = dashboard_data["summary"]
    print_json(summary["key_metrics"], "📈 Key System Metrics")
    
    # Show system health
    print(f"\n🏥 System Health: {summary['health_status'].upper()}")
    
    if summary.get("alerts"):
        print("\n🚨 System Alerts:")
        for alert in summary["alerts"]:
            print(f"  ⚠️  {alert}")
    
    # Show top performers
    if summary.get("top_performers"):
        print("\n🏆 Top Performers:")
        for performer in summary["top_performers"]:
            print(f"  🌟 {performer}")
    
    # Real-time metrics
    print("\n⏱️  Real-Time Metrics:")
    real_time = dashboard.get_real_time_metrics()
    print_json(real_time["last_hour"], "📊 Last Hour Activity")
    
    # Tool performance report
    print_section("Individual Tool Performance")
    tool_report = dashboard.get_tool_performance_report("search_web", hours_back=1)
    if "error" not in tool_report:
        print_json(tool_report["usage_statistics"], "🔍 Search Web Performance")
        print("\n💡 Recommendations:")
        for rec in tool_report["recommendations"]:
            print(f"  • {rec}")
    
    # Agent comparison
    print_section("Agent Performance Comparison")
    agent_comparison = dashboard.get_agent_performance_comparison(hours_back=1)
    if agent_comparison["agent_comparison"]:
        print("🤖 Agent Rankings by Efficiency:")
        for i, (agent, score) in enumerate(agent_comparison["rankings"]["by_efficiency"][:3], 1):
            print(f"  {i}. {agent}: {score:.1%} efficiency")
    
    return dashboard

def demonstrate_feedback_correlation(feedback_system: FeedbackLearningSystem):
    """Demonstrate feedback and tool usage correlation."""
    print_section("Feedback-Tool Usage Correlation Analysis")
    
    # Add sample feedback data
    print("  📝 Adding sample feedback data...")
    
    # High-quality feedback for session 1
    session1_feedback = feedback_system.collect_user_feedback(
        topic="AI Newsletter - High Quality",
        content="Excellent newsletter with comprehensive AI coverage, well-researched content, and engaging writing style.",
        interactive=False  # Use defaults for demo
    )
    
    # Update metadata to link with tool usage session
    feedback_history = feedback_system.logger.get_feedback_history()
    if feedback_history:
        feedback_history[-1].metadata["session_id"] = "newsletter_session_1"
        feedback_history[-1].quality_scores = {
            "clarity": 9.2,
            "accuracy": 9.5,
            "engagement": 8.8,
            "completeness": 9.0
        }
        feedback_history[-1].user_rating = "approved"
        
        # Save updated feedback
        feedback_file = feedback_system.logger.feedback_file
        feedback_data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "version": "1.0"
            },
            "feedback_entries": [
                {
                    "timestamp": entry.timestamp,
                    "newsletter_topic": entry.newsletter_topic,
                    "content_preview": entry.content_preview,
                    "user_rating": entry.user_rating,
                    "quality_scores": entry.quality_scores,
                    "specific_feedback": entry.specific_feedback,
                    "agent_performance": entry.agent_performance,
                    "suggestions": entry.suggestions,
                    "metadata": entry.metadata
                }
                for entry in feedback_history
            ]
        }
        
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
    
    # Lower quality feedback for session 2 (with timeout issues)
    session2_feedback = feedback_system.collect_user_feedback(
        topic="AI Newsletter - Needs Improvement", 
        content="Newsletter had some good content but seemed incomplete due to technical issues during research.",
        interactive=False
    )
    
    # Update for session 2
    feedback_history = feedback_system.logger.get_feedback_history()
    if len(feedback_history) >= 2:
        feedback_history[-1].metadata["session_id"] = "newsletter_session_2"
        feedback_history[-1].quality_scores = {
            "clarity": 6.2,
            "accuracy": 7.0,
            "engagement": 5.8,
            "completeness": 4.5  # Lower due to timeout issues
        }
        feedback_history[-1].user_rating = "needs_revision"
        
        # Update the file again
        feedback_data["feedback_entries"] = [
            {
                "timestamp": entry.timestamp,
                "newsletter_topic": entry.newsletter_topic,
                "content_preview": entry.content_preview,
                "user_rating": entry.user_rating,
                "quality_scores": entry.quality_scores,
                "specific_feedback": entry.specific_feedback,
                "agent_performance": entry.agent_performance,
                "suggestions": entry.suggestions,
                "metadata": entry.metadata
            }
            for entry in feedback_history
        ]
        
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
    
    # Analyze correlations
    print("  🔍 Analyzing tool usage and quality correlations...")
    correlations = feedback_system.analyze_tool_usage_correlations()
    
    if correlations and "tool_correlations" in correlations:
        print_json(correlations["analysis_summary"], "🔗 Correlation Analysis Summary")
        
        if correlations["analysis_summary"].get("recommendations"):
            print("\n💡 System Recommendations:")
            for rec in correlations["analysis_summary"]["recommendations"]:
                print(f"  • {rec}")
    else:
        print("  ℹ️  Insufficient data for correlation analysis (need more feedback entries)")

def demonstrate_executive_reporting(dashboard: ToolUsageAnalyticsDashboard):
    """Demonstrate executive-level reporting."""
    print_section("Executive Reporting and Insights")
    
    # Generate executive summary
    print("  📊 Generating executive summary...")
    summary = dashboard.generate_executive_summary(hours_back=1)
    
    print_json(summary["key_metrics"], "📈 Executive Summary - Key Metrics")
    
    print(f"\n📅 Report Period: {summary['report_period']}")
    print(f"⏱️  Generated: {summary['generated_at']}")
    print(f"📈 Performance Trend: {summary['performance_trend'].upper()}")
    
    print("\n🔍 Top Insights:")
    for i, insight in enumerate(summary["top_insights"], 1):
        print(f"  {i}. {insight}")
    
    print("\n🎯 Executive Recommendations:")
    for i, rec in enumerate(summary["recommendations"], 1):
        print(f"  {i}. {rec}")
    
    # Export data
    print_section("Data Export Capabilities")
    
    print("  💾 Exporting analytics data...")
    output_dir = "logs/demo_exports"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Export JSON
    json_file = dashboard.export_analytics_data(
        format_type="json",
        output_dir=output_dir,
        hours_back=1
    )
    print(f"  ✅ JSON exported to: {json_file}")
    
    # Show file size
    file_size = os.path.getsize(json_file) / 1024  # KB
    print(f"     📦 File size: {file_size:.1f} KB")

def run_performance_demo(tracker: ToolUsageTracker):
    """Demonstrate performance capabilities."""
    print_section("Performance and Scalability Demo")
    
    print("  ⚡ Testing high-volume logging performance...")
    start_time = time.time()
    
    # Log many entries quickly
    for i in range(100):
        tracker.log_tool_usage(
            tool_name=f"perf_tool_{i % 5}",
            agent_name=f"perf_agent_{i % 3}",
            status=ToolExecutionStatus.SUCCESS,
            execution_time=float(i % 10) / 10,
            session_id=f"perf_session_{i // 20}"
        )
    
    logging_time = time.time() - start_time
    print(f"  ✅ Logged 100 entries in {logging_time:.3f} seconds")
    print(f"  📊 Performance: {100/logging_time:.0f} entries/second")
    
    # Test analytics performance
    print("\n  📈 Testing analytics generation performance...")
    start_time = time.time()
    analytics = tracker.generate_usage_analytics(hours_back=1)
    analytics_time = time.time() - start_time
    
    total_entries = analytics["total_tool_calls"]
    print(f"  ✅ Generated analytics for {total_entries} entries in {analytics_time:.3f} seconds")
    print(f"  📊 Performance: {total_entries/analytics_time:.0f} entries processed/second")

def main():
    """Run the complete tool usage tracking demonstration."""
    print_header("Enhanced Tool Usage Tracking System - Live Demo")
    
    print("""
🚀 This demonstration showcases the comprehensive tool usage tracking system
   including real-time monitoring, analytics, correlation analysis, and insights.
   
📊 Features demonstrated:
   • Comprehensive tool usage logging
   • Decorator-based automatic tracking  
   • Real-time analytics and monitoring
   • Agent performance comparison
   • Feedback-quality correlation analysis
   • Executive reporting and insights
   • High-performance data processing
   • Export capabilities
   
⏱️  Estimated runtime: 30-60 seconds
""")
    
    input("Press Enter to start the demonstration...")
    
    # Initialize system components
    print_section("System Initialization")
    
    # Ensure log directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("logs/demo_exports").mkdir(exist_ok=True)
    
    # Initialize tracker
    tracker = ToolUsageTracker(log_file="logs/demo_tool_usage.jsonl")
    print("  ✅ Tool Usage Tracker initialized")
    
    # Initialize feedback system
    feedback_system = FeedbackLearningSystem(feedback_file="logs/demo_feedback.json")
    feedback_system.set_tool_tracker(tracker)
    print("  ✅ Feedback Learning System initialized")
    
    print("  🔗 Systems connected and ready")
    
    # Run demonstrations
    try:
        # 1. Simulate realistic tool usage
        simulate_tool_usage(tracker)
        
        # 2. Demonstrate decorator tracking
        demonstrate_decorator_tracking()
        
        # 3. Show analytics dashboard
        dashboard = demonstrate_analytics_dashboard(tracker, feedback_system)
        
        # 4. Demonstrate feedback correlation
        demonstrate_feedback_correlation(feedback_system)
        
        # 5. Executive reporting
        demonstrate_executive_reporting(dashboard)
        
        # 6. Performance demonstration
        run_performance_demo(tracker)
        
        # Final summary
        print_header("Demo Summary and Next Steps")
        
        final_analytics = tracker.generate_usage_analytics(hours_back=1)
        total_tracked = final_analytics["total_tool_calls"]
        success_rate = final_analytics["overall_success_rate"]
        
        print(f"""
✅ Demo completed successfully!

📊 Session Statistics:
   • Total tool calls tracked: {total_tracked}
   • Overall success rate: {success_rate:.1%}
   • Unique tools used: {len(final_analytics["tool_breakdown"])}
   • Unique agents active: {len(final_analytics["agent_breakdown"])}

🎯 Key Capabilities Demonstrated:
   ✅ Comprehensive tool usage logging with rich metadata
   ✅ Automatic tracking through decorators
   ✅ Real-time analytics and monitoring
   ✅ Performance bottleneck identification
   ✅ Agent efficiency comparison
   ✅ Feedback-quality correlation analysis
   ✅ Executive-level reporting and insights
   ✅ High-performance data processing
   ✅ Flexible data export capabilities

📁 Generated Files:
   • logs/demo_tool_usage.jsonl - Tool usage log
   • logs/demo_feedback.json - Feedback data
   • logs/demo_exports/ - Exported analytics

🚀 The enhanced tool usage tracking system is ready for production use!
   Integration points have been added to agents, MCP orchestrator, and
   feedback systems for comprehensive monitoring and optimization.
""")
        
    except Exception as e:
        print(f"\n❌ Demo encountered an error: {e}")
        print("🔧 This is likely due to missing dependencies or configuration issues.")
        print("📝 Please ensure all required modules are installed and configured.")
        return False
    
    return True

if __name__ == "__main__":
    # Set environment for demo
    os.environ["DEMO_MODE"] = "true"
    
    # Run the demonstration
    success = main()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("🔧 The enhanced tool usage tracking system is ready for integration.")
    else:
        print("\n⚠️  Demo encountered issues. Please check configuration and dependencies.")
    
    sys.exit(0 if success else 1) 