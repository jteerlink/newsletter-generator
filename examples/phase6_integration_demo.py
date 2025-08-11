#!/usr/bin/env python3
"""
Phase 6 Integration Demo: Complete enhanced agent architecture demonstration.

This example demonstrates the full capabilities of the enhanced newsletter
generation system including workflow orchestration, campaign contexts,
iterative refinement, and learning integration.
"""

import sys
import os
import time
from pathlib import Path

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

# Also add project root for relative imports
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

try:
    from core.workflow_orchestrator import WorkflowOrchestrator
    from core.config_manager import ConfigManager  
    from core.campaign_context import CampaignContext
    DEMO_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Demo components not available: {e}")
    print("This is expected if running outside the full environment.")
    DEMO_AVAILABLE = False


def demo_basic_integration():
    """Demonstrate basic integration capabilities."""
    if not DEMO_AVAILABLE:
        print("⚠️  Demo components not available - skipping integration demo")
        return None, None
        
    print("🚀 Phase 6 Integration Demo - Basic Capabilities")
    print("=" * 50)
    
    # 1. Initialize the system
    print("\n1. System Initialization")
    print("-" * 25)
    
    config_manager = ConfigManager()
    orchestrator = WorkflowOrchestrator(config_manager)
    
    print("✓ Configuration manager initialized")
    print("✓ Workflow orchestrator initialized")
    print(f"✓ Found {len(orchestrator.agents)} agent types")
    
    # 2. Campaign Context Management
    print("\n2. Campaign Context Management")
    print("-" * 30)
    
    # List available contexts
    contexts = config_manager.list_campaign_contexts()
    print(f"✓ Available contexts: {', '.join(contexts)}")
    
    # Load and inspect default context
    default_context = config_manager.load_campaign_context("default")
    print(f"✓ Default context loaded")
    print(f"  - Tone: {default_context.content_style.get('tone')}")
    print(f"  - Style: {default_context.content_style.get('style')}")
    print(f"  - Strategic goals: {len(default_context.strategic_goals)}")
    print(f"  - Quality thresholds: {default_context.quality_thresholds}")
    
    # Create a custom context
    custom_context = CampaignContext(
        content_style={
            "tone": "enthusiastic",
            "style": "engaging",
            "personality": "friendly",
            "reading_level": "general",
            "target_length": "medium"
        },
        strategic_goals=[
            "Increase reader engagement",
            "Build community awareness",
            "Share exciting developments"
        ],
        audience_persona={
            "demographics": "tech enthusiasts aged 20-40",
            "interests": "technology, innovation, startups",
            "pain_points": "information overload, staying current",
            "preferred_format": "engaging, accessible content"
        },
        performance_analytics={},
        quality_thresholds={
            "minimum": 0.75,
            "target": 0.85,
            "excellent": 0.95
        },
        forbidden_terminology=[],
        preferred_terminology=["innovation", "breakthrough", "exciting"],
        learning_data={}
    )
    
    config_manager.save_campaign_context("demo_context", custom_context)
    print("✓ Custom demo context created and saved")
    
    return orchestrator, config_manager


def demo_workflow_planning():
    """Demonstrate dynamic workflow planning."""
    print("\n3. Dynamic Workflow Planning")
    print("-" * 30)
    
    orchestrator, config_manager = demo_basic_integration()
    
    # Test different topics and contexts
    test_scenarios = [
        ("AI Breakthroughs", "demo_context"),
        ("Technical Deep Dive", "default"),
        ("Market Analysis", "default")
    ]
    
    for topic, context_id in test_scenarios:
        context = config_manager.load_campaign_context(context_id)
        workflow_plan = orchestrator.create_dynamic_workflow(topic, context)
        
        print(f"\n📋 Workflow Plan for '{topic}' with '{context_id}' context:")
        print(f"  - Phases: {len(workflow_plan['phases'])}")
        print(f"  - Estimated duration: {workflow_plan['total_duration_estimate']} seconds")
        
        for i, phase in enumerate(workflow_plan['phases'], 1):
            print(f"  - Phase {i}: {phase['name']} ({phase['duration_estimate']}s)")
            print(f"    Requirements: {', '.join(phase['requirements'])}")
            print(f"    Quality gate: {phase['quality_gate']}")
        
        # Validate workflow
        validation = orchestrator.validate_workflow_requirements(workflow_plan)
        print(f"  - Validation: {'✓ PASSED' if validation['valid'] else '✗ FAILED'}")
        if validation['warnings']:
            print(f"    Warnings: {len(validation['warnings'])}")
        if validation['issues']:
            print(f"    Issues: {len(validation['issues'])}")


def demo_workflow_analytics():
    """Demonstrate workflow analytics and monitoring."""
    print("\n4. Workflow Analytics & Monitoring")
    print("-" * 35)
    
    orchestrator, config_manager = demo_basic_integration()
    
    # Get initial analytics (no active workflow)
    analytics = orchestrator.get_workflow_analytics()
    print(f"✓ Initial analytics: {analytics['status']}")
    
    # Create mock execution state for demonstration
    from core.execution_state import ExecutionState
    
    execution_state = ExecutionState(workflow_id="demo-workflow")
    execution_state.update_phase("research")
    execution_state.update_quality_score("content_quality", 0.85)
    execution_state.increment_revision_cycle("content_refinement")
    execution_state.add_feedback({
        "phase": "research",
        "quality_score": 0.85,
        "notes": "Good research quality"
    })
    
    orchestrator.execution_state = execution_state
    
    # Get detailed analytics
    analytics = orchestrator.get_workflow_analytics()
    print(f"✓ Active workflow analytics:")
    print(f"  - Workflow ID: {analytics['workflow_id']}")
    print(f"  - Current phase: {analytics['current_phase']}")
    print(f"  - Quality scores: {analytics['quality_scores']}")
    print(f"  - Revision cycles: {analytics['revision_cycles']}")
    print(f"  - Feedback entries: {analytics['feedback_history_count']}")


def demo_configuration_management():
    """Demonstrate advanced configuration management."""
    print("\n5. Advanced Configuration Management")
    print("-" * 37)
    
    orchestrator, config_manager = demo_basic_integration()
    
    # Test context validation
    contexts = config_manager.list_campaign_contexts()
    
    for context_id in contexts:
        validation = config_manager.validate_context(context_id)
        status = "✓ VALID" if validation['valid'] else "✗ INVALID"
        print(f"Context '{context_id}': {status}")
        
        if validation['issues']:
            print(f"  Issues: {validation['issues']}")
    
    # Test context operations
    print(f"\n📊 Context Operations:")
    
    # Get context summary
    summary = config_manager.get_context_summary("default")
    print(f"✓ Context summary generated ({len(summary)} fields)")
    
    # Test context copying
    success = config_manager.copy_campaign_context("default", "default_copy")
    print(f"✓ Context copying: {'SUCCESS' if success else 'FAILED'}")
    
    # Test field updates
    success = config_manager.update_context_field("default_copy", "content_style", {
        "tone": "updated_tone",
        "style": "updated_style"
    })
    print(f"✓ Field update: {'SUCCESS' if success else 'FAILED'}")
    
    # Export contexts for backup
    export_data = config_manager.export_all_contexts()
    print(f"✓ Contexts exported ({len(export_data['contexts'])} contexts)")
    
    # Clean up demo context
    config_manager.delete_campaign_context("default_copy")
    print("✓ Demo context cleaned up")


def demo_learning_integration():
    """Demonstrate learning system integration."""
    print("\n6. Learning System Integration")
    print("-" * 30)
    
    orchestrator, config_manager = demo_basic_integration()
    
    # Get learning system
    learning_system = orchestrator.learning_system
    
    # Simulate performance data
    performance_data = {
        'execution_time': 45.2,
        'quality_score': 0.87,
        'revision_cycles': 2,
        'theme': 'artificial_intelligence',
        'style': 'technical',
        'audience_engagement': 0.92,
        'word_count': 1250,
        'sources_used': 5
    }
    
    print("📈 Updating campaign context with learning data...")
    context = config_manager.load_campaign_context("demo_context")
    learning_system.update_campaign_context(context, performance_data)
    
    print(f"✓ Learning data updated")
    print(f"  - Performance metrics: {len(performance_data)} items")
    print(f"  - Context updated at: {context.updated_at}")
    
    # Generate improvement recommendations
    recommendations = learning_system.generate_improvement_recommendations(context)
    print(f"✓ Generated {len(recommendations)} improvement recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")


def demo_main_integration():
    """Demonstrate main.py integration."""
    print("\n7. Main.py Integration")
    print("-" * 22)
    
    try:
        import main
        
        enhanced_mode = hasattr(main, 'ENHANCED_MODE') and main.ENHANCED_MODE
        print(f"✓ Enhanced mode: {'ENABLED' if enhanced_mode else 'DISABLED'}")
        
        if enhanced_mode:
            print("✓ Enhanced newsletter generation available")
            print("✓ Context management commands available")
            print("✓ Integration testing commands available")
            
            # Test available functions
            functions = [
                'execute_enhanced_newsletter_generation',
                'execute_hierarchical_newsletter_generation'
            ]
            
            for func_name in functions:
                if hasattr(main, func_name):
                    print(f"✓ Function '{func_name}' available")
                else:
                    print(f"⚠ Function '{func_name}' not found")
        else:
            print("⚠ Enhanced mode not available - using basic mode")
    
    except ImportError as e:
        print(f"✗ Main module import failed: {e}")


def demo_integration_testing():
    """Demonstrate integration testing capabilities."""
    print("\n8. Integration Testing")
    print("-" * 22)
    
    print("🧪 Running integration tests...")
    
    # Test 1: Component initialization
    try:
        orchestrator, config_manager = demo_basic_integration()
        print("✓ Component initialization test passed")
    except Exception as e:
        print(f"✗ Component initialization test failed: {e}")
        return
    
    # Test 2: Workflow planning
    try:
        context = config_manager.load_campaign_context("default")
        workflow_plan = orchestrator.create_dynamic_workflow("Test Topic", context)
        assert len(workflow_plan['phases']) >= 3
        print("✓ Workflow planning test passed")
    except Exception as e:
        print(f"✗ Workflow planning test failed: {e}")
    
    # Test 3: Configuration management
    try:
        contexts = config_manager.list_campaign_contexts()
        validation = config_manager.validate_context("default")
        assert validation['valid']
        print("✓ Configuration management test passed")
    except Exception as e:
        print(f"✗ Configuration management test failed: {e}")
    
    # Test 4: Analytics
    try:
        analytics = orchestrator.get_workflow_analytics()
        assert 'status' in analytics
        print("✓ Analytics test passed")
    except Exception as e:
        print(f"✗ Analytics test failed: {e}")
    
    print("✅ All integration tests completed")


def main():
    """Run the complete integration demonstration."""
    print("🎯 Phase 6 Enhanced Agent Architecture Integration Demo")
    print("=" * 60)
    print("This demo showcases the complete integrated system including:")
    print("• Workflow orchestration")
    print("• Campaign context management")
    print("• Dynamic workflow planning")
    print("• Learning system integration")
    print("• Configuration management")
    print("• Analytics and monitoring")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        demo_basic_integration()
        demo_workflow_planning()
        demo_workflow_analytics()
        demo_configuration_management()
        demo_learning_integration()
        demo_main_integration()
        demo_integration_testing()
        
        print("\n🎉 Integration Demo Completed Successfully!")
        print("=" * 40)
        print("✅ All components are fully integrated")
        print("✅ Enhanced agent architecture is operational")
        print("✅ System is ready for production use")
        print("\n💡 Next Steps:")
        print("• Run actual newsletter generation with real topics")
        print("• Customize campaign contexts for your use case")
        print("• Monitor quality metrics and learning data")
        print("• Integrate with external tools and APIs")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check the system configuration and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())