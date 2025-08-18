"""
Enhanced Main orchestration script for the AI Newsletter System (Phase 5)

This demonstrates the enhanced system with:
- Enhanced Workflow Orchestrator with campaign context
- Iterative refinement loop with quality gates
- Learning system integration
- Configuration management
- Full agent coordination
"""

import atexit
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict

# Add src to path so imports work correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure logging first
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

# Import tool usage enhancement components
try:
    from core.claim_validator import ClaimExtractor, SourceValidator, CitationGenerator
    from core.information_enricher import InformationEnricher
    from core.section_aware_refinement import ToolAugmentedRefinementLoop, SectionType
    from core.advanced_quality_gates import AdvancedQualityGate, QualityDimension
    from core.tool_analytics import ToolEffectivenessAnalyzer, ToolType
    from core.tool_cache import get_tool_cache
    from storage import get_storage_provider
    from tools.enhanced_search import MultiProviderSearchEngine
    from core.source_ranker import SourceAuthorityRanker
    TOOL_USAGE_AVAILABLE = True
    logger.info("Tool usage enhancement components loaded successfully")
except ImportError as e:
    logger.warning(f"Tool usage components not available: {e}")
    TOOL_USAGE_AVAILABLE = False

# Import enhanced components
try:
    from core.campaign_context import CampaignContext
    from core.config_manager import ConfigManager
    from core.workflow_orchestrator import WorkflowOrchestrator
    ENHANCED_MODE = True
    logger.info("Enhanced workflow orchestrator loaded successfully")
except ImportError as e:
    logger.warning(f"Enhanced components not available: {e}")
    ENHANCED_MODE = False


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


def execute_enhanced_newsletter_generation(
    topic: str,
    context_id: str = "default",
    output_format: str = "markdown"
) -> Dict[str, Any]:
    """Execute enhanced newsletter generation using the workflow orchestrator."""

    if not ENHANCED_MODE:
        logger.warning(
            "Enhanced mode not available, falling back to basic generation")
        return execute_hierarchical_newsletter_generation(topic)

    start_time = time.time()
    logger.info(f"Starting enhanced newsletter generation for: {topic}")

    try:
        # Initialize orchestrator
        config_manager = ConfigManager()
        orchestrator = WorkflowOrchestrator(config_manager)

        # Execute workflow
        result = orchestrator.execute_newsletter_generation(
            topic, context_id, output_format)

        if result.status == 'completed':
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"newsletter_{topic.replace(' ', '_').lower()}_{
                timestamp}.{output_format}"
            output_path = os.path.join("output", output_filename)

            # Ensure output directory exists
            os.makedirs("output", exist_ok=True)

            # Save the newsletter content
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result.final_content)

            logger.info(
                f"Enhanced newsletter generation completed successfully in {
                    result.execution_time:.2f} seconds")

            return {
                'success': True,
                'output_file': output_path,
                'content': result.final_content,
                'execution_time': result.execution_time,
                'quality_metrics': result.quality_metrics,
                'workflow_id': result.workflow_id,
                'learning_data': result.learning_data,
                'phase_results': result.phase_results
            }
        else:
            logger.error(
                f"Enhanced newsletter generation failed: {
                    result.status}")
            return {
                'success': False,
                'error': f"Workflow failed with status: {result.status}",
                'execution_time': result.execution_time,
                'quality_metrics': result.quality_metrics
            }

    except Exception as e:
        logger.error(f"Enhanced newsletter generation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'execution_time': time.time() - start_time
        }


def execute_hierarchical_newsletter_generation(
        topic: str, audience: str = "technology professionals") -> Dict[str, Any]:
    """Execute tool-augmented newsletter generation with intelligence enhancement."""

    start_time = time.time()
    logger.info(f"Starting tool-augmented newsletter generation for: {topic}")
    
    # Initialize tool usage tracking
    tool_usage_metrics = {
        'vector_queries': 0,
        'web_searches': 0,
        'verified_claims': [],
        'search_providers': [],
        'tool_integration_score': 0.0
    }

    try:
        if TOOL_USAGE_AVAILABLE:
            from enhanced_generation import execute_tool_augmented_generation
            return execute_tool_augmented_generation(topic, audience, tool_usage_metrics)
        else:
            logger.warning("Tool usage components not available, falling back to basic generation")
            from enhanced_generation import execute_basic_generation
            return execute_basic_generation(topic, audience, tool_usage_metrics)

    except Exception as e:
        logger.error(f"Newsletter generation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'execution_time': time.time() - start_time,
            'tool_usage': tool_usage_metrics
        }


def main():
    """Main entry point with enhanced capabilities."""

    if len(sys.argv) < 2:
        mode_status = "Enhanced" if ENHANCED_MODE else "Basic"
        print(f"Newsletter Generation System ({mode_status} Mode)")
        print("Usage:")
        print("  python src/main.py <topic>                    - Generate newsletter")
        print("  python src/main.py <topic> --context <id>     - Use specific context")
        print("  python src/main.py <topic> --format <format>  - Set output format")
        print("  python src/main.py --contexts                 - List available contexts")
        print("  python src/main.py --test-integration         - Test system integration")
        print("  python src/main.py --help                     - Show this help")
        print("")
        print("Examples:")
        print("  python src/main.py 'AI and Machine Learning'")
        print("  python src/main.py 'Latest in Data Science' --context technical")
        print("  python src/main.py 'Business Trends' --context business --format html")
        return

    # Handle special commands
    if sys.argv[1] == "--help" or sys.argv[1] == "-h":
        mode_status = "Enhanced" if ENHANCED_MODE else "Basic"
        print(f"Newsletter Generation System ({mode_status} Mode)")
        print("=" * 50)
        if ENHANCED_MODE:
            print(
                "Generate comprehensive newsletters using enhanced workflow orchestration.")
            print("")
            print("Features:")
            print("  • Enhanced workflow orchestrator with campaign context")
            print("  • Iterative refinement loop with quality gates")
            print("  • Learning system integration")
            print("  • Configuration management")
            print("  • Multiple output formats")
        else:
            print("Generate comprehensive newsletters using basic LLM generation.")
            print("")
            print("Features:")
            print("  • Direct LLM content generation")
            print("  • Basic quality assessment")
        print("")
        print("Usage:")
        print(
            "  python src/main.py <topic> [--context <id>] [--format <format>]")
        print("")
        print("Examples:")
        print("  python src/main.py 'AI and Machine Learning'")
        print("  python src/main.py 'Latest in Data Science' --context technical")
        return

    if sys.argv[1] == "--contexts":
        if ENHANCED_MODE:
            try:
                config_manager = ConfigManager()
                contexts = config_manager.list_campaign_contexts()
                print("Available Campaign Contexts:")
                print("=" * 30)
                for context_id in contexts:
                    summary = config_manager.get_context_summary(context_id)
                    tone = summary.get(
                        'content_style', {}).get(
                        'tone', 'unknown')
                    goals = len(summary.get('strategic_goals', []))
                    thresholds = summary.get('quality_thresholds', {})
                    min_quality = thresholds.get('minimum', 'unknown')
                    print(f"  {context_id}:")
                    print(f"    - Tone: {tone}")
                    print(f"    - Strategic goals: {goals}")
                    print(f"    - Min quality threshold: {min_quality}")
                return
            except Exception as e:
                print(f"Error listing contexts: {e}")
                return
        else:
            print("Context management not available in basic mode")
            return

    if sys.argv[1] == "--test-integration":
        if ENHANCED_MODE:
            print("🧪 Testing Enhanced Integration")
            print("=" * 30)
            try:
                # Test component initialization
                print("1. Testing component initialization...")
                config_manager = ConfigManager()
                orchestrator = WorkflowOrchestrator(config_manager)
                print("   ✓ Components initialized successfully")

                # Test context management
                print("2. Testing context management...")
                contexts = config_manager.list_campaign_contexts()
                default_context = config_manager.load_campaign_context(
                    "default")
                print(
                    f"   ✓ Found {
                        len(contexts)} contexts, default context loaded")

                # Test workflow planning
                print("3. Testing workflow planning...")
                workflow_plan = orchestrator.create_dynamic_workflow(
                    "Test Topic", default_context)
                validation = orchestrator.validate_workflow_requirements(
                    workflow_plan)
                print(f"   ✓ Workflow planned with {
                      len(workflow_plan['phases'])} phases")
                print(f"   ✓ Workflow validation: {validation['valid']}")

                # Test analytics
                print("4. Testing analytics...")
                analytics = orchestrator.get_workflow_analytics()
                print("   ✓ Analytics retrieved successfully")

                print("\n✅ All integration tests passed!")
                return

            except Exception as e:
                print(f"\n❌ Integration test failed: {e}")
                return 1
        else:
            print("Integration testing not available in basic mode")
            return

    # Parse arguments
    args = sys.argv[1:]
    context_id = "default"
    output_format = "markdown"
    topic_parts = []

    i = 0
    while i < len(args):
        if args[i] == "--context" and i + 1 < len(args):
            context_id = args[i + 1]
            i += 2
        elif args[i] == "--format" and i + 1 < len(args):
            output_format = args[i + 1]
            i += 2
        else:
            topic_parts.append(args[i])
            i += 1

    if not topic_parts:
        print("Error: No topic specified")
        return 1

    topic = " ".join(topic_parts)

    # Display execution info
    mode_status = "Enhanced" if ENHANCED_MODE else "Basic"
    print(f"🚀 Newsletter Generation System ({mode_status} Mode)")
    print("=" * 50)
    print(f"📝 Topic: {topic}")
    print(f"🎯 Context: {context_id}")
    print(f"📄 Format: {output_format}")
    if ENHANCED_MODE:
        print(f"🤖 Features: Workflow orchestration, Quality refinement, Learning")
    else:
        print(f"🤖 Features: Basic LLM generation")
    print("=" * 50)

    # Execute the workflow
    if ENHANCED_MODE:
        result = execute_enhanced_newsletter_generation(
            topic, context_id, output_format)
    else:
        result = execute_hierarchical_newsletter_generation(topic)

    if result['success']:
        print(f"\n✅ Newsletter generation completed successfully!")
        print(f"📄 Output saved to: {result['output_file']}")
        print(f"⏱️  Execution time: {result['execution_time']:.2f} seconds")

        # Show enhanced metrics if available
        if ENHANCED_MODE and 'quality_metrics' in result:
            quality_metrics = result['quality_metrics']
            if 'final_score' in quality_metrics or 'overall_score' in quality_metrics:
                score = quality_metrics.get(
                    'final_score', quality_metrics.get(
                        'overall_score', 0))
                print(f"🎯 Quality score: {score:.2f}")

            if 'workflow_id' in result:
                print(f"🔄 Workflow ID: {result['workflow_id']}")

            if 'learning_data' in result and result['learning_data']:
                learning = result['learning_data']
                if 'revision_cycles' in learning:
                    print(f"🔧 Revision cycles: {learning['revision_cycles']}")
    else:
        print(f"\n❌ Newsletter generation failed: {result['error']}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
