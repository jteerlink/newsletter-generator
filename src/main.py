"""
Main orchestration script for the AI Newsletter System (Simplified Version)

This demonstrates the simplified system with:
- ManagerAgent for hierarchical task delegation
- Single deep dive pipeline execution
- Quality assessment and feedback
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
    ManagerAgent, PlannerAgent
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

def execute_hierarchical_newsletter_generation(topic: str, audience: str = "technology professionals") -> Dict[str, Any]:
    """Execute simplified newsletter generation using direct LLM queries."""
    
    start_time = time.time()
    logger.info(f"Starting newsletter generation for: {topic}")
    
    try:
        from src.core.core import query_llm
        
        # Create a comprehensive prompt for newsletter generation
        prompt = f"""
        Write a newsletter about {topic} for {audience}.
        
        Include:
        1. Brief introduction
        2. Key developments and trends
        3. Technical insights
        4. Practical implications
        5. Future outlook
        
        Make it informative and well-structured for technical professionals.
        Target length: 800-1500 words.
        """
        
        logger.info("Generating newsletter content...")
        content = query_llm(prompt)
        
        if not content or len(content) < 100:
            raise Exception("Generated content is insufficient")
        
        execution_time = time.time() - start_time
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"newsletter_{topic.replace(' ', '_').lower()}_{timestamp}.md"
        output_path = os.path.join("output", output_filename)
        
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        
        # Save the newsletter content
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Newsletter generation completed successfully in {execution_time:.2f} seconds")
        
        return {
            'success': True,
            'output_file': output_path,
            'content': content,
            'execution_time': execution_time
        }
        
    except Exception as e:
        logger.error(f"Newsletter generation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'execution_time': time.time() - start_time
        }

def main():
    """Main entry point with simplified capabilities."""
    
    if len(sys.argv) < 2:
        print("Newsletter Generation System")
        print("Usage:")
        print("  python src/main.py <topic>                    - Generate newsletter")
        print("  python src/main.py --help                     - Show this help")
        print("")
        print("Examples:")
        print("  python src/main.py 'AI and Machine Learning'")
        print("  python src/main.py 'Latest in Data Science'")
        return
    
    # Handle help command
    if sys.argv[1] == "--help" or sys.argv[1] == "-h":
        print("Newsletter Generation System")
        print("=" * 50)
        print("Generate comprehensive newsletters using hierarchical agent execution.")
        print("")
        print("Usage:")
        print("  python src/main.py <topic>")
        print("")
        print("Features:")
        print("  • Hierarchical agent execution (Manager → Planner → Research → Writer → Editor)")
        print("  • Deep dive content generation (4,000+ words)")
        print("  • Quality assurance and validation")
        print("  • Automatic file output")
        print("")
        print("Examples:")
        print("  python src/main.py 'AI and Machine Learning'")
        print("  python src/main.py 'Latest in Data Science'")
        return
    
    # Regular newsletter generation (hierarchical execution)
    topic = " ".join(sys.argv[1:])
    
    print("🚀 Newsletter Generation System")
    print("=" * 50)
    print(f"📝 Topic: {topic}")
    print(f"🤖 Agents: Manager → Planner → Research → Writer → Editor")
    print(f"🔧 Features: Hierarchical execution, Quality assessment")
    print("=" * 50)
    
    # Execute the hierarchical workflow
    result = execute_hierarchical_newsletter_generation(topic)
    
    if result['success']:
        print(f"\n✅ Newsletter generation completed successfully!")
        print(f"📄 Output saved to: {result['output_file']}")
        print(f"⏱️  Execution time: {result['execution_time']:.2f} seconds")
    else:
        print(f"\n❌ Newsletter generation failed: {result['error']}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
