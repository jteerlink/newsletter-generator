"""
Enhanced Streamlit Interface for AI Newsletter Generator
Uses modular UI components for a premium user experience
"""

import streamlit as st
import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, Any, List

# Add src to path for imports (go up one directory to find src)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the newsletter generation functions
from src.main import execute_newsletter_generation, execute_hierarchical_newsletter_generation
from src.agents.agents import ResearchAgent, PlannerAgent, WriterAgent, EditorAgent, ManagerAgent

# Import UI components
from ui_components import (
    create_header, create_feature_cards, create_configuration_panel,
    create_advanced_settings_panel, create_status_dashboard, create_configuration_summary,
    create_generate_button, create_progress_tracker, create_content_display_tabs,
    create_performance_dashboard, create_feedback_section, create_footer
)

# Page configuration
st.set_page_config(
    page_title="AI Newsletter Generator",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Ensure full width layout */
    .main > div {
        padding-top: 2rem;
        max-width: 100% !important;
        width: 100% !important;
    }
    
    .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    
    /* Fix column layouts */
    .row-widget.stColumns {
        width: 100% !important;
        gap: 1rem;
    }
    
    .column {
        width: 100% !important;
        padding: 0.5rem;
    }
    
    /* Ensure content spans full width */
    .stMarkdown, .stSelectbox, .stTextInput, .stMultiSelect {
        width: 100% !important;
    }
    
    /* Fix specific component alignments */
    .stSelectbox > div, .stTextInput > div, .stMultiSelect > div {
        width: 100% !important;
    }
    
    /* Remove default margins that cause left alignment */
    .element-container {
        width: 100% !important;
        margin: 0 !important;
    }
    
    /* Ensure container fills full width */
    .main .block-container {
        max-width: none !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    /* Fix markdown containers */
    .stMarkdown > div {
        width: 100% !important;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #0068c9, #29b5e8);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,104,201,0.3);
        border: 2px solid #0068c9;
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        border: 1px solid #dee2e6;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #0068c9;
        box-shadow: 0 2px 8px rgba(0,104,201,0.1);
    }
    
    .stTextInput > div > div {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        border: 1px solid #dee2e6;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div:focus-within {
        border-color: #0068c9;
        box-shadow: 0 2px 8px rgba(0,104,201,0.1);
    }
    
    .stMultiSelect > div > div {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        border: 1px solid #dee2e6;
    }
    
    .stSlider > div > div {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stCheckbox > label {
        font-weight: 500;
        color: #495057;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #dee2e6;
    }
    
    .stTab {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        margin: 0.5rem;
    }
    
    .stExpander {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #0068c9, #29b5e8);
        border-radius: 6px;
    }
    
    .stSpinner > div {
        border-color: #0068c9;
    }
    
    /* Animations for status indicators */
    @keyframes pulse {
        0% { 
            opacity: 1; 
            transform: scale(1);
        }
        50% { 
            opacity: 0.8; 
            transform: scale(1.02);
        }
        100% { 
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .generating {
        animation: pulse 2s infinite;
    }
    
    .status-card {
        animation: fadeInUp 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    .status-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'newsletter_output' not in st.session_state:
    st.session_state.newsletter_output = None
if 'generation_stats' not in st.session_state:
    st.session_state.generation_stats = None
if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False
if 'config' not in st.session_state:
    st.session_state.config = {}
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []

def load_sources_config():
    """Load sources configuration from YAML file"""
    try:
        import yaml
        # Look for sources.yaml in the parent directory's src folder
        sources_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'sources.yaml')
        with open(sources_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading sources config: {e}")
        return {"sources": []}

def get_categories_from_sources(sources_config):
    """Extract unique categories from sources configuration"""
    categories = set()
    for source in sources_config.get('sources', []):
        if 'category' in source:
            categories.add(source['category'])
    return sorted(list(categories))

def generate_newsletter_with_progress(config: Dict[str, Any], advanced_config: Dict[str, Any]):
    """Generate newsletter with enhanced progress tracking"""
    st.session_state.is_generating = True
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        # Step 1: Initialization
        create_progress_tracker("🔄 Initializing AI Agents", 10)
        time.sleep(0.5)
        
        # Step 2: Configuration
        create_progress_tracker("⚙️ Processing Configuration", 20)
        time.sleep(0.5)
        
        # Step 3: Starting Generation
        create_progress_tracker("🚀 Starting Newsletter Generation", 30)
        time.sleep(0.5)
        
        try:
            # Choose workflow based on configuration
            if config['workflow_type'] == "Hierarchical (Manager-Led)":
                create_progress_tracker("🤖 Hierarchical Workflow Active", 50)
                result = execute_hierarchical_newsletter_generation(
                    config['topic'], 
                    config['audience']
                )
            else:
                create_progress_tracker("🤖 Multi-Agent Workflow Active", 50)
                result = execute_newsletter_generation(
                    config['topic'], 
                    advanced_config['collect_feedback']
                )
            
            # Step 4: Processing Results
            create_progress_tracker("📊 Processing Results", 90)
            time.sleep(0.5)
            
            # Step 5: Complete
            create_progress_tracker("✅ Generation Complete!", 100)
            time.sleep(1)
            
            # Store results with better handling
            st.session_state.newsletter_output = result
            
            # Extract content properly from different workflow results
            content = ""
            if result.get('success'):
                if 'content' in result:
                    content = result['content']
                elif 'workflow_result' in result:
                    # Handle hierarchical workflow results
                    workflow_result = result['workflow_result']
                    if 'stream_results' in workflow_result:
                        stream_results = workflow_result['stream_results']
                        if 'writing' in stream_results:
                            content = stream_results['writing'].get('result', '')
                        elif 'editing' in stream_results:
                            content = stream_results['editing'].get('result', '')
            
            st.session_state.generation_stats = {
                'execution_time': result.get('execution_time', 0),
                'success': result.get('success', False),
                'workflow_type': config['workflow_type'],
                'topic': config['topic'],
                'audience': config['audience'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'config': config,
                'advanced_config': advanced_config,
                'content_length': len(content) if content else 0,
                'word_count': len(content.split()) if content else 0
            }
            
            # Clear progress
            progress_container.empty()
            
            if result.get('success'):
                st.success("✅ Newsletter generated successfully!")
                if content:
                    st.info(f"📊 Generated {len(content):,} characters ({len(content.split()):,} words)")
                st.balloons()
                # Force a rerun to show the output
                st.rerun()
            else:
                st.error(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            progress_container.empty()
            st.error(f"❌ Generation failed: {str(e)}")
            st.session_state.generation_stats = {
                'execution_time': 0,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    st.session_state.is_generating = False

def main():
    """Main application function"""
    
    # Create header - ensure full width
    create_header()
    
    # Create feature cards - ensure proper distribution
    create_feature_cards()
    
    # Load sources configuration
    sources_config = load_sources_config()
    available_categories = get_categories_from_sources(sources_config)
    
    # Main content container
    st.markdown("""
    <div style="
        width: 100%;
        margin: 2rem 0;
        padding: 0;
    ">
    """, unsafe_allow_html=True)
    
    # Main layout - proper column distribution
    col1, col2 = st.columns([2.5, 1.5], gap="large")
    
    with col1:
        # Configuration panel
        config = create_configuration_panel(sources_config, available_categories)
        st.session_state.config = config
        
        # Advanced settings
        advanced_config = create_advanced_settings_panel()
        
        # Configuration summary
        create_configuration_summary(config)
        
        # Generate button
        if create_generate_button(config['topic'], st.session_state.is_generating):
            if config['topic']:
                generate_newsletter_with_progress(config, advanced_config)
    
    with col2:
        # Status dashboard
        create_status_dashboard(sources_config, available_categories, st.session_state.is_generating)
        
        # Quick tips container
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            border: 1px solid #dee2e6;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
        """, unsafe_allow_html=True)
        
        with st.expander("💡 Quick Tips", expanded=False):
            st.markdown("""
            **🎯 For Best Results:**
            - Be specific with your topic
            - Choose relevant source categories
            - Allow 5-15 minutes for generation
            - Use hierarchical workflow for complex topics
            
            **📊 Content Quality:**
            - Comprehensive: 15-20k words
            - Standard: 10-15k words  
            - Concise: 5-10k words
            
            **🔧 Pro Tips:**
            - Enable web search for current info
            - Use quality scoring for insights
            - Save intermediate results for debugging
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Output section with enhanced content extraction
    if st.session_state.newsletter_output:
        st.markdown("---")
        
        # Output header
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 2rem 0 1rem 0;
            text-align: center;
            color: white;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            border: 2px solid rgba(255,255,255,0.1);
        ">
            <h2 style="margin: 0; font-size: 1.8rem; font-weight: 700;">
                📰 Generated Newsletter
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        result = st.session_state.newsletter_output
        
        # Debug information (can be removed later)
        with st.expander("🔍 Debug Info", expanded=False):
            st.write("Result keys:", list(result.keys()) if result else "No result")
            if result:
                st.write("Success:", result.get('success'))
                st.write("Has content:", 'content' in result)
                st.write("Has workflow_result:", 'workflow_result' in result)
        
        if result.get('success'):
            # Enhanced content extraction
            content = ""
            
            # Try different content extraction methods
            if 'content' in result and result['content']:
                content = result['content']
                st.info("📋 Content extracted from direct result")
            elif 'workflow_result' in result:
                workflow_result = result['workflow_result']
                st.info("📋 Extracting content from workflow result...")
                
                if 'stream_results' in workflow_result:
                    stream_results = workflow_result['stream_results']
                    
                    # Try to get content from writing stream
                    if 'writing' in stream_results and stream_results['writing'].get('result'):
                        writing_content = stream_results['writing']['result']
                        
                        # Try to get content from editing stream
                        if 'editing' in stream_results and stream_results['editing'].get('result'):
                            editing_content = stream_results['editing']['result']
                            # Combine both if available
                            content = f"{writing_content}\n\n---\n\n## Editorial Review\n\n{editing_content}"
                        else:
                            content = writing_content
                    elif 'editing' in stream_results and stream_results['editing'].get('result'):
                        content = stream_results['editing']['result']
                
                # If still no content, try to extract from other parts
                if not content and isinstance(workflow_result, dict):
                    for key, value in workflow_result.items():
                        if isinstance(value, str) and len(value) > 100:  # Likely content
                            content = value
                            break
            
            if content:
                # Show content statistics
                word_count = len(content.split())
                char_count = len(content)
                
                st.success(f"✅ Newsletter generated successfully! ({char_count:,} characters, {word_count:,} words)")
                
                # Display the content using the enhanced tabs
                create_content_display_tabs(content)
            else:
                st.warning("⚠️ Newsletter generation completed but no content was extracted. Please check the debug info above.")
                
                # Show raw result for debugging
                with st.expander("🔧 Raw Result", expanded=False):
                    st.json(result)
        else:
            error_msg = result.get('error', 'Unknown error')
            st.error(f"❌ Generation failed: {error_msg}")
            
            # Show raw result for debugging failures
            with st.expander("🔧 Raw Error Result", expanded=False):
                st.json(result)
    
    # Performance metrics
    if st.session_state.generation_stats:
        create_performance_dashboard(st.session_state.generation_stats)
    
    # Feedback section
    feedback_data = create_feedback_section()
    if feedback_data:
        st.session_state.feedback_data.append(feedback_data)
        
        # Show feedback analytics if we have data
        if len(st.session_state.feedback_data) > 0:
            with st.expander("📊 Feedback Analytics", expanded=False):
                feedback_df = st.session_state.feedback_data
                avg_quality = sum(f['quality'] for f in feedback_df) / len(feedback_df)
                avg_relevance = sum(f['relevance'] for f in feedback_df) / len(feedback_df)
                avg_completeness = sum(f['completeness'] for f in feedback_df) / len(feedback_df)
                avg_readability = sum(f['readability'] for f in feedback_df) / len(feedback_df)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Quality", f"{avg_quality:.1f}/5")
                with col2:
                    st.metric("Avg Relevance", f"{avg_relevance:.1f}/5")
                with col3:
                    st.metric("Avg Completeness", f"{avg_completeness:.1f}/5")
                with col4:
                    st.metric("Avg Readability", f"{avg_readability:.1f}/5")
    
    # Footer
    create_footer()

if __name__ == "__main__":
    main() 