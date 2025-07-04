"""
Streamlit Interface for AI Newsletter Generator
Modern, comprehensive UI for newsletter generation with all settings and controls
"""

import streamlit as st
import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Add src to path for imports (go up one directory to find src)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the newsletter generation functions
from src.main import execute_newsletter_generation, execute_hierarchical_newsletter_generation
from src.agents.agents import ResearchAgent, PlannerAgent, WriterAgent, EditorAgent, ManagerAgent

# Page configuration
st.set_page_config(
    page_title="AI Newsletter Generator",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #0068c9;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #0068c9, #29b5e8);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,104,201,0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background: linear-gradient(90deg, #56ab2f, #a8e6cf);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: linear-gradient(90deg, #f093fb, #f5576c);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    
    .stTextInput > div > div {
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    
    .stTextArea > div > div {
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    
    .sidebar-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    
    .output-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 1px solid #e1e8ed;
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

def load_sources_config():
    """Load sources configuration from YAML file"""
    try:
        import yaml
        # Look for sources.yaml in the parent directory's src folder
        sources_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'sources.yaml')
        with open(sources_path, 'r') as f:
            return yaml.safe_load(f)
    except:
        return {"sources": []}

def get_categories_from_sources(sources_config):
    """Extract unique categories from sources configuration"""
    categories = set()
    for source in sources_config.get('sources', []):
        if 'category' in source:
            categories.add(source['category'])
    return sorted(list(categories))

def create_metrics_dashboard(stats):
    """Create a metrics dashboard using Plotly"""
    if not stats:
        return None
    
    # Create metrics visualization
    fig = go.Figure()
    
    # Add execution time gauge
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = stats.get('execution_time', 0),
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Execution Time (seconds)"},
        gauge = {
            'axis': {'range': [None, 300]},
            'bar': {'color': "#0068c9"},
            'steps': [
                {'range': [0, 60], 'color': "#a8e6cf"},
                {'range': [60, 180], 'color': "#ffd93d"},
                {'range': [180, 300], 'color': "#ff6b6b"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 240
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI Newsletter Generator</h1>
        <p>Create comprehensive, engaging newsletters with AI-powered research and writing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load sources configuration
    sources_config = load_sources_config()
    available_categories = get_categories_from_sources(sources_config)
    
    # Sidebar Configuration
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h3>⚙️ Configuration</h3>
        <p>Customize your newsletter generation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main topic input
    topic = st.sidebar.text_input(
        "📝 Newsletter Topic",
        placeholder="Enter your newsletter topic...",
        help="The main subject for your newsletter"
    )
    
    # Audience selection
    audience_options = [
        "technology professionals",
        "business executives",
        "general audience",
        "researchers and academics",
        "students and learners",
        "industry specialists",
        "entrepreneurs and startups"
    ]
    
    audience = st.sidebar.selectbox(
        "👥 Target Audience",
        audience_options,
        help="Select your target audience for content optimization"
    )
    
    # Workflow type
    workflow_type = st.sidebar.selectbox(
        "🔄 Workflow Type",
        ["Standard Multi-Agent", "Hierarchical (Manager-Led)"],
        help="Choose between standard multi-agent workflow or hierarchical manager-led workflow"
    )
    
    # Content Settings
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h3>📊 Content Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Content length preference
    content_length = st.sidebar.selectbox(
        "📏 Content Length",
        ["Comprehensive (15-20k words)", "Standard (10-15k words)", "Concise (5-10k words)"],
        help="Choose the desired length for your newsletter"
    )
    
    # Quality focus areas
    quality_focus = st.sidebar.multiselect(
        "🎯 Quality Focus Areas",
        ["Research Depth", "Writing Quality", "Engagement", "Technical Accuracy", "Practical Value"],
        default=["Research Depth", "Writing Quality", "Engagement"],
        help="Select areas to prioritize during generation"
    )
    
    # Source categories
    if available_categories:
        selected_categories = st.sidebar.multiselect(
            "📰 Source Categories",
            available_categories,
            default=available_categories[:3] if len(available_categories) > 3 else available_categories,
            help="Select which source categories to include"
        )
    else:
        selected_categories = []
    
    # Advanced Settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        collect_feedback = st.checkbox(
            "Collect User Feedback",
            value=True,
            help="Enable feedback collection for continuous improvement"
        )
        
        enable_quality_scoring = st.checkbox(
            "Enable Quality Scoring",
            value=True,
            help="Generate detailed quality metrics"
        )
        
        save_intermediate_results = st.checkbox(
            "Save Intermediate Results",
            value=False,
            help="Save results from each agent for debugging"
        )
        
        max_execution_time = st.slider(
            "Max Execution Time (minutes)",
            min_value=5,
            max_value=30,
            value=15,
            help="Maximum time to allow for generation"
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Newsletter Configuration")
        
        # Topic validation
        if topic:
            st.success(f"✅ Topic: {topic}")
        else:
            st.warning("⚠️ Please enter a newsletter topic to begin")
        
        # Configuration summary
        if topic:
            st.markdown("""
            <div class="output-container">
                <h4>📋 Configuration Summary</h4>
                <ul>
                    <li><strong>Topic:</strong> {topic}</li>
                    <li><strong>Audience:</strong> {audience}</li>
                    <li><strong>Workflow:</strong> {workflow}</li>
                    <li><strong>Content Length:</strong> {length}</li>
                    <li><strong>Quality Focus:</strong> {focus}</li>
                    <li><strong>Source Categories:</strong> {categories}</li>
                </ul>
            </div>
            """.format(
                topic=topic,
                audience=audience,
                workflow=workflow_type,
                length=content_length,
                focus=", ".join(quality_focus) if quality_focus else "Default",
                categories=", ".join(selected_categories) if selected_categories else "All"
            ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        
        # Display some stats about the system
        st.metric("Available Sources", len(sources_config.get('sources', [])))
        st.metric("Categories", len(available_categories))
        st.metric("Agents", "5 (Manager, Planner, Research, Writer, Editor)")
        
        # System status
        if st.session_state.is_generating:
            st.markdown("""
            <div class="warning-message">
                <h4>⚡ Generation in Progress</h4>
                <p>Your newsletter is being generated...</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-message">
                <h4>✅ System Ready</h4>
                <p>Ready to generate your newsletter!</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Generation button
    st.markdown("---")
    
    generate_col1, generate_col2, generate_col3 = st.columns([1, 2, 1])
    
    with generate_col2:
        if st.button(
            "🚀 Generate Newsletter",
            disabled=not topic or st.session_state.is_generating,
            help="Click to start generating your newsletter"
        ):
            if topic:
                st.session_state.is_generating = True
                generate_newsletter(topic, audience, workflow_type, collect_feedback, max_execution_time)
    
    # Output section
    if st.session_state.newsletter_output:
        display_newsletter_output()
    
    # Performance metrics
    if st.session_state.generation_stats:
        display_performance_metrics()

def generate_newsletter(topic, audience, workflow_type, collect_feedback, max_execution_time):
    """Generate newsletter with progress tracking"""
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Initializing generation...")
        progress_bar.progress(10)
        
        # Choose workflow
        if workflow_type == "Hierarchical (Manager-Led)":
            status_text.text("🤖 Starting hierarchical workflow...")
            progress_bar.progress(30)
            
            result = execute_hierarchical_newsletter_generation(topic, audience)
        else:
            status_text.text("🤖 Starting standard multi-agent workflow...")
            progress_bar.progress(30)
            
            result = execute_newsletter_generation(topic, collect_feedback)
        
        progress_bar.progress(90)
        status_text.text("✅ Generation complete!")
        
        # Store results
        st.session_state.newsletter_output = result
        st.session_state.generation_stats = {
            'execution_time': result.get('execution_time', 0),
            'success': result.get('success', False),
            'workflow_type': workflow_type,
            'topic': topic,
            'audience': audience,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        progress_bar.progress(100)
        time.sleep(1)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if result.get('success'):
            st.success("✅ Newsletter generated successfully!")
        else:
            st.error(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        st.error(f"❌ Generation failed: {str(e)}")
        st.session_state.generation_stats = {
            'execution_time': 0,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    finally:
        st.session_state.is_generating = False

def display_newsletter_output():
    """Display the generated newsletter output"""
    st.markdown("---")
    st.markdown("## 📰 Generated Newsletter")
    
    result = st.session_state.newsletter_output
    
    if result.get('success'):
        # Content preview
        content = result.get('content', '')
        
        if content:
            # Show content length
            st.info(f"📊 Content Length: {len(content):,} characters (~{len(content.split()):,} words)")
            
            # Content tabs
            tab1, tab2, tab3 = st.tabs(["📖 Full Content", "📋 Preview", "📁 Download"])
            
            with tab1:
                st.markdown("""
                <div class="output-container">
                    <div style="max-height: 600px; overflow-y: auto; padding: 1rem; background: white; border-radius: 8px;">
                """, unsafe_allow_html=True)
                
                # Display full content
                st.markdown(content)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
            
            with tab2:
                # Show first 2000 characters
                preview_length = 2000
                if len(content) > preview_length:
                    st.markdown(f"**Preview (first {preview_length} characters):**")
                    st.markdown(content[:preview_length] + "...")
                    st.info(f"Full content continues for {len(content) - preview_length:,} more characters")
                else:
                    st.markdown("**Complete Content:**")
                    st.markdown(content)
            
            with tab3:
                # Download options
                st.markdown("### 📥 Download Options")
                
                # Markdown download
                st.download_button(
                    label="📄 Download as Markdown",
                    data=content,
                    file_name=f"newsletter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                
                # Text download
                st.download_button(
                    label="📝 Download as Text",
                    data=content,
                    file_name=f"newsletter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                # JSON download with metadata
                json_data = {
                    'content': content,
                    'metadata': st.session_state.generation_stats,
                    'generated_at': datetime.now().isoformat()
                }
                
                st.download_button(
                    label="📊 Download with Metadata (JSON)",
                    data=json.dumps(json_data, indent=2),
                    file_name=f"newsletter_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.warning("⚠️ No content generated")
    else:
        st.error(f"❌ Generation failed: {result.get('error', 'Unknown error')}")

def display_performance_metrics():
    """Display performance metrics and analytics"""
    st.markdown("---")
    st.markdown("## 📊 Performance Metrics")
    
    stats = st.session_state.generation_stats
    
    if stats:
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Execution Time",
                f"{stats.get('execution_time', 0):.1f}s",
                delta=None
            )
        
        with col2:
            st.metric(
                "Status",
                "✅ Success" if stats.get('success') else "❌ Failed",
                delta=None
            )
        
        with col3:
            st.metric(
                "Workflow",
                stats.get('workflow_type', 'Unknown'),
                delta=None
            )
        
        with col4:
            st.metric(
                "Generated",
                stats.get('timestamp', 'Unknown'),
                delta=None
            )
        
        # Performance visualization
        if stats.get('execution_time'):
            fig = create_metrics_dashboard(stats)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed stats
        with st.expander("📈 Detailed Statistics"):
            st.json(stats)

if __name__ == "__main__":
    main() 