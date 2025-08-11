"""
Simplified Newsletter System - Streamlit Interface
Modern, comprehensive UI for the hierarchical newsletter system
Features deep dive pipeline and quality assurance monitoring
"""

import streamlit as st
import sys
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the simplified system components
from src.main import execute_hierarchical_newsletter_generation
from src.quality import QualityAssuranceSystem
from src.core.core import query_llm

# Page configuration
st.set_page_config(
    page_title="Newsletter System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS styling inspired by the infographic
st.markdown("""
<style>
    /* Color scheme from infographic */
    :root {
        --primary-blue: #003F5C;
        --secondary-blue: #2F4B7C;
        --accent-blue: #665191;
        --orange: #FFA600;
        --red: #F95D6A;
        --pink: #D45087;
        --orange-light: #FF7C43;
        --background: #F8F9FA;
        --text-dark: #212529;
        --text-light: #6C757D;
    }
    
    /* Global styles */
    .main > div {
        padding-top: 1rem;
        max-width: 100% !important;
    }
    
    .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
    
    /* Modern header */
    .system-header {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 3rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,63,92,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .system-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.05); opacity: 0.8; }
    }
    
    .system-header h1 {
        font-size: 3rem;
        font-weight: 900;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .system-header p {
        font-size: 1.3rem;
        margin: 1rem 0 0 0;
        opacity: 0.9;
        position: relative;
        z-index: 1;
    }
    
    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }
    
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--orange) 0%, var(--pink) 100%);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
        border: 2px solid var(--orange);
    }
    
    .feature-card h3 {
        color: var(--secondary-blue);
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .feature-card p {
        color: var(--text-light);
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Pipeline selector */
    .pipeline-selector {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 2px solid var(--background);
    }
    
    .pipeline-option {
        background: linear-gradient(135deg, var(--background) 0%, #E9ECEF 100%);
        border: 2px solid transparent;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .pipeline-option.active {
        border-color: var(--orange);
        background: linear-gradient(135deg, var(--orange) 0%, var(--orange-light) 100%);
        color: white;
        box-shadow: 0 6px 20px rgba(255,166,0,0.3);
    }
    
    .pipeline-option:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Content pillar cards */
    .pillar-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .pillar-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid transparent;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .pillar-card.news {
        border-left: 4px solid var(--red);
    }
    
    .pillar-card.tools {
        border-left: 4px solid var(--orange);
    }
    
    .pillar-card.deep-dive {
        border-left: 4px solid var(--accent-blue);
    }
    
    .pillar-card.active {
        border-color: var(--orange);
        background: linear-gradient(135deg, rgba(255,166,0,0.1) 0%, rgba(255,166,0,0.05) 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255,166,0,0.2);
    }
    
    /* Modern buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--orange) 0%, var(--orange-light) 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        box-shadow: 0 4px 15px rgba(255,166,0,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255,166,0,0.4);
        border: 2px solid var(--orange);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Form inputs */
    .stSelectbox > div > div,
    .stTextInput > div > div,
    .stTextArea > div > div {
        background: white;
        border-radius: 12px;
        border: 2px solid var(--background);
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stSelectbox > div > div:hover,
    .stTextInput > div > div:hover,
    .stTextArea > div > div:hover {
        border-color: var(--orange);
        box-shadow: 0 4px 15px rgba(255,166,0,0.1);
    }
    
    /* Quality metrics */
    .quality-metrics {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--secondary-blue) 0%, var(--accent-blue) 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(47,75,124,0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .status-indicator.active {
        background: var(--orange);
    }
    
    .status-indicator.success {
        background: #28a745;
    }
    
    .status-indicator.warning {
        background: var(--red);
    }
    
    /* Content display */
    .content-display {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 2px solid var(--background);
    }
    
    .content-display h2 {
        color: var(--secondary-blue);
        border-bottom: 2px solid var(--orange);
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--orange) 0%, var(--orange-light) 100%);
        border-radius: 10px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--background);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px;
        margin: 0.2rem;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--orange);
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--orange) !important;
        color: white !important;
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
if 'selected_pillar' not in st.session_state:
    st.session_state.selected_pillar = 'news_breakthroughs'
if 'quality_metrics' not in st.session_state:
    st.session_state.quality_metrics = None

def create_system_header():
    """Create the modern system header"""
    st.markdown("""
    <div class="system-header">
        <h1>🚀 Newsletter System</h1>
        <p>AI-powered newsletter generation with hierarchical deep-dive analysis</p>
    </div>
    """, unsafe_allow_html=True)

def create_feature_overview():
    """Create feature overview cards"""
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <h3>🔬 Deep Dive Analysis</h3>
            <p>Comprehensive 4,000+ word technical articles with research, analysis, and expert insights. Weekly in-depth exploration of complex topics.</p>
        </div>
        <div class="feature-card">
            <h3>🤖 Hierarchical Execution</h3>
            <p>ManagerAgent orchestrates specialized agents for research, writing, and editing. Coordinated workflow ensures high-quality output.</p>
        </div>
        <div class="feature-card">
            <h3>📊 Quality Assurance</h3>
            <p>Real-time quality monitoring with technical accuracy validation, mobile readability scoring, and performance analytics.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_pipeline_overview():
    """Create pipeline overview interface"""
    st.markdown("""
    <div class="pipeline-selector">
        <h2 style="color: var(--secondary-blue); margin-bottom: 1.5rem;">🔬 Deep Dive Pipeline</h2>
        <p style="color: var(--text-light); margin-bottom: 2rem;">Comprehensive newsletter generation with hierarchical agent execution</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="pipeline-option active">
        <h3>🔬 Deep Dive Pipeline</h3>
        <p><strong>Target:</strong> 4,000+ words • <strong>Format:</strong> Comprehensive</p>
        <p><strong>Content:</strong> Technical analysis, research, expert insights</p>
        <p><strong>Generation Time:</strong> ~15 minutes</p>
        <p><strong>Agents:</strong> Manager → Planner → Research → Writer → Editor</p>
    </div>
    """, unsafe_allow_html=True)

def create_content_pillar_selector():
    """Create content pillar selection interface"""
    st.markdown("""
    <div class="pipeline-selector">
        <h2 style="color: var(--secondary-blue); margin-bottom: 1.5rem;">📚 Content Pillar Selection</h2>
        <p style="color: var(--text-light); margin-bottom: 2rem;">Select your newsletter's focus area for targeted content generation</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        news_active = "active" if st.session_state.selected_pillar == 'news_breakthroughs' else ""
        if st.button("📰 News & Breakthroughs", key="news_btn", use_container_width=True):
            st.session_state.selected_pillar = 'news_breakthroughs'
            st.rerun()
        
        st.markdown(f"""
        <div class="pillar-card news {news_active}">
            <h4>📰 News & Breakthroughs</h4>
            <p>Latest industry developments, research findings, and technical breakthroughs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        tools_active = "active" if st.session_state.selected_pillar == 'tools_tutorials' else ""
        if st.button("🛠️ Tools & Tutorials", key="tools_btn", use_container_width=True):
            st.session_state.selected_pillar = 'tools_tutorials'
            st.rerun()
        
        st.markdown(f"""
        <div class="pillar-card tools {tools_active}">
            <h4>🛠️ Tools & Tutorials</h4>
            <p>AI/ML tools, frameworks, practical guides, and implementation tutorials</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        deep_active = "active" if st.session_state.selected_pillar == 'deep_dives' else ""
        if st.button("🔬 Deep Dives & Analysis", key="analysis_btn", use_container_width=True):
            st.session_state.selected_pillar = 'deep_dives'
            st.rerun()
        
        st.markdown(f"""
        <div class="pillar-card deep-dive {deep_active}">
            <h4>🔬 Deep Dives & Analysis</h4>
            <p>Comprehensive technical analysis, research surveys, and expert insights</p>
        </div>
        """, unsafe_allow_html=True)

def create_configuration_panel():
    """Create configuration panel for newsletter settings"""
    st.markdown("""
    <div class="pipeline-selector">
        <h2 style="color: var(--secondary-blue); margin-bottom: 1.5rem;">⚙️ Configuration</h2>
        <p style="color: var(--text-light); margin-bottom: 2rem;">Customize your newsletter generation settings</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        topic = st.text_input(
            "📝 Newsletter Topic",
            placeholder="Enter your newsletter topic...",
            help="Main subject for your newsletter content"
        )
        
        audience_options = [
            "AI/ML Engineers",
            "Data Scientists", 
            "Software Developers",
            "Technical Leaders",
            "Research Community",
            "Business Professionals",
            "General Tech Audience"
        ]
        
        audience = st.selectbox(
            "👥 Target Audience",
            audience_options,
            help="Select your target audience for content optimization"
        )
        
        word_count = st.slider(
            "📊 Target Word Count",
            min_value=1000,
            max_value=5000,
            value=4000,
            step=250,
            help="Target word count for the newsletter"
        )
    
    with col2:
        priority = st.selectbox(
            "🎯 Priority Level",
            ["High", "Medium", "Low"],
            index=1,
            help="Priority level for content generation"
        )
        
        special_requirements = st.multiselect(
            "🔧 Special Requirements",
            ["Include code examples", "Academic citations", "Industry case studies", 
             "Performance benchmarks", "Mobile optimization", "Visual elements"],
            help="Additional requirements for your newsletter"
        )
        
        quality_threshold = st.slider(
            "📈 Quality Threshold",
            min_value=0.7,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Minimum quality score for publication"
        )
    
    return {
        'topic': topic,
        'audience': audience,
        'word_count': word_count,
        'priority': priority,
        'special_requirements': special_requirements,
        'quality_threshold': quality_threshold
    }

def create_quality_dashboard():
    """Create quality assurance dashboard"""
    if st.session_state.quality_metrics:
        st.markdown("""
        <div class="quality-metrics">
            <h2 style="color: var(--secondary-blue); margin-bottom: 1.5rem;">📊 Quality Assurance Dashboard</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        
        metrics = st.session_state.quality_metrics
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('technical_accuracy', 0):.1%}</div>
                <div class="metric-label">Technical Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('mobile_readability', 0):.1%}</div>
                <div class="metric-label">Mobile Readability</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('code_validation', 0):.1%}</div>
                <div class="metric-label">Code Validation</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('overall_quality', 0):.1%}</div>
                <div class="metric-label">Overall Quality</div>
            </div>
            """, unsafe_allow_html=True)

def generate_newsletter(config: Dict[str, Any]):
    """Generate newsletter using the hierarchical system"""
    try:
        # Initialize quality system
        quality_system = QualityAssuranceSystem()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Generate content using hierarchical execution
        status_text.text("🔄 Executing hierarchical newsletter generation...")
        progress_bar.progress(10)
        
        # Execute hierarchical newsletter generation
        generation_result = execute_hierarchical_newsletter_generation(
            topic=config['topic'],
            audience=config['audience']
        )
        
        progress_bar.progress(70)
        
        # Step 2: Quality assurance
        status_text.text("🔍 Running quality assurance checks...")
        
        if generation_result and isinstance(generation_result, dict):
            try:
                ready, validation_report = quality_system.validate_newsletter_ready_for_publish(
                    generation_result,
                    'deep_dive'  # Only deep dive pipeline now
                )
                
                quality_metrics_obj = validation_report.get('quality_metrics')

                if quality_metrics_obj:
                    st.session_state.quality_metrics = {
                        'technical_accuracy': quality_metrics_obj.technical_accuracy_score,
                        'mobile_readability': quality_metrics_obj.mobile_readability_score,
                        'code_validation': quality_metrics_obj.code_validation_score,
                        'overall_quality': quality_metrics_obj.overall_quality_score
                    }
                else:
                    st.session_state.quality_metrics = {}
            except Exception as e:
                st.warning(f"Quality assurance check failed: {str(e)}")
                st.session_state.quality_metrics = {}
        
        progress_bar.progress(100)
        status_text.text("✅ Newsletter generation complete!")
        
        return generation_result
        
    except Exception as e:
        st.error(f"Error during newsletter generation: {str(e)}")
        return None

def display_newsletter_content(content: Dict[str, Any]):
    """Display generated newsletter content"""
    st.markdown("""
    <div class="content-display">
        <h2>📰 Generated Newsletter Content</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if content:
        # Create tabs for different content sections
        tabs = st.tabs(["📰 Newsletter", "📊 Quality Report"])
        
        with tabs[0]:
            newsletter_content = content.get('content', '')
            if newsletter_content:
                st.markdown(newsletter_content)
            else:
                st.warning("No newsletter content generated. This might be due to configuration issues.")
        
        with tabs[1]:
            if st.session_state.quality_metrics:
                st.markdown("### Quality Assurance Report")
                for metric, score in st.session_state.quality_metrics.items():
                    st.metric(metric.replace('_', ' ').title(), f"{score:.1%}")
            else:
                st.info("Quality metrics not available for this generation.")
    else:
        st.warning("No content to display. Generate a newsletter to see results.")

def save_newsletter(content: Dict[str, Any], topic: str):
    """Save newsletter to file."""
    try:
        st.info("💾 Saving newsletter...")
        
        # Save to file
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '_')).rstrip()
        filename = f"{timestamp}_{safe_topic}.md"
        filepath = output_dir / filename
        
        newsletter_content = content.get('content', '')
        if newsletter_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(newsletter_content)
            st.success(f"✅ Newsletter saved to `{filepath}`")
            
            # Show file info
            st.info(f"📄 File: {filename}")
            st.info(f"📍 Location: {filepath.absolute()}")
            st.info(f"📊 Size: {len(newsletter_content):,} characters")
        else:
            st.error("❌ No newsletter content to save")
            
    except Exception as e:
        st.error(f"An error occurred during saving: {str(e)}")

def main():
    """Main application function"""
    # System header
    create_system_header()
    
    # Feature overview
    create_feature_overview()
    
    # Pipeline overview
    create_pipeline_overview()
    
    # Content pillar selection
    create_content_pillar_selector()
    
    # Configuration panel
    config = create_configuration_panel()
    
    # Quality dashboard
    create_quality_dashboard()
    
    # Generation section
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Generate Newsletter", use_container_width=True, disabled=st.session_state.is_generating):
            if config['topic']:
                st.session_state.is_generating = True
                
                with st.spinner("Generating newsletter..."):
                    result = generate_newsletter(config)
                    
                    if result:
                        st.session_state.newsletter_output = result
                        st.success("Newsletter generated successfully!")
                    else:
                        st.error("Failed to generate newsletter. Please try again.")
                
                st.session_state.is_generating = False
                st.rerun()
            else:
                st.error("Please enter a newsletter topic to continue.")
    
    # Display content
    if st.session_state.newsletter_output:
        display_newsletter_content(st.session_state.newsletter_output)
        
        st.markdown("---")
        
        if st.button("💾 Save Newsletter", use_container_width=True):
            save_newsletter(st.session_state.newsletter_output, config['topic'])

if __name__ == "__main__":
    main() 