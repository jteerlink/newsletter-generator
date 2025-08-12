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
    
    /* Modern glassmorphism background */
    .stApp {
        background: linear-gradient(135deg, 
            rgba(0,63,92,0.05) 0%, 
            rgba(47,75,124,0.05) 25%,
            rgba(255,166,0,0.05) 50%,
            rgba(212,80,135,0.05) 75%,
            rgba(102,81,145,0.05) 100%);
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
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        border: 1px solid rgba(255,255,255,0.2);
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
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(255,166,0,0.15);
        border: 1px solid rgba(255,166,0,0.3);
        background: rgba(255, 255, 255, 0.98);
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
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    
    /* Modern buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--orange) 0%, var(--orange-light) 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 16px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        border: 1px solid transparent;
        box-shadow: 0 8px 25px rgba(255,166,0,0.25);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 12px 35px rgba(255,166,0,0.3);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .stButton > button:active {
        transform: translateY(-2px) scale(1.01);
    }
    
    /* Form inputs */
    .stSelectbox > div > div,
    .stTextInput > div > div,
    .stTextArea > div > div {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    
    .stSelectbox > div > div:hover,
    .stTextInput > div > div:hover,
    .stTextArea > div > div:hover {
        border-color: rgba(255,166,0,0.3);
        box-shadow: 0 8px 30px rgba(255,166,0,0.1);
        background: rgba(255, 255, 255, 0.95);
        transform: translateY(-2px);
    }
    
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div:focus-within,
    .stTextArea > div > div:focus-within {
        border-color: var(--orange);
        box-shadow: 0 0 0 3px rgba(255,166,0,0.1);
        background: white;
    }
    
    /* Quality metrics */
    .quality-metrics {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--secondary-blue) 0%, var(--accent-blue) 100%);
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 25px rgba(47,75,124,0.25);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: subtle-pulse 3s ease-in-out infinite;
    }
    
    .metric-card:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 35px rgba(47,75,124,0.3);
    }
    
    @keyframes subtle-pulse {
        0%, 100% { transform: scale(1) rotate(0deg); opacity: 0.3; }
        50% { transform: scale(1.1) rotate(180deg); opacity: 0.6; }
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 500;
        position: relative;
        z-index: 1;
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
    
    /* Modern loading animation */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 3rem;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 4px solid rgba(255,166,0,0.2);
        border-top: 4px solid var(--orange);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 1rem;
    }
    
    .loading-dots {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
    }
    
    .loading-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--orange);
        animation: bounce 1.4s ease-in-out infinite both;
    }
    
    .loading-dot:nth-child(1) { animation-delay: -0.32s; }
    .loading-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes bounce {
        0%, 80%, 100% { 
            transform: scale(0.8);
            opacity: 0.5;
        } 
        40% { 
            transform: scale(1.2);
            opacity: 1;
        }
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(248,249,250,0.8);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 0.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.7);
        border-radius: 12px;
        margin: 0.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,166,0,0.1);
        color: var(--orange);
        border: 1px solid rgba(255,166,0,0.2);
        transform: translateY(-1px);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--orange) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(255,166,0,0.3);
        transform: translateY(-2px);
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
if 'quality_metrics' not in st.session_state:
    st.session_state.quality_metrics = None

def create_system_header():
    """Create the modern system header"""
    st.markdown("""
    <div class="system-header">
        <h1>🚀 Newsletter Generator</h1>
        <p>AI-powered newsletter creation with intelligent content generation</p>
    </div>
    """, unsafe_allow_html=True)

def create_loading_animation(message="Generating newsletter..."):
    """Create modern loading animation"""
    return st.markdown(f"""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <h3 style="color: var(--secondary-blue); margin: 0;">{message}</h3>
        <div class="loading-dots">
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_feature_overview():
    """Create feature overview cards"""
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <h3>🤖 AI-Powered Generation</h3>
            <p>Advanced language models create high-quality newsletter content tailored to your audience and focus area.</p>
        </div>
        <div class="feature-card">
            <h3>🎯 Targeted Content</h3>
            <p>Choose your content focus area for personalized newsletters that match your audience's interests and expertise level.</p>
        </div>
        <div class="feature-card">
            <h3>📊 Quality Assurance</h3>
            <p>Built-in quality monitoring with technical accuracy validation, readability scoring, and content optimization.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_setup_explanation():
    """Create friendly setup explanation"""
    st.markdown("""
    <div class="pipeline-selector">
        <h2 style="color: var(--secondary-blue); margin-bottom: 1rem;">🚀 Ready to Create Your Newsletter?</h2>
        <p style="color: var(--text-dark); font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.5rem;">
            Getting started is easy! Just fill out the configuration below with your preferences. 
            Choose your content focus, add a topic, select your audience, and we'll handle the rest. 
            The AI will generate a personalized newsletter that matches your specifications perfectly! ✨
        </p>
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
        # Content focus selection
        focus_options = [
            "General Tech News",
            "AI/ML Developments", 
            "Software Engineering",
            "Data Science",
            "Industry Analysis"
        ]
        
        content_focus = st.selectbox(
            "🎯 Content Focus",
            focus_options,
            index=1,  # Default to AI/ML
            help="Choose the primary focus area for your newsletter content"
        )
        
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
    
    with col2:
        word_count = st.slider(
            "📊 Target Word Count",
            min_value=1000,
            max_value=5000,
            value=3000,
            step=250,
            help="Target word count for the newsletter"
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
        'content_focus': content_focus,
        'topic': topic,
        'audience': audience,
        'word_count': word_count,
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
                    'standard'  # Standard newsletter pipeline
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
    
    # Setup explanation
    create_setup_explanation()
    
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
                
                # Show modern loading animation
                loading_placeholder = st.empty()
                with loading_placeholder:
                    create_loading_animation("🤖 Executing hierarchical generation...")
                
                result = generate_newsletter(config)
                
                # Clear loading animation
                loading_placeholder.empty()
                
                if result:
                    st.session_state.newsletter_output = result
                    st.success("✅ Newsletter generated successfully!")
                else:
                    st.error("❌ Failed to generate newsletter. Please try again.")
                
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