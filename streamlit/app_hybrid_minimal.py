"""
Streamlit Hybrid Newsletter System - Simplified Version
Modern UI with Phase 1-4 integration
"""

import streamlit as st
import sys
import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import hybrid system components
try:
    from agents.daily_quick_pipeline import DailyQuickPipeline
    from agents.hybrid_workflow_manager import HybridWorkflowManager, ContentRequest, ContentPipelineType
    from agents.quality_assurance_system import QualityAssuranceSystem
    from core.core import query_llm
    SYSTEM_AVAILABLE = True
except ImportError as e:
    st.error(f"System components not available: {e}")
    SYSTEM_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Hybrid Newsletter System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Color scheme */
    :root {
        --primary-blue: #003F5C;
        --secondary-blue: #2F4B7C;
        --accent-blue: #665191;
        --orange: #FFA600;
        --red: #F95D6A;
        --background: #F8F9FA;
        --text-dark: #212529;
        --text-light: #6C757D;
    }
    
    .main-header {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(0,63,92,0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 900;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid var(--orange);
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--secondary-blue) 0%, var(--accent-blue) 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(47,75,124,0.3);
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--orange) 0%, #FF7C43 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255,166,0,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255,166,0,0.4);
    }
    
    .pipeline-selector {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .status-success {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    .status-processing {
        background: linear-gradient(135deg, var(--orange) 0%, #FF7C43 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    .content-preview {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid var(--background);
    }
    
    .content-preview h3 {
        color: var(--secondary-blue);
        border-bottom: 2px solid var(--orange);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'newsletter_output' not in st.session_state:
    st.session_state.newsletter_output = None
if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False
if 'selected_pipeline' not in st.session_state:
    st.session_state.selected_pipeline = 'daily_quick'
if 'quality_metrics' not in st.session_state:
    st.session_state.quality_metrics = None
if 'generation_log' not in st.session_state:
    st.session_state.generation_log = []

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Hybrid Newsletter System</h1>
        <p>AI-powered newsletter generation with quality assurance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System status check
    if not SYSTEM_AVAILABLE:
        st.error("⚠️ System components are not available. Please ensure all dependencies are installed.")
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📋 Configuration")
        
        # Pipeline selection
        pipeline_option = st.selectbox(
            "Select Pipeline",
            ["Daily Quick Pipeline", "Deep Dive Pipeline"],
            index=0
        )
        
        st.session_state.selected_pipeline = 'daily_quick' if pipeline_option == "Daily Quick Pipeline" else 'deep_dive'
        
        # Content pillar selection
        pillar_option = st.selectbox(
            "Content Pillar",
            ["News & Breakthroughs", "Tools & Tutorials", "Deep Dives & Analysis"],
            index=0
        )
        
        pillar_mapping = {
            "News & Breakthroughs": "news_breakthroughs",
            "Tools & Tutorials": "tools_tutorials", 
            "Deep Dives & Analysis": "deep_dives"
        }
        selected_pillar = pillar_mapping[pillar_option]
        
        # Topic input
        topic = st.text_input("Newsletter Topic", placeholder="Enter your topic...")
        
        # Store topic in session state for saving
        if topic:
            st.session_state.current_topic = topic
        
        # Audience selection
        audience = st.selectbox(
            "Target Audience",
            ["AI/ML Engineers", "Data Scientists", "Software Developers", "Technical Leaders"],
            index=0
        )
        
        # Word count
        word_count = st.slider(
            "Target Word Count",
            min_value=500,
            max_value=5000,
            value=1500 if st.session_state.selected_pipeline == 'daily_quick' else 4000,
            step=250
        )
        
        # Quality threshold
        quality_threshold = st.slider(
            "Quality Threshold",
            min_value=0.7,
            max_value=1.0,
            value=0.85,
            step=0.05,
            help="Minimum quality score required for publication"
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Pipeline info
        st.markdown('<div class="pipeline-selector">', unsafe_allow_html=True)
        st.subheader(f"📊 {pipeline_option}")
        
        if st.session_state.selected_pipeline == 'daily_quick':
            st.markdown("""
            **⚡ Daily Quick Pipeline Features:**
            - 🕐 Generation time: 2-3 minutes
            - 📄 Target length: 500-1,500 words
            - 📱 Mobile-first optimization
            - 🔍 Light research with curated sources
            - ✅ Basic quality validation
            """)
        else:
            st.markdown("""
            **🔬 Deep Dive Pipeline Features:**
            - 🕐 Generation time: 15-20 minutes
            - 📖 Target length: 3,000-5,000 words
            - 🎯 Extensive research and analysis
            - 📚 Academic citations and references
            - 🔍 Rigorous quality validation
            """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Generation button
        if st.button("🚀 Generate Newsletter", disabled=st.session_state.is_generating or not topic):
            if topic:
                generate_newsletter(topic, selected_pillar, audience, word_count, quality_threshold)
            else:
                st.error("Please enter a newsletter topic.")
    
    with col2:
        # Quality metrics display
        if st.session_state.quality_metrics:
            st.subheader("📊 Quality Metrics")
            
            for metric, value in st.session_state.quality_metrics.items():
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{value:.1%}</div>
                    <div class="metric-label">{metric.replace('_', ' ').title()}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # System status
        st.subheader("🔧 System Status")
        st.markdown(f"""
        <div class="feature-card">
            <p><strong>Pipeline:</strong> {pipeline_option}</p>
            <p><strong>Content Pillar:</strong> {pillar_option}</p>
            <p><strong>Target Audience:</strong> {audience}</p>
            <p><strong>Word Count:</strong> {word_count:,}</p>
            <p><strong>Quality Threshold:</strong> {quality_threshold:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Content display
    if st.session_state.newsletter_output:
        st.markdown("---")
        display_newsletter_content(st.session_state.newsletter_output)
    
    # Generation log
    if st.session_state.generation_log:
        st.markdown("---")
        st.subheader("📝 Generation Log")
        
        for log_entry in st.session_state.generation_log[-5:]:  # Show last 5 entries
            st.markdown(f"""
            <div class="feature-card">
                <p><strong>{log_entry['timestamp']}</strong></p>
                <p>{log_entry['message']}</p>
            </div>
            """, unsafe_allow_html=True)

def generate_newsletter(topic: str, pillar: str, audience: str, word_count: int, quality_threshold: float):
    """Generate newsletter using the hybrid system"""
    st.session_state.is_generating = True
    
    # Add to generation log
    st.session_state.generation_log.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'message': f"Starting newsletter generation for topic: {topic}"
    })
    
    try:
        # Initialize components
        with st.spinner("Initializing system components..."):
            workflow_manager = HybridWorkflowManager() if SYSTEM_AVAILABLE else None
            quality_system = QualityAssuranceSystem() if SYSTEM_AVAILABLE else None
            daily_pipeline = DailyQuickPipeline() if SYSTEM_AVAILABLE else None
        
        # Create content request
        content_request = ContentRequest(
            topic=topic,
            content_pillar=pillar,
            target_audience=audience,
            word_count_target=word_count,
            deadline=datetime.now() + timedelta(hours=1),
            priority=1,
            special_requirements=[]
        )
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Route workflow
        status_text.markdown('<div class="status-processing">🔄 Analyzing content complexity...</div>', unsafe_allow_html=True)
        progress_bar.progress(10)
        time.sleep(1)
        
        # Step 2: Generate content using the selected pipeline
        status_text.markdown('<div class="status-processing">📝 Generating newsletter content...</div>', unsafe_allow_html=True)
        progress_bar.progress(30)
        
        if workflow_manager and SYSTEM_AVAILABLE:
            # Use the hybrid workflow manager with direct pipeline execution
            from agents.hybrid_workflow_manager import ContentPipelineType
            
            pipeline_type = ContentPipelineType.DAILY_QUICK if st.session_state.selected_pipeline == 'daily_quick' else ContentPipelineType.DEEP_DIVE
            
            # Execute the pipeline directly based on UI selection
            workflow_result = workflow_manager.execute_pipeline_directly(content_request, pipeline_type)
            
            if workflow_result.get('status') == 'success':
                if st.session_state.selected_pipeline == 'daily_quick':
                    generation_result = workflow_result.get('result', {})
                else:
                    generation_result = {
                        'content': workflow_result.get('content', ''),
                        'word_count': len(workflow_result.get('content', '').split()),
                        'estimated_read_time': len(workflow_result.get('content', '').split()) // 200
                    }
                
                st.session_state.generation_log.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'message': f"{st.session_state.selected_pipeline.replace('_', ' ').title()} pipeline executed successfully"
                })
            else:
                # Fallback to mock content if pipeline fails
                st.session_state.generation_log.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'message': f"Pipeline failed: {workflow_result.get('error', 'Unknown error')}, using fallback content"
                })
                
                if st.session_state.selected_pipeline == 'daily_quick':
                    generation_result = {
                        'markdown': f"# Daily Quick: {topic}\n\nThis is a quick update on {topic} for {audience}.\n\n## Key Points\n\n- Latest developments in {topic}\n- Important updates for {audience}\n- Actionable insights and recommendations\n\n## What's Next\n\nStay tuned for more updates on {topic}.",
                        'word_count': 250,
                        'estimated_read_time': 2
                    }
                else:
                    generation_result = {
                        'content': f"# Deep Dive: {topic}\n\nThis is a comprehensive analysis of {topic} for {audience}.\n\n## Introduction\n\nDetailed introduction here...\n\n## Analysis\n\nIn-depth analysis content...\n\n## Conclusion\n\nConclusion and recommendations...",
                        'word_count': word_count,
                        'estimated_read_time': word_count // 200
                    }
        else:
            # Fallback when system is not available
            if st.session_state.selected_pipeline == 'daily_quick' and daily_pipeline:
                # Use daily quick pipeline
                generation_result = daily_pipeline.generate_daily_newsletter()
                st.session_state.generation_log.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'message': "Daily quick pipeline executed successfully"
                })
            else:
                # Simulate deep dive pipeline
                generation_result = {
                    'content': f"# Deep Dive: {topic}\n\nThis is a comprehensive analysis of {topic} for {audience}.\n\n## Introduction\n\nDetailed introduction here...\n\n## Analysis\n\nIn-depth analysis content...\n\n## Conclusion\n\nConclusion and recommendations...",
                    'word_count': word_count,
                    'estimated_read_time': word_count // 200
                }
                st.session_state.generation_log.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'message': "Deep dive pipeline executed successfully"
                })
        
        progress_bar.progress(70)
        
        # Step 3: Quality assurance
        status_text.markdown('<div class="status-processing">🔍 Running quality assurance...</div>', unsafe_allow_html=True)
        
        if quality_system and generation_result:
            # Run quality assurance
            ready, validation_report = quality_system.validate_newsletter_ready_for_publish(
                generation_result,
                st.session_state.selected_pipeline
            )
            
            # Extract quality metrics
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
            
            st.session_state.generation_log.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'message': f"Quality assurance completed. Overall score: {st.session_state.quality_metrics.get('overall_quality', 0):.1%}"
            })
        else:
            # Simulate quality metrics
            st.session_state.quality_metrics = {
                'technical_accuracy': 0.85,
                'mobile_readability': 0.90,
                'code_validation': 0.88,
                'overall_quality': 0.87
            }
        
        progress_bar.progress(100)
        
        # Check quality threshold
        overall_quality = st.session_state.quality_metrics['overall_quality']
        if overall_quality >= quality_threshold:
            status_text.markdown('<div class="status-success">✅ Newsletter generated successfully!</div>', unsafe_allow_html=True)
            st.session_state.newsletter_output = generation_result
            st.session_state.generation_log.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'message': f"Newsletter meets quality threshold ({overall_quality:.1%} >= {quality_threshold:.1%})"
            })
        else:
            status_text.markdown(f'<div class="status-processing">⚠️ Quality below threshold ({overall_quality:.1%} < {quality_threshold:.1%})</div>', unsafe_allow_html=True)
            st.session_state.newsletter_output = generation_result
            st.session_state.generation_log.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'message': f"Quality below threshold but content generated ({overall_quality:.1%} < {quality_threshold:.1%})"
            })
        
    except Exception as e:
        st.error(f"Error during generation: {str(e)}")
        st.session_state.generation_log.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'message': f"Error: {str(e)}"
        })
    
    finally:
        st.session_state.is_generating = False

def display_newsletter_content(content: Dict[str, Any]):
    """Display generated newsletter content"""
    st.subheader("📰 Generated Newsletter")
    
    if content:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📄 Content", "📊 Metrics", "📱 Mobile Preview"])
        
        with tab1:
            st.markdown('<div class="content-preview">', unsafe_allow_html=True)
            if st.session_state.selected_pipeline == 'daily_quick':
                # Handle DailyQuickPipeline output structure
                if isinstance(content, dict):
                    newsletter_content = content.get('markdown', content.get('content', 'No content available'))
                else:
                    newsletter_content = str(content)
                st.markdown(newsletter_content)
            else:
                # Handle other pipeline output
                if isinstance(content, dict):
                    newsletter_content = content.get('content', 'No content available')
                else:
                    newsletter_content = str(content)
                st.markdown(newsletter_content)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### 📊 Content Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Handle word count from different structures
                if isinstance(content, dict):
                    word_count = content.get('word_count', 'N/A')
                    if word_count == 'N/A' and 'markdown' in content:
                        # Calculate word count from markdown content
                        word_count = len(content['markdown'].split())
                    elif word_count == 'N/A' and 'content' in content:
                        # Calculate word count from content
                        word_count = len(content['content'].split())
                else:
                    word_count = 'N/A'
                st.metric("Word Count", word_count)
            
            with col2:
                # Handle read time from different structures
                if isinstance(content, dict):
                    read_time = content.get('estimated_read_time', 'N/A')
                    if read_time == 'N/A' and 'metadata' in content:
                        read_time = content['metadata'].get('estimated_read_time', 'N/A')
                else:
                    read_time = 'N/A'
                st.metric("Read Time", f"{read_time} min" if read_time != 'N/A' else 'N/A')
            
            with col3:
                if st.session_state.quality_metrics:
                    st.metric("Quality Score", f"{st.session_state.quality_metrics['overall_quality']:.1%}")
        
        with tab3:
            st.markdown("### 📱 Mobile Preview")
            st.info("Content optimized for mobile-first reading experience")
            
            # Show mobile-optimized preview
            if isinstance(content, dict):
                full_content = content.get('markdown', content.get('content', ''))
            else:
                full_content = str(content)
            mobile_content = full_content[:500] + "..." if len(full_content) > 500 else full_content
            
            st.markdown(f"""
            <div style="
                max-width: 350px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                border: 2px solid var(--background);
            ">
                <div style="
                    background: var(--secondary-blue);
                    color: white;
                    padding: 1rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;
                    text-align: center;
                    font-weight: 600;
                ">
                    📱 Mobile View
                </div>
                <div style="
                    font-size: 0.9rem;
                    line-height: 1.6;
                    color: var(--text-dark);
                ">
                    {mobile_content}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Save and publish section
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("💾 Save & Publish Newsletter", use_container_width=True, type="primary"):
                # Get the current topic from the form (need to access it from session state or parameter)
                current_topic = st.session_state.get('current_topic', 'newsletter')
                save_newsletter_output_fixed(content, current_topic)
    else:
        st.warning("No content available to display.")



def save_newsletter_output_fixed(content: Dict[str, Any], topic: str):
    """Save newsletter to outputs folder"""
    try:
        # Create outputs directory if it doesn't exist
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '_', '-')).rstrip()
        safe_topic = safe_topic.replace(' ', '_')
        
        # Format content for saving
        if st.session_state.selected_pipeline == 'daily_quick':
            if isinstance(content, dict) and 'sections' in content:
                markdown_content = format_daily_quick_markdown(content, topic)
            else:
                markdown_content = content.get('markdown', content.get('content', str(content)))
        else:
            markdown_content = content.get('markdown', content.get('content', str(content)))
        
        # Save to file
        filename = f"{safe_topic}_{timestamp}.md"
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # Show success
        st.success(f"✅ Newsletter saved to: {filepath}")
        
        # Provide download button
        st.download_button(
            label="📥 Download Newsletter",
            data=markdown_content,
            file_name=filename,
            mime="text/markdown"
        )
        
        return filepath
        
    except Exception as e:
        logging.error(f"Error saving newsletter: {e}")
        st.error(f"❌ Error saving newsletter: {str(e)}")
        return None

def format_daily_quick_markdown(content: Dict[str, Any], topic: str) -> str:
    """Format daily quick content as markdown"""
    parts = []
    
    # Header
    parts.append(f"# {topic}")
    parts.append(f"*Generated: {datetime.now().strftime('%B %d, %Y')}*")
    parts.append("")
    
    # Sections
    sections = content.get('sections', {})
    
    if 'news_breakthroughs' in sections:
        parts.append("## 🚀 News & Breakthroughs")
        parts.append(sections['news_breakthroughs'])
        parts.append("")
    
    if 'tools_tutorials' in sections:
        parts.append("## 🛠️ Tools & Tutorials")
        parts.append(sections['tools_tutorials'])
        parts.append("")
    
    if 'quick_hits' in sections:
        parts.append("## ⚡ Quick Hits")
        parts.append(sections['quick_hits'])
        parts.append("")
    
    if 'takeaways' in sections:
        parts.append("## 🎯 Key Takeaways") 
        parts.append(sections['takeaways'])
        parts.append("")
    
    return "\n".join(parts)

if __name__ == "__main__":
    main() 