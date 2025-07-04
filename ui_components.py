"""
Reusable UI Components for Streamlit Newsletter Generator
Modern, responsive components for enhanced user experience
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Any, Optional

def create_header():
    """Create a modern header with branding and navigation"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    ">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">
            🚀 AI Newsletter Generator
        </h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            Create comprehensive, engaging newsletters with AI-powered research and writing
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_feature_cards():
    """Create feature highlight cards with proper column distribution"""
    
    # Create a container with proper styling
    st.markdown("""
    <div style="
        margin: 2rem 0;
        padding: 0;
    ">
    """, unsafe_allow_html=True)
    
    # Use Streamlit columns for proper distribution
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            color: white;
            margin: 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            width: 100%;
        ">
            <h3 style="margin: 0; font-size: 1.3rem; font-weight: 700;">🤖 AI-Powered</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">
                5 specialized agents work together
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            color: white;
            margin: 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            width: 100%;
        ">
            <h3 style="margin: 0; font-size: 1.3rem; font-weight: 700;">📊 Comprehensive</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">
                15-20k words of detailed content
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #fd7e14 0%, #e83e8c 100%);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            color: white;
            margin: 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            width: 100%;
        ">
            <h3 style="margin: 0; font-size: 1.3rem; font-weight: 700;">⚡ Fast</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">
                Generate in under 5 minutes
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def create_configuration_panel(sources_config: Dict, available_categories: List[str]) -> Dict[str, Any]:
    """Create the main configuration panel and return all settings"""
    
    # Configuration header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0 1rem 0;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 2px solid rgba(255,255,255,0.1);
    ">
        <h2 style="margin: 0; font-size: 1.8rem; font-weight: 700;">
            ⚙️ Configuration
        </h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1rem;">
            Customize your newsletter generation settings
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration content container
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 0 0 2rem 0;
        border: 2px solid #dee2e6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        width: 100%;
    ">
    """, unsafe_allow_html=True)
    
    # Main settings in columns with proper spacing
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div style="
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 0 0 1rem 0;
            border: 1px solid #dee2e6;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        ">
            <h4 style="margin: 0 0 1rem 0; color: #495057; font-weight: 600;">📝 Basic Settings</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Topic input with enhanced styling
        topic = st.text_input(
            "Newsletter Topic",
            placeholder="Enter your newsletter topic...",
            help="The main subject for your newsletter",
            key="topic_input"
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
        
        audience = st.selectbox(
            "Target Audience",
            audience_options,
            help="Select your target audience for content optimization"
        )
        
        # Workflow type
        workflow_type = st.selectbox(
            "Workflow Type",
            ["Standard Multi-Agent", "Hierarchical (Manager-Led)"],
            help="Choose between standard multi-agent workflow or hierarchical manager-led workflow"
        )
    
    with col2:
        st.markdown("""
        <div style="
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 0 0 1rem 0;
            border: 1px solid #dee2e6;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        ">
            <h4 style="margin: 0 0 1rem 0; color: #495057; font-weight: 600;">🎯 Content Settings</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Content length preference
        content_length = st.selectbox(
            "Content Length",
            ["Comprehensive (15-20k words)", "Standard (10-15k words)", "Concise (5-10k words)"],
            help="Choose the desired length for your newsletter"
        )
        
        # Quality focus areas
        quality_focus = st.multiselect(
            "Quality Focus Areas",
            ["Research Depth", "Writing Quality", "Engagement", "Technical Accuracy", "Practical Value"],
            default=["Research Depth", "Writing Quality", "Engagement"],
            help="Select areas to prioritize during generation"
        )
        
        # Source categories
        if available_categories:
            selected_categories = st.multiselect(
                "Source Categories",
                available_categories,
                default=available_categories[:3] if len(available_categories) > 3 else available_categories,
                help="Select which source categories to include"
            )
        else:
            selected_categories = []
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return {
        'topic': topic,
        'audience': audience,
        'workflow_type': workflow_type,
        'content_length': content_length,
        'quality_focus': quality_focus,
        'selected_categories': selected_categories
    }

def create_advanced_settings_panel() -> Dict[str, Any]:
    """Create advanced settings panel"""
    with st.expander("🔧 Advanced Settings", expanded=False):
        st.markdown("#### Fine-tune your newsletter generation")
        
        col1, col2 = st.columns(2)
        
        with col1:
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
        
        with col2:
            max_execution_time = st.slider(
                "Max Execution Time (minutes)",
                min_value=5,
                max_value=30,
                value=15,
                help="Maximum time to allow for generation"
            )
            
            enable_web_search = st.checkbox(
                "Enable Web Search",
                value=True,
                help="Allow agents to search the web for current information"
            )
            
            research_depth = st.selectbox(
                "Research Depth",
                ["Standard", "Deep", "Comprehensive"],
                index=1,
                help="How thorough should the research be?"
            )
    
    return {
        'collect_feedback': collect_feedback,
        'enable_quality_scoring': enable_quality_scoring,
        'save_intermediate_results': save_intermediate_results,
        'max_execution_time': max_execution_time,
        'enable_web_search': enable_web_search,
        'research_depth': research_depth
    }

def create_status_dashboard(sources_config: Dict, available_categories: List[str], is_generating: bool) -> None:
    """Create enhanced status dashboard with high visibility"""
    
    # Main status header with larger, more prominent styling
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 2px solid rgba(255,255,255,0.1);
    ">
        <h2 style="margin: 0; font-size: 1.8rem; font-weight: 700;">
            📊 System Status
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced status indicators with larger, more visible metrics
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #dee2e6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    ">
    """, unsafe_allow_html=True)
    
    # Status metrics in a 2x2 grid for better visibility
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 4px 12px rgba(0,123,255,0.3);
        ">
            <h3 style="margin: 0; font-size: 2rem; font-weight: 700;">{}</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;">Sources Available</p>
        </div>
        """.format(len(sources_config.get('sources', []))), unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 4px 12px rgba(23,162,184,0.3);
        ">
            <h3 style="margin: 0; font-size: 2rem; font-weight: 700;">5</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;">AI Agents</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 4px 12px rgba(40,167,69,0.3);
        ">
            <h3 style="margin: 0; font-size: 2rem; font-weight: 700;">{}</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;">Categories</p>
        </div>
        """.format(len(available_categories)), unsafe_allow_html=True)
        
        # Dynamic status indicator with high visibility
        if is_generating:
            status_bg = "linear-gradient(135deg, #dc3545 0%, #c82333 100%)"
            status_text = "GENERATING"
            status_icon = "⚡"
            status_shadow = "rgba(220,53,69,0.3)"
        else:
            status_bg = "linear-gradient(135deg, #28a745 0%, #1e7e34 100%)"
            status_text = "READY"
            status_icon = "✅"
            status_shadow = "rgba(40,167,69,0.3)"
        
        st.markdown(f"""
        <div style="
            background: {status_bg};
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 4px 12px {status_shadow};
            animation: pulse 2s infinite;
        ">
            <h3 style="margin: 0; font-size: 1.5rem; font-weight: 700;">{status_icon} {status_text}</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;">System Status</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Large, prominent system health indicator
    if is_generating:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin: 1.5rem 0;
            text-align: center;
            box-shadow: 0 8px 25px rgba(245,87,108,0.3);
            border: 3px solid rgba(255,255,255,0.2);
            animation: pulse 2s infinite;
        ">
            <h2 style="margin: 0; font-size: 2rem; font-weight: 700;">⚡ GENERATION IN PROGRESS</h2>
            <p style="margin: 1rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                Your newsletter is being generated... Please wait.
            </p>
            <div style="
                width: 100%;
                height: 6px;
                background: rgba(255,255,255,0.3);
                border-radius: 3px;
                margin: 1rem 0;
                overflow: hidden;
            ">
                <div style="
                    width: 100%;
                    height: 100%;
                    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.8), transparent);
                    animation: shimmer 2s infinite;
                "></div>
            </div>
        </div>
        
        <style>
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin: 1.5rem 0;
            text-align: center;
            box-shadow: 0 8px 25px rgba(86,171,47,0.3);
            border: 3px solid rgba(255,255,255,0.2);
        ">
            <h2 style="margin: 0; font-size: 2rem; font-weight: 700;">✅ SYSTEM READY</h2>
            <p style="margin: 1rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                All systems operational. Ready to generate your newsletter!
            </p>
            <div style="
                display: flex;
                justify-content: center;
                align-items: center;
                margin: 1rem 0 0 0;
            ">
                <div style="
                    width: 12px;
                    height: 12px;
                    background: #fff;
                    border-radius: 50%;
                    margin: 0 0.5rem;
                    animation: pulse 1.5s infinite;
                "></div>
                <div style="
                    width: 12px;
                    height: 12px;
                    background: #fff;
                    border-radius: 50%;
                    margin: 0 0.5rem;
                    animation: pulse 1.5s infinite 0.3s;
                "></div>
                <div style="
                    width: 12px;
                    height: 12px;
                    background: #fff;
                    border-radius: 50%;
                    margin: 0 0.5rem;
                    animation: pulse 1.5s infinite 0.6s;
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_configuration_summary(config: Dict[str, Any]) -> None:
    """Create a visual summary of current configuration"""
    if not config.get('topic'):
        st.warning("⚠️ Please enter a newsletter topic to begin")
        return
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 1px solid #e1e8ed;
    ">
        <h4 style="margin: 0 0 1rem 0; color: #2c3e50;">📋 Configuration Summary</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div>
                <strong>Topic:</strong> {topic}<br>
                <strong>Audience:</strong> {audience}<br>
                <strong>Workflow:</strong> {workflow}
            </div>
            <div>
                <strong>Content Length:</strong> {length}<br>
                <strong>Quality Focus:</strong> {focus}<br>
                <strong>Categories:</strong> {categories}
            </div>
        </div>
    </div>
    """.format(
        topic=config['topic'],
        audience=config['audience'],
        workflow=config['workflow_type'],
        length=config['content_length'],
        focus=", ".join(config['quality_focus']) if config['quality_focus'] else "Default",
        categories=", ".join(config['selected_categories']) if config['selected_categories'] else "All"
    ), unsafe_allow_html=True)

def create_generate_button(topic: str, is_generating: bool) -> bool:
    """Create the main generate button with enhanced styling"""
    st.markdown("---")
    
    # Center the button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        button_disabled = not topic or is_generating
        button_text = "🔄 Generating..." if is_generating else "🚀 Generate Newsletter"
        
        if st.button(
            button_text,
            disabled=button_disabled,
            help="Click to start generating your newsletter",
            key="generate_button"
        ):
            return True
    
    return False

def create_progress_tracker(current_step: str, progress: float) -> None:
    """Create an animated progress tracker"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
        text-align: center;
    ">
        <h4 style="margin: 0 0 1rem 0;">{current_step}</h4>
        <div style="
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            padding: 4px;
            margin: 1rem 0;
        ">
            <div style="
                background: linear-gradient(90deg, #56ab2f, #a8e6cf);
                height: 20px;
                border-radius: 6px;
                width: {progress}%;
                transition: width 0.3s ease;
            "></div>
        </div>
        <p style="margin: 0; opacity: 0.9; font-size: 0.9rem;">
            {progress:.1f}% Complete
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_content_display_tabs(content: str) -> None:
    """Create tabbed content display with enhanced formatting"""
    if not content:
        st.warning("⚠️ No content generated")
        return
    
    # Content statistics
    word_count = len(content.split())
    char_count = len(content)
    
    st.info(f"📊 Content Statistics: {char_count:,} characters • {word_count:,} words")
    
    # Content tabs
    tab1, tab2, tab3 = st.tabs(["📖 Full Content", "📋 Preview", "📁 Download"])
    
    with tab1:
        st.markdown("""
        <div style="
            background: white;
            padding: 2rem;
            border-radius: 12px;
            border: 1px solid #e1e8ed;
            max-height: 600px;
            overflow-y: auto;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        ">
        """, unsafe_allow_html=True)
        
        st.markdown(content)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        preview_length = 2000
        if len(content) > preview_length:
            st.markdown(f"**Preview (first {preview_length} characters):**")
            
            st.markdown("""
            <div style="
                background: #f8f9fa;
                padding: 1.5rem;
                border-radius: 8px;
                border-left: 4px solid #0068c9;
                margin: 1rem 0;
            ">
            """, unsafe_allow_html=True)
            
            st.markdown(content[:preview_length] + "...")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.info(f"Full content continues for {len(content) - preview_length:,} more characters")
        else:
            st.markdown("**Complete Content:**")
            st.markdown(content)
    
    with tab3:
        create_download_section(content)

def create_download_section(content: str) -> None:
    """Create download section with multiple format options"""
    st.markdown("### 📥 Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Markdown download
        st.download_button(
            label="📄 Download as Markdown",
            data=content,
            file_name=f"newsletter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            help="Download in Markdown format"
        )
        
        # Text download
        st.download_button(
            label="📝 Download as Text",
            data=content,
            file_name=f"newsletter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            help="Download as plain text"
        )
    
    with col2:
        # JSON download with metadata
        import json
        json_data = {
            'content': content,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'word_count': len(content.split()),
                'char_count': len(content),
                'format': 'newsletter'
            }
        }
        
        st.download_button(
            label="📊 Download with Metadata",
            data=json.dumps(json_data, indent=2),
            file_name=f"newsletter_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Download with metadata in JSON format"
        )
        
        # Create shareable link (placeholder)
        if st.button("🔗 Create Shareable Link", help="Generate a shareable link (coming soon)"):
            st.info("📋 Shareable link feature coming soon!")

def create_performance_dashboard(stats: Dict[str, Any]) -> None:
    """Create comprehensive performance dashboard"""
    st.markdown("## 📊 Performance Analytics")
    
    if not stats:
        st.warning("No performance data available")
        return
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        execution_time = stats.get('execution_time', 0)
        st.metric(
            "Execution Time",
            f"{execution_time:.1f}s",
            delta=f"{'🟢 Fast' if execution_time < 120 else '🟡 Moderate' if execution_time < 240 else '🔴 Slow'}"
        )
    
    with col2:
        success = stats.get('success', False)
        st.metric(
            "Status",
            "✅ Success" if success else "❌ Failed",
            delta=None
        )
    
    with col3:
        workflow = stats.get('workflow_type', 'Unknown')
        st.metric(
            "Workflow",
            workflow.replace("(Manager-Led)", "").replace("Multi-Agent", "Standard"),
            delta=None
        )
    
    with col4:
        timestamp = stats.get('timestamp', 'Unknown')
        st.metric(
            "Generated",
            timestamp.split()[1] if ' ' in timestamp else timestamp,
            delta=None
        )
    
    # Performance visualization
    if stats.get('execution_time'):
        fig = create_execution_time_gauge(stats['execution_time'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed statistics
    with st.expander("📈 Detailed Statistics"):
        st.json(stats)

def create_execution_time_gauge(execution_time: float) -> go.Figure:
    """Create an execution time gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = execution_time,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Execution Time (seconds)"},
        delta = {'reference': 180, 'valueformat': ".1f"},
        gauge = {
            'axis': {'range': [None, 300]},
            'bar': {'color': "#0068c9"},
            'steps': [
                {'range': [0, 120], 'color': "#e8f5e8"},
                {'range': [120, 240], 'color': "#fff3cd"},
                {'range': [240, 300], 'color': "#f8d7da"}
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

def create_feedback_section() -> Optional[Dict[str, Any]]:
    """Create user feedback section"""
    st.markdown("## 💬 Feedback")
    
    with st.expander("📝 Rate this newsletter", expanded=False):
        st.markdown("Help us improve by rating the generated newsletter:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            quality_rating = st.slider(
                "Overall Quality",
                min_value=1,
                max_value=5,
                value=3,
                help="Rate the overall quality of the newsletter"
            )
            
            relevance_rating = st.slider(
                "Relevance",
                min_value=1,
                max_value=5,
                value=3,
                help="How relevant is the content to your needs?"
            )
        
        with col2:
            completeness_rating = st.slider(
                "Completeness",
                min_value=1,
                max_value=5,
                value=3,
                help="How complete and comprehensive is the content?"
            )
            
            readability_rating = st.slider(
                "Readability",
                min_value=1,
                max_value=5,
                value=3,
                help="How easy is it to read and understand?"
            )
        
        feedback_text = st.text_area(
            "Additional Comments",
            placeholder="Share any specific feedback or suggestions...",
            height=100
        )
        
        if st.button("📤 Submit Feedback"):
            feedback_data = {
                'quality': quality_rating,
                'relevance': relevance_rating,
                'completeness': completeness_rating,
                'readability': readability_rating,
                'comments': feedback_text,
                'timestamp': datetime.now().isoformat()
            }
            
            st.success("✅ Thank you for your feedback!")
            return feedback_data
    
    return None

def create_footer():
    """Create footer with additional information"""
    st.markdown("---")
    st.markdown("""
    <div style="
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        margin: 2rem 0;
        color: #6c757d;
    ">
        <p style="margin: 0; font-size: 0.9rem;">
            🚀 <strong>AI Newsletter Generator</strong> - Powered by Advanced AI Agents
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem;">
            Built with ❤️ using Streamlit • Enhanced with Plotly • Optimized for Performance
        </p>
    </div>
    """, unsafe_allow_html=True) 