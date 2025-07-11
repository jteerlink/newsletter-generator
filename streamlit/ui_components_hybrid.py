"""
UI Components for Hybrid Newsletter System
Reusable components with modern design system
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, Any, List, Optional

class ModernUI:
    """Modern UI components with consistent styling"""
    
    # Color scheme from infographic
    COLORS = {
        'primary_blue': '#003F5C',
        'secondary_blue': '#2F4B7C', 
        'accent_blue': '#665191',
        'orange': '#FFA600',
        'red': '#F95D6A',
        'pink': '#D45087',
        'orange_light': '#FF7C43',
        'background': '#F8F9FA',
        'text_dark': '#212529',
        'text_light': '#6C757D'
    }
    
    @staticmethod
    def create_metric_card(title: str, value: str, color: str = 'secondary_blue'):
        """Create a modern metric card"""
        color_value = ModernUI.COLORS[color]
        return f"""
        <div class="metric-card" style="
            background: linear-gradient(135deg, {color_value} 0%, {ModernUI.COLORS['accent_blue']} 100%);
            color: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            text-align: center;
            box-shadow: 0 4px 15px rgba(47,75,124,0.3);
        ">
            <div class="metric-value" style="font-size: 2.5rem; font-weight: 900; margin-bottom: 0.5rem;">
                {value}
            </div>
            <div class="metric-label" style="font-size: 1rem; opacity: 0.9; font-weight: 500;">
                {title}
            </div>
        </div>
        """
    
    @staticmethod
    def create_status_indicator(status: str, label: str = ""):
        """Create a status indicator with animation"""
        color_map = {
            'active': ModernUI.COLORS['orange'],
            'success': '#28a745',
            'warning': ModernUI.COLORS['red'],
            'info': ModernUI.COLORS['secondary_blue']
        }
        
        color = color_map.get(status, ModernUI.COLORS['text_light'])
        
        return f"""
        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
            <div class="status-indicator" style="
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: {color};
                margin-right: 0.5rem;
                animation: pulse 2s ease-in-out infinite;
            "></div>
            <span style="color: {ModernUI.COLORS['text_dark']}; font-weight: 500;">{label}</span>
        </div>
        """
    
    @staticmethod
    def create_progress_card(title: str, current: int, total: int, description: str = ""):
        """Create a progress card with modern styling"""
        percentage = (current / total) * 100 if total > 0 else 0
        
        return f"""
        <div class="progress-card" style="
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 4px solid {ModernUI.COLORS['orange']};
        ">
            <h4 style="color: {ModernUI.COLORS['secondary_blue']}; margin-bottom: 0.5rem;">{title}</h4>
            <div style="
                background: {ModernUI.COLORS['background']};
                border-radius: 10px;
                height: 10px;
                margin: 1rem 0;
                overflow: hidden;
            ">
                <div style="
                    background: linear-gradient(90deg, {ModernUI.COLORS['orange']} 0%, {ModernUI.COLORS['orange_light']} 100%);
                    height: 100%;
                    width: {percentage}%;
                    transition: width 0.3s ease;
                "></div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: {ModernUI.COLORS['text_light']}; font-size: 0.9rem;">{description}</span>
                <span style="color: {ModernUI.COLORS['text_dark']}; font-weight: 600;">{current}/{total}</span>
            </div>
        </div>
        """

class QualityVisualization:
    """Quality metrics visualization components"""
    
    @staticmethod
    def create_quality_gauge(score: float, title: str):
        """Create a quality gauge chart"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            delta = {'reference': 90},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': ModernUI.COLORS['orange']},
                'steps': [
                    {'range': [0, 70], 'color': ModernUI.COLORS['red']},
                    {'range': [70, 85], 'color': ModernUI.COLORS['orange_light']},
                    {'range': [85, 100], 'color': '#28a745'}
                ],
                'threshold': {
                    'line': {'color': ModernUI.COLORS['secondary_blue'], 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            font={'color': ModernUI.COLORS['text_dark']},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    @staticmethod
    def create_pipeline_comparison_chart(daily_stats: Dict, deep_dive_stats: Dict):
        """Create pipeline comparison chart"""
        categories = ['Speed', 'Depth', 'Engagement', 'Technical Accuracy']
        daily_scores = [daily_stats.get(cat.lower().replace(' ', '_'), 0.8) for cat in categories]
        deep_dive_scores = [deep_dive_stats.get(cat.lower().replace(' ', '_'), 0.8) for cat in categories]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=daily_scores,
            theta=categories,
            fill='toself',
            name='Daily Quick Pipeline',
            marker=dict(color=ModernUI.COLORS['orange'])
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=deep_dive_scores,
            theta=categories,
            fill='toself',
            name='Deep Dive Pipeline',
            marker=dict(color=ModernUI.COLORS['secondary_blue'])
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig

class ContentPreview:
    """Content preview and formatting components"""
    
    @staticmethod
    def create_mobile_preview(content: str, subject_line: str = ""):
        """Create mobile-first content preview"""
        # Simulate mobile formatting
        mobile_width = "350px"
        
        return f"""
        <div style="
            max-width: {mobile_width};
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: 1px solid {ModernUI.COLORS['background']};
        ">
            <div style="
                background: {ModernUI.COLORS['secondary_blue']};
                color: white;
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 1rem;
                text-align: center;
                font-weight: 600;
            ">
                📱 Mobile Preview
            </div>
            
            <div style="
                background: {ModernUI.COLORS['background']};
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1rem;
                border-left: 4px solid {ModernUI.COLORS['orange']};
            ">
                <strong>Subject:</strong> {subject_line}
            </div>
            
            <div style="
                font-size: 0.9rem;
                line-height: 1.6;
                color: {ModernUI.COLORS['text_dark']};
                max-height: 400px;
                overflow-y: auto;
            ">
                {content[:500]}...
            </div>
        </div>
        """
    
    @staticmethod
    def create_content_structure_preview(sections: List[Dict[str, Any]]):
        """Create content structure preview"""
        structure_html = """
        <div style="
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
            <h3 style="color: {secondary_blue}; margin-bottom: 1rem;">📄 Content Structure</h3>
        """.format(secondary_blue=ModernUI.COLORS['secondary_blue'])
        
        for i, section in enumerate(sections):
            structure_html += f"""
            <div style="
                background: {ModernUI.COLORS['background']};
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
                border-left: 4px solid {ModernUI.COLORS['orange']};
            ">
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <strong style="color: {ModernUI.COLORS['text_dark']};">
                        {i+1}. {section.get('title', 'Untitled Section')}
                    </strong>
                    <span style="
                        color: {ModernUI.COLORS['text_light']};
                        font-size: 0.9rem;
                    ">
                        ~{section.get('word_count', 0)} words
                    </span>
                </div>
                <p style="
                    color: {ModernUI.COLORS['text_light']};
                    font-size: 0.9rem;
                    margin: 0.5rem 0 0 0;
                ">
                    {section.get('description', 'No description available')}
                </p>
            </div>
            """
        
        structure_html += "</div>"
        return structure_html

class AnimatedComponents:
    """Animated UI components for better UX"""
    
    @staticmethod
    def create_loading_animation(message: str = "Processing..."):
        """Create animated loading component"""
        return f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 2rem 0;
        ">
            <div style="
                width: 50px;
                height: 50px;
                border: 4px solid {ModernUI.COLORS['background']};
                border-top: 4px solid {ModernUI.COLORS['orange']};
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-bottom: 1rem;
            "></div>
            <p style="
                color: {ModernUI.COLORS['text_dark']};
                font-weight: 500;
                margin: 0;
            ">{message}</p>
        </div>
        
        <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        """
    
    @staticmethod
    def create_success_animation(message: str = "Success!"):
        """Create success animation"""
        return f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem;
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(40,167,69,0.3);
            margin: 2rem 0;
            animation: fadeIn 0.5s ease-in-out;
        ">
            <div style="
                font-size: 3rem;
                margin-bottom: 1rem;
                animation: bounce 0.6s ease-in-out;
            ">✅</div>
            <p style="
                font-weight: 600;
                margin: 0;
                font-size: 1.1rem;
            ">{message}</p>
        </div>
        
        <style>
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @keyframes bounce {{
            0%, 20%, 50%, 80%, 100% {{ transform: translateY(0); }}
            40% {{ transform: translateY(-10px); }}
            60% {{ transform: translateY(-5px); }}
        }}
        </style>
        """

def create_newsletter_analytics_dashboard(metrics: Dict[str, Any]):
    """Create comprehensive analytics dashboard"""
    st.markdown("""
    <div style="
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    ">
        <h2 style="color: {secondary_blue}; margin-bottom: 1.5rem;">📊 Newsletter Analytics Dashboard</h2>
    </div>
    """.format(secondary_blue=ModernUI.COLORS['secondary_blue']), unsafe_allow_html=True)
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.plotly_chart(
            QualityVisualization.create_quality_gauge(
                metrics.get('technical_accuracy', 0.85),
                "Technical Accuracy"
            ),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            QualityVisualization.create_quality_gauge(
                metrics.get('mobile_readability', 0.90),
                "Mobile Readability"
            ),
            use_container_width=True
        )
    
    with col3:
        st.plotly_chart(
            QualityVisualization.create_quality_gauge(
                metrics.get('engagement_score', 0.82),
                "Engagement Score"
            ),
            use_container_width=True
        )
    
    with col4:
        st.plotly_chart(
            QualityVisualization.create_quality_gauge(
                metrics.get('overall_quality', 0.87),
                "Overall Quality"
            ),
            use_container_width=True
        )

def create_feature_comparison_table():
    """Create feature comparison table between pipelines"""
    comparison_data = {
        'Feature': [
            'Generation Speed',
            'Content Depth',
            'Word Count',
            'Research Required',
            'Mobile Optimization',
            'Technical Accuracy',
            'Code Examples',
            'Citations'
        ],
        'Daily Quick Pipeline': [
            '⚡ 2-3 minutes',
            '📄 Summary level',
            '📊 500-1,500 words',
            '🔍 Light research',
            '📱 Fully optimized',
            '✅ Basic validation',
            '💻 Code snippets',
            '📚 Key sources'
        ],
        'Deep Dive Pipeline': [
            '🕐 15-20 minutes',
            '🔬 Comprehensive',
            '📖 3,000-5,000 words',
            '🎯 Extensive research',
            '📱 Mobile-friendly',
            '🔍 Rigorous validation',
            '💻 Full examples',
            '📚 Academic citations'
        ]
    }
    
    return comparison_data 