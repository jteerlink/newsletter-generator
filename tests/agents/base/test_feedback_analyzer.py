import pytest
from datetime import datetime, timedelta
from agents.base.feedback_analyzer import FeedbackAnalyzer

@pytest.fixture
def analyzer():
    return FeedbackAnalyzer()

@pytest.fixture
def sample_feedback_history():
    """Create sample feedback history for testing."""
    base_time = datetime.now().timestamp()
    return [
        {
            'workflow_id': 'wf1',
            'feedback': {'rating': 0.8, 'comment': 'This is very helpful and accurate'},
            'timestamp': base_time - 86400 * 7  # 7 days ago
        },
        {
            'workflow_id': 'wf1',
            'feedback': {'rating': 0.9, 'comment': 'Excellent response, very clear and comprehensive'},
            'timestamp': base_time - 86400 * 3  # 3 days ago
        },
        {
            'workflow_id': 'wf1',
            'feedback': {'rating': 0.7, 'comment': 'Good but could be more detailed'},
            'timestamp': base_time - 86400 * 1  # 1 day ago
        }
    ]

def test_sentiment_analysis_positive(analyzer):
    """Test sentiment analysis with positive feedback."""
    text = "This is excellent and very helpful"
    result = analyzer.analyze_sentiment(text)
    
    assert result['sentiment'] == 'positive'
    assert result['score'] > 0
    assert result['positive_count'] > 0
    assert result['negative_count'] == 0

def test_sentiment_analysis_negative(analyzer):
    """Test sentiment analysis with negative feedback."""
    text = "This is bad and useless"
    result = analyzer.analyze_sentiment(text)
    
    assert result['sentiment'] == 'negative'
    assert result['score'] < 0
    assert result['negative_count'] > 0
    assert result['positive_count'] == 0

def test_sentiment_analysis_neutral(analyzer):
    """Test sentiment analysis with neutral feedback."""
    text = "This is okay"
    result = analyzer.analyze_sentiment(text)
    
    assert result['sentiment'] == 'neutral'
    assert abs(result['score']) <= 0.2

def test_sentiment_analysis_empty(analyzer):
    """Test sentiment analysis with empty text."""
    result = analyzer.analyze_sentiment("")
    
    assert result['sentiment'] == 'neutral'
    assert result['score'] == 0.0
    assert result['confidence'] == 0.0

def test_pattern_extraction(analyzer):
    """Test pattern extraction from feedback text."""
    text = "This response is accurate and complete, but could be more clear"
    patterns = analyzer.extract_patterns(text)
    
    assert 'accuracy' in patterns
    assert 'completeness' in patterns
    assert 'clarity' in patterns
    assert 'accurate' in patterns['accuracy']
    assert 'complete' in patterns['completeness']
    assert 'clear' in patterns['clarity']

def test_pattern_extraction_no_patterns(analyzer):
    """Test pattern extraction with no matching patterns."""
    text = "This is just a general comment"
    patterns = analyzer.extract_patterns(text)
    
    assert patterns == {}

def test_trend_analysis_improving(analyzer):
    """Test trend analysis with improving feedback."""
    feedback_history = [
        {
            'feedback': {'rating': 0.5},
            'timestamp': datetime.now().timestamp() - 86400 * 10
        },
        {
            'feedback': {'rating': 0.8},
            'timestamp': datetime.now().timestamp() - 86400 * 5
        },
        {
            'feedback': {'rating': 0.9},
            'timestamp': datetime.now().timestamp() - 86400 * 1
        }
    ]
    
    result = analyzer.analyze_feedback_trends(feedback_history)
    
    assert result['trend'] == 'improving'
    assert result['data_points'] == 3
    assert result['average_rating'] > 0.7

def test_trend_analysis_declining(analyzer):
    """Test trend analysis with declining feedback."""
    feedback_history = [
        {
            'feedback': {'rating': 0.9},
            'timestamp': datetime.now().timestamp() - 86400 * 10
        },
        {
            'feedback': {'rating': 0.6},
            'timestamp': datetime.now().timestamp() - 86400 * 5
        },
        {
            'feedback': {'rating': 0.3},
            'timestamp': datetime.now().timestamp() - 86400 * 1
        }
    ]
    
    result = analyzer.analyze_feedback_trends(feedback_history)
    
    assert result['trend'] == 'declining'
    assert result['data_points'] == 3
    assert result['average_rating'] < 0.7

def test_trend_analysis_insufficient_data(analyzer):
    """Test trend analysis with insufficient data."""
    feedback_history = [
        {
            'feedback': {'rating': 0.8},
            'timestamp': datetime.now().timestamp() - 86400 * 1
        }
    ]
    
    result = analyzer.analyze_feedback_trends(feedback_history)
    
    assert result['trend'] == 'insufficient_data'
    assert result['data_points'] == 1

def test_confidence_adjustment_positive(analyzer, sample_feedback_history):
    """Test confidence adjustment with positive feedback."""
    adjustment = analyzer.calculate_confidence_adjustment(sample_feedback_history, 0.9)
    
    assert adjustment > 0
    assert adjustment <= 0.3

def test_confidence_adjustment_negative(analyzer):
    """Test confidence adjustment with negative feedback."""
    feedback_history = [
        {
            'feedback': {'rating': 0.2, 'comment': 'This is wrong and inaccurate'},
            'timestamp': datetime.now().timestamp() - 86400 * 1
        }
    ]
    
    adjustment = analyzer.calculate_confidence_adjustment(feedback_history, 0.3)
    
    assert adjustment < 0
    assert adjustment >= -0.3

def test_confidence_adjustment_no_feedback(analyzer):
    """Test confidence adjustment with no feedback."""
    adjustment = analyzer.calculate_confidence_adjustment([], 0.5)
    
    assert adjustment == 0.0

def test_feedback_insights(analyzer, sample_feedback_history):
    """Test comprehensive feedback insights generation."""
    insights = analyzer.get_feedback_insights(sample_feedback_history)
    
    assert 'insights' in insights
    assert 'recommendations' in insights
    assert 'trend_analysis' in insights
    assert 'pattern_analysis' in insights
    assert isinstance(insights['insights'], list)
    assert isinstance(insights['recommendations'], list)

def test_feedback_insights_empty(analyzer):
    """Test feedback insights with empty history."""
    insights = analyzer.get_feedback_insights([])
    
    assert insights['insights'] == []
    assert insights['recommendations'] == []

def test_volatility_calculation(analyzer):
    """Test volatility calculation."""
    values = [0.8, 0.9, 0.7, 0.8, 0.9]
    volatility = analyzer._calculate_volatility(values)
    
    assert volatility >= 0
    assert volatility < 1

def test_volatility_calculation_single_value(analyzer):
    """Test volatility calculation with single value."""
    volatility = analyzer._calculate_volatility([0.8])
    
    assert volatility == 0.0 