from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import re
from collections import Counter, defaultdict

class FeedbackAnalyzer:
    """
    Advanced feedback analyzer for agentic RAG system.
    Provides sentiment analysis, pattern recognition, trend analysis, and confidence adjustments.
    """
    
    def __init__(self):
        # Sentiment keywords (stub - in production, use a proper sentiment analysis library)
        self.positive_keywords = [
            'good', 'great', 'excellent', 'perfect', 'accurate', 'helpful', 'useful',
            'clear', 'comprehensive', 'detailed', 'relevant', 'precise', 'thorough'
        ]
        self.negative_keywords = [
            'bad', 'poor', 'incorrect', 'inaccurate', 'unhelpful', 'useless',
            'unclear', 'vague', 'irrelevant', 'confusing', 'incomplete', 'missing'
        ]
        
        # Feedback patterns for analysis
        self.feedback_patterns = {
            'accuracy': r'\b(accurate|inaccurate|correct|wrong|precise|vague)\b',
            'completeness': r'\b(complete|incomplete|missing|thorough|comprehensive)\b',
            'relevance': r'\b(relevant|irrelevant|helpful|useful|useless)\b',
            'clarity': r'\b(clear|unclear|confusing|understandable|vague)\b'
        }
    
    def analyze_sentiment(self, feedback_text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of feedback text.
        Returns sentiment score (-1 to 1) and confidence.
        """
        if not feedback_text:
            return {'score': 0.0, 'confidence': 0.0, 'sentiment': 'neutral'}
        
        text_lower = feedback_text.lower()
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
        
        total_words = len(text_lower.split())
        if total_words == 0:
            return {'score': 0.0, 'confidence': 0.0, 'sentiment': 'neutral'}
        
        # Calculate sentiment score
        sentiment_score = (positive_count - negative_count) / max(total_words, 1)
        sentiment_score = max(-1.0, min(1.0, sentiment_score * 2))  # Scale to -1 to 1
        
        # Determine sentiment label
        if sentiment_score > 0.2:
            sentiment = 'positive'
        elif sentiment_score < -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Calculate confidence based on word count and keyword presence
        confidence = min(1.0, (positive_count + negative_count) / max(total_words, 1))
        
        return {
            'score': sentiment_score,
            'confidence': confidence,
            'sentiment': sentiment,
            'positive_count': positive_count,
            'negative_count': negative_count
        }
    
    def extract_patterns(self, feedback_text: str) -> Dict[str, List[str]]:
        """
        Extract feedback patterns and themes from text.
        """
        if not feedback_text:
            return {}
        
        patterns = {}
        text_lower = feedback_text.lower()
        
        for pattern_name, regex in self.feedback_patterns.items():
            matches = re.findall(regex, text_lower)
            if matches:
                patterns[pattern_name] = list(set(matches))
        
        return patterns
    
    def analyze_feedback_trends(self, feedback_history: List[Dict[str, Any]], 
                               time_window_days: int = 30) -> Dict[str, Any]:
        """
        Analyze feedback trends over time.
        """
        if not feedback_history:
            return {'trend': 'stable', 'confidence': 0.0, 'data_points': 0, 'rating_volatility': 0.0}
        
        # Filter feedback within time window
        cutoff_time = datetime.now() - timedelta(days=time_window_days)
        recent_feedback = [
            f for f in feedback_history 
            if datetime.fromtimestamp(f.get('timestamp', 0)) > cutoff_time
        ]
        
        if len(recent_feedback) < 2:
            return {'trend': 'insufficient_data', 'confidence': 0.0, 'data_points': len(recent_feedback), 'rating_volatility': 0.0}
        
        # Calculate trend based on ratings
        ratings = [f.get('feedback', {}).get('rating', 0.5) for f in recent_feedback]
        timestamps = [f.get('timestamp', 0) for f in recent_feedback]
        
        # Simple linear trend calculation
        if len(ratings) >= 2:
            # Calculate average rating change per day
            time_span = max(timestamps) - min(timestamps)
            if time_span > 0:
                rating_change = ratings[-1] - ratings[0]
                daily_change = rating_change / (time_span / (24 * 3600))
                
                if daily_change > 0.01:
                    trend = 'improving'
                elif daily_change < -0.01:
                    trend = 'declining'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        avg_rating = sum(ratings) / len(ratings)
        confidence = min(1.0, len(recent_feedback) / 10.0)  # More data = higher confidence
        rating_volatility = self._calculate_volatility(ratings)
        
        return {
            'trend': trend,
            'confidence': confidence,
            'data_points': len(recent_feedback),
            'average_rating': avg_rating,
            'rating_volatility': rating_volatility
        }
    
    def calculate_confidence_adjustment(self, feedback_history: List[Dict[str, Any]], 
                                      current_rating: float) -> float:
        """
        Calculate confidence adjustment based on feedback history and current rating.
        Returns adjustment value (-0.3 to +0.3).
        """
        if not feedback_history:
            return 0.0
        
        # Analyze sentiment of recent feedback
        recent_feedback = feedback_history[-5:]  # Last 5 feedback items
        sentiment_scores = []
        
        for feedback in recent_feedback:
            feedback_text = feedback.get('feedback', {}).get('comment', '')
            if feedback_text:
                sentiment = self.analyze_sentiment(feedback_text)
                sentiment_scores.append(sentiment['score'])
        
        # Calculate adjustments based on multiple factors
        adjustments = []
        
        # 1. Current rating adjustment
        rating_adjustment = (current_rating - 0.5) * 0.4  # -0.2 to +0.2
        adjustments.append(rating_adjustment)
        
        # 2. Sentiment adjustment
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            sentiment_adjustment = avg_sentiment * 0.2  # -0.2 to +0.2
            adjustments.append(sentiment_adjustment)
        
        # 3. Trend adjustment
        trend_analysis = self.analyze_feedback_trends(feedback_history)
        if trend_analysis['trend'] == 'improving':
            trend_adjustment = 0.1
        elif trend_analysis['trend'] == 'declining':
            trend_adjustment = -0.1
        else:
            trend_adjustment = 0.0
        adjustments.append(trend_adjustment)
        
        # 4. Consistency adjustment (lower volatility = higher confidence)
        if trend_analysis['rating_volatility'] < 0.1:
            consistency_adjustment = 0.05
        elif trend_analysis['rating_volatility'] > 0.3:
            consistency_adjustment = -0.05
        else:
            consistency_adjustment = 0.0
        adjustments.append(consistency_adjustment)
        
        # Calculate final adjustment
        final_adjustment = sum(adjustments) / len(adjustments)
        return max(-0.3, min(0.3, final_adjustment))
    
    def get_feedback_insights(self, feedback_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive insights from feedback history.
        """
        if not feedback_history:
            return {'insights': [], 'recommendations': []}
        
        insights = []
        recommendations = []
        
        # Analyze patterns in feedback
        all_comments = [f.get('feedback', {}).get('comment', '') for f in feedback_history if f.get('feedback', {}).get('comment')]
        all_patterns = {}
        
        for comment in all_comments:
            patterns = self.extract_patterns(comment)
            for pattern_type, matches in patterns.items():
                if pattern_type not in all_patterns:
                    all_patterns[pattern_type] = []
                all_patterns[pattern_type].extend(matches)
        
        # Generate insights based on patterns
        for pattern_type, matches in all_patterns.items():
            if matches:
                counter = Counter(matches)
                most_common = counter.most_common(1)[0]
                insights.append(f"Most common {pattern_type} feedback: '{most_common[0]}' ({most_common[1]} mentions)")
        
        # Analyze trends
        trend_analysis = self.analyze_feedback_trends(feedback_history)
        if trend_analysis['trend'] != 'insufficient_data':
            insights.append(f"Feedback trend: {trend_analysis['trend']} (confidence: {trend_analysis['confidence']:.2f})")
        
        # Generate recommendations
        avg_rating = trend_analysis.get('average_rating', 0.5)
        if avg_rating < 0.4:
            recommendations.append("Consider improving response accuracy and relevance")
        elif avg_rating < 0.6:
            recommendations.append("Focus on enhancing response completeness and clarity")
        else:
            recommendations.append("Maintain current performance levels")
        
        if trend_analysis.get('rating_volatility', 0) > 0.3:
            recommendations.append("Work on consistency in response quality")
        
        return {
            'insights': insights,
            'recommendations': recommendations,
            'trend_analysis': trend_analysis,
            'pattern_analysis': all_patterns
        }
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (standard deviation) of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5 