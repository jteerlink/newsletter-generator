from ..base.agent_base import AgentBase
from ..base.communication import Message, MessageType
from ..base.feedback_analyzer import FeedbackAnalyzer

class ResponseEvaluatorAgent(AgentBase):
    """
    Agent that assesses LLM output for relevance, completeness, and factuality.
    Adds a self-critique step, confidence scoring, escalation if confidence is low,
    and uses advanced feedback analysis for evaluation. Stores evaluation results in agent memory.
    """
    def __init__(self, agent_id: str = "response_evaluator_agent"):
        super().__init__(agent_id)
        self.feedback_analyzer = FeedbackAnalyzer()

    def run(self):
        # Main execution loop (stub)
        pass

    def receive_message(self, message: dict):
        # Evaluate response and log
        response = message.get('content', {}).get('response', "")
        evaluation = self.evaluate_response(response)
        self.memory.add_evaluation({'response': response, 'evaluation': evaluation})
        return {
            'sender': self.agent_id,
            'recipient': message.get('sender', ''),
            'type': MessageType.RESPONSE.value,
            'content': {'evaluation': evaluation}
        }

    def send_message(self, recipient_id: str, message: dict):
        # Stub for sending messages
        pass

    def evaluate_response(self, response: str) -> dict:
        # Placeholder for evaluation logic
        # In production, use LLM or rules for critique
        confidence = self.confidence_score(response)
        
        # Use advanced feedback analysis
        feedback_analysis = self.get_advanced_feedback_analysis()
        confidence_adjustment = feedback_analysis['confidence_adjustment']
        confidence = min(1.0, max(0.0, confidence + confidence_adjustment))
        
        escalation = False
        escalation_reason = None
        
        # Enhanced escalation logic based on feedback analysis
        if confidence < 0.5:
            escalation = True
            escalation_reason = 'Low confidence'
        elif feedback_analysis['trend_analysis']['trend'] == 'declining' and confidence < 0.7:
            escalation = True
            escalation_reason = 'Declining feedback trend'
        elif feedback_analysis['sentiment_analysis']['sentiment'] == 'negative' and confidence < 0.6:
            escalation = True
            escalation_reason = 'Negative feedback sentiment'
        
        return {
            'relevance': 'unknown',
            'completeness': 'unknown',
            'factuality': 'unknown',
            'self_critique': 'Not implemented (stub)',
            'confidence': confidence,
            'escalation': escalation,
            'escalation_reason': escalation_reason,
            'feedback_analysis': feedback_analysis
        }

    def confidence_score(self, response: str) -> float:
        # Stub: confidence is higher if response is longer
        length = len(response.strip())
        if length > 100:
            return 0.9
        elif length > 30:
            return 0.6
        else:
            return 0.3

    def get_advanced_feedback_analysis(self) -> dict:
        """Get comprehensive feedback analysis including sentiment, patterns, trends, and confidence adjustment."""
        feedback = self.memory.get_feedback()
        
        if not feedback:
            return {
                'confidence_adjustment': 0.0,
                'sentiment_analysis': {'sentiment': 'neutral', 'score': 0.0},
                'trend_analysis': {'trend': 'stable', 'confidence': 0.0},
                'pattern_analysis': {},
                'insights': [],
                'recommendations': []
            }
        
        # Analyze sentiment of recent feedback
        recent_feedback = feedback[-1] if feedback else {}
        feedback_text = recent_feedback.get('feedback', {}).get('comment', '')
        sentiment_analysis = self.feedback_analyzer.analyze_sentiment(feedback_text)
        
        # Analyze trends
        trend_analysis = self.feedback_analyzer.analyze_feedback_trends(feedback)
        
        # Calculate confidence adjustment
        current_rating = recent_feedback.get('feedback', {}).get('rating', 0.5)
        confidence_adjustment = self.feedback_analyzer.calculate_confidence_adjustment(feedback, current_rating)
        
        # Get comprehensive insights
        insights_analysis = self.feedback_analyzer.get_feedback_insights(feedback)
        
        return {
            'confidence_adjustment': confidence_adjustment,
            'sentiment_analysis': sentiment_analysis,
            'trend_analysis': trend_analysis,
            'pattern_analysis': insights_analysis.get('pattern_analysis', {}),
            'insights': insights_analysis.get('insights', []),
            'recommendations': insights_analysis.get('recommendations', [])
        }

    def get_feedback_adjustment(self) -> float:
        """Get confidence adjustment based on historical user feedback (legacy method)."""
        feedback_analysis = self.get_advanced_feedback_analysis()
        return feedback_analysis['confidence_adjustment'] 