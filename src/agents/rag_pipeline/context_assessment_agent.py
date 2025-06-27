from ..base.agent_base import AgentBase
from ..base.communication import Message, MessageType

class ContextAssessmentAgent(AgentBase):
    """
    Agent that decides if more context is needed before LLM call.
    Uses heuristics or LLM-based classification. Logs all assessments in agent memory.
    """
    def __init__(self, agent_id: str = "context_assessment_agent"):
        super().__init__(agent_id)

    def run(self):
        # Main execution loop (stub)
        pass

    def receive_message(self, message: dict):
        # Assess context need and log
        query = message.get('content', {}).get('query', "")
        context_needed = self.assess_context_need(query)
        self.memory.add_context({'query': query, 'context_needed': context_needed})
        return {
            'sender': self.agent_id,
            'recipient': message.get('sender', ''),
            'type': MessageType.RESPONSE.value,
            'content': {'context_needed': context_needed}
        }

    def send_message(self, recipient_id: str, message: dict):
        # Stub for sending messages
        pass

    def assess_context_need(self, query: str) -> bool:
        # Placeholder for context need assessment logic
        # In production, use heuristics or LLM
        return len(query.split()) < 5  # Stub: if query is short, assume more context is needed 