from ..base.agent_base import AgentBase
from ..base.communication import Message, MessageType

class QueryWriterAgent(AgentBase):
    """
    Agent that refines, clarifies, and corrects user queries before retrieval.
    Logs all query transformations in agent memory.
    """
    def __init__(self, agent_id: str = "query_writer_agent"):
        super().__init__(agent_id)

    def run(self):
        # Main execution loop (stub)
        pass

    def receive_message(self, message: dict):
        # Refine and log query transformation
        original_query = message.get('content', {}).get('query', "")
        refined_query = self.refine_query(original_query)
        self.memory.add_query({'original': original_query, 'refined': refined_query})
        return {
            'sender': self.agent_id,
            'recipient': message.get('sender', ''),
            'type': MessageType.RESPONSE.value,
            'content': {'refined_query': refined_query}
        }

    def send_message(self, recipient_id: str, message: dict):
        # Stub for sending messages
        pass

    def refine_query(self, query: str) -> str:
        # Placeholder for query refinement logic
        # In production, use LLM or rules to clarify/correct
        return query.strip()  # Stub: just strip whitespace for now 