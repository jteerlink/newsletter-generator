from ..base.agent_base import AgentBase
from ..base.communication import Message, MessageType

class SourceSelectorAgent(AgentBase):
    """
    Agent that dynamically selects one or more sources (vector DB, web, APIs),
    supports multi-hop retrieval, and logs source selection rationale in memory.
    """
    def __init__(self, agent_id: str = "source_selector_agent"):
        super().__init__(agent_id)

    def run(self):
        # Main execution loop (stub)
        pass

    def receive_message(self, message: dict):
        # Select sources and log rationale
        query = message.get('content', {}).get('query', "")
        sources, rationale = self.select_sources(query)
        self.memory.add_context({'query': query, 'sources': sources, 'rationale': rationale})
        return {
            'sender': self.agent_id,
            'recipient': message.get('sender', ''),
            'type': MessageType.RESPONSE.value,
            'content': {'sources': sources, 'rationale': rationale}
        }

    def send_message(self, recipient_id: str, message: dict):
        # Stub for sending messages
        pass

    def select_sources(self, query: str):
        # Placeholder for dynamic source selection logic
        # In production, use rules or LLM
        sources = ['vector_db']
        rationale = 'Default to vector DB for all queries (stub)'
        return sources, rationale 