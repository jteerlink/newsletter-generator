import time
from .persistence import AgenticPersistence

class AgentMemory:
    """
    Per-agent memory/state object for storing queries, responses, retrieved contexts, evaluation results, and user feedback.
    Supports in-memory storage with SQLite persistence; can be extended for additional persistence backends.
    """
    def __init__(self, agent_id: str = None, persistence: AgenticPersistence = None):
        self.agent_id = agent_id
        self.persistence = persistence
        self.memory = {
            'queries': [],
            'responses': [],
            'contexts': [],
            'evaluations': [],
            'feedback': []
        }
        # Load existing memory from persistence if available
        if self.persistence and self.agent_id:
            self._load_from_persistence()

    def get_current_timestamp(self):
        """Get current timestamp."""
        return time.time()

    def add_query(self, query: dict):
        self.memory['queries'].append(query)
        self._save_to_persistence('queries', query)

    def add_response(self, response: dict):
        self.memory['responses'].append(response)
        self._save_to_persistence('responses', response)

    def add_context(self, context: dict):
        self.memory['contexts'].append(context)
        self._save_to_persistence('contexts', context)

    def add_evaluation(self, evaluation: dict):
        self.memory['evaluations'].append(evaluation)
        self._save_to_persistence('evaluations', evaluation)

    def add_feedback(self, feedback: dict):
        self.memory['feedback'].append(feedback)
        self._save_to_persistence('feedback', feedback)

    def get_queries(self):
        return self.memory['queries']

    def get_responses(self):
        return self.memory['responses']

    def get_contexts(self):
        return self.memory['contexts']

    def get_evaluations(self):
        return self.memory['evaluations']

    def get_feedback(self):
        return self.memory['feedback']

    def clear(self):
        for key in self.memory:
            self.memory[key] = []

    def summary(self):
        return {k: len(v) for k, v in self.memory.items()}

    def _save_to_persistence(self, memory_type: str, data: dict):
        """Save memory item to persistence layer."""
        if self.persistence and self.agent_id:
            try:
                self.persistence.save_agent_memory(self.agent_id, memory_type, data)
            except Exception as e:
                # Log error but don't fail the operation
                import logging
                logging.getLogger(__name__).warning(f"Failed to save to persistence: {e}")

    def _load_from_persistence(self):
        """Load memory from persistence layer."""
        if not self.persistence or not self.agent_id:
            return
        
        try:
            for memory_type in self.memory.keys():
                persisted_data = self.persistence.load_agent_memory(self.agent_id, memory_type)
                self.memory[memory_type] = persisted_data
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to load from persistence: {e}") 