class PromptBuilder:
    """
    Utility for assembling prompts from refined queries and retrieved context.
    Logs prompt construction steps for traceability.
    """
    def __init__(self):
        self.log = []

    def build_prompt(self, refined_query: str, context: str) -> str:
        prompt = f"Context:\n{context}\n\nQuery:\n{refined_query}\n\nAnswer:"
        self.log.append({'refined_query': refined_query, 'context': context, 'prompt': prompt})
        return prompt

    def get_log(self):
        return self.log 