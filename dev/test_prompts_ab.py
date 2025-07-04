from src.core.prompts import get_research_topic_prompt, get_summary_prompt
from src.core.core import query_llm

topic = "The impact of AI on education"
prompt_a = get_research_topic_prompt(topic)
prompt_b = get_summary_prompt(
    f"AI is transforming education by enabling personalized learning experiences, automating administrative tasks, and providing new tools for teachers and students."
)

print("Prompt A:", prompt_a)
result_a = query_llm(prompt_a)
print("Result A:", result_a)

print("\nPrompt B:", prompt_b)
result_b = query_llm(prompt_b)
print("Result B:", result_b)
