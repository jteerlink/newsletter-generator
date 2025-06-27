import asyncio
from datetime import datetime
from agents.planning.master_planner import MasterPlanner
from agents.writing.research_summary_agent import ResearchSummaryAgent
from agents.writing.industry_news_agent import IndustryNewsAgent
from agents.writing.technical_deep_dive_agent import TechnicalDeepDiveAgent
from agents.writing.trend_analysis_agent import TrendAnalysisAgent
from agents.writing.interview_profile_agent import InterviewProfileAgent

# Simulated generic AI content sources
generic_content = [
    {"id": "1", "title": "OpenAI launches new GPT-5 model", "content": "OpenAI has released GPT-5, a new large language model with advanced reasoning capabilities.", "source": "OpenAI Blog", "url": "http://openai.com/gpt-5", "published_date": None},
    {"id": "2", "title": "Google's Gemini Ultra sets new benchmarks", "content": "Gemini Ultra outperforms previous models in language and vision tasks.", "source": "Google AI", "url": "http://ai.google.com/gemini", "published_date": None},
    {"id": "3", "title": "Anthropic's Claude 3.5 Sonnet excels at math", "content": "Claude 3.5 Sonnet achieves state-of-the-art results in mathematical reasoning.", "source": "Anthropic", "url": "http://anthropic.com/claude", "published_date": None},
    {"id": "4", "title": "Meta's Llama 3.1: Planning and Reasoning", "content": "Llama 3.1 introduces advanced planning and reasoning for AI systems.", "source": "Meta AI", "url": "http://ai.meta.com/llama", "published_date": None},
    {"id": "5", "title": "NVIDIA Blackwell B200 powers next-gen AI", "content": "NVIDIA's new GPU architecture accelerates AI training and inference.", "source": "NVIDIA", "url": "http://nvidia.com/blackwell", "published_date": None},
    {"id": "6", "title": "Edge AI goes mainstream", "content": "On-device AI processing is now common in smartphones and IoT devices.", "source": "TechCrunch", "url": "http://techcrunch.com/edge-ai", "published_date": None},
    {"id": "7", "title": "AI in healthcare: New breakthroughs", "content": "AI models are diagnosing diseases and personalizing treatments.", "source": "Nature Medicine", "url": "http://nature.com/ai-healthcare", "published_date": None},
    {"id": "8", "title": "AI safety and alignment research", "content": "Researchers focus on making advanced AI systems safe and aligned with human values.", "source": "AI Alignment Forum", "url": "http://alignmentforum.org/safety", "published_date": None},
    {"id": "9", "title": "AI hardware wars: AMD vs NVIDIA vs Intel", "content": "Competition heats up in the AI accelerator market.", "source": "VentureBeat", "url": "http://venturebeat.com/ai-hardware", "published_date": None},
    {"id": "10", "title": "AI for climate: Modeling and mitigation", "content": "AI is being used to model climate change and develop mitigation strategies.", "source": "Science", "url": "http://science.org/ai-climate", "published_date": None},
]

# Ensure all content items have a valid published_date
def ensure_dates(content_list):
    now = datetime.now()
    for item in content_list:
        if not item.get("published_date"):
            item["published_date"] = now
    return content_list

GENERIC_AI_CONTENT = ensure_dates(generic_content)

# Mocked agent registry (in a real system, this would be dynamic)
AGENT_MAP = {
    'research_summary_agent': ResearchSummaryAgent(),
    'industry_news_agent': IndustryNewsAgent(),
    'technical_deep_dive_agent': TechnicalDeepDiveAgent(),
    'trend_analysis_agent': TrendAnalysisAgent(),
    'interview_profile_agent': InterviewProfileAgent(),
}

async def main():
    print("\n=== GENERIC AI NEWSLETTER SAMPLE RUN ===\n")
    planner = MasterPlanner()
    # Step 1: Plan the newsletter
    plan = planner.plan_newsletter(GENERIC_AI_CONTENT, target_audience="mixed")
    print(f"Newsletter Plan ID: {plan.id}")
    print(f"Title: {plan.title}")
    print(f"Outline Sections: {[s['type'] for s in plan.outline['sections']]}")
    print(f"Assigned Tasks: {list(plan.assigned_tasks.keys())}")
    print(f"Deadlines: {plan.deadlines}")
    print("\n--- Simulating Agent Section Generation ---\n")
    # Step 2: Simulate agent execution for each assigned task
    for agent_id, tasks in plan.assigned_tasks.items():
        agent = AGENT_MAP.get(agent_id)
        if not agent:
            print(f"[WARN] No agent found for {agent_id}")
            continue
        for task in tasks:
            # Simulate agent section generation (mocked, not real web/vector search)
            print(f"[{agent_id}] Generating section for task: {task['id']} (type: {task['type']})")
            # Here, you would call the agent's async task handler with a mock payload
            # For demo, just print a placeholder
            print(f"[{agent_id}] Section: [Simulated content for {task['type']}]\n")
    print("\n=== END OF SAMPLE RUN ===\n")

if __name__ == "__main__":
    asyncio.run(main()) 