import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from agents.writing.industry_news_agent import IndustryNewsAgent
from agents.writing.technical_deep_dive_agent import TechnicalDeepDiveAgent
from agents.writing.trend_analysis_agent import TrendAnalysisAgent
from agents.writing.interview_profile_agent import InterviewProfileAgent

@pytest.mark.asyncio
async def test_industry_news_agent_task_handling():
    agent = IndustryNewsAgent()
    agent.web_search = AsyncMock(return_value={"results": [{"title": "AI Launch", "snippet": "New AI product", "url": "http://news.com"}]})
    agent.vector_search = AsyncMock(return_value={"results": [{"title": "Archived News", "content": "Old AI news", "url": "http://archive.com"}]})
    task_data = {"topic": "AI", "max_length": 400, "task_id": "t1", "sender_id": "planner"}
    await agent._handle_news_task(task_data)
    # No assertion: just ensure no exceptions and logs

@pytest.mark.asyncio
async def test_technical_deep_dive_agent_task_handling():
    agent = TechnicalDeepDiveAgent()
    agent.web_search = AsyncMock(return_value={"results": [{"title": "AI Tutorial", "snippet": "How to use AI", "url": "http://tutorial.com"}]})
    agent.vector_search = AsyncMock(return_value={"results": [{"title": "Code Example", "content": "def ai(): pass", "url": "http://code.com"}]})
    task_data = {"topic": "AI", "max_length": 600, "task_id": "t2", "sender_id": "planner"}
    await agent._handle_deep_dive_task(task_data)

@pytest.mark.asyncio
async def test_trend_analysis_agent_task_handling():
    agent = TrendAnalysisAgent()
    agent.web_search = AsyncMock(return_value={"results": [{"title": "AI Trend", "snippet": "AI is growing", "url": "http://trend.com"}]})
    agent.vector_search = AsyncMock(return_value={"results": [{"title": "Pattern", "content": "Pattern detected", "url": "http://pattern.com"}]})
    task_data = {"topic": "AI", "max_length": 500, "task_id": "t3", "sender_id": "planner"}
    await agent._handle_trend_task(task_data)

@pytest.mark.asyncio
async def test_interview_profile_agent_task_handling():
    agent = InterviewProfileAgent()
    agent.web_search = AsyncMock(return_value={"results": [{"title": "Interview with CEO", "snippet": "Insights from CEO", "url": "http://interview.com"}]})
    agent.vector_search = AsyncMock(return_value={"results": [{"title": "Profile", "content": "Profile of expert", "url": "http://profile.com"}]})
    task_data = {"topic": "AI", "max_length": 400, "task_id": "t4", "sender_id": "planner"}
    await agent._handle_interview_task(task_data)

# Section synthesis logic (unit)
def test_industry_news_section_synthesis():
    agent = IndustryNewsAgent()
    news_results = {"results": [{"title": "AI Launch", "snippet": "New AI product", "url": "http://news.com"}]}
    local_news = {"results": [{"title": "Archived News", "content": "Old AI news", "url": "http://archive.com"}]}
    section = agent._default_section_template("AI", news_results, local_news, 500)
    assert "Industry News" in section
    assert "AI Launch" in section
    assert "Archived News" in section

def test_technical_deep_dive_section_synthesis():
    agent = TechnicalDeepDiveAgent()
    tech_results = {"results": [{"title": "AI Tutorial", "snippet": "How to use AI", "url": "http://tutorial.com"}]}
    local_tech = {"results": [{"title": "Code Example", "content": "def ai(): pass", "url": "http://code.com"}]}
    section = agent._default_section_template("AI", tech_results, local_tech, 600)
    assert "Technical Deep-Dive" in section
    assert "AI Tutorial" in section
    assert "Code Example" in section

def test_trend_analysis_section_synthesis():
    agent = TrendAnalysisAgent()
    trend_results = {"results": [{"title": "AI Trend", "snippet": "AI is growing", "url": "http://trend.com"}]}
    local_trends = {"results": [{"title": "Pattern", "content": "Pattern detected", "url": "http://pattern.com"}]}
    section = agent._default_section_template("AI", trend_results, local_trends, 500)
    assert "Trend Analysis" in section
    assert "AI Trend" in section
    assert "Pattern" in section

def test_interview_profile_section_synthesis():
    agent = InterviewProfileAgent()
    interview_results = {"results": [{"title": "Interview with CEO", "snippet": "Insights from CEO", "url": "http://interview.com"}]}
    local_interviews = {"results": [{"title": "Profile", "content": "Profile of expert", "url": "http://profile.com"}]}
    section = agent._default_section_template("AI", interview_results, local_interviews, 500)
    assert "Interviews & Profiles" in section
    assert "Interview with CEO" in section
    assert "Profile" in section 