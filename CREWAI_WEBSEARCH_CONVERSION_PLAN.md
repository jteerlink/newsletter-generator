# DuckDuckGo to CrewAI Web Search Conversion Plan

## Overview
This document outlines the comprehensive plan to convert the newsletter-generator project from DuckDuckGo search to CrewAI's built-in web search tools to resolve current search errors and improve reliability.

## Current State Analysis

### Existing Implementation
- **Package**: `duckduckgo-search==6.2.13`
- **Custom Tools**: `AgenticSearchTool` class for iterative search
- **Functions**: `search_web()`, `search_web_with_alternatives()`, `async_search_web()`
- **Issues**: Rate limiting, connection errors, API reliability problems

### Current File Structure
```
src/
├── tools/
│   └── tools.py          # Current DuckDuckGo implementation
├── agents/
│   └── agents.py         # Agent implementations using tools
└── tests/
    └── test_integration_phase2.py  # Integration tests
```

## Phase 1: Foundation Setup

### 1.1 Install CrewAI Dependencies
```bash
pip install crewai crewai[tools]
```

### 1.2 Update requirements.txt
Add to `requirements.txt`:
```
crewai>=0.82.0
crewai[tools]>=0.82.0
```

### 1.3 Choose Search Provider
**Selected**: `SerperDevTool` (Google Search via Serper API)

**Rationale**:
- Most reliable and commonly used
- Professional-grade API
- Good free tier (2,500 searches/month)
- Excellent error handling

**Alternatives considered**:
- `BraveSearchTool` (Brave Search API)
- `DuckDuckGoSearchRun` (LangChain integration)

## Phase 2: API Setup

### 2.1 Set Up API Keys
1. Get API key from https://serper.dev/
2. Set environment variable:
   ```bash
   export SERPER_API_KEY="your-api-key-here"
   ```

### 2.2 Create Environment Configuration
Create `.env` file in the root directory:
```bash
# Copy example and edit
cp .env.example .env
# Or create manually:
echo "SERPER_API_KEY=your-api-key-here" > .env
```

Add to `.env`:
```
# SerperDev API Key for Google Search
SERPER_API_KEY=your-serper-api-key-here
```

**Setup Instructions:**
1. Visit https://serper.dev/ and create a free account
2. Copy your API key from the dashboard
3. Replace `your-serper-api-key-here` with your actual key
4. Free tier provides 2,500 searches/month

## Phase 3: Core Implementation

### 3.1 Create New CrewAI Tools
Replace current `src/tools/tools.py` functions with CrewAI-compatible tools:

```python
from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew

# Initialize search tool
search_tool = SerperDevTool()

# Create CrewAI-compatible wrapper
def crewai_search_web(query: str, max_results: int = 5) -> str:
    """CrewAI web search wrapper"""
    try:
        # Use CrewAI tool
        results = search_tool.run(query)
        return format_search_results(results, max_results)
    except Exception as e:
        logger.error(f"CrewAI search error: {e}")
        return f"Search unavailable: {str(e)}"
```

### 3.2 Update Agent Integration
Modify `src/agents/agents.py` to use CrewAI tools:

```python
from crewai_tools import SerperDevTool

class CrewAIAgent:
    def __init__(self):
        self.search_tool = SerperDevTool()
        self.agent = Agent(
            role="Research Agent",
            goal="Provide comprehensive research",
            tools=[self.search_tool],
            backstory="Expert researcher with web search capabilities"
        )
```

## Phase 4: Migration Strategy

### 4.1 Gradual Migration Approach
- Keep existing DuckDuckGo functions as fallbacks initially
- Implement feature flags to switch between search providers
- Add comprehensive error handling and logging

### 4.2 Update Tool Registry
```python
AVAILABLE_TOOLS = {
    'crewai_search_web': crewai_search_web,
    'serper_search': SerperDevTool(),
    'legacy_search_web': search_web,  # Fallback
    'search_knowledge_base': search_knowledge_base
}
```

### 4.3 Backward Compatibility
- Maintain existing function signatures
- Preserve output formats
- Keep error handling patterns

## Phase 5: Testing & Validation

### 5.1 Update Tests
- Modify `tests/test_integration_phase2.py` for CrewAI tools
- Add mock tests for API failures
- Test rate limiting and error handling

### 5.2 Integration Testing
- Test with `test_tool_calls.py`
- Validate agent task execution
- Check search result formatting

### 5.3 Performance Testing
- Compare response times
- Test with various query types
- Validate result quality

## Phase 6: Cleanup & Documentation

### 6.1 Remove Legacy Code
- Remove `duckduckgo-search` dependency
- Clean up old `AgenticSearchTool` class
- Update imports and references

### 6.2 Update Documentation
- Update README with new setup instructions
- Document API key requirements
- Update troubleshooting guides

## Implementation Benefits

1. **Reliability**: CrewAI tools are more stable and maintained
2. **Rate Limiting**: Built-in handling of API limits
3. **Error Handling**: Better error management and fallbacks
4. **Scalability**: Professional-grade search APIs
5. **Integration**: Seamless with existing CrewAI migration plans

## Potential Challenges & Solutions

### Challenge 1: API Costs
- **Solution**: SerperDev has free tier (2,500 searches/month)
- **Monitoring**: Implement usage tracking

### Challenge 2: API Key Management
- **Solution**: Use environment variables and `.env` files
- **Security**: Never commit API keys to version control

### Challenge 3: Rate Limiting
- **Solution**: Implement exponential backoff and caching
- **Fallback**: Use multiple search providers

### Challenge 4: Format Differences
- **Solution**: Create adapters to maintain existing interfaces
- **Testing**: Comprehensive format validation

## Migration Timeline

- **Phase 1-2**: 1-2 hours (setup and API keys)
- **Phase 3-4**: 4-6 hours (core implementation and migration)
- **Phase 5**: 2-3 hours (testing and validation)
- **Phase 6**: 1-2 hours (cleanup and documentation)

**Total Estimated Time**: 8-13 hours

## File Changes Required

### New Files
- `.env` (API keys)
- `src/tools/crewai_tools.py` (new CrewAI tools)

### Modified Files
- `requirements.txt` (add CrewAI dependencies)
- `src/tools/tools.py` (update tool registry)
- `src/agents/agents.py` (update agent implementations)
- `tests/test_integration_phase2.py` (update tests)
- `test_tool_calls.py` (update test script)

### Removed Files
- None initially (gradual migration)

## Rollback Plan

If issues arise during migration:
1. Revert to DuckDuckGo functions using feature flags
2. Use git to rollback specific changes
3. Maintain both implementations temporarily
4. Gradual rollback by reverting phases in reverse order

## Success Criteria

- [ ] All existing tests pass
- [ ] Search functionality works reliably
- [ ] No increase in response times
- [ ] Error handling improved
- [ ] Rate limiting issues resolved
- [ ] API costs within budget
- [ ] Documentation updated

## Post-Migration Maintenance

1. Monitor API usage and costs
2. Update API keys as needed
3. Keep CrewAI tools updated
4. Monitor search result quality
5. Adjust rate limiting as needed

---

**Document Created**: January 2025
**Last Updated**: January 2025
**Status**: Ready for Implementation 