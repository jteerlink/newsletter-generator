# AI Multi-Agent Newsletter System: Implementation Plan (with Coding & Testing)

This document provides a concrete, step-by-step guide for building the AI Multi-Agent Newsletter System. It translates the project overview into an actionable development plan, detailing the specific libraries, commands, architectural patterns, and testing protocols to be used at each stage.

## Phase 1: Foundational LLM Setup

**Objective:** Establish a stable, local LLM environment and create the core Python scripts for interaction. This phase is about ensuring the engine of the system is running reliably before building anything on top of it.

### Task 1.1: Environment Setup

**Instructions for Coding Agent:**
- Execute `ollama serve` to ensure the service is running.
- Execute `python -m venv venv` to create a virtual environment.
- Activate the environment (`source venv/bin/activate` or `.\venv\Scripts\activate`).
- Create a `requirements.txt` file with the content:
  ```
  ollama
  python-dotenv
  pytest
  ```
- Execute `pip install -r requirements.txt`.

### Task 1.2: Model Selection and Testing

**Instructions for Coding Agent:**
- Execute `ollama pull llama3`, `ollama pull gemma3n`, and `ollama pull deepseek-r1`.
- Create a new file `benchmark.py`.
- In `benchmark.py`, write a script that imports `ollama` and `time`. The script should define a list of test prompts. It will then loop through a list of model names (`['llama3', 'gemma3n', 'deepseek-r1']`), and for each model, loop through the test prompts. For each prompt, it will record the time before and after an `ollama.chat()` call and print the duration.

### Task 1.3: Basic Python Integration

**Instructions for Coding Agent:**
- Create a `.env` file with the lines: `OLLAMA_MODEL="deepseek-r1"` (default), `OLLAMA_MODEL_GEMMA3N="gemma3n"`, `OLLAMA_MODEL_DEEPSEEK_R1="deepseek-r1"`.
- Create a file named `core.py`.
- In `core.py`, import `ollama`, `os`, `logging`, and `load_dotenv` from `dotenv`.
- Call `load_dotenv()`.
- Configure logging to write to `interaction.log` with a INFO level.
- Create a function `query_llm(prompt: str) -> str`. This function will read the model name from the environment variable. It will contain a `try...except ollama.ResponseError as e:` block. On success, it logs the response and returns it. On failure, it logs the error `e` and returns an error message string.

### Task 1.4: Prompt Engineering Foundation

**Instructions for Coding Agent:**
- Create a `prompts.py` file.
- Inside, define several functions that take variables as arguments and return formatted prompt strings (f-strings), such as `get_research_topic_prompt(topic: str)`.
- Create a script `test_prompts_ab.py` that imports functions from `prompts.py` and `query_llm` from `core.py`. The script should define two variations of a prompt for the same task, call `query_llm` for each, and print the results side-by-side for comparison.

### Testing Strategy (Phase 1)

**Unit Testing:**
- Create a `tests/` directory with an empty `__init__.py` file.
- Create `tests/test_core.py`.
- In `test_core.py`, use `pytest` and `unittest.mock.patch` to test `query_llm`.
- Write a test that mocks `ollama.chat` to return a successful response and assert that `query_llm` returns the expected text.
- Write another test that makes the mocked `ollama.chat` raise a `ResponseError` and assert that `query_llm` returns your designated error string.
- Create `tests/test_prompts.py` to test the functions in `prompts.py`, asserting they return strings and contain the input variables.

**Integration Testing:**
- Create `tests/test_integration_phase1.py`.
- This test will import `query_llm` and call it with a simple prompt like "Hello".
- It will not use mocks. It will assert that the returned string is not empty and does not contain the error message. This validates that the connection to the running Ollama service is working correctly.

## Phase 2: Agent with Tools (Search & RAG)

**Objective:** Give the foundational LLM access to external information, enabling it to answer questions about recent events and consult a private knowledge base.

**Revised Task Order:**

### Task 2.1: Vector Database Setup

**Instructions for Coding Agent:**
- Add `chromadb`, `sentence-transformers`, and `langchain_text_splitters` to `requirements.txt` and install.
- Create `vector_db.py`.
- Inside, create a function `get_db_collection(path="/path/to/db", name="newsletter_content")` that returns a ChromaDB collection object using a persistent client.
- **Chunking Strategy:** Use `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=100` for all text chunking.

### Task 2.2: Content Ingestion Pipeline

**Instructions for Coding Agent:**
- Create `ingest.py`.
- The script should accept a file path as a command-line argument.
- It will read the file's content, then call the `add_text_to_db` function from `vector_db.py` to process and store it using the chunking strategy above.

### Task 2.3: Web Search Integration

**Instructions for Coding Agent:**
- Add `duckduckgo-search` to `requirements.txt` and install.
- Create `tools.py`. Inside, import `DDGS` from `duckduckgo_search` and `@tool` from `crewai_tools`.
- Define a function `search_web(query: str) -> str`. Decorate it with `@tool("Web Search Tool")`.
- The function should use `DDGS().text()` to get search results, then format the top 3-5 results into a single string containing the title, href, and body for each.

### Task 2.4: CrewAI Framework and Agent Setup

**Instructions for Coding Agent:**
- Add `crewai` and `crewai[tools]` to `requirements.txt` and install.
- Create `agents.py`. Import `Agent` from `crewai` and your tools from `tools.py`.
- Define a `ResearchAgent` with the specified role, goal, and the list of tools from `tools.py`. Configure its `llm` parameter to use Ollama via CrewAI's integration.
- Create `main.py`. Import the agent, `Task`, and `Crew`.
- Define a research task with a specific topic, assigning it to your `ResearchAgent`.
- Instantiate the `Crew` with the agent and task, then call `crew.kickoff()`.

### Task 2.5: Integration and End-to-End Testing

**Instructions for Coding Agent:**
- After each major step, add and run relevant unit tests.
- Create a final integration test that:
  - Programmatically creates a temporary ChromaDB instance.
  - Uses `ingest.py` logic to add a known text document (e.g., "The sky is blue and contains clouds.") to the database.
  - Instantiates the full `ResearchAgent` and task.
  - Runs the crew with a query like "What color is the sky?".
  - Asserts that the final output from the crew contains the word "blue". This verifies the entire RAG pipeline is working end-to-end.

### Testing Strategy (Phase 2)

**Unit Testing:**
- `tests/test_vector_db.py`: Test collection creation and chunking logic.
- `tests/test_ingestion.py`: Test the ingestion script by having it read a temporary file and mocking the `add_text_to_db` call to verify the file content is read and passed correctly.
- `tests/test_tools.py`: Mock the `DDGS` and `chromadb` clients. For `search_web`, provide a sample search result payload and assert that your function correctly parses and formats it. For vector DB search, assert the query is passed correctly.
- `tests/test_agents.py`: Test that your `ResearchAgent` instantiates correctly and has the right tools assigned to it.

**Integration Testing:**
- `tests/test_integration_phase2.py`: Run the full pipeline as described above.

## Phase 2.5: Advanced Content Scraping with Crawl4AI

**Objective:** Implement a sophisticated web scraping system using crawl4ai to replace existing scrapers with enhanced content extraction, JavaScript handling, and intelligent content processing.

### Prerequisites
- Phase 2 vector database and content ingestion pipeline must be completed
- Existing `sources.yaml` configuration and `DataProcessor` classes are in place

### Task 2.5.1: Crawl4AI Environment Setup

**Instructions for Coding Agent:**
- Add the following dependencies to `requirements.txt`:
  ```
  crawl4ai>=0.6.3
  playwright>=1.40.0
  pydantic>=2.5.0
  ```
- Execute `pip install -r requirements.txt`
- Run `playwright install` to set up browser dependencies

### Task 2.5.2: Core Crawl4AI Scraper Implementation

**Instructions for Coding Agent:**
- Create `src/scrapers/crawl4ai_scraper.py`
- Implement `Crawl4aiScraper` class that inherits from existing scraper interface:
  ```python
  class Crawl4aiScraper(BaseScraper):
      async def extract_from_source(self, source: SourceConfig) -> List[Article]
      async def extract_from_multiple_sources(self, sources: List[SourceConfig]) -> List[Article]
  ```
- Use `AsyncWebCrawler` with `DefaultMarkdownGenerator` and `PruningContentFilter` for clean content extraction
- Implement batch processing with `arun_many()` for efficient parallel scraping
- Add comprehensive error handling and retry logic

### Task 2.5.3: Content Extraction Strategies

**Instructions for Coding Agent:**
- Create `src/scrapers/extraction_strategies.py`
- Implement multiple extraction strategies:
  - **BasicMarkdownStrategy**: Simple markdown conversion for basic content
  - **EnhancedContentStrategy**: Uses `PruningContentFilter` for cleaner output
  - **StructuredDataStrategy**: Uses `JsonCssExtractionStrategy` for specific data extraction
  - **LLMPoweredStrategy**: Uses `LLMExtractionStrategy` for complex content analysis
- Create strategy factory for automatic strategy selection based on source configuration

### Task 2.5.4: Configuration Enhancement

**Instructions for Coding Agent:**
- Create `src/scrapers/crawl4ai_config.py`
- Extend existing `SourceConfig` class to support crawl4ai-specific settings:
  ```python
  class EnhancedSourceConfig(SourceConfig):
      scraper_type: str = "traditional"  # "crawl4ai" or "traditional"
      extraction_strategy: str = "basic"
      crawl4ai_config: Optional[Dict[str, Any]] = None
  ```
- Add configuration validation and defaults
- Implement configuration migration utilities

### Task 2.5.5: Enhanced Data Processing Integration

**Instructions for Coding Agent:**
- Modify `src/scrapers/data_processor.py` to handle crawl4ai results
- Add crawl4ai-specific metadata to article storage:
  ```python
  metadata = {
      "extraction_method": "crawl4ai",
      "content_type": "markdown", 
      "quality_indicators": {
          "word_count": len(content.split()),
          "links_found": len(result.links.internal + result.links.external),
          "media_count": len(result.media.images)
      }
  }
  ```
- Implement enhanced content quality scoring based on crawl4ai extraction results
- Add local storage for raw crawl4ai outputs in `data/crawl4ai_raw/`

### Task 2.5.6: Vector Database Integration Enhancement

**Instructions for Coding Agent:**
- Enhance existing `VectorStore` integration in `src/storage/vector_store.py`
- Use crawl4ai's clean markdown output for better chunking quality
- Implement enhanced metadata enrichment:
  - Document structure preservation
  - Link relationship mapping
  - Content quality indicators
- Add deduplication based on content hashes from crawl4ai results

### Task 2.5.7: Parallel Scraper Implementation

**Instructions for Coding Agent:**
- Create `src/scrapers/hybrid_scraper.py` to support both traditional and crawl4ai scrapers
- Implement smart fallback logic: crawl4ai primary, traditional backup
- Add performance comparison logging
- Enable gradual migration from traditional to crawl4ai scrapers per source

### Testing Strategy (Phase 2.5)

**Unit Testing:**
- `tests/scrapers/test_crawl4ai_scraper.py`: Test core scraper functionality with mocked crawl4ai responses
- `tests/scrapers/test_extraction_strategies.py`: Test each extraction strategy independently
- `tests/scrapers/test_crawl4ai_config.py`: Test configuration validation and migration
- `tests/scrapers/test_hybrid_scraper.py`: Test fallback logic and strategy selection

**Integration Testing:**
- `tests/test_integration_crawl4ai.py`: End-to-end test with real websites:
  - Test markdown extraction quality compared to traditional scraper
  - Verify vector database integration with crawl4ai content
  - Test batch processing performance
  - Validate error handling and retry mechanisms

**Performance Testing:**
- `tests/test_crawl4ai_performance.py`: Compare extraction speed and quality:
  - Traditional scraper vs crawl4ai scraper
  - Sequential vs parallel processing
  - Memory usage and resource consumption

### Acceptance Criteria (Phase 2.5)

- [ ] Crawl4ai scraper successfully extracts content from all existing sources in `sources.yaml`
- [ ] Content quality improved (measured by word count, structure preservation, cleanliness)
- [ ] JavaScript-heavy sites successfully scraped (test with dynamic content sites)
- [ ] Integration with existing vector database maintains all functionality
- [ ] Performance equal or better than existing scrapers
- [ ] Graceful fallback to traditional scrapers when crawl4ai fails
- [ ] All existing tests continue to pass
- [ ] New comprehensive test coverage for crawl4ai components

### Migration Strategy

1. **Week 1**: Implement core infrastructure (Tasks 2.5.1-2.5.3)
2. **Week 2**: Configuration and integration (Tasks 2.5.4-2.5.6)  
3. **Week 3**: Testing and optimization (Task 2.5.7 + comprehensive testing)
4. **Week 4**: Gradual source migration and performance validation

## Phase 3: Multi-Agent Team (Core System)

**Objective:** To assemble the full, specialized team of agents that collaborate to produce a complete newsletter draft.

### Task 3.1-3.3: Agent Roles, Communication, and Orchestration

**Instructions for Coding Agent:**
- In `agents.py`, define the `PlannerAgent`, `WriterAgent`, and `EditorAgent` classes, each with a detailed system prompt in its backstory. The Planner and Writer will not have tools.
- In `main.py`, define four distinct `Task` objects: `plan_task`, `research_task`, `write_task`, `edit_task`.
- Set the task dependencies using the `context` parameter. `research_task` will have `context=[plan_task]`. `write_task` will have `context=[research_task]`. `edit_task` will have `context=[write_task]`.
- Assemble the final `Crew` with the list of all four agents and the list of the four tasks in order. The `kickoff()` call will now orchestrate the full workflow.

### Task 3.4: Quality Assurance System

**Instructions for Coding Agent:**
- Refine the `EditorAgent`'s goal and backstory in `agents.py`.
- The prompt should explicitly instruct the agent to check for specific criteria (Clarity, Factual Correctness, Tone) and to produce a final section in its output titled "QUALITY SCORECARD" with a rating for each.

### Task 3.5: Human Review Interface

**Instructions for Coding Agent:**
- In `main.py`, after `crew.kickoff()` returns its result, add code to write the string result to a file named `final_newsletter.md`.
- After saving, `print()` the result to the console and use `input("Approve this draft? (y/n): ")` to capture feedback. Log this feedback to a simple text file for now.

### Testing Strategy (Phase 3)

**Unit Testing:**
- `tests/test_agents.py`: Add simple tests to ensure the new agents (Planner, Writer, Editor) can be instantiated without errors.

**Integration Testing:**
- `tests/test_integration_phase3.py`: This is the most critical test yet.
- The test will set up and run the entire four-agent crew on a fixed topic (e.g., "The impact of AI on software development").
- The test will take time to run. It should assert the following:
  - The `kickoff()` method completes without raising an exception.
  - The returned result is a non-empty string.
  - The final result string contains keywords that indicate each agent did its job, such as "Outline", "Sources", and "QUALITY SCORECARD".
  - The `final_newsletter.md` file is created.

## Phase 4: Advanced Features & Self-Improvement

**Objective:** Evolve the system from a static workflow to a dynamic, learning system that improves over time.

### Task 4.1: Agentic RAG Implementation

**Instructions for Coding Agent:**
- Refactor the `search_web` function in `tools.py` into a class, `AgenticSearchTool`.
- The `run` method of this class will contain a for loop (e.g., for 3 iterations).
- Inside the loop, it performs a search, then makes a separate LLM call to evaluate if the results are sufficient. If not, it uses the LLM to generate a better query for the next iteration. It accumulates results from all iterations.
- The `ResearchAgent`'s tool will be an instance of this class.

### Task 4.2: Advanced Inter-Agent Communication

**Instructions for Coding Agent:**
- In `main.py`, import `Process` from `crewai`.
- Define a new `ManagerAgent` in `agents.py`.
- Instead of a linear task list, structure the `Crew` with `process=Process.hierarchical` and set the `manager_llm` to your Ollama model. The manager agent will now orchestrate the subordinate agents.

### Task 4.3: Self-Learning Loop Implementation

**Instructions for Coding Agent:**
- Create `feedback.py`. The `log_feedback` function from Phase 3 will now write to a JSON file, storing the final output and the y/n feedback together.
- Create `analyze_feedback.py`. This script will read the JSON log, find all drafts marked "n", and use an LLM to "identify a common theme in the rejected drafts and suggest one change to the Editor agent's prompt to fix it."

### Task 4.4: Advanced Quality Control

**Instructions for Coding Agent:**
- In `main.py`, before defining the `edit_task`, add code to read a "gold_standard_example.md" file from disk.
- Modify the description of the `edit_task` to include the content of this example file, and instruct the Editor agent to compare the draft to this example.

### Task 4.5: Scalability and Performance Optimization

**Instructions for Coding Agent:**
- Add `from functools import lru_cache` to `tools.py`.
- Add the `@lru_cache(maxsize=16)` decorator to the main search method in your `AgenticSearchTool` class.
- Refactor tool methods that perform I/O (web searches, DB queries) to be `async def`. Mark the corresponding tasks in `main.py` as `async_execution=True`. CrewAI will handle the concurrent execution.

### Testing Strategy (Phase 4)

**Unit Testing:**
- `tests/test_tools.py`: Add a test for the `AgenticSearchTool` class. Mock the search and the LLM evaluation call, and assert that the tool performs multiple loops when the evaluator mock returns "insufficient".
- `tests/test_feedback.py`: Test the JSON logging and analysis functions.

**Integration Testing:**
- `tests/test_integration_phase4_performance.py`:
  - Run the full Phase 3 crew and record the execution time.
  - Run the full Phase 4 crew (with async and caching) on the same topic.
  - Assert that the Phase 4 execution time is significantly less than the Phase 3 time.
- A manual test will be required to validate the hierarchical process, observing the logs to ensure the manager agent is delegating tasks as expected.