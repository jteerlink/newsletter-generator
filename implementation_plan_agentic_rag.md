# Agentic RAG Implementation Plan

## Overview
This document outlines the steps and architectural changes required to convert the current newsletter system to an **Agentic RAG (Retrieval-Augmented Generation)** architecture. The plan incorporates advanced agentic behaviors, modularity, and feedback mechanisms for robust, iterative, and high-quality information retrieval and generation.

---

## Open Source Tools & Frameworks
To ensure the system is cost-effective and fully local, the following open source tools and frameworks will be leveraged:

### Large Language Models (LLMs)
- **Ollama**: Local LLM runner supporting models like Llama 2, Mistral, and others ([ollama.com](https://ollama.com/))
- **Open LLMs**: Models such as Llama 2, Mistral, DeepSeek, and others that can be run locally via Ollama or similar frameworks

### Vector Database & Embeddings
- **ChromaDB**: Open source vector database for storing and retrieving embeddings ([chromadb.com](https://www.trychroma.com/))
- **Sentence-Transformers**: For generating embeddings locally (e.g., all-MiniLM-L6-v2)
- **FAISS** (optional): Facebook AI Similarity Search for high-performance local vector search ([github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss))

### Agent Framework & Orchestration
- **Custom Python Agent Framework**: Built on top of your existing agent base classes, with extensibility for agentic behaviors
- **FastAPI**: For building local APIs and agent orchestration endpoints ([fastapi.tiangolo.com](https://fastapi.tiangolo.com/))
- **Asyncio**: For asynchronous agent communication and orchestration

### Data Processing & Retrieval
- **Scrapy**: For web scraping and data collection ([scrapy.org](https://scrapy.org/))
- **DuckDuckGo Search (duckduckgo-search)**: For web search without API costs ([github.com/deedy5/duckduckgo-search](https://github.com/deedy5/duckduckgo-search))
- **NLTK / spaCy**: For NLP preprocessing and text analysis ([nltk.org](https://www.nltk.org/), [spacy.io](https://spacy.io/))

### Logging, Monitoring, and Analytics
- **Python Logging**: Standard library for logging agent actions and system events
- **Prometheus + Grafana** (optional): For local monitoring and dashboarding ([prometheus.io](https://prometheus.io/), [grafana.com](https://grafana.com/))

### Storage & State
- **SQLite**: Lightweight, file-based database for agent state and metadata
- **Redis** (optional): For fast, in-memory state management ([redis.io](https://redis.io/))

### Testing
- **Pytest**: For unit and integration testing ([pytest.org](https://docs.pytest.org/))

All selected tools are open source, free to use, and can be run entirely on local hardware without incurring any cloud or API costs.

---

## Phase 1: Foundation & Agentic Infrastructure

### 1.1 Agent State & Memory
- **Implement per-agent memory/state objects**
  - Store past queries, responses, retrieved contexts, and evaluation results
  - Use in-memory or persistent storage (e.g., Redis, SQLite, or file-based)
  - Integrate memory access into agent base classes

### 1.2 Agent Communication & Orchestration
- **Enhance agent communication protocol**
  - Support message types for critique, delegation, escalation, and feedback
  - Add correlation IDs for tracking multi-step workflows
- **Create/extend an Orchestrator (e.g., RAGOrchestrator)**
  - Manage agent handoffs, iteration control, and workflow state

---

## Phase 2: Agentic RAG Pipeline Implementation

### 2.1 Query Rewriting Agent
- **Develop QueryWriterAgent**
  - Refine, clarify, and correct user queries before retrieval
  - Log all query transformations in agent memory

### 2.2 Context Need Assessment Agent
- **Develop ContextAssessmentAgent**
  - Decide if more context is needed before LLM call
  - Use heuristics or LLM-based classification

### 2.3 Source Selection & Multi-Source Retrieval
- **Develop SourceSelectorAgent**
  - Dynamically select one or more sources (vector DB, web, APIs)
  - Support multi-hop retrieval and merging of results
  - Log source selection rationale in memory

### 2.4 Dynamic Tool Selection & Chaining
- **Allow agents to select and chain MCP tools**
  - Implement logic for tool selection based on query type and context
  - Enable chaining (e.g., web search → vector search)

### 2.5 Prompt Construction & LLM Call
- **Develop PromptBuilder utility/agent**
  - Assemble prompt from refined query and retrieved context
  - Log prompt construction steps

### 2.6 Response Evaluation & Self-Critique
- **Develop ResponseEvaluatorAgent**
  - Assess LLM output for relevance, completeness, and factuality
  - Add self-critique step (optionally use a different LLM or prompt)
  - Store evaluation results in agent memory

### 2.7 Iteration & Control Flow
- **Implement iteration control in orchestrator**
  - Set max iteration count per query
  - Allow agents to gracefully exit or escalate if answer is not found

---

## Phase 3: Advanced Agentic Features

### 3.1 Uncertainty Estimation & Escalation
- **Add confidence scoring to ResponseEvaluatorAgent**
  - If confidence is low, escalate to human or return a fallback message

### 3.2 Agent Collaboration & Specialization
- **Enable agent delegation and collaboration**
  - Use agent registry to match tasks to specialized agents (e.g., technical, news, summary)
  - Allow agents to request help or delegate subtasks

### 3.3 Logging, Monitoring, and Analytics
- **Implement centralized logging for all agent actions, tool calls, and decisions**
  - Track metrics: retrieval latency, answer quality, iteration counts, escalation rates
  - Use logs for debugging and system improvement

### 3.4 User Feedback Integration
- **Add user feedback collection on responses**
  - Store feedback in agent memory and/or analytics DB
  - Use feedback to inform future agent decisions and retraining

---

## Phase 4: Extensibility & Testing

### 4.1 Extensible Agent & Tool Framework
- **Refactor agents and tools to use abstract base classes and plugin/registration system**
  - Make it easy to add new agent types and retrieval tools

### 4.2 Testing & Validation
- **Develop unit and integration tests for all new agentic behaviors**
  - Test multi-agent workflows, memory persistence, feedback loops, and escalation

---

## Milestones & Success Criteria

- [ ] Agents maintain and utilize state/memory across iterations
- [ ] Self-critique and feedback loops improve answer quality
- [ ] Multi-source and multi-hop retrieval supported
- [ ] Dynamic tool selection and chaining operational
- [ ] Uncertainty estimation and escalation in place
- [ ] Agent collaboration and delegation functional
- [ ] Centralized logging and analytics available
- [ ] User feedback loop integrated
- [ ] System is extensible and well-tested

---

## Next Steps
1. Review and approve this plan
2. Assign tasks and begin Phase 1 implementation
3. Schedule regular reviews to track progress and adjust as needed 