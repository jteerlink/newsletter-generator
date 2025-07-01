# AI Multi-Agent Newsletter System

## Project Overview
This project is an AI-powered multi-agent system designed to automate the research, drafting, and editing of newsletter content. The system leverages local LLMs (via Ollama) and is structured for extensibility, robust testing, and future multi-agent collaboration.

---

## Phase 1: Foundational LLM Setup

### 1. Environment Setup
- Ensure [Ollama](https://ollama.com/) is installed and running:
  ```sh
  ollama serve
  ```
- Create and activate a Python virtual environment:
  ```sh
  python -m venv venv
  source venv/bin/activate  # On Windows: .\venv\Scripts\activate
  ```
- Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```

### 2. Model Selection
- Pull the required models:
  ```sh
  ollama pull llama3
  ollama pull gemma3n
  ollama pull deepseek-r1
  ```

### 3. Configuration
- Create a `.env` file in the project root with the following content:
  ```ini
  OLLAMA_MODEL=llama3
  OLLAMA_MODEL_GEMMA3N=gemma3n
  OLLAMA_MODEL_DEEPSEEK_R1=deepseek-r1
  ```
  - `OLLAMA_MODEL` is the default model used by the system.

---

## Usage

### Run Benchmarks
Benchmark all models and prompts, saving results to CSV:
```sh
python benchmark.py
```
Results will be saved to `benchmark_results.csv`.

### Run Tests
Run all unit and integration tests:
```sh
pytest --maxfail=3 --disable-warnings -v
```

### Try Prompt Engineering
Compare prompt variations and LLM responses:
```sh
python test_prompts_ab.py
```

---

## Troubleshooting
- **Ollama not running:** Ensure you have started the Ollama server with `ollama serve`.
- **Model not found:** Make sure you have pulled all required models.
- **.env issues:** Double-check your `.env` file for typos and correct variable names.
- **Test failures:** Review error messages for missing dependencies or misconfigurations.

---

## Next Steps
- Proceed to Phase 2 for vector database and web search integration.
- See `product_plan.md` for the full implementation roadmap. 