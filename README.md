# AI Multi-Agent Newsletter Generator

## Project Overview
This project is an AI-powered multi-agent system designed to automate the research, drafting, and editing of newsletter content. The system leverages local LLMs (via Ollama), web scraping with Crawl4AI, and features a modern Streamlit web interface for easy interaction.

## 🌟 Key Features
- **Multi-Agent Architecture**: Specialized agents for research, writing, and editing
- **Web Interface**: Modern Streamlit UI with real-time progress tracking
- **Advanced Web Scraping**: Crawl4AI integration for comprehensive content extraction
- **Flexible Workflows**: Standard multi-agent or hierarchical manager-led workflows
- **Content Customization**: Audience targeting, length control, and quality focus areas
- **Performance Analytics**: Built-in benchmarking and feedback collection

## 📁 Project Structure
```
newsletter-generator/
├── src/                     # Core application code
│   ├── agents/             # Multi-agent system implementation
│   ├── core/               # Core functionality and prompts
│   ├── scrapers/           # Web scraping and content extraction
│   ├── storage/            # Vector database and storage
│   └── tools/              # AI tools and utilities
├── streamlit/              # Web interface
│   ├── streamlit_app_enhanced.py  # Enhanced UI (recommended)
│   ├── ui_components.py           # Reusable UI components
│   └── launch_streamlit.py        # Smart launcher
├── docs/                   # Documentation and analysis
├── dev/                    # Development and testing files
├── tests/                  # Unit and integration tests
├── configs/                # Configuration files
└── data/                   # Data storage and outputs
```

## 🚀 Quick Start

### 1. Environment Setup
- Install [Ollama](https://ollama.com/) and start the server:
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

### 2. Model Setup
Pull the required models:
```sh
ollama pull llama3
ollama pull gemma3n
ollama pull deepseek-r1
```

### 3. Configuration
Create a `.env` file in the project root:
```ini
OLLAMA_MODEL=llama3
OLLAMA_MODEL_GEMMA3N=gemma3n
OLLAMA_MODEL_DEEPSEEK_R1=deepseek-r1
```

### 4. Launch the Web Interface
```sh
python streamlit/launch_streamlit.py
```

Or run directly:
```sh
streamlit run streamlit/streamlit_app_enhanced.py
```

## 💻 Usage Options

### Web Interface (Recommended)
The Streamlit interface provides:
- Topic input with validation
- Audience selection (7+ types)
- Workflow choice (Standard vs Hierarchical)
- Content length and quality controls
- Real-time progress tracking
- Multi-format output with download options

### Command Line
```sh
python src/main.py
```

### Development & Testing
```sh
# Run all tests
pytest --maxfail=3 --disable-warnings -v

# Run benchmarks
python dev/benchmark.py

# Test specific integrations
python dev/test_crawl4ai_integration.py
```

## 📖 Documentation
- **[docs/](docs/)** - Architecture, integration guides, and project analysis
- **[streamlit/README.md](streamlit/README.md)** - Web interface documentation
- **[dev/README.md](dev/README.md)** - Development and testing guide

## 🧪 Development
- **Development files**: Located in `dev/` directory
- **Testing**: Comprehensive test suite in `tests/`
- **Benchmarking**: Performance analysis tools in `dev/`

## 🔧 Troubleshooting
- **Ollama not running**: Ensure `ollama serve` is running
- **Model not found**: Verify models are pulled with `ollama list`
- **Import errors**: Check virtual environment activation and dependencies
- **Streamlit issues**: Try the launcher script for dependency checking

## 📈 Next Steps
- Explore the web interface for easy newsletter generation
- Review documentation in `docs/` for system architecture
- Check `docs/product_plan.md` for future roadmap
- Contribute to development using files in `dev/`

---

*For detailed setup instructions and advanced features, see the documentation in the `docs/` directory.* 