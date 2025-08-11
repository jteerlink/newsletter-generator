"""Helper utilities for managing imports and dependencies."""

import importlib
import logging
from typing import Any, Dict, List, Optional

from .exceptions import ImportError

logger = logging.getLogger(__name__)


class ImportHelper:
    """Helper class for managing imports and dependencies."""

    @staticmethod
    def safe_import(
            module_name: str,
            package_name: str = None) -> Optional[Any]:
        """Safely import a module with fallback handling."""
        try:
            if package_name:
                module = importlib.import_module(module_name, package_name)
            else:
                module = importlib.import_module(module_name)
            logger.debug(f"Successfully imported {module_name}")
            return module
        except ImportError as e:
            logger.warning(f"Failed to import {module_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error importing {module_name}: {e}")
            return None

    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """Check if all required dependencies are available."""
        dependencies = {
            "ollama": "ollama",
            "chromadb": "chromadb",
            "streamlit": "streamlit",
            "requests": "requests",
            "beautifulsoup4": "bs4",
            "feedparser": "feedparser",
            "python-dotenv": "dotenv",
            "pydantic": "pydantic",
            "scikit-learn": "sklearn",
            "crawl4ai": "crawl4ai"
        }

        results = {}
        for name, module in dependencies.items():
            results[name] = ImportHelper.safe_import(module) is not None

        return results

    @staticmethod
    def get_optional_dependencies() -> Dict[str, bool]:
        """Check optional dependencies."""
        optional_deps = {
            "plotly": "plotly",
            "pandas": "pandas",
            "numpy": "numpy",
            "matplotlib": "matplotlib"
        }

        results = {}
        for name, module in optional_deps.items():
            results[name] = ImportHelper.safe_import(module) is not None

        return results

    @staticmethod
    def check_import_errors() -> Dict[str, str]:
        """Check for specific import errors and return error messages."""
        error_messages = {}

        # Check for common import issues
        try:
            import chromadb
        except ImportError as e:
            error_messages["chromadb"] = str(e)

        try:
            import ollama
        except ImportError as e:
            error_messages["ollama"] = str(e)

        try:
            import crawl4ai
        except ImportError as e:
            error_messages["crawl4ai"] = str(e)

        return error_messages

    @staticmethod
    def get_missing_dependencies() -> List[str]:
        """Get list of missing required dependencies."""
        dependencies = ImportHelper.check_dependencies()
        missing = [name for name, available in dependencies.items()
                   if not available]
        return missing

    @staticmethod
    def validate_environment() -> Dict[str, Any]:
        """Validate the current environment and dependencies."""
        required_deps = ImportHelper.check_dependencies()
        optional_deps = ImportHelper.get_optional_dependencies()
        import_errors = ImportHelper.check_import_errors()
        missing_deps = ImportHelper.get_missing_dependencies()

        return {
            "required_dependencies": required_deps,
            "optional_dependencies": optional_deps,
            "import_errors": import_errors,
            "missing_dependencies": missing_deps,
            "environment_valid": len(missing_deps) == 0
        }


def safe_import_core():
    """Safely import core modules with proper error handling."""
    try:
        from .core import query_llm
        return query_llm
    except ImportError as e:
        logger.error(f"Failed to import core module: {e}")
        raise ImportError(f"Core module import failed: {e}")


def safe_import_agents():
    """Safely import agent modules."""
    try:
        from agents.agents import EditorAgent, ResearchAgent, SimpleAgent, WriterAgent
        return SimpleAgent, ResearchAgent, WriterAgent, EditorAgent
    except ImportError as e:
        logger.error(f"Failed to import agent modules: {e}")
        raise ImportError(f"Agent module import failed: {e}")


def safe_import_tools():
    """Safely import tool modules."""
    try:
        from tools.tools import search_knowledge_base, search_web
        return search_web, search_knowledge_base
    except ImportError as e:
        logger.error(f"Failed to import tool modules: {e}")
        raise ImportError(f"Tool module import failed: {e}")


def safe_import_scrapers():
    """Safely import scraper modules."""
    try:
        from scrapers.crawl4ai_web_scraper import WebScraperWrapper
        return WebScraperWrapper
    except ImportError as e:
        logger.error(f"Failed to import scraper modules: {e}")
        raise ImportError(f"Scraper module import failed: {e}")


def safe_import_storage():
    """Safely import storage modules."""
    try:
        from storage import ChromaStorageProvider
        return ChromaStorageProvider
    except ImportError as e:
        logger.error(f"Failed to import storage modules: {e}")
        raise ImportError(f"Storage module import failed: {e}")


def check_system_health() -> Dict[str, Any]:
    """Check overall system health and dependency status."""
    health_report = {
        "required_dependencies": ImportHelper.check_dependencies(),
        "optional_dependencies": ImportHelper.get_optional_dependencies(),
        "core_modules": {},
        "issues": []
    }

    # Check core module imports
    try:
        health_report["core_modules"]["query_llm"] = safe_import_core(
        ) is not None
    except Exception as e:
        health_report["issues"].append(f"Core module import failed: {e}")
        health_report["core_modules"]["query_llm"] = False

    try:
        health_report["core_modules"]["agents"] = safe_import_agents() is not None
    except Exception as e:
        health_report["issues"].append(f"Agent module import failed: {e}")
        health_report["core_modules"]["agents"] = False

    try:
        health_report["core_modules"]["tools"] = safe_import_tools() is not None
    except Exception as e:
        health_report["issues"].append(f"Tool module import failed: {e}")
        health_report["core_modules"]["tools"] = False

    try:
        health_report["core_modules"]["scrapers"] = safe_import_scrapers(
        ) is not None
    except Exception as e:
        health_report["issues"].append(f"Scraper module import failed: {e}")
        health_report["core_modules"]["scrapers"] = False

    try:
        health_report["core_modules"]["storage"] = safe_import_storage(
        ) is not None
    except Exception as e:
        health_report["issues"].append(f"Storage module import failed: {e}")
        health_report["core_modules"]["storage"] = False

    return health_report
