import os
from pathlib import Path
from dotenv import load_dotenv
import yaml
from typing import List, Dict, Optional, Any
from loguru import logger

# --- Project Root ---
# Determine the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# --- Load Environment Variables ---
# Load .env file from the project root
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loaded environment variables from: {env_path}")
else:
    logger.info(f".env file not found at {env_path}. Relying on system environment variables.")

# --- Configuration File ---
# Load a YAML configuration file (e.g., config.yaml) from the project root
# This allows for more complex configurations than just .env
CONFIG_FILE_PATH = PROJECT_ROOT / "config.yaml"
APP_CONFIG: Dict[str, Any] = {}

if CONFIG_FILE_PATH.exists():
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            APP_CONFIG = yaml.safe_load(f)
        logger.info(f"Loaded application configuration from: {CONFIG_FILE_PATH}")
    except Exception as e:
        logger.error(f"Error loading {CONFIG_FILE_PATH}: {e}. Using defaults and environment variables.")
else:
    logger.info(f"Configuration file {CONFIG_FILE_PATH} not found. Using defaults and environment variables.")


# --- Helper Function to Get Config Value ---
def get_config_value(key_path: str, default: Optional[Any] = None, env_var: Optional[str] = None) -> Any:
    """
    Retrieves a configuration value from environment variables (highest priority),
    then from the YAML config file, and finally uses a default value.
    key_path: Dot-separated path for nested YAML keys (e.g., "embedding.model_name").
    env_var: The specific environment variable to check.
    """
    # 1. Check environment variable
    if env_var:
        value = os.getenv(env_var)
        if value is not None:
            # Attempt to parse common types
            if value.lower() in ['true', 'false']:
                return value.lower() == 'true'
            if value.isdigit():
                return int(value)
            try:
                return float(value)
            except ValueError:
                pass # Not a float
            return value # Return as string

    # 2. Check YAML config
    keys = key_path.split('.')
    val = APP_CONFIG
    try:
        for key in keys:
            val = val[key]
        if val is not None:
            return val
    except (KeyError, TypeError):
        pass # Key not found or val is not a dict

    # 3. Return default
    if default is not None:
        return default

    # If no default and key not found, it might be an issue
    # logger.warning(f"Configuration key '{key_path}' not found and no default provided.")
    return None


# --- API Keys ---
OPENAI_API_KEY = get_config_value("api_keys.openai", env_var="OPENAI_API_KEY")
GOOGLE_API_KEY = get_config_value("api_keys.google", env_var="GOOGLE_API_KEY")
# Add other API keys as needed (e.g., Anthropic, Cohere)
# ANTHROPIC_API_KEY = get_config_value("api_keys.anthropic", env_var="ANTHROPIC_API_KEY")

OPENAI_BASE_URL = get_config_value("url.openai", env_var="OPENAI_BASE_URL")

# --- Paths ---
# Default paths, can be overridden by config.yaml or environment variables
DATA_DIR = Path(get_config_value("paths.data_dir", PROJECT_ROOT / "data", "RAG_DATA_DIR"))
REPOS_DIR = Path(get_config_value("paths.repos_dir", DATA_DIR / "repositories", "RAG_REPOS_DIR"))
INDEX_DIR = Path(get_config_value("paths.index_dir", DATA_DIR / "indexes", "RAG_INDEX_DIR"))
LOG_DIR = Path(get_config_value("paths.log_dir", PROJECT_ROOT / "logs", "RAG_LOG_DIR"))
GRAMMAR_DIR = Path(get_config_value("paths.grammar_dir", PROJECT_ROOT / "grammars", "RAG_GRAMMAR_DIR")) # For compiled tree-sitter grammars

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPOS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
GRAMMAR_DIR.mkdir(parents=True, exist_ok=True) # User might place compiled .so/.dll files here

# --- Logging Configuration ---
LOG_LEVEL = get_config_value("logging.level", "INFO", "RAG_LOG_LEVEL").upper()
LOG_FILE_PATH = LOG_DIR / get_config_value("logging.file_name", "rag_app.log", "RAG_LOG_FILE")

logger.add(LOG_FILE_PATH, level=LOG_LEVEL, rotation="10 MB", retention="7 days", compression="zip")
logger.info(f"Logging configured. Level: {LOG_LEVEL}, File: {LOG_FILE_PATH}")


# --- Embedding Model Configuration ---
EMBEDDING_MODEL_PROVIDER = get_config_value("embedding.provider", "openai", "RAG_EMBEDDING_PROVIDER") # e.g., "openai", "huggingface", "google"
EMBEDDING_MODEL_NAME = get_config_value("embedding.model_name", "text-embedding-3-small", "RAG_EMBEDDING_MODEL_NAME")
EMBEDDING_DIMENSIONS = get_config_value("embedding.dimensions", 1536, "RAG_EMBEDDING_DIMENSIONS") # Adjust if using different OpenAI models or Sentence Transformers
EMBEDDING_BATCH_SIZE = get_config_value("embedding.batch_size", 32, "RAG_EMBEDDING_BATCH_SIZE")

# --- Generator (LLM) Model Configuration ---
GENERATOR_MODEL_NAME = get_config_value("generator.model_name", "gpt-4o-mini", "RAG_GENERATOR_MODEL_NAME")
GENERATOR_TEMPERATURE = get_config_value("generator.temperature", 0.7, "RAG_GENERATOR_TEMPERATURE")
GENERATOR_PROMPT = get_config_value("generator.prompt", """
You are an expert AI assistant specializing in analyzing and explaining code repositories.
Your responses should be accurate, concise, and directly answer the user's query based on the provided context.
Format your answers using Markdown. Cite specific file paths or code constructs when relevant.
""", "RAG_GENERATOR_PROMPT")

# --- Tree-sitter Configuration ---
# List of supported languages for AST chunking and their grammar file names (if manually managed)
# Users should install language bindings via pip (e.g., tree-sitter-python)
# Or place compiled .so/.dll files in GRAMMAR_DIR and map them here.
# `tree_sitter_languages` package can simplify this.
TREE_SITTER_LANGUAGES: Dict[str, Dict[str, str]] = get_config_value(
    "tree_sitter.languages",
    {
        "python": {"extensions": [".py"], "grammar_name": "python"}, # Assumes tree_sitter_languages or pip install
        "java": {"extensions": [".java"], "grammar_name": "java"},
        "javascript": {"extensions": [".js", ".jsx", ".ts", ".tsx"], "grammar_name": "javascript"}, # tree-sitter-javascript often covers typescript too
        "typescript": {"extensions": [".ts", ".tsx"], "grammar_name": "typescript"},
        # Add more languages here
        # "c": {"extensions": [".c", ".h"], "grammar_file": "c.so", "language_name_in_lib": "c"},
        # "cpp": {"extensions": [".cpp", ".hpp", ".cc", ".hh"], "grammar_file": "cpp.so", "language_name_in_lib": "cpp"},
    }
)
# Max tokens for a single AST node to be considered one chunk. Larger nodes might be further split by TextSplitter.
AST_CHUNK_MAX_TOKENS = get_config_value("tree_sitter.ast_chunk_max_tokens", 1000, "RAG_AST_CHUNK_MAX_TOKENS")
# Min tokens for an AST node (like function/class) to be considered a distinct chunk.
AST_CHUNK_MIN_TOKENS = get_config_value("tree_sitter.ast_chunk_min_tokens", 30, "RAG_AST_CHUNK_MIN_TOKENS")

# --- Text Splitter Configuration (Fallback for non-code or very large AST nodes) ---
TEXT_CHUNK_SIZE = get_config_value("text_splitter.chunk_size", 500, "RAG_TEXT_CHUNK_SIZE") # In tokens
TEXT_CHUNK_OVERLAP = get_config_value("text_splitter.chunk_overlap", 50, "RAG_TEXT_CHUNK_OVERLAP") # In tokens

# --- Indexing Configuration ---
# FAISS Index
FAISS_INDEX_FILENAME = get_config_value("indexing.faiss.filename", "vector_index.faiss")
METADATA_FILENAME = get_config_value("indexing.faiss.metadata_filename", "metadata.json")
# BM25 Index
BM25_INDEX_FILENAME = get_config_value("indexing.bm25.filename", "bm25_index.pkl")

# --- Retrieval Configuration ---
RETRIEVAL_VECTOR_TOP_K = get_config_value("retrieval.vector_top_k", 20, "RAG_RETRIEVAL_VECTOR_TOP_K")
RETRIEVAL_BM25_TOP_K = get_config_value("retrieval.bm25_top_k", 5, "RAG_RETRIEVAL_BM25_TOP_K")
# RRF k parameter (controls how many documents from each retriever are considered in fusion, not the final k)
# RRF usually uses a small constant like 60 for its internal calculations, not directly related to final top-k.
# The final number of documents to LLM will be controlled by a re-ranker or a simple slice.
RRF_CONSTANT_K = get_config_value("retrieval.rrf_k_constant", 60) # RRF k parameter

RETRIEVAL_INDEXES = get_config_value("retrieval.indexes", ["vector", "bm25"], "RAG_RETRIEVAL_INDEXES")

# --- File Processing Configuration ---
# Default excluded directories and files (can be extended in config.yaml)
DEFAULT_EXCLUDED_DIRS: List[str] = get_config_value("file_processing.default_excluded_dirs", [
    ".git", ".idea", ".vscode", "__pycache__", "node_modules", "venv", ".venv",
    "target", "build", "dist", "docs", "examples", "tests", "test"
    # Add more common ones if needed
])
DEFAULT_EXCLUDED_FILES: List[str] = get_config_value("file_processing.default_excluded_files", [
    "*.min.js", "*.min.css", "*.lock", "*.log", ".*", "LICENSE"
    # Add more common ones if needed
])
# Max file size in MB to process
MAX_FILE_SIZE_MB = get_config_value("file_processing.max_file_size_mb", 5, "RAG_MAX_FILE_SIZE_MB")


# --- API Server Configuration ---
API_HOST = get_config_value("api_server.host", "0.0.0.0", "RAG_API_HOST")
API_PORT = get_config_value("api_server.port", 8000, "RAG_API_PORT")
API_RELOAD = get_config_value("api_server.reload", False, "RAG_API_RELOAD") # For development

# --- Validation and Logging of Config ---
def log_important_configs():
    logger.info("--- Key Configurations ---")
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Data Directory: {DATA_DIR}")
    logger.info(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    logger.info(f"Generator Model: {GENERATOR_MODEL_NAME}")
    logger.info(f"Supported AST Languages: {list(TREE_SITTER_LANGUAGES.keys())}")
    logger.info(f"API Server: http://{API_HOST}:{API_PORT}")
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key is not set")
    logger.info("--- End Key Configurations ---")


def get_openai_llm_client(apikey: Optional[str] = None, baseurl: Optional[str] = None):
    """Initializes and returns an OpenAI client for LLM generation."""
    if not apikey and not OPENAI_API_KEY:
        logger.error("OpenAI API key not provided. Cannot use OpenAI for generation.")
        raise ValueError("OpenAI API key not set.")
    try:
        from openai import AsyncOpenAI # Use AsyncOpenAI for FastAPI integration
        _openai_llm_client = AsyncOpenAI(api_key=apikey or OPENAI_API_KEY, base_url=baseurl or OPENAI_BASE_URL)
        logger.info("AsyncOpenAI client for LLM generation initialized.")
    except ImportError:
        logger.error("OpenAI Python package not installed. `pip install openai`")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize AsyncOpenAI client: {e}")
        raise
    return _openai_llm_client

def get_openai_embeddings_client(apikey: Optional[str] = None, baseurl: Optional[str] = None):
    """Initializes and returns an OpenAI client."""
    if not apikey and not OPENAI_API_KEY:
        logger.error("OpenAI API key not provided. Cannot use OpenAI embeddings.")
        raise ValueError("OpenAI API key not set.")
    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=apikey or OPENAI_API_KEY, base_url=baseurl or OPENAI_BASE_URL)
        logger.info("OpenAI client initialized.")
    except ImportError:
        logger.error("OpenAI Python package not installed. `pip install openai`")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise
    return _openai_client

# Call at the end of the module to log configs when imported
log_important_configs()