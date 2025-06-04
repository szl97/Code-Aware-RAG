# **Code-Aware RAG Project**

English | [中文](./README.zh.md)

This project aims to build an advanced Retrieval-Augmented Generation (RAG) system specifically designed for code repositories. Unlike traditional RAG systems that treat code as plain text, the core objective of this project is to achieve "Code-Aware" capabilities, enabling deeper understanding of code structure, semantics, and dependencies to provide more accurate and intelligent Q&A and analysis functionalities.

## **✨ Core Features**

* **Intelligent Code Chunking**:
  * Uses tree-sitter to parse code's Abstract Syntax Tree (AST).
  * Chunks code by logical units (functions, classes, methods) rather than fixed lengths, ensuring context integrity.
  * Preserves hierarchical context information (parent class, filename, line numbers).
* **Hybrid Search Strategy**:
  * Combines **Vector Search** (using FAISS for semantic similarity) and **Sparse Search** (using BM25 for precise keyword matching).
  * Intelligently merges results from both approaches using **Reciprocal Rank Fusion (RRF)** algorithm to improve recall and accuracy.
* **Multi-language Support (Extensible)**:
  * tree-sitter's design makes it easy to extend language support by adding new grammar libraries.
* **Modular and Configurable Design**:
  * Clear separation of data processing, indexing, retrieval, and generation modules.
  * Flexible configuration through .env and config.yaml files for API keys, model selection, paths, chunking, and retrieval parameters.
* **Asynchronous API Interface**:
  * Built with FastAPI for non-blocking repository setup and streaming query responses.

## **Project Structure**

```
.
├── main.py                    # API service entry point
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables example (copy to .env)
├── config.example.yaml       # Application config example (optionally copy to config.yaml)
├── grammars/                 # (Optional) Manual compiled tree-sitter grammars (.so, .dll)
│
└── src/                      # Source code directory
    ├── __init__.py           # Makes src a package
    ├── api.py                # FastAPI application interface definitions
    ├── config.py             # Project configuration loading and management
    ├── pipeline.py           # RAG core processing flow orchestration (RAGPipeline)
    │
    ├── data_processing/      # Data preprocessing module
    │   ├── __init__.py
    │   ├── document_loader.py # Load and filter files from codebase (LoadedDocument)
    │   └── chunkers.py       # Intelligent code chunking (TreeSitterChunker, TokenSplitter, DocumentChunk)
    │
    ├── indexing/             # Index building module
    │   ├── __init__.py
    │   ├── vector_index.py   # Vector indexing (FaissVectorIndex, using FAISS)
    │   └── sparse_index.py   # Sparse indexing (BM25Index, using rank_bm25)
    │
    ├── retrieval/            # Retrieval module
    │   ├── __init__.py
    │   └── retriever.py      # Hybrid retriever (HybridRetriever)
    │
    ├── generation/           # LLM generation module
    │   ├── __init__.py
    │   └── generator.py      # LLM interaction and Prompt construction (LLMGenerator)
    │
    └── templates/            # Jinja2 Prompt templates directory
        └── rag_prompt_template.jinja2 # Default RAG Prompt template
```

## **🚀 Quick Start**

1. **Clone the Project**:
   git clone <your-repository-url>
   cd <project-directory>

2. **Create and Activate Virtual Environment** (recommended):
   python -m venv venv
   source venv/bin/activate  # Linux/macOS

3. **Install Dependencies**:
   pip install -r requirements.txt

   If some languages don't have pre-compiled pip packages, you might need to compile their tree-sitter grammars from source and place the generated shared library files (.so or .dll) in the grammars/ directory, then configure accordingly in config.yaml (or src/config.py).

   Please run python download_nltk_data.py to download NLTK data.

4. **Configure Environment**:
   **Edit .env file**
   * Copy .env.example to .env and fill in your API keys (e.g., OPENAI_API_KEY).
     cp .env.example .env

   **Edit config.yaml file**
   * (Optional) Copy config.yaml.example to config.yaml and modify application settings as needed (model names, paths, chunking parameters, etc.).
     cp config.yaml.example config.yaml

5. **Start API Service**:
   python main.py

   Service runs by default at http://0.0.0.0:8000 (refer to API_HOST and API_PORT settings in src/config.py).

6. **Use the API**:
   * Setup and Index Repository:
     Send request to POST /repository/setup endpoint.
     Request body example:
     {
       "repo_id": "bella-issues-bot",
       "repo_url_or_path": "https://github.com/bella-top/bella-issues-bot.git",
       "force_reclone": false,
       "force_reindex": false
     }
     `repo_id` is your unique identifier for this repository.

   * Query Indexed Repository:
     Send request to POST /query/stream endpoint.
     Request body example:
     {
       "repo_id": "bella-issues-bot",
       "query_text": "Introduce the workflow of bella-issues-bot"
     }

     Response will be streaming text generated by the LLM.

## **🛠️ Tech Stack**

* **Python 3.9+**
* **FastAPI**: High-performance web framework for building APIs.
* **Uvicorn**: ASGI server.
* **Pydantic**: Data validation and model definition.
* **Loguru**: More elegant logging.
* **Tree-sitter**: Code parsing and AST construction for intelligent chunking.
* **FAISS**: Facebook AI Similarity Search for efficient vector similarity search.
* **Rank-BM25**: Implementation of BM25 sparse retrieval algorithm.
* **Sentence Transformers / OpenAI API / Google Generative AI SDK**: For text embeddings and LLM interaction.
* **Jinja2**: Prompt template engine.
* **GitPython**: Git repository interaction.
* **PyYAML**: YAML configuration file parsing.
* **python-dotenv**: .env file loading.

## **🔮 Future Enhancements (Based on Optimization Strategy)**

This project lays a solid foundation and can be further optimized and extended in the following directions:

* **Phase One: Advanced Optimization**
  * **Context Re-ranking**: Use Cross-Encoder models to re-rank initial retrieval results, improving context quality for LLM input.
  * **Multi-Vector Representation & Summary Augmentation**: Create multiple vector representations for code blocks including code itself and auto-generated summaries to enhance retrieval matching.
  * **Context Window Awareness**: Dynamically handle retrieved content exceeding LLM context window limits (truncation, summarization, etc.).
* **Phase Two: Frontier Exploration**
  * **Build Code Knowledge Graph (CKG)**: Extract entities (files, classes, functions) and relationships (calls, inheritance, imports) to support deeper code dependency and impact analysis.
  * **Control Flow & Data Flow Analysis**: Incorporate deeper program analysis techniques to understand code execution logic.

## **🤝 Contributing**

Contributions are welcome! Please check the (to-be-created) contribution guidelines and code of conduct before submitting Pull Requests.