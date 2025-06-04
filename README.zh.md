# **针对代码仓库的RAG项目 (Code-Aware RAG)**

中文 | [English](./README.md)

本项目旨在构建一个先进的、针对代码仓库的检索增强生成 (RAG) 系统。与传统RAG将代码视为纯文本不同，本项目的核心目标是实现“代码感知”（Code-Aware）能力，更深入地理解代码的结构、语义和依赖关系，从而提供更精准、更智能的问答和分析功能。

## **✨ 核心特性**

* **智能代码分块 (Intelligent Code Chunking)**:  
  * 使用 tree-sitter 解析代码的抽象语法树 (AST)。  
  * 按代码的逻辑单元（如函数、类、方法）进行分块，而非固定长度，确保了上下文的完整性。  
  * 保留代码块的层级上下文信息（如所属类、文件名、起止行号）。  
* **混合检索策略 (Hybrid Search)**:  
  * 结合**向量检索** (基于FAISS，捕捉语义相似性) 和 **稀疏检索** (基于BM25，擅长关键词精确匹配)。  
  * 通过**倒数排序融合 (Reciprocal Rank Fusion - RRF)** 算法智能合并两路检索结果，提升检索的召回率和准确率。  
* **基于LLM的查询改写 (Query Rewriting)**:  
  * 由于自然语言召回代码文件效率不高，利用大语言模型将用户查询转化为更高效的检索查询。  
  * 支持提供自定义改写提示词，以开启并且定制查询转换策略。  
  * 通过更好地将查询与代码的语义结构对齐，提高检索质量。  
* **多语言支持 (可扩展)**:  
  * tree-sitter 的设计使其易于通过添加新的语言语法库来扩展支持的编程语言。  
* **模块化与可配置设计**:  
  * 清晰分离数据处理、索引、检索、生成等模块。  
  * 通过 .env 和 config.yaml 文件进行灵活配置，包括API密钥、模型选择、路径、分块与检索参数等。  
* **异步API接口**:  
  * 使用 FastAPI 构建异步API，提供非阻塞的仓库设置和流式查询响应。

## **项目结构**

```
.
├── main.py                    # API服务启动入口
├── requirements.txt           # Python依赖
├── .env.example               # 环境变量示例文件 (用户需复制为 .env)
├── config.example.yaml        # 应用配置示例文件 (用户可选复制为 config.yaml)
├── grammars/                  # (可选) 存放手动编译的tree-sitter语法库 (.so, .dll)
│
└── src/                       # 源代码目录
    ├── __init__.py            # 使src成为一个包
    ├── api.py                 # FastAPI 应用接口定义
    ├── config.py              # 项目配置加载与管理
    ├── pipeline.py            # RAG核心处理流程编排 (RAGPipeline)
    │
    ├── data_processing/       # 数据预处理模块
    │   ├── __init__.py
    │   ├── document_loader.py # 从代码库加载和过滤文件 (LoadedDocument)
    │   └── chunkers.py        # 智能代码分块 (TreeSitterChunker, TokenSplitter, DocumentChunk)
    │
    ├── indexing/              # 索引构建模块
    │   ├── __init__.py
    │   ├── vector_index.py    # 向量索引 (FaissVectorIndex, 使用FAISS)
    │   └── sparse_index.py    # 稀疏索引 (BM25Index, 使用rank_bm25)
    │
    ├── retrieval/             # 检索模块
    │   ├── __init__.py
    │   └── retriever.py       # 混合检索器 (HybridRetriever)
    │
    ├── generation/            # LLM生成模块
    │   ├── __init__.py
    │   └── generator.py       # LLM交互与Prompt构建 (LLMGenerator)
    │
    └── templates/             # Jinja2 Prompt模板目录
        └── rag_prompt_template.jinja2 # 默认的RAG Prompt模板
```

## **🚀 快速开始**

1. **克隆项目**:  
   ```shell
   git clone <your-repository-url>  
   cd <project-directory>
   ```
2. **创建并激活虚拟环境** (推荐):  
   ```shell
   python -m venv venv  
   source venv/bin/activate  # Linux/macOS
   ```

3. **安装依赖**:  
   ```shell
   pip install -r requirements.txt
   python download_nltk_data.py
   ```

   * 如果某些语言没有预编译的pip包，你可能需要从源码编译其 tree-sitter 语法库，并将生成的共享库文件（如 .so 或 .dll）放置在 grammars/ 目录，并在 config.yaml (或 src/config.py) 中进行相应配置。  

   * 请运行 python download_nltk_data.py 下载NLTK数据。

4. **配置环境**:

   **编辑 config.yaml 文件（推荐）**
    * 复制 config.yaml.example 为 config.yaml，并根据需要修改应用配置（如模型名称、路径、分块参数等）。  
      cp config.yaml.example config.yaml  
   
   **编辑 .env 文件**
   * 复制 .env.example 为 .env，并填入你的API密钥 (如 OPENAI_API_KEY 等)。  
     cp .env.example .env

   **注意**
   * 如果不设置apikey，那么服务会以`未配置apikey模式`启动，该模式下用户请求需要在请求头中通过`Bear Token`提供apikey。
     

5. **启动API服务**:  
   python main.py

   服务默认运行在 http://0.0.0.0:8000 (具体请参考 src/config.py 中的 API_HOST 和 API_PORT 设置)。  
6. **使用API**:  
   * 设置并索引仓库:  
     * 向 POST /repository/setup 端点发送请求。
     * 请求头: `Authorization: Bearer {apikey}` （仅在`未配置apikey模式`下需要）
     * 请求体示例:  
     ```json
     {  
       "repo_id": "bella-issues-bot",  
       "repo_url_or_path": "https://github.com/szl97/bella-issues-bot.git",  
       "force_reclone": false,  
       "force_reindex": false  
     }  
     ```
     `repo_id` 是你为这个仓库指定的唯一标识符。

   * 查询已索引的仓库:  
     * 向 POST /query/stream 端点发送请求。  
     * 请求头: `Authorization: Bearer {apikey}` （仅在`未配置apikey模式`下需要）
     * 请求体示例:  
     ```json
     {
       "repo_id": "bella-issues-bot",
       "sys_prompt": "xxxx",   // 当不想使用默认的系统提示词时需要提供
       "query_text": "Introduce the workflow of bella-issues-bot",
       "rewrite_prompt": "xxx" // 当想要改写查询用于召回时需要提供
     }
     ```

     响应将是LLM生成的流式文本。

## **🛠️ 技术栈**

* **Python 3.9+**  
* **FastAPI**: 高性能Web框架，用于构建API。  
* **Uvicorn**: ASGI服务器。  
* **Pydantic**: 数据校验与模型定义。  
* **Loguru**: 更优雅的日志记录。  
* **Tree-sitter**: 代码解析与AST（抽象语法树）构建，用于智能分块。  
* **FAISS**: Facebook AI Similarity Search，用于高效的向量相似性搜索。  
* **Rank-BM25**: 实现BM25稀疏检索算法。  
* **Sentence Transformers / OpenAI API / Google Generative AI SDK**: 用于生成文本嵌入和与大语言模型交互。  
* **Jinja2**: Prompt模板引擎。  
* **GitPython**: 与Git仓库交互。  
* **PyYAML**: 解析YAML配置文件。  
* **python-dotenv**: 加载 .env 文件。
* * **nltk**: 用于NLP操作，如分词、提取词干和n元组

## **🔮 未来增强方向 (基于优化策略方案)**

本项目奠定了坚实的基础，未来可以从以下方向进一步优化和扩展：

* **阶段一：高级优化**  
  * **上下文重排 (Context Re-ranking)**: 使用Cross-Encoder模型对初步检索结果进行重排序，提升送入LLM的上下文质量。  
  * **多向量表示与摘要增强 (Multi-Vector Representation & Summary Augmentation)**: 为代码块创建代码本身、自动生成摘要等多种向量表示，增强检索匹配能力。  
  * **上下文窗口感知**: 动态处理超出LLM上下文窗口限制的检索内容（截断、摘要等）。  
* **阶段二：前沿探索**  
  * **构建代码知识图谱 (Code Knowledge Graph - CKG)**: 抽取代码中的实体（文件、类、函数）和关系（调用、继承、导入），构建图谱以支持更深层次的代码依赖和影响分析。  
  * **控制流与数据流分析**: 结合更深入的程序分析技术，理解代码执行逻辑。

## **🤝 贡献**

欢迎对本项目进行贡献！请在提交Pull Request前查阅（待创建的）贡献指南和行为准则。

希望这个README能够清晰地介绍你的项目！
