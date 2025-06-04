from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

# Import project modules
from src import config
from src.data_processing.document_loader import (
    LoadedDocument,
    clone_repository,
    load_documents_from_repo
)
from src.data_processing.chunkers import (
    CodeRAGChunker,
    DocumentChunk
)
from src.indexing.vector_index import FaissVectorIndex
from src.indexing.sparse_index import BM25Index
from src.retrieval.retriever import HybridRetriever


class RAGPipeline:
    """
    Orchestrates the entire RAG pipeline from repository processing to retrieval.
    """
    def __init__(
            self,
            repo_id: str, # A unique identifier for the repository being processed/queried
            indexes: list[str] = config.RETRIEVAL_INDEXES,
            index_base_dir: Path = config.INDEX_DIR,
            repos_base_dir: Path = config.REPOS_DIR,
            model: str = config.GENERATOR_MODEL_NAME,
            temperature: float = config.GENERATOR_TEMPERATURE
    ):
        """
        Initializes the RAG pipeline for a specific repository.

        Args:
            repo_id (str): A unique identifier for the repository. This will be used
                           to create a dedicated subdirectory within the index_base_dir
                           and repos_base_dir. For example, if repo_url is
                           "https://github.com/user/myrepo", repo_id could be "user_myrepo".
            indexes (list[str]): The indexes types used by retriever.
            index_base_dir (Path): The base directory where all indexes are stored.
            repos_base_dir (Path): The base directory where cloned repositories are stored.
        """
        if not repo_id:
            raise ValueError("repo_id cannot be empty.")

        self.repo_id = repo_id.replace("/", "_").replace(":", "_") # Sanitize repo_id for directory naming
        self.repository_path: Optional[Path] = None # Path to the cloned/local repository

        self.index_dir = index_base_dir / self.repo_id
        self.cloned_repo_path = repos_base_dir / self.repo_id

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.cloned_repo_path.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.temperature = temperature

        logger.info(f"RAGPipeline initialized for repo_id: '{self.repo_id}'.")
        logger.info(f"  Index directory: {self.index_dir}")
        logger.info(f"  Cloned repo directory: {self.cloned_repo_path}")

        # Initialize components (they will load indexes if they exist)
        self.chunker = CodeRAGChunker()
        self.indexes = indexes
        # The retriever will try to load indexes from its self.index_dir upon initialization
        self.retriever = HybridRetriever(index_dir=self.index_dir, indexes=self.indexes)


    def setup_repository(
            self,
            repo_url_or_path: str,
            access_token: Optional[str] = None,
            force_reclone: bool = False,
            force_reindex: bool = False,
            apikey: Optional[str] = None
    ) -> bool:
        """
        Sets up the repository: clones if it's a URL, then processes and indexes it.

        Args:
            repo_url_or_path (str): URL of the Git repository or path to a local repository.
            access_token (Optional[str]): Access token for private Git repositories.
            force_reclone (bool): If True, will delete existing cloned repo and reclone.
            force_reindex (bool): If True, will rebuild indexes even if they exist.

        Returns:
            bool: True if setup (cloning and indexing) was successful, False otherwise.
        """
        logger.info(f"Setting up repository: {repo_url_or_path} for repo_id: {self.repo_id}")

        # 1. Handle repository path (clone if URL, or use local path)
        is_url = repo_url_or_path.startswith("http://") or repo_url_or_path.startswith("https://")

        if is_url:
            self.repository_path = self.cloned_repo_path
            if force_reclone and self.repository_path.exists():
                logger.info(f"Force reclone: Deleting existing repository at {self.repository_path}")
                import shutil
                try:
                    shutil.rmtree(self.repository_path)
                    self.repository_path.mkdir(parents=True, exist_ok=True) # Recreate dir after delete
                except Exception as e:
                    logger.error(f"Failed to delete existing repository {self.repository_path}: {e}")
                    return False

            if not clone_repository(repo_url_or_path, self.repository_path, access_token):
                logger.error(f"Failed to clone repository: {repo_url_or_path}")
                return False
            logger.info(f"Repository successfully set up at: {self.repository_path}")
        else:
            local_path = Path(repo_url_or_path)
            if not local_path.exists() or not local_path.is_dir():
                logger.error(f"Local repository path does not exist or is not a directory: {local_path}")
                return False
            self.repository_path = local_path
            logger.info(f"Using local repository at: {self.repository_path}")

        # 2. Check if indexing is needed
        # Indexes are specific to this repo_id, stored in self.index_dir
        vector_index_file = self.index_dir / config.FAISS_INDEX_FILENAME
        bm25_index_file = self.index_dir / config.BM25_INDEX_FILENAME
        metadata_file = self.index_dir / config.METADATA_FILENAME

        if not force_reindex and vector_index_file.exists() and bm25_index_file.exists() and metadata_file.exists():
            logger.info(f"Indexes already exist for repo_id '{self.repo_id}' at {self.index_dir}. Skipping indexing.")
            # Ensure retriever loads these existing indexes
            if not self.retriever.vector_index or not self.retriever.bm25_index:
                logger.info("Retriever indexes were not loaded initially, attempting to load now.")
                self.retriever._load_indexes() # Try to load them again
            return True

        logger.info(f"Proceeding with indexing for repo_id '{self.repo_id}'. Force reindex: {force_reindex}")
        return self._build_indexes_for_repository(force_rebuild=force_reindex, apikey=apikey)

    def _build_indexes_for_repository(self, force_rebuild: bool = False, apikey:Optional[str] = None) -> bool:
        """
        Internal method to load documents, chunk them, and build/save all indexes.
        """
        if not self.repository_path:
            logger.error("Repository path not set. Cannot build indexes.")
            return False

        logger.info(f"Loading documents from: {self.repository_path}")
        docs_iterator = load_documents_from_repo(
            repo_path=self.repository_path,
            # Pass configured exclusions, or allow overrides if method signature changes
            excluded_dirs=config.DEFAULT_EXCLUDED_DIRS,
            excluded_files=config.DEFAULT_EXCLUDED_FILES,
            max_file_size_mb=config.MAX_FILE_SIZE_MB
        )
        loaded_documents: List[LoadedDocument] = list(docs_iterator)

        if not loaded_documents:
            logger.warning(f"No documents loaded from {self.repository_path}. Indexing cannot proceed.")
            return False
        logger.info(f"Loaded {len(loaded_documents)} documents.")

        logger.info("Chunking documents...")
        document_chunks: List[DocumentChunk] = self.chunker.chunk_documents(loaded_documents)
        if not document_chunks:
            logger.warning("No chunks produced from documents. Indexing cannot proceed.")
            return False
        logger.info(f"Produced {len(document_chunks)} chunks.")

        # Initialize indexers to point to the correct subdirectory for this repo_id
        vector_indexer = FaissVectorIndex(index_dir=self.index_dir, embedding_dim=config.EMBEDDING_DIMENSIONS)
        bm25_indexer = BM25Index(index_dir=self.index_dir)

        logger.info("Building Vector Index...")
        if not vector_indexer.build_index(document_chunks, force_rebuild=force_rebuild, apikey=apikey):
            logger.error("Failed to build Vector Index.")
            return False
        logger.info("Vector Index built successfully.")

        logger.info("Building BM25 Index...")
        if not bm25_indexer.build_index(document_chunks, force_rebuild=force_rebuild): # BM25 will use/overwrite metadata from vector_indexer if paths are same
            logger.error("Failed to build BM25 Index.")
            return False
        logger.info("BM25 Index built successfully.")

        # After building, ensure the retriever loads these new/updated indexes
        logger.info("Reloading indexes in retriever after build...")
        self.retriever._load_indexes()

        return True

    def query(self,
              query_text: str,
              apikey: Optional[str] =None,
              top_n_final: Optional[int] = None,
              vector_top_k: Optional[int] = None,
              bm25_top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Performs a hybrid retrieval query against the configured repository's indexes.

        Args:
            query_text (str): The user's query.
            top_n_final (Optional[int]): The final number of top documents to return after fusion.
                                         Defaults to self.vector_top_k or self.bm25_top_k (whichever is larger).
            vector_top_k (Optional[int]): vector_top_k
            bm25_top_k (Optional[int]): bm25_top_k
        Returns:
            List[Dict[str, Any]]: A list of fused and ranked chunk metadata.
        """
        if self.retriever._load_indexes():
            logger.info("Indexes loaded successfully on demand.")
        else:
            logger.error(f"Failed to load indexes on demand for repo_id '{self.repo_id}'.")
            return []

        return self.retriever.retrieve(query_text, top_n_final=top_n_final, vector_top_k=vector_top_k, bm25_top_k=bm25_top_k, apikey=apikey)


# --- Example Usage ---
if __name__ == "__main__":
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")

    # --- Test Configuration ---
    # Use a unique repo_id for testing to avoid conflicts
    TEST_REPO_ID = "test_pipeline_repo_loguru"
    # Small, public repo for quick testing
    TEST_REPO_URL = "https://github.com/szl97/bella-issues-bot.git"

    # Alternative: Test with a local path
    # TEST_LOCAL_REPO_PATH_STR = "./test_repo_project_pipeline" # Create this path with some files
    # TEST_LOCAL_REPO_PATH = Path(TEST_LOCAL_REPO_PATH_STR)
    # if not TEST_LOCAL_REPO_PATH.exists():
    #     TEST_LOCAL_REPO_PATH.mkdir(parents=True, exist_ok=True)
    #     (TEST_LOCAL_REPO_PATH / "main.py").write_text("def main_func():\n    print('Pipeline test')")
    #     (TEST_LOCAL_REPO_PATH / "README.md").write_text("# Pipeline Test Repo")
    # repo_to_process = str(TEST_LOCAL_REPO_PATH.resolve())
    # TEST_REPO_ID = "local_pipeline_test"

    repo_to_process = TEST_REPO_URL # Choose URL or local path for testing

    # Initialize pipeline
    logger.info(f"\n--- Initializing RAGPipeline for repo_id: {TEST_REPO_ID} ---")
    pipeline = RAGPipeline(repo_id=TEST_REPO_ID)

    # Setup repository (clones and/or indexes)
    # Set force_reindex=True if you want to rebuild indexes every time during testing.
    # Set force_reclone=True if you want to re-download the repo every time.
    logger.info(f"\n--- Setting up repository: {repo_to_process} ---")
    setup_success = pipeline.setup_repository(
        repo_url_or_path=repo_to_process,
        force_reclone=False, # Set to True to always re-download for testing
        force_reindex=False  # Set to True to always re-build indexes for testing
    )

    if setup_success:
        logger.info("Repository setup and indexing completed successfully (or indexes already existed).")

        # Perform some queries
        queries_to_test = [
            "How to use it?",
            "If I want to change the model, what to do?"
        ]

        for q_text in queries_to_test:
            logger.info(f"\n--- Querying pipeline for: '{q_text}' ---")
            results = pipeline.query(q_text, top_n_final=3)
            if results:
                logger.info(f"Found {len(results)} relevant chunks:")
                for i, chunk_meta in enumerate(results):
                    logger.info(f"  Rank {i+1}:")
                    logger.info(f"    Original File: {chunk_meta.get('original_file_path')}")
                    logger.info(f"    Chunk ID: {chunk_meta.get('chunk_id')}")
                    logger.info(f"    Type: {chunk_meta.get('code_construct_type', 'text')}")
                    if chunk_meta.get('code_construct_name'):
                        logger.info(f"    Name: {chunk_meta.get('code_construct_name')}")
                    # Content is not in metadata by default, but you could add a preview if needed
            else:
                logger.info("  No relevant chunks found for this query.")
    else:
        logger.error("Failed to setup repository and build indexes.")

    # --- Optional: Clean up test directories ---
    # import shutil
    # test_pipeline_index_dir = config.INDEX_DIR / TEST_REPO_ID
    # test_pipeline_repo_dir = config.REPOS_DIR / TEST_REPO_ID
    # if test_pipeline_index_dir.exists():
    #     shutil.rmtree(test_pipeline_index_dir)
    #     logger.info(f"Cleaned up test index directory: {test_pipeline_index_dir}")
    # if test_pipeline_repo_dir.exists() and repo_to_process == TEST_REPO_URL : # Only remove if it was cloned
    #     shutil.rmtree(test_pipeline_repo_dir)
    #     logger.info(f"Cleaned up test cloned repository: {test_pipeline_repo_dir}")
    # if Path(TEST_LOCAL_REPO_PATH_STR).exists() and repo_to_process != TEST_REPO_URL:
    #      shutil.rmtree(TEST_LOCAL_REPO_PATH_STR)
    #      logger.info(f"Cleaned up local test repo: {TEST_LOCAL_REPO_PATH_STR}")