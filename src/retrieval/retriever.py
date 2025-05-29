from typing import List, Tuple, Dict, Any, Optional
from loguru import logger
from pathlib import Path

# Import configurations and necessary classes
from src import config
from src.indexing.vector_index import FaissVectorIndex, generate_embeddings # generate_embeddings for query
from src.indexing.sparse_index import BM25Index, preprocess_text_for_bm25 # preprocess_text_for_bm25 for query
from src.data_processing.chunkers import DocumentChunk # For type hinting, though retriever returns metadata dicts

# Define a type for a search result item (score, metadata_dict)
SearchResultItem = Tuple[float, Dict[str, Any]]

class HybridRetriever:
    """
    Orchestrates hybrid retrieval using both dense (vector) and sparse (BM25) search,
    and fuses the results using Reciprocal Rank Fusion (RRF).
    """
    def __init__(
            self,
            index_dir: Path = config.INDEX_DIR,
            vector_top_k: int = config.RETRIEVAL_VECTOR_TOP_K,
            bm25_top_k: int = config.RETRIEVAL_BM25_TOP_K,
            rrf_k_constant: int = config.RRF_CONSTANT_K # RRF k parameter
    ):
        self.index_dir = index_dir
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k_constant = rrf_k_constant # Constant for RRF scoring

        self.vector_index: Optional[FaissVectorIndex] = None
        self.bm25_index: Optional[BM25Index] = None

        self._load_indexes()

    def _load_indexes(self) -> bool:
        """Loads both vector and sparse indexes."""
        logger.info(f"Initializing and loading indexes from: {self.index_dir}")

        # Load Vector Index
        try:
            self.vector_index = FaissVectorIndex(index_dir=self.index_dir, embedding_dim=config.EMBEDDING_DIMENSIONS)
            if not self.vector_index.load_index():
                logger.error("Failed to load FaissVectorIndex. Vector search will not be available.")
                self.vector_index = None # Ensure it's None if load fails
            else:
                logger.info("FaissVectorIndex loaded successfully.")
        except Exception as e:
            logger.error(f"Exception during FaissVectorIndex initialization or loading: {e}")
            self.vector_index = None

        # Load BM25 Index
        try:
            self.bm25_index = BM25Index(index_dir=self.index_dir)
            if not self.bm25_index.load_index():
                logger.error("Failed to load BM25Index. Sparse search will not be available.")
                self.bm25_index = None # Ensure it's None if load fails
            else:
                logger.info("BM25Index loaded successfully.")
        except Exception as e:
            logger.error(f"Exception during BM25Index initialization or loading: {e}")
            self.bm25_index = None

        if self.vector_index and self.bm25_index:
            # Basic consistency check: metadata count.
            # This assumes metadata is shared and built from the same chunk list.
            if len(self.vector_index.chunk_metadata) != len(self.bm25_index.chunk_metadata):
                logger.warning(
                    f"Metadata count mismatch between vector index ({len(self.vector_index.chunk_metadata)}) "
                    f"and BM25 index ({len(self.bm25_index.chunk_metadata)}). "
                    "This might indicate issues if they are supposed to be perfectly aligned."
                )
            return True
        elif self.vector_index:
            logger.warning("Only Vector Index was loaded. BM25 search will be skipped.")
            return True
        elif self.bm25_index:
            logger.warning("Only BM25 Index was loaded. Vector search will be skipped.")
            return True
        else:
            logger.error("Neither Vector Index nor BM25 Index could be loaded. Retrieval will not work.")
            return False


    def retrieve(self, query_text: str, top_n_final: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Performs hybrid retrieval for a given query.

        Args:
            query_text (str): The user's query.
            top_n_final (Optional[int]): The final number of top documents to return after fusion.
                                         Defaults to self.vector_top_k or self.bm25_top_k (whichever is larger).

        Returns:
            List[Dict[str, Any]]: A list of fused and ranked chunk metadata dictionaries.
                                  Each dictionary is a chunk's metadata.
        """
        if not self.vector_index and not self.bm25_index:
            logger.error("No indexes are loaded. Cannot retrieve.")
            return []

        if top_n_final is None:
            top_n_final = max(self.vector_top_k, self.bm25_top_k, 5) # Default to at least 5 if k's are small

        logger.info(f"Retrieving for query: '{query_text[:100]}...'")

        vector_results: List[SearchResultItem] = []
        if self.vector_index:
            try:
                # Faiss search returns (distance, metadata)
                # Lower distance is better.
                vector_results = self.vector_index.search(query_text, top_k=self.vector_top_k)
                logger.debug(f"Vector search returned {len(vector_results)} results.")
            except Exception as e:
                logger.error(f"Error during vector search: {e}")

        bm25_results: List[SearchResultItem] = []
        if self.bm25_index:
            try:
                # BM25 search returns (score, metadata)
                # Higher score is better.
                bm25_results = self.bm25_index.search(query_text, top_k=self.bm25_top_k)
                logger.debug(f"BM25 search returned {len(bm25_results)} results.")
            except Exception as e:
                logger.error(f"Error during BM25 search: {e}")

        if not vector_results and not bm25_results:
            logger.info("No results from either vector or BM25 search.")
            return []

        fused_results = self._reciprocal_rank_fusion(
            [vector_results, bm25_results], # Pass as a list of result lists
            k=self.rrf_k_constant
        )

        # Sort fused results by score (descending) and take top_n_final
        fused_results.sort(key=lambda x: x[0], reverse=True)

        final_chunk_metadata_list: List[Dict[str, Any]] = [meta for score, meta in fused_results[:top_n_final]]

        logger.info(f"Hybrid retrieval returned {len(final_chunk_metadata_list)} fused results (target top_n_final: {top_n_final}).")
        return final_chunk_metadata_list

    def _reciprocal_rank_fusion(self,
                                search_results_lists: List[List[SearchResultItem]],
                                k: int = 60
                                ) -> List[SearchResultItem]:
        """
        Performs Reciprocal Rank Fusion on multiple lists of search results.

        Args:
            search_results_lists (List[List[SearchResultItem]]):
                A list where each element is a list of (score, metadata_dict) tuples
                from a single search engine.
                Important: Assumes higher scores are better for BM25, lower scores (distances) are better for FAISS.
                This function will normalize FAISS distances.
            k (int): The RRF constant (default is 60, as used in many papers).

        Returns:
            List[SearchResultItem]: A list of (fused_score, metadata_dict) tuples,
                                    ranked by the fused RRF score (higher is better).
        """
        if not any(search_results_lists): # If all lists are empty
            return []

        # Use a unique identifier for each chunk. 'file_path' (chunk's unique path) should work.
        # If chunks could have identical file_path (e.g. from different original files but same chunk name),
        # a more robust unique ID is needed (e.g., hash of content or a UUID assigned during chunking).
        # For now, assuming chunk's 'file_path' is sufficiently unique for RRF.
        # The metadata dict itself can be hashed if it's hashable, or use a tuple of key fields.
        # Let's use 'original_file_path' + 'chunk_id' as a robust key.

        def get_doc_id(metadata: Dict[str, Any]) -> Tuple[str, int]:
            # Create a unique ID for a document based on its metadata
            return (str(metadata.get('original_file_path', '')), int(metadata.get('chunk_id', -1)))

        fused_scores: Dict[Tuple[str, int], float] = {}
        doc_metadata_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}

        for i, result_list in enumerate(search_results_lists):
            is_vector_search = (i == 0 and self.vector_index is not None) # Assuming vector results are first

            # Sort vector results by distance (ascending) if they are distances
            # BM25 scores are already higher-is-better
            # This simple check might need refinement if result_lists order changes
            # or if scores are already normalized.
            # For FAISS L2, score is distance (lower is better).
            # For BM25, score is relevance (higher is better).

            # We need to rank items. For FAISS (distance), rank is based on ascending distance.
            # For BM25 (score), rank is based on descending score.

            ranked_list = []
            if is_vector_search: # FAISS L2 distances
                # Sort by distance (score field in SearchResultItem) ascending
                sorted_by_score = sorted(result_list, key=lambda x: x[0])
                ranked_list = [(idx + 1, item[1]) for idx, item in enumerate(sorted_by_score)] # (rank, metadata)
            else: # BM25 scores or other higher-is-better scores
                # Sort by score descending
                sorted_by_score = sorted(result_list, key=lambda x: x[0], reverse=True)
                ranked_list = [(idx + 1, item[1]) for idx, item in enumerate(sorted_by_score)] # (rank, metadata)


            for rank, metadata in ranked_list:
                doc_id = get_doc_id(metadata)
                if doc_id not in doc_metadata_cache:
                    doc_metadata_cache[doc_id] = metadata

                # RRF formula: sum(1 / (k + rank))
                rrf_score = 1.0 / (k + rank)

                if doc_id in fused_scores:
                    fused_scores[doc_id] += rrf_score
                else:
                    fused_scores[doc_id] = rrf_score

        # Convert fused_scores dict to a list of (score, metadata)
        final_fused_results: List[SearchResultItem] = []
        for doc_id, score in fused_scores.items():
            if doc_id in doc_metadata_cache: # Should always be true
                final_fused_results.append((score, doc_metadata_cache[doc_id]))
            else: # Should not happen
                logger.warning(f"Metadata not found in cache for doc_id {doc_id} during RRF. This is unexpected.")


        # Sort by RRF score in descending order
        final_fused_results.sort(key=lambda x: x[0], reverse=True)

        return final_fused_results


# --- Example Usage ---
if __name__ == "__main__":
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO") # Print to stdout for testing

    # This example assumes that FaissVectorIndex and BM25Index have been built
    # and their index files exist in config.INDEX_DIR / "test_vector_index"
    # and config.INDEX_DIR / "test_bm25_index" respectively from previous examples.
    # If not, run their __main__ blocks first.

    test_vector_index_dir = config.INDEX_DIR / "test_vector_index"
    test_bm25_index_dir = config.INDEX_DIR / "test_bm25_index" # BM25 uses the same metadata for now

    # Check if dummy index files exist (created by vector_index.py and sparse_index.py examples)
    if not (test_vector_index_dir / config.FAISS_INDEX_FILENAME).exists() or \
            not (test_bm25_index_dir / config.BM25_INDEX_FILENAME).exists() or \
            not (test_vector_index_dir / config.METADATA_FILENAME).exists(): # Metadata is shared
        logger.error("Dummy index files not found. Please run the main blocks of:")
        logger.error("  - src/indexing/vector_index.py")
        logger.error("  - src/indexing/sparse_index.py")
        logger.error("To create them in their respective test directories before running this retriever example.")
        logger.error(f"Expected vector index: {test_vector_index_dir / config.FAISS_INDEX_FILENAME}")
        logger.error(f"Expected bm25 index: {test_bm25_index_dir / config.BM25_INDEX_FILENAME}")
        logger.error(f"Expected metadata: {test_vector_index_dir / config.METADATA_FILENAME}")

        # Create minimal dummy indexes for the retriever to load if they don't exist
        # This is just to allow the retriever example to run without prior manual steps.
        # In a real scenario, these would be properly built.
        logger.warning("Attempting to create minimal dummy indexes for retriever example to run...")

        from src.data_processing.chunkers import DocumentChunk
        from src.indexing.vector_index import FaissVectorIndex
        from src.indexing.sparse_index import BM25Index

        dummy_chunks_data = [
            {"file_path": Path("test.py_chunk_1"), "content": "def python hello(): return 'world'", "language": "python", "original_file_path": Path("test.py"), "chunk_id":1, "size_bytes":30, "absolute_path": Path("/abs/test.py")},
            {"file_path": Path("test.py_chunk_2"), "content": "class Greeter: def greet(self): print('Python Greeter class hello world')", "language": "python", "original_file_path": Path("test.py"), "chunk_id":2, "size_bytes":50, "absolute_path": Path("/abs/test.py")},
            {"file_path": Path("other.js_chunk_1"), "content": "function test() { return 1+1; } // javascript test function", "language": "javascript", "original_file_path": Path("other.js"), "chunk_id":1, "size_bytes":40, "absolute_path": Path("/abs/other.js")},
        ]
        test_chunks = [DocumentChunk(**data) for data in dummy_chunks_data]

        temp_vector_indexer = FaissVectorIndex(index_dir=test_vector_index_dir, embedding_dim=config.EMBEDDING_DIMENSIONS)
        temp_vector_indexer.build_index(test_chunks, force_rebuild=True)

        temp_bm25_indexer = BM25Index(index_dir=test_bm25_index_dir) # Uses same metadata path by default
        temp_bm25_indexer.build_index(test_chunks, force_rebuild=True)
        logger.info("Minimal dummy indexes created.")


    # Initialize Hybrid Retriever - it will try to load from the *same* index_dir
    # For this example, we assume both index types (Faiss, BM25) and their shared metadata
    # are in a single directory structure.
    # If they were in separate dirs, HybridRetriever would need two dir paths.
    # Let's use the vector index dir, assuming metadata is there and BM25 model is also there.
    # This implies config.BM25_INDEX_FILENAME and config.FAISS_INDEX_FILENAME are in the same dir.

    # For the test, we'll point to the vector index's directory, assuming BM25 model is also there.
    # This requires that both FaissVectorIndex and BM25Index were built to use the same metadata file
    # and their respective model files are in the same directory.
    # Our current setup for BM25Index and FaissVectorIndex uses config.METADATA_FILENAME in their respective index_dir.
    # If test_vector_index_dir and test_bm25_index_dir are different, this won't work directly
    # unless metadata is copied or they are configured to use the same metadata path.
    # For simplicity, let's assume they are built to use the same index_dir for this test.
    # The `BM25Index` and `FaissVectorIndex` save their respective models (bm25.pkl, vector.faiss)
    # and *both* save/load metadata to `config.METADATA_FILENAME` within their `index_dir`.
    # So, if their `index_dir` is the same, they share metadata.

    # Let's ensure the test setup uses the same directory for both for this example.
    # We'll use `test_vector_index_dir` as the common ground.
    # The BM25 example saves its model to its own `test_bm25_index_dir`.
    # The Faiss example saves its model to its own `test_vector_index_dir`.
    # They both save metadata to `config.METADATA_FILENAME` within their respective dirs.
    # This means the metadata files would be different if dirs are different.

    # For the HybridRetriever to work correctly with current shared metadata assumption,
    # both index types should reference the same metadata file.
    # This implies they should ideally be in the same directory or configured to point to a shared metadata file.

    # Simplification for testing: Assume both index files (FAISS & BM25 model) and ONE metadata file
    # are located in `test_hybrid_index_dir`.
    # You'd need to ensure your build process places them accordingly.

    test_hybrid_index_dir = config.INDEX_DIR / "test_hybrid_retriever_indexes"
    # Re-run dummy index creation to ensure they are in the same dir for this test
    logger.info(f"Ensuring dummy indexes are in: {test_hybrid_index_dir}")
    from src.data_processing.chunkers import DocumentChunk # Re-import if needed
    from src.indexing.vector_index import FaissVectorIndex # Re-import if needed
    from src.indexing.sparse_index import BM25Index # Re-import if needed

    dummy_chunks_data_hr = [
        {"file_path": Path("hr_test.py_chunk_1"), "content": "def python_hello_function(): return 'world'", "language": "python", "original_file_path": Path("hr_test.py"), "chunk_id":1, "size_bytes":30, "absolute_path": Path("/abs/hr_test.py")},
        {"file_path": Path("hr_test.py_chunk_2"), "content": "class PythonGreeter: def greet(self): print('Python Greeter class says hello world')", "language": "python", "original_file_path": Path("hr_test.py"), "chunk_id":2, "size_bytes":50, "absolute_path": Path("/abs/hr_test.py")},
        {"file_path": Path("hr_other.js_chunk_1"), "content": "function javascript_test_function() { return 1+1; }", "language": "javascript", "original_file_path": Path("hr_other.js"), "chunk_id":1, "size_bytes":40, "absolute_path": Path("/abs/hr_other.js")},
    ]
    test_chunks_hr = [DocumentChunk(**data) for data in dummy_chunks_data_hr]

    hr_vector_indexer = FaissVectorIndex(index_dir=test_hybrid_index_dir, embedding_dim=config.EMBEDDING_DIMENSIONS)
    hr_vector_indexer.build_index(test_chunks_hr, force_rebuild=True)

    hr_bm25_indexer = BM25Index(index_dir=test_hybrid_index_dir)
    hr_bm25_indexer.build_index(test_chunks_hr, force_rebuild=True) # This will overwrite metadata if not careful
    logger.info("Minimal dummy indexes for HybridRetriever created in shared directory.")


    logger.info("\n--- Initializing Hybrid Retriever ---")
    hybrid_retriever = HybridRetriever(index_dir=test_hybrid_index_dir) # Point to the shared dir

    if hybrid_retriever.vector_index and hybrid_retriever.bm25_index:
        logger.info("Hybrid Retriever initialized successfully with both indexes.")

        queries_hr = [
            "python hello function",
            "javascript test",
            "Greeter class",
            "non_existent_term_for_testing_fusion"
        ]

        for query in queries_hr:
            logger.info(f"\n--- Hybrid Retrieving for query: '{query}' ---")
            fused_results = hybrid_retriever.retrieve(query, top_n_final=3)

            if not fused_results:
                logger.info("  No fused results found.")
            for i, meta_chunk in enumerate(fused_results):
                # The fused_results from retrieve() are just the metadata dictionaries
                logger.info(f"  Rank {i+1}:")
                logger.info(f"    Original File: {meta_chunk.get('original_file_path')}")
                logger.info(f"    Chunk ID: {meta_chunk.get('chunk_id')}")
                logger.info(f"    File Path (Chunk): {meta_chunk.get('file_path')}")
                # To see content, you'd need to load it based on metadata, or include it in metadata (not recommended for large content)
                # logger.info(f"    Content Preview (if available in meta): {meta_chunk.get('content_preview', 'N/A')}")
            logger.info("---")

        # Example: Check if a known relevant document is ranked highly for a specific query
        results_for_greeter_hr = hybrid_retriever.retrieve("PythonGreeter class", top_n_final=1)
        if results_for_greeter_hr:
            assert "hr_test.py_chunk_2" in str(results_for_greeter_hr[0]['file_path']), "Hybrid search result mismatch for 'PythonGreeter class'"
            logger.info("Basic hybrid search assertion passed.")
        else:
            logger.error("Hybrid search for 'PythonGreeter class' returned no results.")

    else:
        logger.error("Hybrid Retriever failed to initialize one or both indexes. Cannot run test.")

    # Clean up test hybrid index directory (optional)
    # import shutil
    # if test_hybrid_index_dir.exists():
    #      shutil.rmtree(test_hybrid_index_dir)
    #      logger.info(f"Cleaned up test hybrid index directory: {test_hybrid_index_dir}")
