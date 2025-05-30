import pickle
from pathlib import Path
import json
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi # Using the Okapi BM25 variant

# Import configurations and DocumentChunk model
from src import config
from src.data_processing.chunkers import DocumentChunk # Pydantic model for chunks

# --- Simple Text Preprocessing for BM25 ---
def preprocess_text_for_bm25(text: str) -> List[str]:
    """
    Simple preprocessing: lowercase and split by whitespace.
    More advanced preprocessing (stemming, stopword removal) can be added if needed,
    but often simple tokenization works well for BM25 with code/technical text.
    """
    if not text:
        return []
    return text.lower().split()


class BM25Index:
    """
    Manages the BM25 sparse index.
    """
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.bm25_model_file_path = self.index_dir / config.BM25_INDEX_FILENAME
        # We also need to store the mapping from BM25's internal document ID
        # back to our chunk metadata or a persistent chunk ID.
        # For simplicity, we'll store the same metadata list as Faiss,
        # assuming the order of chunks fed to BM25 is the same as for Faiss.
        self.metadata_file_path = self.index_dir / config.METADATA_FILENAME # Reuse from vector_index for consistency

        self.bm25: Optional[BM25Okapi] = None
        self.chunk_corpus_tokenized: List[List[str]] = [] # Stores tokenized version of each chunk's content
        self.chunk_metadata: List[Dict[str, Any]] = [] # Stores metadata for each chunk

        self.index_dir.mkdir(parents=True, exist_ok=True)

    def build_index(self, chunks: List[DocumentChunk], force_rebuild: bool = False) -> bool:
        """
        Builds or rebuilds the BM25 index from a list of DocumentChunks.
        """
        if not force_rebuild and self.bm25_model_file_path.exists() and self.metadata_file_path.exists():
            logger.info(f"BM25 index already exists at {self.index_dir}. Skipping build. Use force_rebuild=True to override.")
            return True

        if not chunks:
            logger.warning("No chunks provided to build the BM25 index.")
            return False

        logger.info(f"Building BM25 index with {len(chunks)} chunks...")

        self.chunk_corpus_tokenized = [preprocess_text_for_bm25(chunk.content) for chunk in chunks]

        # Filter out empty tokenized documents, which can cause issues with BM25
        valid_indices = [i for i, tokens in enumerate(self.chunk_corpus_tokenized) if tokens]
        if not valid_indices:
            logger.warning("All chunks resulted in empty token lists after preprocessing. BM25 index cannot be built.")
            return False

        self.chunk_corpus_tokenized = [self.chunk_corpus_tokenized[i] for i in valid_indices]

        # Prepare metadata only for chunks that are actually indexed
        self.chunk_metadata = []
        for i_original, chunk in enumerate(chunks):
            if i_original not in valid_indices:
                continue
            meta_item = chunk.model_dump(exclude={'absolute_path'})
            meta_item['original_file_path'] = str(chunk.original_file_path)
            meta_item['file_path'] = str(chunk.file_path)
            # The 'id' here corresponds to the index in self.chunk_corpus_tokenized and self.chunk_metadata
            meta_item['bm25_doc_id'] = len(self.chunk_metadata)
            self.chunk_metadata.append(meta_item)

        if not self.chunk_corpus_tokenized: # Should be caught by valid_indices check, but as a safeguard
            logger.error("No valid tokenized content to build BM25 index.")
            return False

        try:
            self.bm25 = BM25Okapi(self.chunk_corpus_tokenized)
            logger.info(f"BM25 index built. Indexed {len(self.chunk_corpus_tokenized)} documents.")
        except Exception as e:
            logger.error(f"Failed to initialize BM25Okapi model: {e}")
            return False

        self.save_index()
        return True

    def save_index(self):
        """Saves the BM25 model and associated metadata to disk."""
        if self.bm25:
            logger.info(f"Saving BM25 model to: {self.bm25_model_file_path}")
            with open(self.bm25_model_file_path, 'wb') as f:
                pickle.dump({
                    "bm25_model": self.bm25,
                    "corpus_tokenized_checksum": len(self.chunk_corpus_tokenized) # Simple checksum
                }, f)
        else:
            logger.warning("No BM25 model to save.")

        if self.chunk_metadata:
            logger.info(f"Saving BM25 metadata to: {self.metadata_file_path} (shared with vector index)")
            # This assumes that the order and content of chunks for BM25 and Faiss are identical.
            # If they can diverge, BM25 should have its own metadata file.
            # For now, we assume they are built from the same chunk list.
            with open(self.metadata_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunk_metadata, f, indent=2)
        else:
            logger.warning("No BM25 metadata to save.")


    def load_index(self) -> bool:
        """Loads the BM25 model and metadata from disk."""
        if not self.bm25_model_file_path.exists() or not self.metadata_file_path.exists():
            logger.warning(f"BM25 index or metadata files not found in {self.index_dir}. Cannot load index.")
            return False

        try:
            logger.info(f"Loading BM25 model from: {self.bm25_model_file_path}")
            with open(self.bm25_model_file_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.bm25 = saved_data["bm25_model"]

            logger.info(f"Loading BM25 metadata from: {self.metadata_file_path}")
            with open(self.metadata_file_path, 'r', encoding='utf-8') as f:
                self.chunk_metadata = json.load(f)

            # Rebuild the tokenized corpus if needed for some BM25 implementations,
            # or ensure the loaded model doesn't strictly require it for get_scores.
            # rank_bm25 typically doesn't need the corpus again after fitting for get_scores.
            # However, it's good practice to ensure consistency.
            # For now, we assume the pickled BM25 object is self-contained for scoring.
            logger.info(f"BM25 index and metadata loaded. Model ready for {len(self.chunk_metadata)} documents.")

            # Sanity check (optional):
            # if saved_data.get("corpus_tokenized_checksum") != len(self.chunk_metadata):
            #    logger.warning("Mismatch between loaded BM25 model's expected corpus size and loaded metadata size.")

            return True
        except FileNotFoundError:
            logger.error(f"BM25 model or metadata file not found during load.")
            return False
        except Exception as e:
            logger.error(f"Error loading BM25 index or metadata: {e}")
            self.bm25 = None
            self.chunk_metadata = []
            return False

    def search(self, query_text: str, top_k: int = config.RETRIEVAL_BM25_TOP_K) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Searches the index for a query text using BM25.

        Args:
            query_text (str): The text to search for.
            top_k (int): The number of top results to return.

        Returns:
            List[Tuple[float, Dict[str, Any]]]: A list of (score, metadata) tuples.
                                                Score is BM25 relevance score (higher is better).
        """
        if self.bm25 is None:
            logger.warning("BM25 Index not loaded or built. Cannot perform search.")
            if not self.load_index(): # Try to load if not already
                return []

        if self.bm25 is None: # Check again after load attempt
            logger.error("Failed to load BM25 index for search.")
            return []

        if not self.chunk_metadata:
            logger.error("BM25 metadata is empty. Cannot map search results.")
            return []

        logger.debug(f"BM25 searching for query: '{query_text[:50]}...' (top_k={top_k})")
        tokenized_query = preprocess_text_for_bm25(query_text)

        if not tokenized_query:
            logger.debug("Tokenized query is empty. BM25 search will likely return no results.")
            return []

        try:
            # Get scores for all documents in the corpus
            doc_scores = self.bm25.get_scores(tokenized_query)
        except Exception as e:
            logger.error(f"Error getting BM25 scores: {e}")
            return []

        # Get top_k indices and scores
        # Ensure we don't request more items than available documents.
        actual_top_k = min(top_k, len(doc_scores))
        if actual_top_k == 0:
            return []

        top_n_indices = np.argsort(doc_scores)[::-1][:actual_top_k] # Sort descending, take top_k

        results = []
        for i in top_n_indices:
            score = doc_scores[i]
            # BM25 scores can be negative if a document has no query terms.
            # We might want to filter out non-positive scores depending on strategy.
            # For now, returning them as is.
            if i < len(self.chunk_metadata):
                results.append((float(score), self.chunk_metadata[i]))
            else:
                logger.warning(f"BM25 returned index {i} which is out of bounds for metadata ({len(self.chunk_metadata)}).")

        logger.debug(f"BM25 search returned {len(results)} results.")
        return results


# --- Example Usage ---
if __name__ == "__main__":
    from src.data_processing.chunkers import DocumentChunk # Re-import for clarity if running standalone
    from pathlib import Path

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")

    # Create dummy chunks for testing (same as vector_index example for consistency)
    dummy_chunks_data = [
        {"file_path": Path("test.py_chunk_1"), "content": "def hello(): return 'world'", "language": "python", "original_file_path": Path("test.py"), "chunk_id":1, "size_bytes":30, "absolute_path": Path("/abs/test.py")},
        {"file_path": Path("test.py_chunk_2"), "content": "class Greeter: def greet(self): print('Python Greeter class hello')", "language": "python", "original_file_path": Path("test.py"), "chunk_id":2, "size_bytes":50, "absolute_path": Path("/abs/test.py")},
        {"file_path": Path("other.js_chunk_1"), "content": "function test() { return 1+1; } // javascript test function", "language": "javascript", "original_file_path": Path("other.js"), "chunk_id":1, "size_bytes":40, "absolute_path": Path("/abs/other.js")},
        {"file_path": Path("empty_content.txt_chunk_1"), "content": "", "language": "text", "original_file_path": Path("empty_content.txt"), "chunk_id":1, "size_bytes":0, "absolute_path": Path("/abs/empty_content.txt")}, # Empty content
    ]
    test_chunks = [DocumentChunk(**data) for data in dummy_chunks_data]

    # Initialize BM25 Index
    test_bm25_index_dir = config.INDEX_DIR / "test_bm25_index"
    bm25_indexer = BM25Index(index_dir=test_bm25_index_dir)

    # Build index
    logger.info("\n--- Building BM25 Index ---")
    build_success = bm25_indexer.build_index(test_chunks, force_rebuild=True)

    if build_success:
        logger.info("BM25 Index built successfully.")

        # Test loading the index
        logger.info("\n--- Testing BM25 Index Loading ---")
        loaded_bm25_indexer = BM25Index(index_dir=test_bm25_index_dir)
        load_success = loaded_bm25_indexer.load_index()

        if load_success:
            logger.info("BM25 Index loaded successfully.")
            # Test search
            logger.info("\n--- Testing BM25 Search ---")

            queries = [
                "python hello function",
                "javascript test",
                "empty", # Should not match much if empty content was filtered
                "Greeter class"
            ]

            for query in queries:
                search_results = loaded_bm25_indexer.search(query, top_k=2)
                logger.info(f"Search results for '{query}':")
                if not search_results:
                    logger.info("  No results found.")
                for score, meta in search_results:
                    logger.info(f"  Score (BM25): {score:.4f}, Chunk Original Path: {meta['original_file_path']}, Chunk ID: {meta['chunk_id']}")
                logger.info("---")

            # A simple check
            results_for_greeter = loaded_bm25_indexer.search("Greeter class", top_k=1)
            if results_for_greeter:
                assert "test.py_chunk_2" in str(results_for_greeter[0][1]['file_path']), "BM25 search result mismatch for 'Greeter class'"
                logger.info("Basic BM25 search assertion passed.")
            else:
                logger.error("BM25 search for 'Greeter class' returned no results.")


        else:
            logger.error("Failed to load the BM25 index for testing.")
    else:
        logger.error("Failed to build the BM25 index for testing.")

    # Clean up test index directory (optional)
    # import shutil
    # if test_bm25_index_dir.exists():
    #     shutil.rmtree(test_bm25_index_dir)
    #     logger.info(f"Cleaned up test BM25 index directory: {test_bm25_index_dir}")
