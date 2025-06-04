import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import faiss
import numpy as np
from loguru import logger

from src import config
from src.data_processing.chunkers import DocumentChunk  # Pydantic model for chunks

def generate_embeddings(
        texts: List[str],
        model_name: str = config.EMBEDDING_MODEL_NAME,
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
        dimensions: int = config.EMBEDDING_DIMENSIONS,
        apikey: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Generates embeddings for a list of texts using the configured provider.
    """
    if not texts:
        return np.array([])

    logger.info(f"Generating embeddings for {len(texts)} texts using {model_name} (batch size: {batch_size}).")

    client = config.get_openai_embeddings_client(apikey = apikey)
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                input=batch_texts,
                model=model_name,
                encoding_format="float",
                dimensions=dimensions # some models support this
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            logger.debug(f"Generated embeddings for batch {i//batch_size + 1}")
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings for batch: {e}")
            # Decide on error handling: skip batch, return None, or raise
            return None # Or handle more gracefully
    embeddings_list = all_embeddings

    if not embeddings_list:
        logger.warning("No embeddings were generated.")
        return np.array([])

    try:
        return np.array(embeddings_list).astype('float32')
    except ValueError as ve:
        logger.error(f"Could not convert embeddings to numpy array. Check for consistent embedding dimensions: {ve}")
        # Log dimensions of first few embeddings if possible
        if embeddings_list:
            for i, emb in enumerate(embeddings_list[:3]):
                logger.debug(f"Embedding {i} length: {len(emb) if isinstance(emb, list) else 'N/A'}")
        return None


class FaissVectorIndex:
    """
    Manages the FAISS vector index and associated metadata.
    """
    def __init__(self, index_dir: Path,
                 embedding_dim: int = config.EMBEDDING_DIMENSIONS,
                 model_name: str = config.EMBEDDING_MODEL_NAME,
                 batch_size: int = config.EMBEDDING_BATCH_SIZE):
        self.index_dir = index_dir
        self.embedding_dim = embedding_dim
        self.index_file_path = self.index_dir / config.FAISS_INDEX_FILENAME
        self.metadata_file_path = self.index_dir / config.METADATA_FILENAME
        self.model = model_name
        self.batch_size = batch_size

        self.index: Optional[faiss.Index] = None
        self.chunk_metadata: List[Dict[str, Any]] = [] # Stores metadata for each vector

        self.index_dir.mkdir(parents=True, exist_ok=True)

    def build_index(self, chunks: List[DocumentChunk], force_rebuild: bool = False, apikey: Optional[str] = None) -> bool:
        """
        Builds or rebuilds the FAISS index from a list of DocumentChunks.
        """
        if not force_rebuild and self.index_file_path.exists() and self.metadata_file_path.exists():
            logger.info(f"Index already exists at {self.index_dir}. Skipping build. Use force_rebuild=True to override.")
            # Optionally load existing index here if needed for immediate use,
            # but typically build is called when creating, and load when using.
            return True

        if not chunks:
            logger.warning("No chunks provided to build the index.")
            return False

        logger.info(f"Building FAISS index with {len(chunks)} chunks...")
        chunk_texts = [chunk.content for chunk in chunks]

        embeddings = generate_embeddings(chunk_texts, model_name=self.model, batch_size=self.batch_size, dimensions=self.embedding_dim, apikey=apikey)

        if embeddings is None or embeddings.shape[0] == 0:
            logger.error("Failed to generate embeddings. Index not built.")
            return False

        if embeddings.shape[1] != self.embedding_dim:
            logger.error(f"Generated embeddings dimension ({embeddings.shape[1]}) does not match configured dimension ({self.embedding_dim}). Index not built.")
            # Update config or check embedding model if this happens
            # config.EMBEDDING_DIMENSIONS = embeddings.shape[1] # Risky, better to fix model or config
            return False

        # Initialize FAISS index (IndexFlatL2 is a common choice for exact search)
        # For larger datasets, consider more advanced index types like IndexIVFFlat.
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings)
        logger.info(f"FAISS index built. Total vectors: {self.index.ntotal}")

        # Prepare and save metadata
        self.chunk_metadata = []
        for i, chunk in enumerate(chunks):
            # Store essential metadata. Convert Path objects to strings for JSON serialization.
            meta_item = chunk.model_dump(exclude={'absolute_path'}) # Exclude large fields or non-serializable
            meta_item['original_file_path'] = str(chunk.original_file_path)
            meta_item['file_path'] = str(chunk.file_path) # This is the chunk's unique path
            meta_item['vector_id'] = i # Simple mapping: vector index in FAISS = index in this list
            self.chunk_metadata.append(meta_item)

        self.save_index()
        return True

    def save_index(self):
        """Saves the FAISS index and metadata to disk."""
        if self.index:
            logger.info(f"Saving FAISS index to: {self.index_file_path}")
            faiss.write_index(self.index, str(self.index_file_path))
        else:
            logger.warning("No FAISS index to save.")

        if self.chunk_metadata:
            logger.info(f"Saving metadata to: {self.metadata_file_path}")
            with open(self.metadata_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunk_metadata, f, indent=2)
        else:
            logger.warning("No metadata to save.")

    def load_index(self) -> bool:
        """Loads the FAISS index and metadata from disk."""
        if not self.index_file_path.exists() or not self.metadata_file_path.exists():
            logger.warning(f"Index files not found in {self.index_dir}. Cannot load index.")
            return False

        try:
            logger.info(f"Loading FAISS index from: {self.index_file_path}")
            self.index = faiss.read_index(str(self.index_file_path))
            logger.info(f"FAISS index loaded. Total vectors: {self.index.ntotal}, Dimensions: {self.index.d}")

            if self.index.d != self.embedding_dim:
                logger.warning(f"Loaded index dimension ({self.index.d}) differs from configured ({self.embedding_dim}). This might cause issues.")
                # self.embedding_dim = self.index.d # Update if needed

            logger.info(f"Loading metadata from: {self.metadata_file_path}")
            with open(self.metadata_file_path, 'r', encoding='utf-8') as f:
                self.chunk_metadata = json.load(f)
            logger.info(f"Metadata loaded for {len(self.chunk_metadata)} chunks.")
            return True
        except Exception as e:
            logger.error(f"Error loading FAISS index or metadata: {e}")
            self.index = None
            self.chunk_metadata = []
            return False

    def search(self,
               query_text: str,
               top_k: int = config.RETRIEVAL_VECTOR_TOP_K,
               model: str = config.EMBEDDING_MODEL_NAME,
               batch_size: int = config.EMBEDDING_BATCH_SIZE,
               dimensions: int = config.EMBEDDING_DIMENSIONS,
               apikey: Optional[str] = None) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Searches the index for a query text.

        Args:
            query_text (str): The text to search for.
            top_k (int): The number of top results to return.

        Returns:
            List[Tuple[float, Dict[str, Any]]]: A list of (score, metadata) tuples.
                                                Score is distance (lower is better for L2).
        """
        if self.index is None:
            logger.warning("FAISS Index not loaded or built. Cannot perform search.")
            if not self.load_index(): # Try to load if not already
                return []

        if self.index is None: # Check again after attempting load
            logger.error("Failed to load index for search.")
            return []


        logger.debug(f"Searching index for query: '{query_text[:50]}...' (top_k={top_k})")
        query_embedding = generate_embeddings(texts =[query_text], apikey=apikey, model_name=model, batch_size=batch_size, dimensions=dimensions)

        if query_embedding is None or query_embedding.shape[0] == 0:
            logger.error("Failed to generate embedding for query. Search aborted.")
            return []

        # FAISS search returns distances (D) and indices (I)
        # Distances are L2 distances (lower is better)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        if indices.size > 0:
            for i in range(indices.shape[1]): # Iterate through top_k results for the query
                vector_id = indices[0, i]
                distance = distances[0, i]
                if vector_id < 0 or vector_id >= len(self.chunk_metadata): # faiss can return -1 if not enough results
                    logger.warning(f"Invalid vector_id {vector_id} from FAISS search. Skipping.")
                    continue

                # Convert L2 distance to a similarity score (e.g., 1 / (1 + distance)) if desired,
                # or just use distance. For now, returning distance.
                results.append((float(distance), self.chunk_metadata[vector_id]))

        logger.debug(f"Search returned {len(results)} results.")
        return results


# --- Example Usage ---
if __name__ == "__main__":
    from src.data_processing.chunkers import DocumentChunk
    from pathlib import Path
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")

    # Create dummy chunks for testing
    dummy_chunks_data = [
        {"file_path": Path("test.py_chunk_1"), "content": "def hello(): return 'world'", "language": "python", "original_file_path": Path("test.py"), "chunk_id":1, "size_bytes":30, "absolute_path": Path("/abs/test.py")},
        {"file_path": Path("test.py_chunk_2"), "content": "class Greeter: def greet(self): print('Hello')", "language": "python", "original_file_path": Path("test.py"), "chunk_id":2, "size_bytes":50, "absolute_path": Path("/abs/test.py")},
        {"file_path": Path("other.js_chunk_1"), "content": "function test() { return 1+1; }", "language": "javascript", "original_file_path": Path("other.js"), "chunk_id":1, "size_bytes":40, "absolute_path": Path("/abs/other.js")},
    ]
    test_chunks = [DocumentChunk(**data) for data in dummy_chunks_data]

    # Initialize Vector Index
    # Ensure config.INDEX_DIR is writable or change to a temp dir for testing
    test_index_dir = config.INDEX_DIR / "test_vector_index"
    vector_indexer = FaissVectorIndex(index_dir=test_index_dir, embedding_dim=config.EMBEDDING_DIMENSIONS)

    # Build index
    logger.info("\n--- Building Vector Index ---")
    build_success = vector_indexer.build_index(test_chunks, force_rebuild=True)

    if build_success:
        logger.info("Index built successfully.")

        # Test loading the index (as if in a new session)
        logger.info("\n--- Testing Index Loading ---")
        loaded_vector_indexer = FaissVectorIndex(index_dir=test_index_dir, embedding_dim=config.EMBEDDING_DIMENSIONS)
        load_success = loaded_vector_indexer.load_index()

        if load_success:
            logger.info("Index loaded successfully.")
            # Test search
            logger.info("\n--- Testing Search ---")
            query = "python hello world function"
            search_results = loaded_vector_indexer.search(query, top_k=2)

            logger.info(f"Search results for '{query}':")
            for score, meta in search_results:
                logger.info(f"  Score (Distance): {score:.4f}, Chunk Original Path: {meta['original_file_path']}, Chunk ID: {meta['chunk_id']}")
                # logger.info(f"     Content: {meta['content'][:50]}...") # content is not in meta by default

            # A simple check
            if search_results:
                assert "test.py" in str(search_results[0][1]['original_file_path']), "Search result mismatch"
                logger.info("Basic search assertion passed.")

        else:
            logger.error("Failed to load the index for testing.")
    else:
        logger.error("Failed to build the index for testing.")

    # Clean up test index directory (optional)
    # import shutil
    # if test_index_dir.exists():
    #     shutil.rmtree(test_index_dir)
    #     logger.info(f"Cleaned up test index directory: {test_index_dir}")
