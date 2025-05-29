import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Any

import tiktoken  # For token counting in TextSplitter
from loguru import logger
from pydantic import Field
from tree_sitter import Language, Parser, Node

# Import configurations and LoadedDocument model
from src import config
from src.data_processing.document_loader import LoadedDocument  # Assuming this is the Pydantic model

# --- Helper for Token Counting (consistent across chunkers) ---
_token_encoders: Dict[str, Any] = {}

def count_document_tokens(text: str, model_name: str = config.EMBEDDING_MODEL_NAME) -> int:
    """Counts tokens for a given text using tiktoken, caching encoders."""
    # This function might be slightly different from one in config if that one is for LLMs
    # and this one is specifically for embedding model token limits.
    # For simplicity, we'll assume config.EMBEDDING_MODEL_NAME is appropriate.
    if model_name not in _token_encoders:
        try:
            _token_encoders[model_name] = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(f"Warning: model {model_name} not found. Using cl100k_base encoding.")
            _token_encoders[model_name] = tiktoken.get_encoding("cl100k_base")

    try:
        return len(_token_encoders[model_name].encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}. Approximating with len/4.")
        return len(text) // 4


# --- Pydantic Model for Chunks ---
# We can reuse LoadedDocument or create a new Chunk model if metadata differs significantly.
# For now, let's assume a chunk is also a form of "document" for embedding.
# If we need more specific chunk metadata, we'd define a new Pydantic model here.
class DocumentChunk(LoadedDocument):
    """Represents a chunk of a document, ready for embedding."""
    original_file_path: Path = Field(description="Relative path of the original file.")
    chunk_id: int = Field(description="Sequential ID of the chunk within the original document.")
    start_char_offset: Optional[int] = Field(None, description="Start character offset in the original content.")
    end_char_offset: Optional[int] = Field(None, description="End character offset in the original content.")
    # Tree-sitter specific metadata
    code_construct_type: Optional[str] = Field(None, description="Type of code construct (e.g., function, class).")
    code_construct_name: Optional[str] = Field(None, description="Name of the code construct.")
    start_line: Optional[int] = Field(None, description="Start line number in the original file.")
    end_line: Optional[int] = Field(None, description="End line number in the original file.")

    class Config:
        arbitrary_types_allowed = True


# --- Abstract Base Class for Chunkers ---
class BaseChunker(ABC):
    @abstractmethod
    def chunk_document(self, document: LoadedDocument) -> List[DocumentChunk]:
        """Chunks a single LoadedDocument into multiple DocumentChunks."""
        pass

    def chunk_documents(self, documents: List[LoadedDocument]) -> List[DocumentChunk]:
        """Chunks a list of LoadedDocuments."""
        all_chunks = []
        for doc in documents:
            try:
                chunks = self.chunk_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to chunk document {doc.file_path}: {e}")
        return all_chunks


# --- TreeSitterChunker ---
# Global cache for loaded tree-sitter languages and parsers
_ts_languages: Dict[str, Language] = {}
_ts_parsers: Dict[str, Parser] = {}

def get_tree_sitter_parser(language_name: str) -> Optional[Parser]:
    """
    Loads and returns a tree-sitter parser for the given language.
    Uses tree_sitter_languages if available, otherwise expects grammars
    to be manually managed (e.g., in config.GRAMMAR_DIR).
    """
    if language_name in _ts_parsers:
        return _ts_parsers[language_name]

    lang_config = config.TREE_SITTER_LANGUAGES.get(language_name)
    if not lang_config:
        logger.warning(f"No tree-sitter configuration found for language: {language_name}")
        return None

    ts_lang_obj: Optional[Language] = None
    grammar_name_in_lib = lang_config.get("grammar_name", language_name)

    try:
        try:
            from tree_sitter_languages import get_language
            ts_lang_obj = get_language(grammar_name_in_lib)
            logger.info(f"Successfully loaded '{grammar_name_in_lib}' grammar using tree_sitter_languages.")
        except Exception as e:
            logger.error(f"Failed to load grammar for '{grammar_name_in_lib}' via tree_sitter_languages: {e}")
            
            grammar_file = lang_config.get("grammar_file")
            if grammar_file:
                grammar_path = config.GRAMMAR_DIR / grammar_file
                if grammar_path.exists():
                    try:
                        ts_lang_obj = Language(str(grammar_path), grammar_name_in_lib)
                        logger.info(f"Successfully loaded grammar from {grammar_path} for {grammar_name_in_lib}")
                    except Exception as e2:
                        logger.error(f"Failed to load grammar from {grammar_path} for {grammar_name_in_lib}: {e2}")
                else:
                    logger.error(f"Grammar file {grammar_path} not found for language {language_name}.")
            else:
                logger.warning(f"No grammar_file specified for {language_name} and tree_sitter_languages failed.")
    except Exception as e:
        logger.error(f"Failed to load grammar for '{grammar_name_in_lib}': {e}")

    if ts_lang_obj:
        parser = Parser()
        parser.set_language(ts_lang_obj)
        _ts_parsers[language_name] = parser
        return parser

    logger.error(f"Could not initialize tree-sitter parser for language: {language_name}")
    return None


class TreeSitterChunker(BaseChunker):
    """
    Chunks code documents based on Abstract Syntax Trees (AST) using tree-sitter.
    """
    def __init__(
            self,
            max_chunk_tokens: int = config.AST_CHUNK_MAX_TOKENS,
            min_chunk_tokens: int = config.AST_CHUNK_MIN_TOKENS,
            fallback_chunker: Optional[BaseChunker] = None # For nodes too large or too small after AST
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.fallback_chunker = fallback_chunker
        logger.info(f"TreeSitterChunker initialized. Max tokens: {max_chunk_tokens}, Min tokens: {min_chunk_tokens}")

        # Define node types that represent significant, self-contained code blocks
        # This can be extended per language.
        self.significant_node_types_map: Dict[str, List[str]] = {
            "python": ["function_definition", "class_definition"],
            "java": ["method_declaration", "class_declaration", "interface_declaration", "enum_declaration", "constructor_declaration"],
            "javascript": ["function_declaration", "class_declaration", "method_definition", "lexical_declaration", "variable_declaration"], # lex/var for top-level const/let/var
            "typescript": ["function_declaration", "class_declaration", "method_definition", "interface_declaration", "enum_declaration", "lexical_declaration", "variable_declaration"],
            # Add more languages and their respective AST node types
        }

    def _get_node_name(self, node: Node, language: str) -> str:
        """Extracts a name for a given AST node (e.g., function name, class name)."""
        # This is language-specific.
        if language in ["python", "javascript", "typescript", "java"]: # Common field name
            name_node = node.child_by_field_name("name")
            if name_node:
                return name_node.text.decode('utf-8')

        # Fallback or more specific logic for other languages/nodes
        # For example, for a Java method, the name is within a specific child sequence.
        # This might need more sophisticated AST traversal for some languages/constructs.
        # For now, a simple "name" field check.
        return "anonymous_construct"


    def _extract_chunks_from_node(
            self,
            node: Node,
            document: LoadedDocument,
            chunk_id_counter: List[int] # Use a list to pass by reference for mutable int
    ) -> List[DocumentChunk]:
        """
        Recursively traverses the AST and extracts chunks.
        """
        chunks: List[DocumentChunk] = []
        language = document.language
        if not language: # Should not happen if document_loader sets it
            return chunks

        significant_node_types = self.significant_node_types_map.get(language, [])
        node_text = node.text.decode('utf-8')
        node_tokens = count_document_tokens(node_text)

        is_significant_node = node.type in significant_node_types

        if is_significant_node and node_tokens >= self.min_chunk_tokens:
            if node_tokens <= self.max_chunk_tokens:
                chunk_id_counter[0] += 1
                chunks.append(DocumentChunk(
                    file_path=Path(f"{document.file_path}_chunk_{chunk_id_counter[0]}"), # Create a unique path for the chunk
                    absolute_path=document.absolute_path, # Original file path
                    content=node_text,
                    language=language,
                    size_bytes=len(node_text.encode('utf-8')),
                    original_file_path=document.file_path,
                    chunk_id=chunk_id_counter[0],
                    start_char_offset=node.start_byte,
                    end_char_offset=node.end_byte,
                    code_construct_type=node.type,
                    code_construct_name=self._get_node_name(node, language),
                    start_line=node.start_point[0] + 1, # tree-sitter is 0-indexed
                    end_line=node.end_point[0] + 1
                ))
                return chunks # This node is a good chunk, don't break it further unless it's too big (handled by max_tokens check)
            else:
                # Significant node is too large, try to break it down by its children.
                # If children also don't yield good chunks, this large node might be passed to fallback_chunker.
                logger.debug(f"Node type '{node.type}' in {document.file_path} is too large ({node_tokens} tokens). Processing children.")
                # Proceed to process children

        # If not a significant node on its own, or if it was too large and we are processing children:
        for child_node in node.children:
            chunks.extend(self._extract_chunks_from_node(child_node, document, chunk_id_counter))

        # If after processing all children, this node itself (if not significant initially)
        # forms a coherent block of text that's not too small, and it wasn't broken down,
        # and it's not the root of a huge file that should be text-split.
        # This part is tricky. For now, we primarily rely on significant_node_types.
        # If no significant children were found and this node is the root of a small file,
        # it might be passed to the fallback chunker.

        return chunks

    def chunk_document(self, document: LoadedDocument) -> List[DocumentChunk]:
        if not document.language or document.language not in config.TREE_SITTER_LANGUAGES:
            logger.debug(f"Language '{document.language}' not configured for TreeSitter. Using fallback for: {document.file_path}")
            if self.fallback_chunker:
                return self.fallback_chunker.chunk_document(document)
            return [ # Create a single chunk from the whole document if no fallback
                DocumentChunk(
                    **document.model_dump(),
                    original_file_path=document.file_path,
                    chunk_id=0,
                    start_char_offset=0,
                    end_char_offset=len(document.content.encode('utf-8'))
                )
            ]

        parser = get_tree_sitter_parser(document.language)
        if not parser:
            logger.warning(f"No tree-sitter parser for language '{document.language}'. Using fallback for: {document.file_path}")
            if self.fallback_chunker:
                return self.fallback_chunker.chunk_document(document)
            return [DocumentChunk(**document.model_dump(), original_file_path=document.file_path, chunk_id=0, start_char_offset=0, end_char_offset=len(document.content.encode('utf-8')))]

        logger.debug(f"AST Chunking: {document.file_path} (Language: {document.language})")
        try:
            tree = parser.parse(bytes(document.content, "utf8"))
            root_node = tree.root_node

            chunk_id_counter = [0] # Mutable int passed by reference
            ast_chunks = self._extract_chunks_from_node(root_node, document, chunk_id_counter)

            if not ast_chunks: # If AST chunking produced nothing (e.g. empty file, or very small file with no significant constructs)
                logger.info(f"AST chunking yielded no specific chunks for {document.file_path}. Using fallback or treating as single chunk.")
                if self.fallback_chunker:
                    return self.fallback_chunker.chunk_document(document)
                # If no fallback, and the document is small enough, treat it as one chunk.
                # Otherwise, it might be an issue if it's a large code file that couldn't be AST-parsed.
                doc_tokens = count_document_tokens(document.content)
                if doc_tokens <= self.max_chunk_tokens:
                    return [DocumentChunk(**document.model_dump(), original_file_path=document.file_path, chunk_id=0, start_char_offset=0, end_char_offset=len(document.content.encode('utf-8')))]
                else:
                    logger.warning(f"Document {document.file_path} yielded no AST chunks and is too large ({doc_tokens} tokens). Consider a fallback text splitter for such cases if not already configured.")
                    # If no fallback, it becomes one large chunk, which might be problematic for embedding.
                    return [DocumentChunk(**document.model_dump(), original_file_path=document.file_path, chunk_id=0, start_char_offset=0, end_char_offset=len(document.content.encode('utf-8')))]


            # Post-process AST chunks: if any AST chunk is still too large, use fallback_chunker on it.
            final_chunks = []
            for chunk in ast_chunks:
                chunk_tokens = count_document_tokens(chunk.content)
                if chunk_tokens > self.max_chunk_tokens and self.fallback_chunker:
                    logger.debug(f"AST chunk '{chunk.code_construct_name}' from {document.file_path} is still too large ({chunk_tokens} tokens). Applying fallback chunker.")
                    # Create a temporary LoadedDocument for the fallback_chunker
                    temp_doc_for_fallback = LoadedDocument(
                        file_path=chunk.original_file_path, # Keep original file context
                        absolute_path=chunk.absolute_path,
                        content=chunk.content,
                        language=chunk.language,
                        size_bytes=chunk.size_bytes
                    )
                    fallback_sub_chunks = self.fallback_chunker.chunk_document(temp_doc_for_fallback)
                    # Need to adjust metadata of fallback_sub_chunks to reflect they are part of the larger AST chunk
                    for i, sub_chunk in enumerate(fallback_sub_chunks):
                        sub_chunk.original_file_path = chunk.original_file_path
                        sub_chunk.chunk_id = int(f"{chunk.chunk_id}{i:03d}") # Composite ID
                        # Offsets would need to be relative to the original document, which is complex here.
                        # For now, keep them relative to the chunk.content fed to fallback.
                        sub_chunk.code_construct_type = f"{chunk.code_construct_type}_sub_split"
                        sub_chunk.code_construct_name = chunk.code_construct_name
                        sub_chunk.start_line = chunk.start_line # Approximate, as line numbers change with sub-splitting
                        sub_chunk.end_line = chunk.end_line
                    final_chunks.extend(fallback_sub_chunks)
                else:
                    final_chunks.append(chunk)

            logger.info(f"AST chunking for {document.file_path} produced {len(final_chunks)} chunks.")
            return final_chunks

        except Exception as e:
            logger.error(f"Error during AST chunking for {document.file_path}: {e}. Using fallback.")
            if self.fallback_chunker:
                return self.fallback_chunker.chunk_document(document)
            return [DocumentChunk(**document.model_dump(), original_file_path=document.file_path, chunk_id=0, start_char_offset=0, end_char_offset=len(document.content.encode('utf-8')))]


# --- TokenSplitter (Fallback) ---
class TokenSplitter(BaseChunker):
    """
    Splits text documents based on token count with overlap.
    Serves as a fallback for non-code text or when AST chunking is not applicable/effective.
    """
    def __init__(
            self,
            chunk_size: int = config.TEXT_CHUNK_SIZE,
            chunk_overlap: int = config.TEXT_CHUNK_OVERLAP,
            embedding_model_name: str = config.EMBEDDING_MODEL_NAME
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.embedding_model_name)
        except KeyError:
            logger.warning(
                f"Warning: model {self.embedding_model_name} not found for tiktoken. Using cl100k_base."
            )
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        logger.info(f"TokenSplitter initialized. Chunk size: {chunk_size} tokens, Overlap: {chunk_overlap} tokens.")

    def chunk_document(self, document: LoadedDocument) -> List[DocumentChunk]:
        if not document.content.strip(): # Handle empty or whitespace-only content
            logger.debug(f"Document {document.file_path} is empty or whitespace-only. Skipping chunking.")
            return []

        tokens = self.tokenizer.encode(document.content)
        chunks = []
        chunk_id = 0

        start_token_idx = 0
        while start_token_idx < len(tokens):
            end_token_idx = min(start_token_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_token_idx:end_token_idx]

            # Decode tokens back to text for the chunk
            # Handling potential errors during decode if tokens are split mid-character for some encodings
            try:
                chunk_text = self.tokenizer.decode(chunk_tokens)
            except Exception as e:
                logger.error(f"Error decoding token chunk for {document.file_path}: {e}. Trying to decode with error handling.")
                # Try to decode smaller parts or skip problematic tokens if this becomes an issue.
                # For now, a simple robust decode:
                chunk_text = "".join(
                    self.tokenizer.decode_single_token_bytes(token).decode('utf-8', errors='replace')
                    for token in chunk_tokens
                )


            # Approximate char offsets - this is not perfectly accurate but good for reference
            # A more precise way would be to map token indices back to char offsets in original text
            # For now, simple approximation based on previous chunks' text length
            start_char = sum(len(c.content) for c in chunks) if chunks else 0
            # This approximation for start_char is flawed if there's overlap.
            # A better way for start/end char offset is needed if it's critical.
            # For now, let's make them None for TokenSplitter or mark as approximate.

            chunk_id += 1
            chunks.append(DocumentChunk(
                file_path=Path(f"{document.file_path}_chunk_{chunk_id}"), # Unique path for the chunk
                absolute_path=document.absolute_path,
                content=chunk_text,
                language=document.language,
                size_bytes=len(chunk_text.encode('utf-8')),
                original_file_path=document.file_path,
                chunk_id=chunk_id,
                # Character offsets are hard to get right with token-based splitting & overlap.
                # Set to None or clearly mark as approximate if implemented.
                start_char_offset=None, # Placeholder
                end_char_offset=None,   # Placeholder
            ))

            if end_token_idx == len(tokens): # Reached the end
                break

            start_token_idx += (self.chunk_size - self.chunk_overlap)

        logger.debug(f"TokenSplitter for {document.file_path} produced {len(chunks)} chunks.")
        return chunks


# --- Main Chunker (Orchestrator) ---
class CodeRAGChunker(BaseChunker):
    """
    Orchestrates chunking: uses TreeSitterChunker for code,
    and a fallback (e.g., TokenSplitter) for other text or large code elements.
    """
    def __init__(self):
        self.text_splitter = TokenSplitter() # Default fallback
        self.tree_sitter_chunker = TreeSitterChunker(fallback_chunker=self.text_splitter)
        logger.info("CodeRAGChunker initialized with TreeSitterChunker and TokenSplitter fallback.")

    def chunk_document(self, document: LoadedDocument) -> List[DocumentChunk]:
        if document.language in config.TREE_SITTER_LANGUAGES:
            logger.debug(f"Using TreeSitterChunker for: {document.file_path} (Lang: {document.language})")
            return self.tree_sitter_chunker.chunk_document(document)
        else:
            logger.debug(f"Using TokenSplitter (fallback) for: {document.file_path} (Lang: {document.language or 'unknown'})")
            return self.text_splitter.chunk_document(document)


# --- Example Usage ---
if __name__ == "__main__":
    from pathlib import Path

    # Setup logger for testing
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="DEBUG")

    # Create dummy documents (mimicking output from document_loader)
    dummy_py_content = """
def factorial(n):
    # This is a comment
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        # Another comment
        return a - b

# A very long function that might exceed max_chunk_tokens
def very_long_function_example():
    s = "long_string" * 500 # This will make the text large
    print(s)
    # imagine more code here making it exceed AST_CHUNK_MAX_TOKENS
    return len(s)
"""
    dummy_md_content = "# Title\n\nThis is some markdown content with paragraphs.\n\n- Item 1\n- Item 2\n\n"

    # Ensure dummy files have unique paths for chunk naming
    doc1_path = Path("dummy_module.py")
    doc2_path = Path("notes.md")

    documents_to_chunk = [
        LoadedDocument(file_path=doc1_path, absolute_path=Path("/abs") / doc1_path, content=dummy_py_content, language="python", size_bytes=len(dummy_py_content.encode())),
        LoadedDocument(file_path=doc2_path, absolute_path=Path("/abs") / doc2_path, content=dummy_md_content, language="markdown", size_bytes=len(dummy_md_content.encode())),
        LoadedDocument(file_path=Path("empty.py"), absolute_path=Path("/abs/empty.py"), content="", language="python", size_bytes=0),
        LoadedDocument(file_path=Path("small.py"), absolute_path=Path("/abs/small.py"), content="x=1", language="python", size_bytes=3),

    ]

    # Initialize the main chunker
    main_chunker = CodeRAGChunker()

    all_final_chunks: List[DocumentChunk] = main_chunker.chunk_documents(documents_to_chunk)

    logger.info(f"\n--- Total Chunks Produced: {len(all_final_chunks)} ---")
    for i, chunk in enumerate(all_final_chunks):
        logger.info(f"Chunk {i+1}:")
        logger.info(f"  Original File: {chunk.original_file_path}")
        logger.info(f"  Chunk ID in Original: {chunk.chunk_id}")
        logger.info(f"  Language: {chunk.language}")
        logger.info(f"  Type: {chunk.code_construct_type or 'text'}")
        if chunk.code_construct_name:
            logger.info(f"  Name: {chunk.code_construct_name}")
        if chunk.start_line and chunk.end_line:
            logger.info(f"  Lines: {chunk.start_line}-{chunk.end_line}")
        logger.info(f"  Content Preview: {chunk.content[:100].replace(os.linesep, ' ')}...")
        logger.info(f"  Token count (approx): {count_document_tokens(chunk.content)}")
        logger.info("-" * 20)

