import asyncio  # For managing pipeline instances
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

# Import project modules
from src import config  # To initialize logging and access configs
from src.generation.generator import LLMGenerator
from src.pipeline import RAGPipeline

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Code-Aware RAG API",
    description="API for interacting with the Code-Aware RAG system.",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State / Cache for RAG Pipelines ---
# In a production system, this would be more robust (e.g., using Redis, a proper cache, or a manager class)
# For simplicity, we use a dictionary to store initialized pipelines by repo_id.
# This means pipelines are stateful per API worker process.
# Key: repo_id (str), Value: RAGPipeline instance
active_pipelines: Dict[str, RAGPipeline] = {}
pipeline_locks: Dict[str, asyncio.Lock] = {} # To prevent concurrent setup for the same repo_id

# --- Pydantic Models for API Requests and Responses ---
class RepositorySetupRequest(BaseModel):
    repo_id: str = Field(..., description="A unique identifier for the repository (e.g., 'github_user_repo'). This will be used for directory names.")
    repo_url_or_path: str = Field(..., description="URL of the Git repository or an absolute path to a local repository.")
    access_token: Optional[str] = Field(None, description="Access token for private Git repositories.")
    force_reclone: bool = Field(False, description="If True, delete and reclone the repository if it already exists locally.")
    force_reindex: bool = Field(False, description="If True, rebuild all indexes even if they exist.")

class RepositorySetupResponse(BaseModel):
    repo_id: str
    message: str
    index_status: str # e.g., "Indexed", "Already Existed", "Failed"
    repository_path: Optional[str] = None # Path where the repo is stored/cloned

class QueryRequest(BaseModel):
    repo_id: str = Field(..., description="The unique identifier of the repository to query (must have been set up first).")
    sys_prompt: Optional[str] = Field(config.GENERATOR_PROMPT, description="The system query for LLM generation.")
    query_text: str = Field(..., description="The user's query about the code repository.")
    top_n_final: Optional[int] = Field(config.RETRIEVAL_VECTOR_TOP_K, description="Number of final context chunks to consider for generation.")
    indexes: list[str] = Field(config.RETRIEVAL_INDEXES, description="The indexes which is enabled for retrieval.")
    vector_top_k: int = Field(config.RETRIEVAL_VECTOR_TOP_K, description="Top_k for vector index.")
    bm25_top_k: int = Field(config.BM25_INDEX_FILENAME, description="Top_k for bm25 sparse index.")


# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    logger.info("Code-Aware RAG API starting up...")
    # Potentially pre-load some default or frequently used pipelines here if desired.
    # For now, pipelines are loaded on demand.

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Code-Aware RAG API shutting down...")
    # Clean up resources if necessary

@app.post("/repository/setup", response_model=RepositorySetupResponse)
async def setup_repository_endpoint(request: RepositorySetupRequest):
    """
    Sets up a repository: clones it (if a URL is provided) and builds its search indexes.
    This operation can be time-consuming.
    """
    repo_id = request.repo_id.replace("/", "_").replace(":", "_") # Sanitize

    # Get or create a lock for this repo_id
    if repo_id not in pipeline_locks:
        pipeline_locks[repo_id] = asyncio.Lock()

    async with pipeline_locks[repo_id]: # Ensure only one setup process for a repo_id at a time
        logger.info(f"Received setup request for repo_id: {repo_id}, source: {request.repo_url_or_path}")

        pipeline = active_pipelines.get(repo_id)
        if not pipeline:
            try:
                pipeline = RAGPipeline(repo_id=repo_id) # Initializes with correct index_dir
                active_pipelines[repo_id] = pipeline
                logger.info(f"Created new RAGPipeline instance for repo_id: {repo_id}")
            except Exception as e:
                logger.error(f"Failed to initialize RAGPipeline for {repo_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Pipeline initialization error: {str(e)}")
        else:
            logger.info(f"Using existing RAGPipeline instance for repo_id: {repo_id}")

        try:
            # The setup_repository method is synchronous. For a truly non-blocking API,
            # this long-running task should be run in a separate thread or process pool.
            # For FastAPI, `run_in_threadpool` from `starlette.concurrency` is an option.
            # For this example, we'll call it directly, which might block the event loop
            # if setup is very long.
            # from starlette.concurrency import run_in_threadpool
            # success = await run_in_threadpool(
            #     pipeline.setup_repository,
            #     repo_url_or_path=request.repo_url_or_path,
            #     access_token=request.access_token,
            #     force_reclone=request.force_reclone,
            #     force_reindex=request.force_reindex
            # )

            # Direct call (can block if setup is long)
            success = pipeline.setup_repository(
                repo_url_or_path=request.repo_url_or_path,
                access_token=request.access_token,
                force_reclone=request.force_reclone,
                force_reindex=request.force_reindex
            )

            if success:
                # Check if indexes actually exist now to determine status
                index_status = "Indexed Successfully"
                if not request.force_reindex and \
                        (pipeline.index_dir / config.FAISS_INDEX_FILENAME).exists() and \
                        (pipeline.index_dir / config.BM25_INDEX_FILENAME).exists():
                    # If not forced and files exist, it implies they were used or just verified
                    # The RAGPipeline.setup_repository already logs if it skips indexing.
                    # We can refine this status based on more detailed feedback from setup_repository if needed.
                    if not request.force_reindex: # If not forced, and they existed, it means they were used.
                        index_status = "Indexes Already Existed or Verified"


                return RepositorySetupResponse(
                    repo_id=repo_id,
                    message="Repository setup process completed.",
                    index_status=index_status,
                    repository_path=str(pipeline.repository_path) if pipeline.repository_path else None
                )
            else:
                # If setup_repository returns False, it means something critical failed.
                # The RAGPipeline should log specifics.
                raise HTTPException(status_code=500, detail="Repository setup failed. Check server logs for details.")

        except ValueError as ve: # Catch specific errors like invalid paths
            logger.error(f"ValueError during repository setup for {repo_id}: {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Unexpected error during repository setup for {repo_id}: {e}")
            # Log the full traceback for debugging
            logger.exception("Repository setup exception:")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/query/stream")
async def query_repository_stream(request: QueryRequest) -> StreamingResponse:

    """
    Queries an already set up repository and streams the LLM's response.
    """
    repo_id = request.repo_id.replace("/", "_").replace(":", "_") # Sanitize
    logger.info(f"Received streaming query for repo_id: '{repo_id}', query: '{request.query_text[:50]}...'")

    pipeline = active_pipelines.get(repo_id)
    if not pipeline:
        logger.warning(f"No active RAGPipeline found for repo_id: {repo_id}. Repository might not be set up.")
        # Attempt to create and load on the fly if indexes exist
        try:
            pipeline = RAGPipeline(repo_id=repo_id, indexes=request.indexes)
            if not pipeline.retriever.vector_index and not pipeline.retriever.bm25_index:
                # This means _load_indexes inside HybridRetriever failed to find existing indexes
                raise HTTPException(status_code=404, detail=f"Repository with repo_id '{repo_id}' not found or not indexed. Please set it up first.")
            active_pipelines[repo_id] = pipeline # Cache it if successfully loaded
            logger.info(f"Dynamically loaded RAGPipeline for repo_id: {repo_id} for query.")
        except Exception as e:
            logger.error(f"Failed to dynamically load RAGPipeline for {repo_id}: {e}")
            raise HTTPException(status_code=404, detail=f"Repository with repo_id '{repo_id}' not found or not indexed. Please set it up first.")


    # 1. Retrieve context chunks
    try:
        # The query method in RAGPipeline is synchronous.
        # For a truly non-blocking API, this should also be run in a threadpool.
        # from starlette.concurrency import run_in_threadpool
        # context_chunks_meta = await run_in_threadpool(
        #     pipeline.query,
        #     query_text=request.query_text,
        #     top_n_final=request.top_n_final
        # )

        # Direct call (can block if retrieval is very slow, though usually fast)
        context_chunks_meta: List[Dict[str, Any]] = pipeline.query(
            query_text=request.query_text,
            top_n_final=request.top_n_final,
            vector_top_k=request.vector_top_k,
            bm25_top_k=request.bm25_top_k
        )
        logger.info(f"Retrieved {len(context_chunks_meta)} context chunks for query.")

    except Exception as e:
        logger.error(f"Error during retrieval for repo_id {repo_id}: {e}")
        logger.exception("Retrieval exception:")
        raise HTTPException(status_code=500, detail=f"Error retrieving context: {str(e)}")

    # 2. Initialize LLM Generator (uses settings from config.py by default)
    try:
        llm_generator = LLMGenerator() # Uses defaults from config.py
    except Exception as e:
        logger.error(f"Failed to initialize LLMGenerator: {e}")
        raise HTTPException(status_code=500, detail=f"LLM Generator initialization error: {str(e)}")

    # 3. Stream response from LLM
    try:
        response_stream_iterator = llm_generator.generate_response_stream(
            sys_prompy=request.sys_prompt,
            user_query=request.query_text,
            context_chunks=context_chunks_meta
        )
        return StreamingResponse(response_stream_iterator, media_type="text/plain")
    except Exception as e:
        logger.error(f"Error during LLM response generation for repo_id {repo_id}: {e}")
        logger.exception("LLM generation exception:")
        # Note: If the error happens inside the async generator, it might be harder to catch here.
        # The generator itself should handle internal errors and yield error messages if possible.
        raise HTTPException(status_code=500, detail=f"Error generating LLM response: {str(e)}")


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "message": "Code-Aware RAG API is running."}

# --- Main function for local Uvicorn execution (optional, usually use main.py) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API server directly from api.py for debugging...")
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, reload=config.API_RELOAD)