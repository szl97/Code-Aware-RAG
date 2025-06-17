import asyncio  # For managing pipeline instances
from typing import List, Dict, Any, Optional, Union
import threading

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field
from fastapi.security import OAuth2PasswordBearer

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

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/api-doc/dummy-token-url")

# --- Global State / Cache for RAG Pipelines ---
# In a production system, this would be more robust (e.g., using Redis, a proper cache, or a manager class)
# For simplicity, we use a dictionary to store initialized pipelines by repo_id.
# This means pipelines are stateful per API worker process.
# Key: repo_id (str), Value: RAGPipeline instance
pipeline_locks: Dict[str, asyncio.Lock] = {} # To prevent concurrent setup for the same repo_id

# Track repository setup status
# Key: repo_id (str), Value: Dict with status information
setup_status: Dict[str, Dict[str, Any]] = {}

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
    index_status: str # e.g., "Indexed", "Already Existed", "Failed", "In Progress"
    repository_path: Optional[str] = None # Path where the repo is stored/cloned
    task_id: Optional[str] = None # ID to track background task

class RepositoryStatusResponse(BaseModel):
    repo_id: str
    status: str # "pending", "completed", "failed"
    message: str
    index_status: Optional[str] = None
    repository_path: Optional[str] = None

class QueryRequest(BaseModel):
    repo_id: str = Field(..., description="The unique identifier of the repository to query (must have been set up first).")
    sys_prompt: str = Field(config.GENERATOR_PROMPT, description="The system query for LLM generation.")
    query_text: str = Field(..., description="The user's query about the code repository.")
    top_n_final: int = Field(config.RETRIEVAL_VECTOR_TOP_K, description="Number of final context chunks to consider for generation.")
    indexes: list[str] = Field(config.RETRIEVAL_INDEXES, description="The indexes which is enabled for retrieval.")
    vector_top_k: int = Field(config.RETRIEVAL_VECTOR_TOP_K, description="Top_k for vector index.")
    bm25_top_k: int = Field(config.RETRIEVAL_BM25_TOP_K, description="Top_k for bm25 sparse index.")
    rewrite_query: Optional[str] = Field(None, description="Rewrite query for retrieval.")
    rewrite_prompt: Optional[str] = Field(None, description="Prompt used to rewrite query for retrieval.")


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

@app.post("/v1/code-rag/repository/setup", response_model=RepositorySetupResponse)
async def setup_repository_endpoint(request: RepositorySetupRequest, apikey: Optional[str] = Depends(oauth2_scheme)):
    """
    Sets up a repository: clones it (if a URL is provided) and builds its search indexes.
    This operation is executed in the background.
    """
    repo_id = request.repo_id.replace("/", "_").replace(":", "_") # Sanitize
    
    # Generate a task ID
    task_id = f"{repo_id}_{threading.get_ident()}"
    
    # Get or create a lock for this repo_id
    if repo_id not in pipeline_locks:
        pipeline_locks[repo_id] = asyncio.Lock()

    # Check if setup is already in progress
    if repo_id in setup_status and setup_status[repo_id].get("status") == "pending":
        return RepositorySetupResponse(
            repo_id=repo_id,
            message="Repository setup already in progress",
            index_status="In Progress",
            task_id=setup_status[repo_id].get("task_id")
        )

    logger.info(f"Received setup request for repo_id: {repo_id}, source: {request.repo_url_or_path}")
    
    try:
        pipeline = RAGPipeline(repo_id=repo_id) # Initializes with correct index_dir
        logger.info(f"Created new RAGPipeline instance for repo_id: {repo_id}")
    except Exception as e:
        logger.error(f"Failed to initialize RAGPipeline for {repo_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline initialization error: {str(e)}")
    
    # Store initial status
    setup_status[repo_id] = {
        "status": "pending",
        "message": "Repository setup started",
        "index_status": "In Progress",
        "task_id": task_id,
        "repository_path": None
    }
    
    # Define the background task function
    def background_setup():
        try:
            # The actual setup work
            success = pipeline.setup_repository(
                repo_url_or_path=request.repo_url_or_path,
                access_token=request.access_token,
                force_reclone=request.force_reclone,
                force_reindex=request.force_reindex,
                apikey=apikey
            )
            
            if success:
                # Check if indexes actually exist now to determine status
                index_status = "Indexed Successfully"
                if not request.force_reindex and \
                        (pipeline.index_dir / config.FAISS_INDEX_FILENAME).exists() and \
                        (pipeline.index_dir / config.BM25_INDEX_FILENAME).exists():
                    # If not forced and files exist, they were used or verified
                    if not request.force_reindex:
                        index_status = "Indexes Already Existed or Verified"
                
                # Update status to completed
                setup_status[repo_id] = {
                    "status": "completed",
                    "message": "Repository setup process completed",
                    "index_status": index_status,
                    "task_id": task_id,
                    "repository_path": str(pipeline.repository_path) if pipeline.repository_path else None
                }
            else:
                # Update status to failed
                setup_status[repo_id] = {
                    "status": "failed",
                    "message": "Repository setup failed",
                    "index_status": "Failed",
                    "task_id": task_id,
                    "repository_path": str(pipeline.repository_path) if pipeline.repository_path else None
                }
        except Exception as e:
            logger.error(f"Error in background setup for {repo_id}: {e}")
            logger.exception("Background setup exception:")
            # Update status to failed with error message
            setup_status[repo_id] = {
                "status": "failed",
                "message": f"Error: {str(e)}",
                "index_status": "Failed",
                "task_id": task_id,
                "repository_path": str(pipeline.repository_path) if pipeline.repository_path else None
            }
    
    # Start the background task
    thread = threading.Thread(target=background_setup)
    thread.daemon = True  # Daemonize thread to not block shutdown
    thread.start()
    
    # Return immediate response
    return RepositorySetupResponse(
        repo_id=repo_id,
        message="Repository setup started in background",
        index_status="In Progress",
        repository_path=None,
        task_id=task_id
    )


@app.post("/v1/code-rag/query/stream")
async def query_repository_stream(request: QueryRequest, apikey: Optional[str] = Depends(oauth2_scheme)) -> StreamingResponse:

    """
    Queries an already set up repository and streams the LLM's response.
    """
    repo_id = request.repo_id.replace("/", "_").replace(":", "_") # Sanitize
    logger.info(f"Received streaming query for repo_id: '{repo_id}', query: '{request.query_text[:50]}...'")

    try:
        pipeline = RAGPipeline(repo_id=repo_id, indexes=request.indexes)
        if not pipeline.retriever.vector_index and not pipeline.retriever.bm25_index:
            # This means _load_indexes inside HybridRetriever failed to find existing indexes
            raise HTTPException(status_code=404, detail=f"Repository with repo_id '{repo_id}' not found or not indexed. Please set it up first.")
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
        retriever_query = request.query_text
        if request.rewrite_query:
            retriever_query = request.rewrite_query
        elif request.rewrite_prompt:
            retriever_query = await pipeline.retriever.rewrite_query(sys_prompt=request.rewrite_prompt, user_query=request.query_text, apikey=apikey)
        context_chunks_meta: List[Dict[str, Any]] = pipeline.query(
            query_text=retriever_query,
            top_n_final=request.top_n_final,
            vector_top_k=request.vector_top_k,
            bm25_top_k=request.bm25_top_k,
            apikey=apikey
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
            apikey=apikey,
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


@app.get("/v1/code-rag/repository/status/{repo_id}")
async def check_repository_setup_status(repo_id: str):
    """
    Check the status of a repository setup process.
    """
    sanitized_repo_id = repo_id.replace("/", "_").replace(":", "_")  # Sanitize in the same way
    
    if sanitized_repo_id not in setup_status:
        raise HTTPException(status_code=404, detail=f"No setup process found for repository ID: {repo_id}")
    
    status_info = setup_status[sanitized_repo_id]
    
    return RepositoryStatusResponse(
        repo_id=repo_id,
        status=status_info.get("status", "unknown"),
        message=status_info.get("message", ""),
        index_status=status_info.get("index_status"),
        repository_path=status_info.get("repository_path")
    )

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "message": "Code-Aware RAG API is running."}

# --- Main function for local Uvicorn execution (optional, usually use main.py) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API server directly from api.py for debugging...")
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, reload=config.API_RELOAD)