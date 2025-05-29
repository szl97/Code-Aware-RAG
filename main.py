import uvicorn
from loguru import logger

# Import the FastAPI app instance from src.api
# Also import config to ensure it's loaded and for server settings
try:
    from src import config # This will also initialize logging via config.py
    from src.api import app
except ImportError as e:
    # This fallback is for cases where main.py might be run in a way that src isn't immediately on PYTHONPATH
    # e.g. directly `python main.py` from root without `PYTHONPATH=.` or `-m`
    import sys
    from pathlib import Path
    # Add project root to sys.path if src is not found
    project_root = Path(__file__).parent.resolve()
    if str(project_root / "src") not in sys.path and str(project_root) not in sys.path :
        sys.path.insert(0, str(project_root))
        logger.info(f"Added {project_root} to sys.path")

    try:
        from src import config
        from src.api import app
    except ImportError as e_inner:
        logger.error(f"Failed to import 'app' from 'src.api' or 'config' from 'src': {e_inner}")
        logger.error("Ensure your PYTHONPATH is set up correctly or run the app as a module, e.g., 'python -m main'.")
        sys.exit(1)


if __name__ == "__main__":
    logger.info(f"Starting Code-Aware RAG API server via main.py...")
    logger.info(f"Host: {config.API_HOST}")
    logger.info(f"Port: {config.API_PORT}")
    logger.info(f"Reload (Development): {config.API_RELOAD}")

    uvicorn.run(
        "src.api:app",  # Path to the app instance (module:variable)
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD,
        log_level=config.LOG_LEVEL.lower(), # Uvicorn log level
        # workers=config.API_WORKERS # For production, you might configure workers
    )

    # Alternatively, if you want to run the app object directly (less common for uvicorn.run from script):
    # uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, reload=config.API_RELOAD)
    # The string "src.api:app" is generally preferred as it allows Uvicorn's reloader to work more effectively.
