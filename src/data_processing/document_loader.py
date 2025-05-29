import os
from pathlib import Path
from typing import List, Optional, Iterator
from urllib.parse import urlparse, urlunparse

import chardet  # For robust encoding detection
import git  # Using GitPython library
from loguru import logger
from pydantic import BaseModel, Field

# Import configurations from src.config
from src import config


# --- Pydantic Model for Loaded Documents ---
class LoadedDocument(BaseModel):
    """Represents a single document loaded from the repository."""
    file_path: Path = Field(description="Relative path of the file within the repository.")
    absolute_path: Path = Field(description="Absolute path of the file on the system.")
    content: str = Field(description="Text content of the file.")
    language: Optional[str] = Field(None, description="Detected programming language or file type.")
    size_bytes: int = Field(description="Size of the file in bytes.")

    # Add a unique ID for each document, can be hash of content or path
    # For simplicity, we'll use relative path as a key part of an ID later if needed.

    class Config:
        arbitrary_types_allowed = True


# --- Repository Cloning ---
def clone_repository(repo_url: str, local_path: Path, access_token: Optional[str] = None) -> bool:
    """
    Clones a Git repository to a specified local path.
    Uses GitPython for more robust interaction.

    Args:
        repo_url (str): The URL of the Git repository.
        local_path (Path): The local directory where the repository will be cloned.
        access_token (str, optional): Access token for private repositories.
                                      (Note: GitPython handles auth via credential helpers or SSH keys primarily.
                                       For token auth over HTTPS, URL modification is needed as shown,
                                       but be cautious with token exposure.)
    Returns:
        bool: True if cloning was successful or repo already exists, False otherwise.
    """
    if local_path.exists() and any(local_path.iterdir()):
        logger.info(f"Repository already exists at {local_path}. Skipping clone.")
        try:
            # Optionally, you could add logic here to pull latest changes
            # repo = git.Repo(local_path)
            # origin = repo.remotes.origin
            # origin.pull()
            # logger.info(f"Pulled latest changes for repository at {local_path}")
            pass
        except git.InvalidGitRepositoryError:
            logger.warning(f"{local_path} exists but is not a valid Git repository. Will attempt to clone.")
            # Potentially remove the directory if it's not a git repo and should be
        except Exception as e:
            logger.warning(f"Could not pull latest changes for {local_path}: {e}")
        return True

    logger.info(f"Cloning repository from {repo_url} to {local_path}...")
    try:
        # Modify URL for token authentication if token is provided
        # This is a common pattern, but ensure your Git version and server support it.
        # SSH keys are generally preferred for private repos.
        clone_url_with_token = repo_url
        if access_token:
            parsed_url = urlparse(repo_url)
            # Example for GitHub, GitLab might need 'oauth2:<token>'
            # This part might need adjustment based on the Git hosting provider
            if "github.com" in parsed_url.netloc:
                clone_url_with_token = urlunparse(
                    (parsed_url.scheme, f"{access_token}@{parsed_url.netloc}", parsed_url.path, '', '', '')
                )
            elif "gitlab.com" in parsed_url.netloc:
                clone_url_with_token = urlunparse(
                    (parsed_url.scheme, f"oauth2:{access_token}@{parsed_url.netloc}", parsed_url.path, '', '', '')
                )
            # Add other providers as needed

        git.Repo.clone_from(clone_url_with_token, local_path, depth=1) # Shallow clone for speed
        logger.info(f"Repository cloned successfully to {local_path}.")
        return True
    except git.GitCommandError as e:
        logger.error(f"Git command error during cloning {repo_url} to {local_path}: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Failed to clone repository {repo_url} to {local_path}: {e}")
        return False


# --- File Discovery and Loading ---
def get_language_from_extension(file_path: Path) -> Optional[str]:
    """
    Determines the language based on file extension from the config.
    """
    ext = file_path.suffix.lower()
    for lang, lang_details in config.TREE_SITTER_LANGUAGES.items():
        if ext in lang_details.get("extensions", []):
            return lang
    # Fallback for common non-tree-sitter handled types or general text
    if ext == ".md": return "markdown"
    if ext == ".txt": return "text"
    if ext in [".json", ".yaml", ".yml"]: return ext[1:]
    return None


def is_file_excluded(
        relative_file_path: Path,
        excluded_dirs: List[str],
        excluded_files: List[str]
) -> bool:
    """
    Checks if a file should be excluded based on configured patterns.
    """
    # Check against excluded directories
    for excluded_dir_pattern in excluded_dirs:
        # Simple check: if any part of the path matches the excluded dir pattern
        # More sophisticated matching (e.g. glob) can be added if needed
        if excluded_dir_pattern in relative_file_path.parts:
            return True
        try: # Try glob matching for patterns like '*/test/*'
            if relative_file_path.match(f"*{os.sep}{excluded_dir_pattern}{os.sep}*") or \
                    relative_file_path.match(f"{excluded_dir_pattern}{os.sep}*"):
                return True
        except Exception: # Path.match can fail on complex patterns not meant for it
            pass


    # Check against excluded files (name or pattern)
    file_name = relative_file_path.name
    for excluded_file_pattern in excluded_files:
        if file_name == excluded_file_pattern: # Exact match
            return True
        if Path(file_name).match(excluded_file_pattern): # Glob pattern match
            return True
        if excluded_file_pattern.startswith(".") and file_name.startswith(excluded_file_pattern): # Hidden files like .env
            return True

    return False


def read_file_content(file_path: Path) -> Optional[str]:
    """
    Reads file content with robust encoding detection.
    """
    try:
        with open(file_path, 'rb') as f_byte:
            raw_bytes = f_byte.read()

        # Detect encoding
        detected_encoding = chardet.detect(raw_bytes)['encoding']
        if detected_encoding:
            try:
                return raw_bytes.decode(detected_encoding)
            except UnicodeDecodeError:
                logger.warning(f"Could not decode {file_path} with detected encoding {detected_encoding}. Trying utf-8.")
                # Fallback to utf-8, then to ignore errors
                try:
                    return raw_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    logger.error(f"Failed to decode {file_path} with utf-8. Decoding with error replacement.")
                    return raw_bytes.decode('utf-8', errors='replace')
        else:
            # If chardet fails, try common encodings
            try:
                return raw_bytes.decode('utf-8')
            except UnicodeDecodeError:
                logger.error(f"chardet failed to detect encoding for {file_path} and utf-8 failed. Decoding with error replacement.")
                return raw_bytes.decode('utf-8', errors='replace') # Last resort
    except IOError as e:
        logger.error(f"IOError reading file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error reading file {file_path}: {e}")
        return None


def load_documents_from_repo(
        repo_path: Path,
        excluded_dirs: Optional[List[str]] = None,
        excluded_files: Optional[List[str]] = None,
        max_file_size_mb: Optional[float] = None
) -> Iterator[LoadedDocument]:
    """
    Walks through a local repository path, filters, and loads documents.

    Args:
        repo_path (Path): The absolute path to the local repository.
        excluded_dirs (Optional[List[str]]): List of directory patterns to exclude.
                                             Defaults to config.DEFAULT_EXCLUDED_DIRS.
        excluded_files (Optional[List[str]]): List of file patterns to exclude.
                                              Defaults to config.DEFAULT_EXCLUDED_FILES.
        max_file_size_mb (Optional[float]): Maximum file size in MB to process.
                                            Defaults to config.MAX_FILE_SIZE_MB.

    Yields:
        Iterator[LoadedDocument]: An iterator of LoadedDocument objects.
    """
    _excluded_dirs = excluded_dirs if excluded_dirs is not None else config.DEFAULT_EXCLUDED_DIRS
    _excluded_files = excluded_files if excluded_files is not None else config.DEFAULT_EXCLUDED_FILES
    _max_size_bytes = (max_file_size_mb if max_file_size_mb is not None else config.MAX_FILE_SIZE_MB) * 1024 * 1024

    logger.info(f"Loading documents from repository: {repo_path}")
    logger.debug(f"Excluded directories: {_excluded_dirs}")
    logger.debug(f"Excluded files: {_excluded_files}")
    logger.debug(f"Max file size (bytes): {_max_size_bytes}")

    processed_files_count = 0
    skipped_files_count = 0

    for root, dirs, files in os.walk(repo_path, topdown=True):
        # Modify dirs in-place to prevent walking into excluded directories
        # This is more efficient than checking each file path later
        current_rel_dir_parts = Path(root).relative_to(repo_path).parts

        dirs[:] = [
            d for d in dirs
            if not any(ex_dir in (Path(root) / d).relative_to(repo_path).parts for ex_dir in _excluded_dirs) and
               not any(Path(d).match(ex_dir_pattern) for ex_dir_pattern in _excluded_dirs if '*' in ex_dir_pattern) # Handle glob patterns for dirs
        ]

        for filename in files:
            absolute_file_path = Path(root) / filename
            relative_file_path = absolute_file_path.relative_to(repo_path)

            # File exclusion check
            if is_file_excluded(relative_file_path, _excluded_dirs, _excluded_files):
                logger.trace(f"Skipping (excluded): {relative_file_path}")
                skipped_files_count += 1
                continue

            # File size check
            try:
                file_size_bytes = absolute_file_path.stat().st_size
                if file_size_bytes == 0:
                    logger.trace(f"Skipping (empty file): {relative_file_path}")
                    skipped_files_count += 1
                    continue
                if file_size_bytes > _max_size_bytes:
                    logger.trace(f"Skipping (too large: {file_size_bytes / (1024*1024):.2f}MB): {relative_file_path}")
                    skipped_files_count += 1
                    continue
            except FileNotFoundError:
                logger.warning(f"File not found during size check (possibly a broken symlink): {absolute_file_path}")
                skipped_files_count += 1
                continue

            # Try to read content
            content = read_file_content(absolute_file_path)
            if content is None: # Skip if read failed
                skipped_files_count += 1
                continue

            language = get_language_from_extension(absolute_file_path)

            # Yield the document if it has content and a detected language (or is a text type)
            # We might want to process files even if language is None if they are plain text.
            # For now, let's assume we only care about files with recognized extensions for AST or text.
            if language or (not language and absolute_file_path.suffix.lower() in ['.txt', '.md']): # Process .txt and .md even if no specific lang
                if not language and absolute_file_path.suffix.lower() in ['.txt', '.md']:
                    language = absolute_file_path.suffix.lower()[1:]

                processed_files_count += 1
                yield LoadedDocument(
                    file_path=relative_file_path,
                    absolute_path=absolute_file_path,
                    content=content,
                    language=language,
                    size_bytes=file_size_bytes
                )
                if processed_files_count % 100 == 0:
                    logger.info(f"Processed {processed_files_count} files so far...")
            else:
                logger.trace(f"Skipping (unsupported extension or no content): {relative_file_path}")
                skipped_files_count += 1

    logger.info(f"Finished loading documents. Processed: {processed_files_count}, Skipped: {skipped_files_count}")


# --- Main function for testing this module ---
if __name__ == "__main__":
    logger.remove() # Remove default handler
    logger.add(lambda msg: print(msg, end=""), level="INFO") # Print to stdout for testing

    # Example Usage:
    # 1. Define a test repository URL and local path
    #    Make sure to use a small, public repository for quick testing.
    #    Or a local path to a small code project.
    # TEST_REPO_URL = "https://github.com/loguru/loguru.git" # Example public repo
    TEST_LOCAL_REPO_PATH_STR = "./test_repo_project" # Example local path
    TEST_LOCAL_REPO_PATH = Path(TEST_LOCAL_REPO_PATH_STR)

    # Create a dummy local repo for testing if it doesn't exist
    if not TEST_LOCAL_REPO_PATH.exists():
        TEST_LOCAL_REPO_PATH.mkdir(parents=True, exist_ok=True)
        (TEST_LOCAL_REPO_PATH / "main.py").write_text("def hello():\n    print('Hello World')\n\nclass MyClass:\n    pass")
        (TEST_LOCAL_REPO_PATH / "utils.js").write_text("function greet() { console.log('Greetings!'); }")
        (TEST_LOCAL_REPO_PATH / "README.md").write_text("# Test Project")
        (TEST_LOCAL_REPO_PATH / ".env").write_text("SECRET_KEY=123")
        (TEST_LOCAL_REPO_PATH / "data.json").write_text('{"key": "value"}')
        (TEST_LOCAL_REPO_PATH / "src").mkdir(exist_ok=True)
        (TEST_LOCAL_REPO_PATH / "src" / "app.py").write_text("print('app')")
        (TEST_LOCAL_REPO_PATH / "node_modules" / "some_lib.js").mkdir(parents=True, exist_ok=True)
        (TEST_LOCAL_REPO_PATH / "node_modules" / "some_lib.js").write_text("// some lib")


    # 2. Test cloning (if using a URL)
    # cloned_successfully = clone_repository(TEST_REPO_URL, config.REPOS_DIR / "loguru_test")
    # if cloned_successfully:
    #     repo_to_load = config.REPOS_DIR / "loguru_test"
    # else:
    #     logger.error("Cloning failed, cannot proceed with loading from remote.")
    #     repo_to_load = None

    # For local testing:
    repo_to_load = TEST_LOCAL_REPO_PATH

    if repo_to_load and repo_to_load.exists():
        logger.info(f"\n--- Loading documents from: {repo_to_load} ---")

        # Override exclusions for testing if needed
        custom_excluded_dirs = config.DEFAULT_EXCLUDED_DIRS # Or provide a custom list
        custom_excluded_files = config.DEFAULT_EXCLUDED_FILES # Or provide a custom list

        # Example: exclude all JS files for this run
        # custom_excluded_files.append("*.js")

        doc_iterator = load_documents_from_repo(
            repo_path=repo_to_load,
            excluded_dirs=custom_excluded_dirs,
            excluded_files=custom_excluded_files,
            max_file_size_mb=config.MAX_FILE_SIZE_MB
        )

        loaded_docs_list: List[LoadedDocument] = []
        for i, doc in enumerate(doc_iterator):
            loaded_docs_list.append(doc)
            logger.info(f"Loaded: {doc.file_path} (Lang: {doc.language}, Size: {doc.size_bytes}B)")
            # logger.info(f"   Content preview: {doc.content[:100].replace(os.linesep, ' ')}...")
            if i >= 10: # Print details for first few docs
                logger.info("... (omitting further individual printouts for brevity)")
                # break

        logger.info(f"\nTotal documents loaded: {len(loaded_docs_list)}")

        # Verify exclusion worked (e.g. .env should not be loaded)
        assert not any(d.file_path.name == ".env" for d in loaded_docs_list), ".env file should be excluded"
        assert not any("node_modules" in d.file_path.parts for d in loaded_docs_list), "node_modules should be excluded"
        logger.info("Exclusion assertion passed.")

    else:
        logger.error(f"Test repository path {repo_to_load} does not exist.")

    # Clean up dummy repo
    # import shutil
    # if TEST_LOCAL_REPO_PATH.exists() and TEST_LOCAL_REPO_PATH_STR == "./test_repo_project":
    #     shutil.rmtree(TEST_LOCAL_REPO_PATH)
    #     logger.info(f"Cleaned up dummy test repository: {TEST_LOCAL_REPO_PATH}")
