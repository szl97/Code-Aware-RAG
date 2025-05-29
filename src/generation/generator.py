from typing import List, Dict, Any, Optional, AsyncIterator
from loguru import logger
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path

from pydantic import json

# Import project modules
from src import config
from src.data_processing.chunkers import DocumentChunk

# --- LLM Client Abstraction (Simplified) ---
# Similar to embedding, a more formal factory or separate classes might be used in a larger system.
_openai_llm_client = None
_google_llm_model = None # For Google's GenerativeModel

def get_openai_llm_client():
    """Initializes and returns an OpenAI client for LLM generation."""
    global _openai_llm_client
    if _openai_llm_client is None:
        if not config.OPENAI_API_KEY:
            logger.error("OpenAI API key not configured. Cannot use OpenAI for generation.")
            raise ValueError("OpenAI API key not set.")
        try:
            from openai import AsyncOpenAI # Use AsyncOpenAI for FastAPI integration
            if not config.OPENAI_BASE_URL:
                _openai_llm_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            else:
                _openai_llm_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
            logger.info("AsyncOpenAI client for LLM generation initialized.")
        except ImportError:
            logger.error("OpenAI Python package not installed. `pip install openai`")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize AsyncOpenAI client: {e}")
            raise
    return _openai_llm_client

def get_google_llm_model():
    """Initializes and returns a Google GenerativeModel instance."""
    global _google_llm_model
    if _google_llm_model is None:
        if not config.GOOGLE_API_KEY:
            logger.error("Google API key not configured. Cannot use Google for generation.")
            raise ValueError("Google API key not set.")
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.GOOGLE_API_KEY)
            _google_llm_model = genai.GenerativeModel(
                config.GENERATOR_MODEL_NAME,
                generation_config={ # Ensure generation_config is correctly structured
                    "temperature": config.GENERATOR_TEMPERATURE,
                    "max_output_tokens": config.GENERATOR_MAX_TOKENS,
                    # Add top_p, top_k if supported and configured
                }
            )
            logger.info(f"Google GenerativeModel '{config.GENERATOR_MODEL_NAME}' initialized.")
        except ImportError:
            logger.error("Google Generative AI package not installed. `pip install google-generativeai`")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Google GenerativeModel: {e}")
            raise
    return _google_llm_model

# --- Prompt Templating ---
# Assume prompt templates are stored in a 'templates' directory within 'src'
TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

# Create a default prompt template file if it doesn't exist
DEFAULT_PROMPT_TEMPLATE_FILE = TEMPLATE_DIR / "rag_prompt_template.jinja2"
if not DEFAULT_PROMPT_TEMPLATE_FILE.exists():
    DEFAULT_PROMPT_TEMPLATE_FILE.write_text(
        """You are an expert AI assistant specializing in analyzing and explaining code repositories.
        Your responses should be accurate, concise, and directly answer the user's query based on the provided context.
        Format your answers using Markdown. Cite specific file paths or code constructs when relevant.
        
        ## Context from the Repository:
        {% if context_chunks %}
        {% for chunk in context_chunks %}
        ---
        File: {{ chunk.original_file_path }}
        {% if chunk.code_construct_type and chunk.code_construct_name %}
        Type: {{ chunk.code_construct_type }}
        Name: {{ chunk.code_construct_name }}
        {% if chunk.start_line and chunk.end_line %}
        Lines: {{ chunk.start_line }}-{{ chunk.end_line }}
        {% endif %}
        {% endif %}
        
        Content:
        ```{{ chunk.language if chunk.language else '' }}
        {{ chunk.content }}
        ```
        ---
        {% endfor %}
        {% else %}
        No specific context was retrieved from the repository for this query. Try to answer based on general knowledge if applicable, or state that the information is not found in the repository.
        {% endif %}
        
        ## User Query:
        {{ user_query }}
        
        ## Your Answer (in Markdown):
        """
    )
    logger.info(f"Created default prompt template: {DEFAULT_PROMPT_TEMPLATE_FILE}")


jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(['html', 'xml', 'jinja2']), # Jinja2 is for the template itself
    trim_blocks=True,
    lstrip_blocks=True
)


class LLMGenerator:
    """
    Handles prompt construction and interaction with the configured LLM.
    """
    def __init__(
            self,
            provider: str = config.GENERATOR_MODEL_PROVIDER,
            model_name: str = config.GENERATOR_MODEL_NAME,
            temperature: float = config.GENERATOR_TEMPERATURE,
            max_tokens: int = config.GENERATOR_MAX_TOKENS,
            prompt_template_name: str = "rag_prompt_template.jinja2"
    ):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            self.prompt_template = jinja_env.get_template(prompt_template_name)
            logger.info(f"LLMGenerator initialized with provider: {provider}, model: {model_name}, template: {prompt_template_name}")
        except Exception as e:
            logger.error(f"Failed to load prompt template '{prompt_template_name}': {e}")
            raise

    def _construct_prompt(self, user_query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Constructs the full prompt using the Jinja2 template.
        context_chunks are metadata dicts from the retriever.
        """
        try:
            # Ensure context_chunks are serializable for the template if they are Pydantic models
            # The retriever returns List[Dict[str, Any]], which should be fine.
            # If DocumentChunk objects were passed, convert them:
            # serializable_chunks = [chunk.model_dump() for chunk in context_chunks]

            prompt = self.prompt_template.render(
                user_query=user_query,
                context_chunks=context_chunks # Already dicts from retriever
            )
            logger.debug(f"Constructed prompt:\n{prompt[:500]}...") # Log a preview
            return prompt
        except Exception as e:
            logger.error(f"Error rendering prompt template: {e}")
            # Fallback prompt if template rendering fails
            return f"Context: {json.dumps(context_chunks, indent=2)}\n\nQuery: {user_query}\n\nAnswer:"


    async def generate_response_stream(
            self,
            user_query: str,
            context_chunks: List[Dict[str, Any]] # Metadata dicts from retriever
    ) -> AsyncIterator[str]:
        """
        Generates a response from the LLM in a streaming fashion.

        Args:
            user_query (str): The user's original query.
            context_chunks (List[Dict[str, Any]]): List of metadata dictionaries for context chunks.

        Yields:
            AsyncIterator[str]: An async iterator of response text chunks.
        """
        full_prompt = self._construct_prompt(user_query, context_chunks)

        logger.info(f"Generating streaming response from {self.provider}/{self.model_name}...")

        if self.provider == "openai":
            client = get_openai_llm_client()
            try:
                stream = await client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        # OpenAI prefers a system message for overall instructions,
                        # and user message for the combined context+query.
                        # Our template currently puts everything into one block.
                        # For better OpenAI results, might refine template for system/user roles.
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                )
                async for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        yield content
                logger.info("OpenAI streaming response finished.")
            except Exception as e:
                logger.error(f"Error during OpenAI streaming: {e}")
                yield f"\nAn error occurred with the OpenAI API: {str(e)}"

        elif self.provider == "google":
            model = get_google_llm_model()
            try:
                # Google's API might have a slightly different way to specify stream=True
                # and handle chat history if needed.
                # For a single turn from a complex prompt, generate_content is fine.
                # Ensure the model is configured for streaming if it's a separate setting.
                # The `stream=True` parameter is on the `generate_content` call.
                responses = await model.generate_content_async(full_prompt, stream=True) # Pass stream=True here
                async for chunk in responses: # Iterate over the async generator
                    if hasattr(chunk, 'text') and chunk.text:
                        yield chunk.text
                    # elif hasattr(chunk, 'parts'): # Check for parts if text is not directly available
                    #     for part in chunk.parts:
                    #         if hasattr(part, 'text') and part.text:
                    #             yield part.text
                logger.info("Google streaming response finished.")
            except Exception as e:
                logger.error(f"Error during Google streaming: {e}")
                yield f"\nAn error occurred with the Google API: {str(e)}"
        else:
            logger.error(f"Unsupported LLM provider for streaming: {self.provider}")
            yield f"\nError: LLM provider '{self.provider}' is not supported for streaming generation."


    async def generate_response_non_streaming(
            self,
            user_query: str,
            context_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generates a complete response from the LLM (non-streaming).
        """
        full_prompt = self._construct_prompt(user_query, context_chunks)
        logger.info(f"Generating non-streaming response from {self.provider}/{self.model_name}...")

        full_response_text = ""

        if self.provider == "openai":
            client = get_openai_llm_client()
            try:
                completion = await client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False # Ensure stream is False
                )
                if completion.choices and completion.choices[0].message:
                    full_response_text = completion.choices[0].message.content or ""
                logger.info("OpenAI non-streaming response received.")
            except Exception as e:
                logger.error(f"Error during OpenAI non-streaming generation: {e}")
                full_response_text = f"An error occurred with the OpenAI API: {str(e)}"

        elif self.provider == "google":
            model = get_google_llm_model()
            try:
                response = await model.generate_content_async(full_prompt, stream=False) # Ensure stream is False
                full_response_text = response.text if hasattr(response, 'text') else ""
                if not full_response_text and hasattr(response, 'parts'): # Fallback for complex responses
                    full_response_text = "".join([part.text for part in response.parts if hasattr(part, 'text')])

                logger.info("Google non-streaming response received.")
            except Exception as e:
                logger.error(f"Error during Google non-streaming generation: {e}")
                full_response_text = f"An error occurred with the Google API: {str(e)}"
        else:
            logger.error(f"Unsupported LLM provider: {self.provider}")
            full_response_text = f"Error: LLM provider '{self.provider}' is not supported."

        return full_response_text.strip()


# --- Example Usage ---
if __name__ == "__main__":
    import asyncio

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")

    # Ensure API keys are set in your .env file or environment for this test
    # For example, OPENAI_API_KEY or GOOGLE_API_KEY

    # Create dummy context (metadata dicts, as returned by retriever)
    dummy_context = [
        {
            "original_file_path": "src/utils.py",
            "chunk_id": 1,
            "language": "python",
            "code_construct_type": "function_definition",
            "code_construct_name": "helper_function",
            "start_line": 10,
            "end_line": 15,
            "content": "def helper_function(param1, param2):\n    # Does something useful\n    return param1 + param2"
        },
        {
            "original_file_path": "README.md",
            "chunk_id": 1,
            "language": "markdown",
            "content": "This project aims to provide a RAG system for code. It uses advanced techniques."
        }
    ]
    test_user_query = "How does the helper_function work and what is the project about?"

    async def run_tests():
        # Test with configured provider (ensure your .env or config.yaml points to a working one)
        logger.info(f"\n--- Testing LLMGenerator with configured provider: {config.GENERATOR_MODEL_PROVIDER} ---")
        generator = LLMGenerator()

        logger.info("\n--- Testing Non-Streaming Response ---")
        response = await generator.generate_response_non_streaming(test_user_query, dummy_context)
        logger.info(f"Non-streaming response:\n{response}")

        logger.info("\n--- Testing Streaming Response ---")
        full_streamed_response = ""
        async for chunk_text in generator.generate_response_stream(test_user_query, dummy_context):
            print(chunk_text, end="", flush=True) # Print stream directly
            full_streamed_response += chunk_text
        print("\n--- End of Stream ---")
        # logger.info(f"Full streamed response collected:\n{full_streamed_response}")

        # Add assertions if you have expected outputs or specific behaviors to check
        assert len(full_streamed_response) > 0, "Streamed response should not be empty"

    # Run the async tests
    try:
        asyncio.run(run_tests())
    except ValueError as ve: # Catch API key errors from client init
        logger.error(f"Test failed due to configuration error: {ve}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during tests: {e}")