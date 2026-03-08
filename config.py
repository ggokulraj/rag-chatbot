import os
from dotenv import load_dotenv

# Load variables from .env (if present)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Allow tests to run without API key when using mock models
if not OPENAI_API_KEY and not os.getenv("PYTEST_CURRENT_TEST"):
    # Only require API key in production/normal operation
    pass

OPENAI_LLM_MODEL = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
CHROMA_PATH = "./chroma_db"
DATA_PATH = "./data"
COLLECTION_NAME = "rag_docs"
