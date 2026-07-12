"""Central config. Reads from environment variables (and a .env file in dev).

Keeping all tunable knobs here means an interviewer can see, in one place,
every decision you made: which models, chunk size, how many chunks to retrieve.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Database ---
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://docqa:docqa@localhost:5432/docqa",
)

# --- Provider (OpenAI-compatible) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # 1536 dims
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")

# --- Web search (Tavily) ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- Pipeline knobs (these are your interview talking points) ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))        # chars per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # chars shared between neighbours
TOP_K = int(os.getenv("TOP_K", "4"))                    # chunks fed to the LLM
EMBEDDING_DIM = 1536
