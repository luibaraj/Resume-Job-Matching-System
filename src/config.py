"""
Project-wide configuration constants.

Environment variables (VOYAGE_API_KEY, DB_PATH, GREENHOUSE_BOARD_TOKENS)
are NOT read here — scripts load them via load_dotenv() and pass values as
explicit arguments to src/ functions.
"""

# Voyage AI / Embedding
VOYAGE_MODEL: str = "voyage-3.5-lite"
EMBEDDING_DIM: int = 1024           # Output dimension for voyage-3.5-lite
VOYAGE_BATCH_SIZE: int = 128        # Hard API limit per embed() call
EMBED_MAX_RETRIES: int = 3
EMBED_RETRY_BASE_DELAY: float = 2.0 # Base seconds for exponential back-off

# Database
DB_DEFAULT_PATH: str = "data/jobs.db"

# Chroma
CHROMA_COLLECTION_NAME: str = "jobs"
CHROMA_DEFAULT_DIR: str = "data/chroma"

# Retrieval
RETRIEVE_TOP_K: int = 10
HNSW_EF_CONSTRUCTION: int = 400  # HNSW index build quality
HNSW_EF: int = 400                # HNSW query recall