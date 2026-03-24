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
RETRIEVE_TOP_K: int = 100
HNSW_EF_CONSTRUCTION: int = 400  # HNSW index build quality
HNSW_EF: int = 400                # HNSW query recall

# Reranking
COHERE_RERANK_MODEL: str = "rerank-english-v3.0"
RERANK_TOP_N: int = 10

# Metadata extraction
DEGREE_UNKNOWN: int = 0
DEGREE_BACHELOR: int = 1
DEGREE_MASTER: int = 2
DEGREE_PHD: int = 3

SENIORITY_UNKNOWN: int = 0
SENIORITY_ENTRY: int = 1
SENIORITY_MID: int = 2
SENIORITY_SENIOR: int = 3

YEARS_UNKNOWN: int = -1

# Generation
OLLAMA_MODEL: str = "llama3.2:3b-instruct-q4_K_M"
GENERATION_TEMPERATURE: float = 0.7
GENERATION_TOP_P: float = 0.9
GENERATION_MAX_TOKENS: int = 150
SKELETON_MAX_TOKENS: int = 200
MAX_BATCH_SIZE: int = 2
CORPUS_LIMITATION_MESSAGE: str = (
    "No strongly matching jobs found in current corpus. "
    "This indicates corpus limitations, not poor fit. "
    "Recommend expanding job database or checking back later."
)