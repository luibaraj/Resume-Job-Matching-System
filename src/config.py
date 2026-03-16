import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # Database
    db_path: str = field(
        default_factory=lambda: os.environ.get("DB_PATH", "/data/jobs.db")
    )

    # Greenhouse collection
    greenhouse_board_tokens: list[str] = field(
        default_factory=lambda: [
            t.strip()
            for t in os.environ.get("GREENHOUSE_BOARD_TOKENS", "").split(",")
            if t.strip()
        ]
    )
    request_timeout_seconds: int = field(
        default_factory=lambda: int(os.environ.get("REQUEST_TIMEOUT", "30"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.environ.get("MAX_RETRIES", "3"))
    )
    retry_backoff_seconds: float = field(
        default_factory=lambda: float(os.environ.get("RETRY_BACKOFF", "2.0"))
    )
    max_workers: int = field(
        default_factory=lambda: int(os.environ.get("COLLECTION_MAX_WORKERS", "8"))
    )
    request_delay_seconds: float = field(
        default_factory=lambda: float(os.environ.get("COLLECTION_REQUEST_DELAY", "0.1"))
    )

    # Preprocessing
    preprocessing_workers: int = field(
        default_factory=lambda: int(os.environ.get("PREPROCESSING_WORKERS", str(os.cpu_count() or 1)))
    )
    preprocessing_chunk_size: int = field(
        default_factory=lambda: int(os.environ.get("PREPROCESSING_CHUNK_SIZE", "500"))
    )
    preprocessing_max_retries: int = field(
        default_factory=lambda: int(os.environ.get("PREPROCESSING_MAX_RETRIES", "2"))
    )

    # Extraction
    google_api_key: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_API_KEY", "")
    )
    extraction_model_id: str = field(
        default_factory=lambda: os.environ.get("EXTRACTION_MODEL_ID", "gemini-2.5-flash")
    )
    extraction_chunk_size: int = field(
        default_factory=lambda: int(os.environ.get("EXTRACTION_CHUNK_SIZE", "50"))
    )
    extraction_max_retries: int = field(
        default_factory=lambda: int(os.environ.get("EXTRACTION_MAX_RETRIES", "2"))
    )

    # Embedding
    embedding_model_id: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_MODEL_ID", "Qwen/Qwen3-Embedding-0.6B")
    )
    embedding_chunk_size: int = field(
        default_factory=lambda: int(os.environ.get("EMBEDDING_CHUNK_SIZE", "256"))
    )
    embedding_batch_size: int = field(
        default_factory=lambda: int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))
    )
    embedding_max_retries: int = field(
        default_factory=lambda: int(os.environ.get("EMBEDDING_MAX_RETRIES", "2"))
    )
    embedding_num_workers: int = field(
        default_factory=lambda: int(os.environ.get("EMBEDDING_NUM_WORKERS", "4"))
    )

    # Retrieval
    retrieval_top_k: int = field(
        default_factory=lambda: int(os.environ.get("RETRIEVAL_TOP_K", "100"))
    )
    retrieval_user_profile_path: str = field(
        default_factory=lambda: os.environ.get("RETRIEVAL_USER_PROFILE_PATH", "data/user_profile.txt")
    )
    retrieval_rrf_k: int = field(
        default_factory=lambda: int(os.environ.get("RETRIEVAL_RRF_K", "60"))
    )

    # Reranking
    reranking_model_id: str = field(
        default_factory=lambda: os.environ.get("RERANKING_MODEL_ID", "Qwen/Qwen3-Reranker-0.6B")
    )
    reranking_top_k: int = field(
        default_factory=lambda: int(os.environ.get("RERANKING_TOP_K", "20"))
    )
    reranking_batch_size: int = field(
        default_factory=lambda: int(os.environ.get("RERANKING_BATCH_SIZE", "8"))
    )

    # Logging
    log_level: str = field(
        default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO")
    )


def load_config() -> Config:
    """Instantiate and validate Config from environment.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    config = Config()
    if config.preprocessing_workers < 1:
        raise ValueError("PREPROCESSING_WORKERS must be >= 1")
    if config.preprocessing_chunk_size < 1:
        raise ValueError("PREPROCESSING_CHUNK_SIZE must be >= 1")
    if config.preprocessing_max_retries < 0:
        raise ValueError("PREPROCESSING_MAX_RETRIES must be >= 0")
    if not config.google_api_key:
        raise ValueError("GOOGLE_API_KEY must be set")
    if config.extraction_chunk_size < 1:
        raise ValueError("EXTRACTION_CHUNK_SIZE must be >= 1")
    if config.extraction_max_retries < 0:
        raise ValueError("EXTRACTION_MAX_RETRIES must be >= 0")
    if config.embedding_chunk_size < 1:
        raise ValueError("EMBEDDING_CHUNK_SIZE must be >= 1")
    if config.embedding_batch_size < 1:
        raise ValueError("EMBEDDING_BATCH_SIZE must be >= 1")
    if config.embedding_max_retries < 0:
        raise ValueError("EMBEDDING_MAX_RETRIES must be >= 0")
    if config.embedding_num_workers < 1:
        raise ValueError("EMBEDDING_NUM_WORKERS must be >= 1")
    if config.retrieval_top_k < 1:
        raise ValueError("RETRIEVAL_TOP_K must be >= 1")
    if not config.retrieval_user_profile_path:
        raise ValueError("RETRIEVAL_USER_PROFILE_PATH must not be empty")
    if config.retrieval_rrf_k < 1:
        raise ValueError("RETRIEVAL_RRF_K must be >= 1")
    if config.reranking_top_k < 1:
        raise ValueError("RERANKING_TOP_K must be >= 1")
    if config.reranking_batch_size < 1:
        raise ValueError("RERANKING_BATCH_SIZE must be >= 1")
    return config
