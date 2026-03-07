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
    return config
