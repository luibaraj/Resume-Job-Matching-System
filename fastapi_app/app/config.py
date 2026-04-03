"""
Configuration settings for the FastAPI application.
"""
import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings

# Project root (three levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API keys (optional with defaults for testing)
    VOYAGE_API_KEY: str = ""
    COHERE_API_KEY: str = ""
    # GREENHOUSE_BOARD_TOKENS not needed for matching API

    # Database
    DB_PATH: str = str(_PROJECT_ROOT / "data" / "jobs.db")
    CHROMA_DIR: str = str(_PROJECT_ROOT / "data" / "chroma_db")

    # Embedding model
    VOYAGE_MODEL: str = "voyage-3.5-lite"
    VOYAGE_BATCH_SIZE: int = 128

    # Retrieval
    CHROMA_COLLECTION_NAME: str = "job_embeddings"
    RETRIEVE_TOP_K: int = 100
    HNSW_EF: int = 200
    HNSW_EF_CONSTRUCTION: int = 200

    # Reranking
    RERANK_TOP_N: int = 10
    COHERE_RERANK_MODEL: str = "rerank-english-v3.0"

    # Generation
    OLLAMA_MODEL: str = "llama3.2:3b-instruct-q4_K_M"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Caching
    EMBEDDING_CACHE_PATH: str = str(_PROJECT_ROOT / "data" / "user_profile_embedding.npy")
    HASH_CACHE_PATH: str = str(_PROJECT_ROOT / "data" / "user_profile_embedding_hash.txt")

    # FastAPI
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    class Config:
        env_file = _PROJECT_ROOT / ".env"
        env_file_encoding = "utf-8"


settings = Settings()
