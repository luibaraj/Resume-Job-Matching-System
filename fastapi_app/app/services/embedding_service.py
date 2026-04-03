"""
Service for embedding resume text using Voyage AI.
"""
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to sys.path to import src modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.embedding import create_client, embed_batch
from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for embedding resume text with caching."""

    def __init__(self):
        self.client = create_client(settings.VOYAGE_API_KEY)

    def _resume_hash(self, resume_text: str) -> str:
        """Compute MD5 hash of resume text for cache invalidation."""
        return hashlib.md5(resume_text.encode()).hexdigest()

    def load_or_embed_resume(
        self,
        resume_text: str,
        cache_path: Optional[str] = None,
        hash_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Load cached resume embedding if available and valid, otherwise embed and cache.

        Args:
            resume_text: Resume text to embed.
            cache_path: Path to cached embedding file.
            hash_path: Path to cached hash file.

        Returns:
            Embedding vector (numpy array, float32).
        """
        if cache_path is None:
            cache_path = settings.EMBEDDING_CACHE_PATH
        if hash_path is None:
            hash_path = settings.HASH_CACHE_PATH

        current_hash = self._resume_hash(resume_text)
        cache = Path(cache_path)
        hash_file = Path(hash_path)

        # Cache hit: both files exist and hash matches
        if cache.exists() and hash_file.exists():
            saved_hash = hash_file.read_text().strip()
            if saved_hash == current_hash:
                logger.info("Loading cached resume embedding...")
                try:
                    return np.load(str(cache))
                except Exception as e:
                    logger.warning(
                        "Failed to load cached embedding: %s. Re-embedding.",
                        e,
                    )
                    # Fall through to re-embed

        # Cache miss or load failure: embed, save, and cache the hash
        logger.info("Embedding resume...")
        embeddings = embed_batch(self.client, [resume_text])
        embedding = embeddings[0]

        # Ensure parent directory exists
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache), embedding)
        hash_file.write_text(current_hash)
        logger.info("Resume embedding cached to %s", cache_path)

        return embedding
