"""
Embedding module: Voyage AI client creation and text embedding.

All functions are stateless and side-effect-free (no DB, no env loading).
"""

import logging
import time

import numpy as np
import voyageai

from config import (
    EMBEDDING_DIM,
    EMBED_MAX_RETRIES,
    EMBED_RETRY_BASE_DELAY,
    VOYAGE_MODEL,
)

logger = logging.getLogger(__name__)


def create_client(api_key: str) -> voyageai.Client:
    """
    Create and return a Voyage AI client.

    Args:
        api_key: The Voyage AI API key (VOYAGE_API_KEY).

    Returns:
        An authenticated voyageai.Client instance.

    Raises:
        ValueError: If api_key is empty or None.
    """
    if not api_key:
        raise ValueError("VOYAGE_API_KEY must be set and non-empty")
    return voyageai.Client(api_key=api_key)


def embed_batch(
    client: voyageai.Client,
    texts: list[str],
    model: str = VOYAGE_MODEL,
    max_retries: int = EMBED_MAX_RETRIES,
    retry_base_delay: float = EMBED_RETRY_BASE_DELAY,
) -> list[np.ndarray]:
    """
    Embed a batch of texts using Voyage AI, returning one numpy array per text.

    Retries on transient errors (rate limits, network errors) with exponential
    back-off. Raises on permanent failure after all retries are exhausted.

    Args:
        client: An authenticated voyageai.Client.
        texts: List of strings to embed. Must be non-empty, length <= 128.
        model: Voyage AI model name.
        max_retries: Maximum number of retry attempts on transient errors.
        retry_base_delay: Base delay in seconds for exponential back-off.

    Returns:
        List of numpy arrays (float32), one per input text, in the same order.

    Raises:
        ValueError: If texts is empty.
        Exception: If all retries fail.
    """
    if not texts:
        raise ValueError("texts must be non-empty")

    attempt = 0
    while True:
        try:
            result = client.embed(texts, model=model, input_type=None)
            # result.embeddings is a list of list[float]
            return [np.array(vec, dtype=np.float32) for vec in result.embeddings]
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "embed_batch failed after %d attempts (batch_size=%d, model=%s): %s (%s)",
                    max_retries,
                    len(texts),
                    model,
                    type(exc).__name__,
                    exc,
                )
                raise
            delay = retry_base_delay * (2 ** (attempt - 1))
            logger.warning(
                "embed_batch attempt %d/%d failed (batch_size=%d, %s): %s. Retrying in %.1fs...",
                attempt,
                max_retries,
                len(texts),
                type(exc).__name__,
                exc,
                delay,
            )
            time.sleep(delay)


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """
    Serialize a numpy float32 embedding to raw bytes for SQLite BLOB storage.

    Args:
        embedding: A 1-D numpy array of float32 values.

    Returns:
        Raw bytes (little-endian float32 values).
    """
    return embedding.astype(np.float32).tobytes()


def deserialize_embedding(blob: bytes, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """
    Deserialize a SQLite BLOB back into a numpy float32 array.

    Args:
        blob: Raw bytes from the embedding column.
        dim: Expected embedding dimension (default 1024 for voyage-3.5-lite).

    Returns:
        A 1-D numpy array of shape (dim,) and dtype float32.

    Raises:
        ValueError: If the blob length does not match the expected dimension.
    """
    expected_bytes = dim * 4  # float32 = 4 bytes each
    if len(blob) != expected_bytes:
        raise ValueError(
            f"Expected {expected_bytes} bytes for dim={dim}, got {len(blob)}"
        )
    return np.frombuffer(blob, dtype=np.float32).copy()
