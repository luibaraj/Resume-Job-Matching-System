"""
Cached embedding utilities for evaluation pipeline.

Provides generic embed_with_cache function and thin wrappers for positives and resumes.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
import embedding
from eval import data_loading, eval_config

logger = logging.getLogger(__name__)


def embed_with_cache(
    voyage_client,
    df: pd.DataFrame,
    id_col: str,
    text_col: str,
    cache_path: str,
    hash_path: str,
    model: str = None,
    skip_empty: bool = False,
) -> dict:
    """
    Embed a DataFrame column with .npz/.hash caching.

    Args:
        voyage_client: VoyageAI client
        df: DataFrame containing text and ID columns
        id_col: Name of ID column (str or int)
        text_col: Name of text column to embed
        cache_path: Path to .npz cache file
        hash_path: Path to .hash file (stores hash of column for invalidation)
        model: VoyageAI model name
        skip_empty: If True, skip rows with empty text and log warnings

    Returns:
        dict mapping ID values to embeddings. Caller responsible for int casting if needed.
    """
    if model is None:
        model = config.VOYAGE_MODEL

    cache_p = Path(cache_path)
    hash_p = Path(hash_path)

    # Compute hash of text column
    sorted_texts = df.sort_values(id_col)[text_col].values.astype(str)
    current_hash = data_loading.compute_hash("|".join(sorted_texts).encode("utf-8"))

    # Check cache
    if cache_p.exists() and hash_p.exists():
        try:
            with open(hash_p) as f:
                cached_hash = f.read().strip()
            if cached_hash == current_hash:
                logger.info(f"Loading embeddings from cache: {cache_path}")
                cached = np.load(cache_p)
                result = {k: cached[k] for k in cached.files}
                logger.info(f"Loaded {len(result)} embeddings from cache")
                return result
        except Exception as e:
            logger.warning(f"Cache load failed: {e}; will re-embed")

    # Embed
    logger.info(f"Embedding from column: {text_col}")
    embeddings_dict = {}
    empty_count = 0

    df = df.sort_values(id_col)
    for idx in range(0, len(df), config.VOYAGE_BATCH_SIZE):
        batch = df.iloc[idx : idx + config.VOYAGE_BATCH_SIZE]
        texts = []
        ids = []

        for _, row in batch.iterrows():
            text = row[text_col]
            if not text or not str(text).strip():
                if skip_empty:
                    logger.warning(
                        f"Skipping empty {text_col} for {id_col}={row[id_col]}"
                    )
                    empty_count += 1
                    continue
                else:
                    text = ""

            texts.append(str(text))
            ids.append(row[id_col])

        if texts:
            embeddings = embedding.embed_batch(voyage_client, texts, model=model)
            for id_val, emb in zip(ids, embeddings):
                embeddings_dict[id_val] = emb

        logger.info(f"Embedded {len(embeddings_dict)}/{len(df)} items")

    if empty_count > 0:
        logger.warning(f"Skipped {empty_count} items with empty {text_col}")

    # Cache
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    # Save with string keys (numpy keys must be strings)
    np.savez(cache_p, **{str(k): v for k, v in embeddings_dict.items()})
    with open(hash_p, "w") as f:
        f.write(current_hash)

    logger.info(f"Cached {len(embeddings_dict)} embeddings to {cache_path}")

    return embeddings_dict


def embed_positives(
    voyage_client,
    positives_df: pd.DataFrame,
    model: str = None,
    cache_path: str = None,
    hash_path: str = None,
) -> dict[str, np.ndarray]:
    """
    Embed synthetic positives, using cache if job_description hash matches.

    Returns:
        {positive_uuid: embedding_float32}
    """
    if cache_path is None:
        cache_path = eval_config.TUNE_POSITIVE_EMBEDDINGS_CACHE
    if hash_path is None:
        hash_path = eval_config.TUNE_POSITIVE_EMBEDDINGS_HASH

    embeddings = embed_with_cache(
        voyage_client,
        positives_df,
        id_col="id",
        text_col="job_description",
        cache_path=cache_path,
        hash_path=hash_path,
        model=model,
        skip_empty=True,
    )
    # Ensure keys are strings (UUIDs)
    return {str(k): v for k, v in embeddings.items()}


def embed_resumes(
    voyage_client,
    resumes_df: pd.DataFrame,
    model: str = None,
    cache_path: str = None,
    hash_path: str = None,
) -> dict[int, np.ndarray]:
    """
    Embed all resumes, using cache if resume text hash matches.

    Returns:
        {resume_id: embedding_float32}
    """
    if cache_path is None:
        cache_path = eval_config.TUNE_RESUME_EMBEDDINGS_CACHE
    if hash_path is None:
        hash_path = eval_config.TUNE_RESUME_EMBEDDINGS_HASH

    embeddings = embed_with_cache(
        voyage_client,
        resumes_df,
        id_col="id",
        text_col="resume",
        cache_path=cache_path,
        hash_path=hash_path,
        model=model,
        skip_empty=False,
    )
    # Cast keys to int (resume IDs are ints in the DB)
    return {int(k): v for k, v in embeddings.items()}
