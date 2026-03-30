"""
ChromaDB collection management for evaluation pipeline.

Handles building, caching, and updating collections used in retrieval evaluation.
"""

import logging
import sqlite3
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd

import config
import embedding
import regex_extraction
from eval import data_loading, eval_config

logger = logging.getLogger(__name__)


def get_or_build_tune_collection(
    tune_jobs_df: pd.DataFrame,
    db_path: str,
    chroma_dir: str = eval_config.CHROMA_TUNE_EVAL_DIR,
    hash_path: str = eval_config.TUNE_SAMPLED_JOBS_HASH,
    collection_name: str = eval_config.CHROMA_TUNE_EVAL_COLLECTION,
    force_rebuild: bool = False,
) -> chromadb.Collection:
    """
    Get or build the tune eval ChromaDB collection, using cache if jobs haven't changed.

    The collection contains sampled jobs with ID prefix "job_{id}".

    Args:
        tune_jobs_df: DataFrame with job_id column
        db_path: Path to jobs.db
        chroma_dir: ChromaDB persistent directory
        hash_path: Path to hash file for cache invalidation
        collection_name: ChromaDB collection name
        force_rebuild: Force rebuild even if cache exists

    Returns:
        ChromaDB collection object
    """
    hash_p = Path(hash_path)

    # Compute hash of job_id column
    sorted_ids = np.sort(tune_jobs_df["job_id"].values)
    current_hash = data_loading.compute_hash("|".join(map(str, sorted_ids)).encode("utf-8"))

    # Check if collection exists and hash matches
    if not force_rebuild and hash_p.exists():
        try:
            with open(hash_p) as f:
                cached_hash = f.read().strip()
            if cached_hash == current_hash:
                chroma_client = chromadb.PersistentClient(path=chroma_dir)
                try:
                    collection = chroma_client.get_collection(name=collection_name)
                    logger.info("Reusing existing tune eval collection")
                    return collection
                except Exception:
                    pass
        except Exception:
            pass

    logger.info("Building tune eval ChromaDB collection")
    chroma_client = chromadb.PersistentClient(path=chroma_dir)

    # Delete existing collection if present
    try:
        chroma_client.delete_collection(name=collection_name)
        logger.info("Deleted existing collection")
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name=collection_name, metadata={"hnsw_construction": config.HNSW_EF}
    )

    # Fetch job data (embeddings + metadata)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    job_ids = tune_jobs_df["job_id"].tolist()
    ids_to_upsert = []
    embeddings_to_upsert = []
    metadatas_to_upsert = []

    rows = data_loading.chunked_select(
        conn,
        """
        SELECT id, title, location, source_url, board_token, cleaned_description,
               embedding
        FROM jobs WHERE id IN ({})
        """,
        job_ids,
    )

    for row in rows:
        ids_to_upsert.append(f"job_{row[0]}")
        embeddings_to_upsert.append(
            embedding.deserialize_embedding(row[6], dim=config.EMBEDDING_DIM).tolist()
        )
        metadatas_to_upsert.append(
            {
                "title": row[1] or "",
                "location": row[2] or "",
                "source_url": row[3] or "",
                "board_token": row[4] or "",
                "cleaned_description": row[5] or "",
                "required_degree": regex_extraction.extract_degree_requirement(row[5] or ""),
                "seniority_level": (
                    regex_extraction.extract_seniority_level(row[5] or "")
                    or regex_extraction.extract_seniority_from_title(row[1] or "")
                ),
                "min_years_experience": regex_extraction.extract_years_experience(row[5] or ""),
            }
        )

    conn.close()

    collection.upsert(
        ids=ids_to_upsert,
        embeddings=embeddings_to_upsert,
        documents=[m["cleaned_description"] for m in metadatas_to_upsert],
        metadatas=metadatas_to_upsert,
    )

    logger.info(f"Upserted {len(ids_to_upsert)} jobs to tune eval collection")

    # Write hash
    hash_p.parent.mkdir(parents=True, exist_ok=True)
    with open(hash_p, "w") as f:
        f.write(current_hash)

    return collection


def swap_positives(
    collection: chromadb.Collection,
    prev_positive_ids: list[str],
    current_positives: pd.DataFrame,
    positive_embeddings: dict[str, np.ndarray],
) -> list[str]:
    """
    Delete previous positives and upsert current positives to the collection.

    Args:
        collection: ChromaDB collection
        prev_positive_ids: List of previous positive ChromaDB IDs to delete
        current_positives: DataFrame of current positives
        positive_embeddings: dict mapping positive ID to embedding

    Returns:
        List of current positive ChromaDB IDs (for use as prev_positive_ids next iteration)
    """
    # Delete previous
    if prev_positive_ids:
        collection.delete(ids=prev_positive_ids)

    # Upsert current
    ids_to_upsert = []
    embeddings_to_upsert = []
    metadatas_to_upsert = []

    for _, row in current_positives.iterrows():
        pos_id = row["id"]
        if pos_id not in positive_embeddings:
            logger.warning(f"No embedding for positive {pos_id}; skipping")
            continue

        ids_to_upsert.append(f"pos_{pos_id}")
        embeddings_to_upsert.append(positive_embeddings[pos_id].tolist())
        metadatas_to_upsert.append(
            {
                "title": str(row["title"]) if row["title"] else "",
                "location": "",
                "source_url": "",
                "board_token": "",
                "cleaned_description": str(row["job_description"]),
                "required_degree": regex_extraction.extract_degree_requirement(str(row["job_description"])),
                "seniority_level": (
                    regex_extraction.extract_seniority_level(str(row["job_description"]))
                    or regex_extraction.extract_seniority_from_title(str(row["title"]) if row["title"] else "")
                ),
                "min_years_experience": regex_extraction.extract_years_experience(str(row["job_description"])),
            }
        )

    collection.upsert(
        ids=ids_to_upsert,
        embeddings=embeddings_to_upsert,
        documents=[m["cleaned_description"] for m in metadatas_to_upsert],
        metadatas=metadatas_to_upsert,
    )

    return ids_to_upsert
