"""
Data loading utilities for evaluation pipeline.

Provides job sampling from SQLite, embedding loading, and chunked SQL query utilities.
"""

import hashlib
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import config
import embedding
from eval import eval_config

logger = logging.getLogger(__name__)

SQLITE_CHUNK_SIZE = 500


def compute_hash(data: bytes) -> str:
    """Compute MD5 hash of data."""
    return hashlib.md5(data).hexdigest()


def chunked_select(
    conn: sqlite3.Connection,
    query_template: str,
    ids: list,
    chunk_size: int = SQLITE_CHUNK_SIZE,
) -> list:
    """
    Execute a chunked SELECT IN query on a list of IDs.

    Args:
        conn: SQLite connection
        query_template: SQL query with {} placeholder for IN clause
                       e.g. "SELECT id, name FROM jobs WHERE id IN ({})"
        ids: List of IDs to query
        chunk_size: Max number of IDs per query (SQLite placeholder limit)

    Returns:
        Flattened list of all rows from all chunks
    """
    rows = []
    cursor = conn.cursor()

    for i in range(0, len(ids), chunk_size):
        chunk = ids[i : i + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        query = query_template.format(placeholders)
        cursor.execute(query, chunk)
        rows.extend(cursor.fetchall())

    return rows


def sample_jobs(
    db_path: str,
    tune_n: int = eval_config.TUNE_SAMPLE_N,
    test_n: int = eval_config.TEST_SAMPLE_N,
    seed: int = eval_config.SAMPLE_SEED,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample tune and test jobs from jobs.db, caching to CSV if needed.

    Returns:
        (tune_jobs_df, test_jobs_df) with columns: job_id, cleaned_description
    """
    tune_path = Path(eval_config.TUNE_SAMPLED_JOBS_PATH)
    test_path = Path(eval_config.TEST_SAMPLED_JOBS_PATH)

    if tune_path.exists() and test_path.exists() and not force:
        logger.info("Loading sampled jobs from existing CSVs")
        tune_df = pd.read_csv(tune_path)
        test_df = pd.read_csv(test_path)
        return tune_df, test_df

    logger.info("Sampling jobs from jobs.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM jobs WHERE embedded=1 AND embedding IS NOT NULL ORDER BY id"
    )
    all_job_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    if len(all_job_ids) < tune_n + test_n:
        logger.warning(
            f"Only {len(all_job_ids)} embedded jobs available; need {tune_n + test_n}"
        )
        tune_n = min(tune_n, len(all_job_ids) // 2)
        test_n = len(all_job_ids) - tune_n

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(len(all_job_ids), size=tune_n + test_n, replace=False)
    sampled_ids = [all_job_ids[i] for i in sorted(sampled_indices)]
    tune_ids = sampled_ids[:tune_n]
    test_ids = sampled_ids[tune_n:]

    # Fetch descriptions
    conn = sqlite3.connect(db_path)
    tune_df = _fetch_jobs_by_id(conn, tune_ids)
    test_df = _fetch_jobs_by_id(conn, test_ids)
    conn.close()

    # Write CSVs
    tune_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    tune_df.to_csv(tune_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Sampled {len(tune_df)} tune jobs and {len(test_df)} test jobs")

    return tune_df, test_df


def _fetch_jobs_by_id(conn: sqlite3.Connection, job_ids: list[int]) -> pd.DataFrame:
    """Fetch jobs by ID with columns: job_id, cleaned_description."""
    rows = chunked_select(
        conn,
        "SELECT id, cleaned_description FROM jobs WHERE id IN ({})",
        job_ids,
    )
    return pd.DataFrame(rows, columns=["job_id", "cleaned_description"])


def load_sampled_job_embeddings(
    db_path: str, job_ids: list[int]
) -> dict[int, np.ndarray]:
    """Load embeddings for sampled jobs from jobs.db BLOB column."""
    embeddings = {}
    conn = sqlite3.connect(db_path)

    rows = chunked_select(
        conn,
        "SELECT id, embedding FROM jobs WHERE id IN ({})",
        job_ids,
    )
    for job_id, blob in rows:
        embeddings[job_id] = embedding.deserialize_embedding(blob, dim=config.EMBEDDING_DIM)

    conn.close()
    return embeddings
