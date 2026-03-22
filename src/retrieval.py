"""
Retrieval module: Chroma collection management and dense vector search.

All functions are stateless and side-effect-free beyond Chroma persistence.
No environment loading, path resolution, or DB connection management is performed here.
"""

import logging
import sqlite3
from typing import Optional, TypedDict

import chromadb
import numpy as np

from config import (
    CHROMA_COLLECTION_NAME,
    DEGREE_UNKNOWN,
    EMBEDDING_DIM,
    SENIORITY_UNKNOWN,
    YEARS_UNKNOWN,
)
from embedding import deserialize_embedding
from regex_extraction import (
    extract_degree_requirement,
    extract_seniority_level,
    extract_years_experience,
)

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME: str = CHROMA_COLLECTION_NAME
SQLITE_CHUNK_SIZE: int = 500


class JobResult(TypedDict):
    """Result of a vector similarity query."""

    id: str
    distance: float
    title: str
    location: str
    source_url: str
    board_token: str
    cleaned_description: str
    required_degree: int
    seniority_level: int
    min_years_experience: int


def build_collection(
    conn: sqlite3.Connection,
    chroma_client: chromadb.PersistentClient,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    ef_construction: int = 100,
) -> chromadb.Collection:
    """
    Load embedded jobs from SQLite and upsert them into a Chroma collection.

    Fetches all jobs with embedded=1 and embedding IS NOT NULL from SQLite,
    deserializes embeddings, and upserts them to Chroma in chunks to manage memory.
    The operation is idempotent: calling this function twice on unchanged data is a no-op
    (Chroma upsert overwrites by ID).

    Args:
        conn: An open sqlite3.Connection to the jobs database.
        chroma_client: A chromadb.PersistentClient instance.
        collection_name: Name of the Chroma collection. Defaults to "jobs".
        ef_construction: HNSW construction parameter controlling index build quality.
            Higher values (e.g., 200) produce better quality indices at the cost of slower build.
            Defaults to 100.

    Returns:
        The chromadb.Collection object (either newly created or retrieved).

    Raises:
        sqlite3.DatabaseError: If the query fails.
        ValueError: If embeddings cannot be deserialized (wrong size, corrupted blob).
    """
    conn.row_factory = sqlite3.Row
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw_construction": ef_construction},
    )

    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, title, location, source_url, board_token, cleaned_description, embedding
        FROM jobs
        WHERE embedded = 1 AND embedding IS NOT NULL
        """
    )

    total_upserted = 0

    while True:
        rows = cursor.fetchmany(SQLITE_CHUNK_SIZE)
        if not rows:
            break

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for row in rows:
            ids.append(str(row["id"]))
            embeddings.append(
                deserialize_embedding(row["embedding"], dim=EMBEDDING_DIM).tolist()
            )
            desc = row["cleaned_description"] or ""
            documents.append(desc)
            metadatas.append(
                {
                    "title": row["title"] or "",
                    "location": row["location"] or "",
                    "source_url": row["source_url"] or "",
                    "board_token": row["board_token"] or "",
                    "cleaned_description": desc,
                    "required_degree": extract_degree_requirement(desc),
                    "seniority_level": extract_seniority_level(desc),
                    "min_years_experience": extract_years_experience(desc),
                }
            )

        collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

        total_upserted += len(rows)
        logger.info(
            "Upserted %d rows to collection '%s' (batch of %d)",
            total_upserted,
            collection_name,
            len(rows),
        )

    logger.info(
        "Collection '%s' sync complete: %d total jobs indexed", collection_name, total_upserted
    )

    return collection


def query_collection(
    collection: chromadb.Collection,
    query_embedding: np.ndarray,
    top_k: int = 10,
    ef: int = 10,
    where: Optional[dict] = None,
) -> list[JobResult]:
    """
    Query a Chroma collection for the most similar jobs using dense vectors.

    Args:
        collection: A chromadb.Collection instance.
        query_embedding: A 1-D numpy array of shape (EMBEDDING_DIM,) and dtype float32.
        top_k: Number of top results to return. Defaults to 10.
        ef: HNSW search parameter controlling query recall. Higher values (e.g., 50)
            produce more accurate results at the cost of slower queries. Defaults to 10.
        where: Optional ChromaDB where filter dict. If provided, only jobs matching
            the filter are returned. Defaults to None (no filtering).

    Returns:
        A list of JobResult dicts, ordered by ascending distance (most similar first).

    Raises:
        ValueError: If query_embedding has incorrect shape or dtype.
    """
    if query_embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Expected query_embedding shape ({EMBEDDING_DIM},), got {query_embedding.shape}"
        )

    collection.modify(metadata={"hnsw:ef": ef})
    query_kwargs: dict = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": top_k,
        "include": ["metadatas", "distances"],
    }
    if where is not None:
        query_kwargs["where"] = where
    result = collection.query(**query_kwargs)

    # result.ids, result.distances, result.metadatas are all lists of lists (one per query)
    ids = result["ids"][0]
    distances = result["distances"][0]
    metadatas = result["metadatas"][0]

    return [
        JobResult(
            id=doc_id,
            distance=float(dist),
            title=meta.get("title", ""),
            location=meta.get("location", ""),
            source_url=meta.get("source_url", ""),
            board_token=meta.get("board_token", ""),
            cleaned_description=meta.get("cleaned_description", ""),
            required_degree=meta.get("required_degree", DEGREE_UNKNOWN),
            seniority_level=meta.get("seniority_level", SENIORITY_UNKNOWN),
            min_years_experience=meta.get("min_years_experience", YEARS_UNKNOWN),
        )
        for doc_id, dist, meta in zip(ids, distances, metadatas)
    ]
