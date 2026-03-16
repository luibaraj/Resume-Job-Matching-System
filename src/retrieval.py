import logging
import os

import numpy as np

from src.embedding import load_model
from src.utils import setup_logging


def build_user_embedding_string(profile_text: str) -> str:
    """Return the embedding input string for a user profile.

    Module-level for testability.

    Args:
        profile_text: Raw text content of the user profile file

    Returns:
        Stripped profile text ready for encoding
    """
    return profile_text.strip()


def load_user_profile(path: str) -> str:
    """Read the user profile file as UTF-8 text.

    Args:
        path: Path to the user profile file

    Returns:
        File contents as a string

    Raises:
        FileNotFoundError: If the file does not exist
    """
    with open(path, encoding="utf-8") as f:
        return f.read()


def embed_user_profile(model, profile_text: str) -> np.ndarray:
    """Encode the user profile into a normalized float32 vector.

    Args:
        model: SentenceTransformer model instance
        profile_text: Stripped profile text

    Returns:
        float32 unit vector of shape [dim]
    """
    result = model.encode(
        [profile_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return result[0].astype(np.float32)


def load_corpus_embeddings(db, model_id: str) -> tuple[list[int], np.ndarray]:
    """Load all job embeddings into a 2D numpy matrix.

    Fetches blobs from DB, closes the connection, then deserializes into
    a stacked matrix. ~67MB for 16K jobs x 1024 dims.

    Args:
        db: DatabaseManager instance
        model_id: Model identifier to filter embeddings by

    Returns:
        Tuple of (job_ids list, corpus matrix of shape [N, dim])
        Returns ([], empty array) if no embeddings found.
    """
    rows = db.get_all_embeddings(model_id)
    if not rows:
        return [], np.empty((0,), dtype=np.float32)

    job_ids = [row[0] for row in rows]
    vectors = [np.frombuffer(row[1], dtype=np.float32) for row in rows]
    corpus_matrix = np.stack(vectors)
    return job_ids, corpus_matrix


def dense_top_k(
    query_vec: np.ndarray,
    corpus_matrix: np.ndarray,
    job_ids: list[int],
    top_k: int,
) -> list[tuple[int, float, int]]:
    """Select top-k jobs by cosine similarity (dot product of normalized vectors).

    Args:
        query_vec: Unit vector of shape [dim]
        corpus_matrix: Normalized matrix of shape [N, dim]
        job_ids: Job IDs corresponding to rows of corpus_matrix
        top_k: Number of results to return

    Returns:
        List of (job_id, score, rank) tuples sorted by score descending, 1-based rank
    """
    top_k = min(top_k, len(job_ids))
    scores = corpus_matrix @ query_vec

    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [
        (job_ids[idx], float(scores[idx]), rank + 1)
        for rank, idx in enumerate(top_indices)
    ]


def build_bm25_index(job_texts: list[tuple[int, str]]):
    """Build an in-memory BM25 index over cleaned job descriptions.

    Tokenizes by lowercasing and splitting on whitespace.

    Args:
        job_texts: List of (job_id, cleaned_description) tuples

    Returns:
        Tuple of (ordered_job_ids, BM25Okapi index)
    """
    from rank_bm25 import BM25Okapi

    job_ids = [row[0] for row in job_texts]
    tokenized = [row[1].lower().split() for row in job_texts]
    bm25 = BM25Okapi(tokenized)
    return job_ids, bm25


def sparse_top_k(
    bm25,
    query_tokens: list[str],
    job_ids: list[int],
    top_k: int,
) -> list[tuple[int, float, int]]:
    """Select top-k jobs by BM25 score.

    Args:
        bm25: BM25Okapi index
        query_tokens: Tokenized query (lowercased whitespace-split profile text)
        job_ids: Job IDs corresponding to BM25 corpus rows
        top_k: Number of results to return

    Returns:
        List of (job_id, score, rank) tuples sorted by score descending, 1-based rank
    """
    top_k = min(top_k, len(job_ids))
    scores = bm25.get_scores(query_tokens)

    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [
        (job_ids[idx], float(scores[idx]), rank + 1)
        for rank, idx in enumerate(top_indices)
    ]


def reciprocal_rank_fusion(
    dense_results: list[tuple[int, float, int]],
    sparse_results: list[tuple[int, float, int]],
    rrf_k: int,
    top_k: int,
) -> list[tuple[int, float, int]]:
    """Fuse dense and sparse ranked lists via Reciprocal Rank Fusion.

    RRF score = 1/(rrf_k + dense_rank) + 1/(rrf_k + sparse_rank).
    Jobs appearing in only one list still receive their single contribution.

    Args:
        dense_results: (job_id, score, rank) list from dense retrieval
        sparse_results: (job_id, score, rank) list from sparse retrieval
        rrf_k: RRF constant (standard default: 60)
        top_k: Final number of results to return

    Returns:
        List of (job_id, rrf_score, rank) tuples, top_k results, 1-based rank
    """
    rrf_scores: dict[int, float] = {}

    for job_id, _, rank in dense_results:
        rrf_scores[job_id] = rrf_scores.get(job_id, 0.0) + 1.0 / (rrf_k + rank)

    for job_id, _, rank in sparse_results:
        rrf_scores[job_id] = rrf_scores.get(job_id, 0.0) + 1.0 / (rrf_k + rank)

    sorted_jobs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top = sorted_jobs[:top_k]

    return [(job_id, score, rank + 1) for rank, (job_id, score) in enumerate(top)]


def retrieve(db, run_id: int, config) -> tuple[int, int]:
    """Run hybrid retrieval: BM25 + dense embeddings fused via RRF.

    Args:
        db: DatabaseManager instance
        run_id: Pipeline run ID for audit logging
        config: Config instance with retrieval fields set

    Returns:
        Tuple of (matches_written, 0)
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger = setup_logging(log_level=log_level, name="retrieve")

    # Load and tokenize user profile
    profile_text = load_user_profile(config.retrieval_user_profile_path)
    embedding_text = build_user_embedding_string(profile_text)
    query_tokens = embedding_text.lower().split()
    logger.info("User profile loaded from %s", config.retrieval_user_profile_path)

    # Embed user profile
    model = load_model(config.embedding_model_id)
    logger.info("Embedding model loaded: %s", config.embedding_model_id)
    query_vec = embed_user_profile(model, embedding_text)

    # Load dense corpus
    dense_job_ids, corpus_matrix = load_corpus_embeddings(db, config.embedding_model_id)
    if not dense_job_ids:
        logger.warning(
            "No embeddings found in DB for model %s — skipping retrieval",
            config.embedding_model_id,
        )
        return 0, 0
    logger.info(
        "Loaded %d job embeddings (corpus shape: %s)",
        len(dense_job_ids), corpus_matrix.shape,
    )

    # Load sparse corpus (only jobs that have embeddings, so corpora stay aligned)
    sparse_job_texts = db.get_all_cleaned_descriptions()
    if not sparse_job_texts:
        logger.warning("No cleaned descriptions found — skipping retrieval")
        return 0, 0
    logger.info("Loaded %d cleaned descriptions for BM25 index", len(sparse_job_texts))

    # Build BM25 index
    bm25_job_ids, bm25 = build_bm25_index(sparse_job_texts)
    logger.info("BM25 index built")

    # Retrieve candidates from each path (2x top_k for wider recall before fusion)
    candidate_k = min(config.retrieval_top_k * 2, len(dense_job_ids))
    dense_results = dense_top_k(query_vec, corpus_matrix, dense_job_ids, candidate_k)
    sparse_results = sparse_top_k(bm25, query_tokens, bm25_job_ids, candidate_k)
    logger.info(
        "Dense candidates: %d, sparse candidates: %d",
        len(dense_results), len(sparse_results),
    )

    # Fuse via RRF
    final = reciprocal_rank_fusion(
        dense_results, sparse_results, config.retrieval_rrf_k, config.retrieval_top_k
    )
    logger.info("RRF fusion complete: %d final matches", len(final))

    # Persist results
    matches = [
        (job_id, score, rank, config.embedding_model_id)
        for job_id, score, rank in final
    ]
    db.insert_job_matches(matches)
    logger.info("Wrote %d job matches to DB", len(matches))

    return len(matches), 0


def main() -> None:
    """CLI entrypoint called by Docker container.

    Reads config, initializes DB, runs hybrid retrieval, updates pipeline_runs.
    Exits with code 0 on success, 1 on failure.
    """
    from datetime import datetime

    from src.config import load_config
    from src.database import DatabaseManager

    logger = setup_logging(name="retrieval_main")
    try:
        config = load_config()
        logger.setLevel(config.log_level)

        db = DatabaseManager(config.db_path)
        db.initialize_schema()

        run_date = datetime.utcnow().strftime("%Y-%m-%d")
        run_id = db.create_pipeline_run(run_date, "retrieval")

        processed, skipped = retrieve(db, run_id, config)
        db.finish_pipeline_run(run_id, "success", jobs_processed=processed, jobs_skipped=skipped)
        logger.info("Retrieval step completed: %d matches written", processed)

    except Exception as e:
        logger.exception("Retrieval step failed")
        try:
            from src.config import load_config
            from src.database import DatabaseManager

            config = load_config()
            db = DatabaseManager(config.db_path)
            run_date = datetime.utcnow().strftime("%Y-%m-%d")
            run_id = db.create_pipeline_run(run_date, "retrieval")
            db.finish_pipeline_run(run_id, "failed", 0, 0, str(e))
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
