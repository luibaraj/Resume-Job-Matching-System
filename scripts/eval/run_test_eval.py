"""
Test set evaluation script for the Resume-Job Matching System.

Measures retrieval quality (precision@5, recall@10) on the held-out test set
using a corpus of 1000 sampled real jobs + synthetic positives. Uses ChromaDB for
retrieval, matching the production pipeline.

Pipeline:
1. Sample 1000 tune + 1000 test jobs from jobs.db
2. Embed synthetic positives and resumes for the test set (cached)
3. Build ChromaDB collection from test-sampled jobs (cached, separate from tune)
4. For each resume:
   a. Swap positives in the collection (remove prev, add current)
   b. Dense retrieval via query_collection()
   c. Optional Cohere reranking
   d. Compute precision@5, recall@10
5. Aggregate metrics and write diagnostic outputs
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
from dotenv import load_dotenv

import mlflow
from config import (
    DB_DEFAULT_PATH,
    HNSW_EF,
    HNSW_EF_CONSTRUCTION,
    RETRIEVE_TOP_K,
    RERANK_TOP_N,
    RERANK_INTER_REQUEST_DELAY,
    VOYAGE_MODEL,
    EMBEDDING_DIM,
    COHERE_RERANK_MODEL,
)
from embedding import create_client
from eval.collection import get_or_build_tune_collection, swap_positives
from eval.embedding_cache import embed_positives, embed_resumes
from eval.reporting import write_results_json, write_missed_positives_csv
from eval.eval_config import (
    K_PRECISION,
    K_RECALL,
    RESULTS_DIR,
    TEST_POSITIVES_PATH,
    TEST_RESUMES_PATH,
    TEST_SAMPLE_N,
    TEST_POSITIVE_EMBEDDINGS_CACHE,
    TEST_POSITIVE_EMBEDDINGS_HASH,
    TEST_RESUME_EMBEDDINGS_CACHE,
    TEST_RESUME_EMBEDDINGS_HASH,
    CHROMA_TEST_EVAL_DIR,
    CHROMA_TEST_EVAL_COLLECTION,
    TEST_SAMPLED_JOBS_HASH,
    TEST_RESULTS_JSON,
    TEST_MISSED_CSV,
    SAMPLE_SEED,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from eval.data_loading import sample_jobs
from eval.metrics import batch_compute_metrics_at_k, compute_metrics_at_k
from eval.types import PositiveRetrievalStatus, ResumeEvalResult
from reranking import batch_rerank_jobs, create_rerank_client
from retrieval import JobResult, query_collection

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test set evaluation for Resume-Job Matching System"
    )
    parser.add_argument(
        "--db-path", default=DB_DEFAULT_PATH, help="Path to jobs.db"
    )
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        help="Skip Cohere reranking stage (retrieval-only eval)",
    )
    parser.add_argument(
        "--force-resample",
        action="store_true",
        help="Force re-sample jobs even if CSVs exist",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_env() -> tuple[str, Optional[str]]:
    """Load .env and return API keys. Abort if VOYAGE_API_KEY missing."""
    load_dotenv()
    voyage_key = os.environ.get("VOYAGE_API_KEY")
    cohere_key = os.environ.get("COHERE_API_KEY")

    if not voyage_key:
        raise ValueError("VOYAGE_API_KEY must be set in .env")

    if not cohere_key:
        logger.warning("COHERE_API_KEY not set; reranking will be skipped")

    return voyage_key, cohere_key


def retrieve_for_resume(
    resume_row: pd.Series,
    resume_positives_df: pd.DataFrame,
    resume_embeddings: dict[int, np.ndarray],
    positive_embeddings: dict[str, np.ndarray],
    collection: chromadb.Collection,
) -> Optional[tuple[list[JobResult], set[str]]]:
    """
    Swap positives and run dense retrieval for a single resume.

    Returns:
        (retrieved_jobs, positive_chroma_ids) or None if retrieval fails.
    """
    resume_id = int(resume_row["id"])
    logger.info(
        f"Retrieving for resume {resume_id} ({resume_row['seniority']}/{resume_row['domain']})"
    )

    try:
        # Swap positives
        resume_positives = resume_positives_df[
            resume_positives_df["resume_id"] == resume_id
        ]
        if len(resume_positives) == 0:
            logger.warning("No positives for resume %s", resume_id)
            return None

        current_positive_ids = swap_positives(
            collection, [], resume_positives, positive_embeddings
        )

        # Get query embedding
        query_emb = resume_embeddings[resume_id]

        # Dense retrieval
        retrieved = query_collection(collection, query_emb, top_k=RETRIEVE_TOP_K, ef=HNSW_EF)

        return (retrieved, set(current_positive_ids))

    except Exception as e:
        logger.error("Error retrieving for resume %s: %s", resume_id, e, exc_info=True)
        return None


def classify_positive_retrieval(
    retrieved: list[JobResult],
    positive_chroma_ids: set[str],
) -> dict[str, dict]:
    """
    Classify which positives were retrieved and at what rank.

    Returns:
        {positive_chroma_id: {"embedding_rank": int | None, "embedding_hit": bool}}
    """
    statuses = {pos_id: {"embedding_rank": None, "embedding_hit": False}
                for pos_id in positive_chroma_ids}

    for rank, job in enumerate(retrieved, start=1):
        if job["id"] in positive_chroma_ids:
            statuses[job["id"]]["embedding_rank"] = rank
            statuses[job["id"]]["embedding_hit"] = True

    return statuses


def classify_reranker_outcome(
    statuses: dict[str, dict],
    reranked: list[JobResult],
    skip_rerank: bool,
) -> dict[str, dict]:
    """
    Augment statuses with reranker outcomes and miss_type classification.

    Returns updated statuses dict.
    """
    for status in statuses.values():
        status["rerank_rank"] = None
        status["reranker_hit"] = None
        status["miss_type"] = "embedding_miss"

    if skip_rerank:
        # No reranking: embedding hits are hits
        for status in statuses.values():
            if status["embedding_hit"]:
                status["miss_type"] = "hit"
        return statuses

    # Classify reranker outcomes
    for rank, job in enumerate(reranked, start=1):
        if job["id"] in statuses:
            statuses[job["id"]]["rerank_rank"] = rank
            statuses[job["id"]]["reranker_hit"] = True

    for pos_id, status in statuses.items():
        if status["embedding_hit"]:
            if status["reranker_hit"]:
                status["miss_type"] = "hit"
            else:
                status["miss_type"] = "reranker_miss"
        else:
            status["miss_type"] = "embedding_miss"

    return statuses


def score_resume(
    resume_row: pd.Series,
    resume_positives_df: pd.DataFrame,
    positives_df: pd.DataFrame,
    retrieved: list[JobResult],
    positive_chroma_ids: set[str],
    reranked: list[JobResult],
    skip_rerank: bool,
) -> Optional[ResumeEvalResult]:
    """
    Score a resume's retrieval and compute metrics.

    Classifies hits, computes precision@k and recall@k metrics.

    Returns:
        ResumeEvalResult or None if scoring fails.
    """
    resume_id = int(resume_row["id"])

    try:
        # Classify embedding hits
        statuses = classify_positive_retrieval(retrieved, positive_chroma_ids)

        # Classify reranker outcome
        statuses = classify_reranker_outcome(statuses, reranked, skip_rerank)

        # Build final ranked list (from reranked)
        final_ranked = [job["id"] for job in reranked]

        # Compute metrics
        relevant_ids = positive_chroma_ids
        metrics = compute_metrics_at_k(
            retrieved_ids=final_ranked,
            relevant_ids=relevant_ids,
            k_values=[K_PRECISION, K_RECALL],
        )

        # Get resume's positives for this evaluation
        resume_positives = resume_positives_df[
            resume_positives_df["resume_id"] == resume_id
        ]

        # Build PositiveRetrievalStatus records
        positive_statuses = []
        for _, pos_row in resume_positives.iterrows():
            pos_chroma_id = f"pos_{pos_row['id']}"
            status = statuses[pos_chroma_id]

            # Parse skills
            primary_skills = (
                str(pos_row["primary_skills"]).split("; ")
                if pd.notna(pos_row["primary_skills"])
                else []
            )

            positive_statuses.append(
                PositiveRetrievalStatus(
                    positive_id=pos_row["id"],
                    resume_id=resume_id,
                    resume_seniority=resume_row["seniority"],
                    resume_domain=resume_row["domain"],
                    positive_title=str(pos_row["title"]),
                    positive_seniority=str(pos_row["seniority"]),
                    positive_domain=str(pos_row["domain"]),
                    primary_skills=primary_skills,
                    embedding_rank=status["embedding_rank"],
                    embedding_hit=status["embedding_hit"],
                    rerank_rank=status["rerank_rank"],
                    reranker_hit=status["reranker_hit"],
                    miss_type=status["miss_type"],
                    seniority_gap=resume_row["seniority"] != pos_row["seniority"],
                    domain_gap=resume_row["domain"] != pos_row["domain"],
                )
            )

        return ResumeEvalResult(
            resume_id=resume_id,
            seniority=str(resume_row["seniority"]),
            domain=str(resume_row["domain"]),
            precision_at_5=metrics["precision"][K_PRECISION],
            recall_at_10=metrics["recall"][K_RECALL],
            num_positives=len(resume_positives),
            positives=positive_statuses,
        )

    except Exception as e:
        logger.error("Error scoring resume %s: %s", resume_id, e, exc_info=True)
        return None




def main() -> None:
    """Main orchestration function."""
    args = parse_args()
    setup_logging(args.log_level)

    logger.info("=" * 80)
    logger.info("Resume-Job Matching: Test Set Evaluation")
    logger.info("=" * 80)

    # Setup
    voyage_api_key, cohere_api_key = load_env()
    skip_rerank = args.skip_rerank or not cohere_api_key
    if not cohere_api_key and not args.skip_rerank:
        logger.warning("Cohere API key missing; reranking disabled")

    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    run_name = f"test_eval_ef{HNSW_EF}_topk{RETRIEVE_TOP_K}_rerank{RERANK_TOP_N}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "eval_set":    "test",
            "skip_rerank": str(skip_rerank),
            "db_path":     args.db_path,
        })
        mlflow.log_params({
            "voyage_model":          VOYAGE_MODEL,
            "embedding_dim":         EMBEDDING_DIM,
            "retrieve_top_k":        RETRIEVE_TOP_K,
            "hnsw_ef":               HNSW_EF,
            "hnsw_ef_construction":  HNSW_EF_CONSTRUCTION,
            "rerank_model":          COHERE_RERANK_MODEL,
            "rerank_top_n":          RERANK_TOP_N,
            "skip_rerank":           skip_rerank,
            "k_precision":           K_PRECISION,
            "k_recall":              K_RECALL,
            "test_sample_n":         TEST_SAMPLE_N,
            "sample_seed":           SAMPLE_SEED,
        })

        # Create results directory
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

        voyage_client = create_client(voyage_api_key)

        try:
            # Load test data
            logger.info("Loading test data")
            resumes_df = pd.read_csv(TEST_RESUMES_PATH)
            positives_df = pd.read_csv(TEST_POSITIVES_PATH)
            logger.info("Loaded %d resumes and %d positives", len(resumes_df), len(positives_df))

            # Sample jobs (returns both tune and test; we take the test set)
            _, test_jobs_df = sample_jobs(
                args.db_path, force=args.force_resample
            )

            # Embed positives and resumes with test-specific cache paths
            positive_embeddings = embed_positives(
                voyage_client,
                positives_df,
                cache_path=TEST_POSITIVE_EMBEDDINGS_CACHE,
                hash_path=TEST_POSITIVE_EMBEDDINGS_HASH,
            )
            resume_embeddings = embed_resumes(
                voyage_client,
                resumes_df,
                cache_path=TEST_RESUME_EMBEDDINGS_CACHE,
                hash_path=TEST_RESUME_EMBEDDINGS_HASH,
            )

            # Build test collection (override all tune-specific defaults)
            collection = get_or_build_tune_collection(
                test_jobs_df,
                args.db_path,
                chroma_dir=CHROMA_TEST_EVAL_DIR,
                hash_path=TEST_SAMPLED_JOBS_HASH,
                collection_name=CHROMA_TEST_EVAL_COLLECTION,
                force_rebuild=args.force_resample,
            )

            # Phase 1: Retrieval loop
            logger.info("Phase 1: Starting retrieval for all resumes")
            retrieval_results: list[tuple[pd.Series, Optional[tuple[list[JobResult], set[str]]]]] = []

            for _, resume_row in resumes_df.iterrows():
                result = retrieve_for_resume(
                    resume_row,
                    positives_df,
                    resume_embeddings,
                    positive_embeddings,
                    collection,
                )
                retrieval_results.append((resume_row, result))

            # Phase 2: Batch reranking
            all_reranked: dict[int, list[JobResult]] = {}
            if not skip_rerank and cohere_api_key:
                logger.info("Phase 2: Starting batch reranking")
                cohere_client = create_rerank_client(cohere_api_key)
                valid = [(row, ret) for row, ret in retrieval_results if ret is not None]
                queries_and_jobs = [(row["resume"], ret[0]) for row, ret in valid]
                logger.info(
                    "Batch reranking %d resumes (inter-request delay: %.1fs)",
                    len(queries_and_jobs),
                    RERANK_INTER_REQUEST_DELAY,
                )
                reranked_lists = batch_rerank_jobs(
                    queries_and_jobs,
                    client=cohere_client,
                    inter_request_delay=RERANK_INTER_REQUEST_DELAY,
                )
                for (row, _), reranked in zip(valid, reranked_lists):
                    all_reranked[int(row["id"])] = reranked
            else:
                logger.info("Phase 2: Skipping reranking")

            # Phase 3: Scoring loop
            logger.info("Phase 3: Starting scoring and metric computation")
            all_results: list[ResumeEvalResult] = []

            for resume_row, retrieval_result in retrieval_results:
                if retrieval_result is None:
                    continue

                retrieved, positive_chroma_ids = retrieval_result
                resume_id = int(resume_row["id"])
                reranked = all_reranked.get(resume_id, retrieved)

                result = score_resume(
                    resume_row,
                    positives_df,
                    positives_df,
                    retrieved,
                    positive_chroma_ids,
                    reranked,
                    skip_rerank,
                )

                if result:
                    all_results.append(result)

            logger.info("Successfully evaluated %d/%d resumes", len(all_results), len(resumes_df))

            # Compute aggregate metrics
            logger.info("Computing aggregate metrics")
            batch_retrieved_clean = []
            batch_relevant = []

            for result in all_results:
                ranked = [p["positive_id"] for p in result["positives"] if p["miss_type"] == "hit"]
                batch_retrieved_clean.append([f"pos_{pid}" for pid in ranked])
                batch_relevant.append({f"pos_{p['positive_id']}" for p in result["positives"]})

            batch_metrics = batch_compute_metrics_at_k(
                batch_retrieved_clean,
                batch_relevant,
                k_values=[K_PRECISION, K_RECALL],
            )

            # Write results (using reporting module with test paths)
            write_results_json(all_results, batch_metrics, skip_rerank, output_path=TEST_RESULTS_JSON)
            write_missed_positives_csv(all_results, positives_df, output_path=TEST_MISSED_CSV)

            logger.info("=" * 80)
            logger.info("Mean Precision@%d: %.3f", K_PRECISION, batch_metrics['mean_precision'][K_PRECISION])
            logger.info("Mean Recall@%d: %.3f", K_RECALL, batch_metrics['mean_recall'][K_RECALL])
            logger.info("=" * 80)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error("Fatal error: %s", e, exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
