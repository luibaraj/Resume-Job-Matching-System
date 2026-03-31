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
    return parser.parse_args()


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
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
            logger.warning(f"No positives for resume {resume_id}")
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
        logger.error(f"Error retrieving for resume {resume_id}: {e}", exc_info=True)
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
        logger.error(f"Error scoring resume {resume_id}: {e}", exc_info=True)
        return None


def _sanitize_metric_name(name: str) -> str:
    """Sanitize a string for use in MLflow metric names."""
    return name.replace(" ", "_").replace("/", "_")


def write_test_results_json(
    results: list[ResumeEvalResult],
    batch_metrics,
    skip_rerank: bool,
) -> None:
    """Write test evaluation results to JSON and log to MLflow."""
    # Miss analysis
    total_positives = sum(r["num_positives"] for r in results)
    hits = sum(
        sum(1 for p in r["positives"] if p["miss_type"] == "hit") for r in results
    )
    embedding_misses = sum(
        sum(1 for p in r["positives"] if p["miss_type"] == "embedding_miss")
        for r in results
    )
    reranker_misses = sum(
        sum(1 for p in r["positives"] if p["miss_type"] == "reranker_miss")
        for r in results
    )

    # Miss rates by seniority and domain
    miss_by_seniority = {}
    miss_by_domain = {}

    for result in results:
        for positive in result["positives"]:
            seniority = positive["positive_seniority"]
            domain = positive["positive_domain"]

            if seniority not in miss_by_seniority:
                miss_by_seniority[seniority] = {"total": 0, "misses": 0}
            if domain not in miss_by_domain:
                miss_by_domain[domain] = {"total": 0, "misses": 0}

            miss_by_seniority[seniority]["total"] += 1
            miss_by_domain[domain]["total"] += 1

            if positive["miss_type"] != "hit":
                miss_by_seniority[seniority]["misses"] += 1
                miss_by_domain[domain]["misses"] += 1

    miss_rate_by_seniority = {
        s: c["misses"] / c["total"] if c["total"] > 0 else 0.0
        for s, c in miss_by_seniority.items()
    }
    miss_rate_by_domain = {
        d: c["misses"] / c["total"] if c["total"] > 0 else 0.0
        for d, c in miss_by_domain.items()
    }

    output = {
        "run_metadata": {
            "timestamp": datetime.now().isoformat(),
            "seed": SAMPLE_SEED,
            "k_precision": K_PRECISION,
            "k_recall": K_RECALL,
            "skip_rerank": skip_rerank,
            "num_resumes": len(results),
            "num_sampled_test_jobs": TEST_SAMPLE_N,
        },
        "aggregate": {
            "mean_precision_at_5": batch_metrics["mean_precision"][K_PRECISION],
            "mean_recall_at_10": batch_metrics["mean_recall"][K_RECALL],
            "num_queries": batch_metrics["num_queries"],
        },
        "per_resume": [
            {
                "resume_id": r["resume_id"],
                "seniority": r["seniority"],
                "domain": r["domain"],
                "precision_at_5": r["precision_at_5"],
                "recall_at_10": r["recall_at_10"],
                "num_positives": r["num_positives"],
                "positive_statuses": [
                    {
                        "positive_id": p["positive_id"],
                        "title": p["positive_title"],
                        "embedding_rank": p["embedding_rank"],
                        "embedding_hit": p["embedding_hit"],
                        "rerank_rank": p["rerank_rank"],
                        "reranker_hit": p["reranker_hit"],
                        "miss_type": p["miss_type"],
                    }
                    for p in r["positives"]
                ],
            }
            for r in results
        ],
        "miss_analysis": {
            "total_positives": total_positives,
            "total_hits": hits,
            "embedding_misses": embedding_misses,
            "reranker_misses": reranker_misses,
            "miss_rate_by_seniority": miss_rate_by_seniority,
            "miss_rate_by_domain": miss_rate_by_domain,
        },
    }

    with open(TEST_RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Wrote results to {TEST_RESULTS_JSON}")

    # MLflow logging
    if mlflow.active_run():
        # Artifacts
        mlflow.log_artifact(str(TEST_RESULTS_JSON), artifact_path="results")

        # Aggregate metrics
        mlflow.log_metrics({
            f"mean_precision_at_{K_PRECISION}": batch_metrics["mean_precision"][K_PRECISION],
            f"mean_recall_at_{K_RECALL}": batch_metrics["mean_recall"][K_RECALL],
            "num_queries": float(batch_metrics["num_queries"]),
            "total_positives":     float(total_positives),
            "total_hits":          float(hits),
            "embedding_misses":    float(embedding_misses),
            "reranker_misses":     float(reranker_misses),
            "embedding_miss_rate": embedding_misses / total_positives if total_positives else 0.0,
            "reranker_miss_rate":  reranker_misses / total_positives if total_positives else 0.0,
        })

        # Per-domain and per-seniority miss rates
        for domain, rate in miss_rate_by_domain.items():
            mlflow.log_metric(f"miss_rate_domain_{_sanitize_metric_name(domain)}", rate)
        for seniority, rate in miss_rate_by_seniority.items():
            mlflow.log_metric(f"miss_rate_seniority_{_sanitize_metric_name(seniority)}", rate)

        # Distributional summary of per-resume metrics
        precisions = [r["precision_at_5"] for r in results]
        recalls    = [r["recall_at_10"]   for r in results]
        mlflow.log_metrics({
            "min_precision":         float(min(precisions)),
            "max_precision":         float(max(precisions)),
            "std_precision":         float(np.std(precisions)),
            "min_recall":            float(min(recalls)),
            "max_recall":            float(max(recalls)),
            "std_recall":            float(np.std(recalls)),
            "pct_perfect_precision": sum(p == 1.0 for p in precisions) / len(precisions),
        })

        # Per-resume table
        cols = ["resume_id", "seniority", "domain",
                f"precision_at_{K_PRECISION}",
                f"recall_at_{K_RECALL}", "num_positives"]
        rows = [
            [r["resume_id"], r["seniority"], r["domain"],
             r["precision_at_5"], r["recall_at_10"], r["num_positives"]]
            for r in results
        ]
        mlflow.log_table(data={"columns": cols, "data": rows},
                         artifact_file="results/per_resume_metrics.json")


def write_test_missed_positives_csv(
    results: list[ResumeEvalResult], positives_df: pd.DataFrame
) -> None:
    """Write missed positives to CSV for analysis and log to MLflow."""
    rows = []

    for result in results:
        for positive in result["positives"]:
            if positive["miss_type"] == "hit":
                continue

            # Look up full positive record from positives_df
            pos_record = positives_df[positives_df["id"] == positive["positive_id"]]
            if pos_record.empty:
                continue

            pos_row = pos_record.iloc[0]

            rows.append(
                {
                    "positive_id": positive["positive_id"],
                    "resume_id": result["resume_id"],
                    "resume_seniority": result["seniority"],
                    "resume_domain": result["domain"],
                    "positive_title": positive["positive_title"],
                    "positive_seniority": positive["positive_seniority"],
                    "positive_domain": positive["positive_domain"],
                    "primary_skills": "; ".join(positive["primary_skills"]),
                    "secondary_skills": pos_row["secondary_skills"],
                    "responsibilities": pos_row["responsibilities"],
                    "miss_type": positive["miss_type"],
                    "embedding_rank": positive["embedding_rank"],
                    "rerank_rank": positive["rerank_rank"],
                    "seniority_gap": positive["seniority_gap"],
                    "domain_gap": positive["domain_gap"],
                    "job_description": pos_row["job_description"],
                }
            )

    missed_df = pd.DataFrame(rows)
    missed_df.to_csv(TEST_MISSED_CSV, index=False)
    logger.info(f"Wrote {len(missed_df)} missed positives to {TEST_MISSED_CSV}")

    # MLflow logging
    if mlflow.active_run():
        mlflow.log_artifact(str(TEST_MISSED_CSV), artifact_path="results")
        mlflow.log_metric("num_missed_positives", float(len(missed_df)))


def main() -> None:
    """Main orchestration function."""
    setup_logging()
    args = parse_args()

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
            logger.info(f"Loaded {len(resumes_df)} resumes and {len(positives_df)} positives")

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

            logger.info(f"Successfully evaluated {len(all_results)}/{len(resumes_df)} resumes")

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

            # Write results
            write_test_results_json(all_results, batch_metrics, skip_rerank)
            write_test_missed_positives_csv(all_results, positives_df)

            logger.info("=" * 80)
            logger.info(f"Mean Precision@{K_PRECISION}: {batch_metrics['mean_precision'][K_PRECISION]:.3f}")
            logger.info(f"Mean Recall@{K_RECALL}: {batch_metrics['mean_recall'][K_RECALL]:.3f}")
            logger.info("=" * 80)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
