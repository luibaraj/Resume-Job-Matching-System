"""
Tuning evaluation script for the Resume-Job Matching System.

Measures retrieval quality (precision@5, recall@10) on the tune set (30 resumes, 150 positives)
using a corpus of 1000 sampled real jobs + synthetic positives. Uses ChromaDB for retrieval,
matching the production pipeline.

Pipeline:
1. Sample 1000 tune + 1000 test jobs from jobs.db
2. Embed 150 synthetic positives and 30 resumes (cached)
3. Build ChromaDB collection from sampled jobs (cached)
4. For each resume:
   a. Swap positives in the collection (remove prev, add current)
   b. Dense retrieval via query_collection()
   c. Optional Cohere reranking
   d. Compute precision@5, recall@10
5. Aggregate metrics and write diagnostic outputs
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
from dotenv import load_dotenv

from config import (
    DB_DEFAULT_PATH,
    HNSW_EF,
    RETRIEVE_TOP_K,
    RERANK_TOP_N,
    VOYAGE_MODEL,
)
from embedding import create_client
from eval.collection import get_or_build_tune_collection, swap_positives
from eval.embedding_cache import embed_positives, embed_resumes
from eval.eval_config import (
    K_PRECISION,
    K_RECALL,
    RESULTS_DIR,
    TUNE_POSITIVES_PATH,
    TUNE_RESUMES_PATH,
)
from eval.data_loading import sample_jobs
from eval.metrics import batch_compute_metrics_at_k, compute_metrics_at_k
from eval.reporting import write_missed_positives_csv, write_results_json
from eval.types import PositiveRetrievalStatus, ResumeEvalResult
from reranking import rerank_jobs
from retrieval import JobResult, query_collection

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tune set evaluation for Resume-Job Matching System"
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


def evaluate_resume(
    resume_row: pd.Series,
    resume_positives_df: pd.DataFrame,
    resume_embeddings: dict[int, np.ndarray],
    positive_embeddings: dict[str, np.ndarray],
    positives_df: pd.DataFrame,
    collection: chromadb.Collection,
    voyage_client,
    cohere_api_key: Optional[str],
    skip_rerank: bool,
) -> Optional[ResumeEvalResult]:
    """
    Evaluate a single resume against the collection.

    Returns ResumeEvalResult or None if evaluation fails.
    """
    resume_id = int(resume_row["id"])
    logger.info(
        f"Evaluating resume {resume_id} ({resume_row['seniority']}/{resume_row['domain']})"
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

        # Classify embedding hits
        positive_chroma_ids = set(current_positive_ids)
        statuses = classify_positive_retrieval(retrieved, positive_chroma_ids)

        # Reranking (optional)
        reranked = retrieved
        if not skip_rerank and cohere_api_key:
            reranked = rerank_jobs(
                query=resume_row["resume"],
                jobs=retrieved,
                top_n=RERANK_TOP_N,
                api_key=cohere_api_key,
            )

        # Classify reranker outcome
        statuses = classify_reranker_outcome(statuses, reranked, skip_rerank)

        # Build final ranked list (from reranked or retrieved)
        final_ranked = [job["id"] for job in reranked]

        # Compute metrics
        relevant_ids = set(current_positive_ids)
        metrics = compute_metrics_at_k(
            retrieved_ids=final_ranked,
            relevant_ids=relevant_ids,
            k_values=[K_PRECISION, K_RECALL],
        )

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
        logger.error(f"Error evaluating resume {resume_id}: {e}", exc_info=True)
        return None


def main() -> None:
    """Main orchestration function."""
    setup_logging()
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Resume-Job Matching: Tune Set Evaluation")
    logger.info("=" * 80)

    # Setup
    voyage_api_key, cohere_api_key = load_env()
    skip_rerank = args.skip_rerank or not cohere_api_key
    if not cohere_api_key and not args.skip_rerank:
        logger.warning("Cohere API key missing; reranking disabled")

    # Create results directory
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    voyage_client = create_client(voyage_api_key)

    try:
        # Load tune data
        logger.info("Loading tune data")
        resumes_df = pd.read_csv(TUNE_RESUMES_PATH)
        positives_df = pd.read_csv(TUNE_POSITIVES_PATH)
        logger.info(f"Loaded {len(resumes_df)} resumes and {len(positives_df)} positives")

        # Sample jobs
        tune_jobs_df, _ = sample_jobs(
            args.db_path, force=args.force_resample
        )

        # Embed positives and resumes
        positive_embeddings = embed_positives(
            voyage_client, positives_df
        )
        resume_embeddings = embed_resumes(
            voyage_client, resumes_df
        )

        # Build collection
        collection = get_or_build_tune_collection(
            tune_jobs_df, args.db_path, force_rebuild=args.force_resample
        )

        # Evaluation loop
        logger.info("Starting per-resume evaluation loop")
        all_results: list[ResumeEvalResult] = []

        for _, resume_row in resumes_df.iterrows():
            result = evaluate_resume(
                resume_row,
                positives_df,
                resume_embeddings,
                positive_embeddings,
                positives_df,
                collection,
                voyage_client,
                cohere_api_key,
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
        write_results_json(all_results, batch_metrics, skip_rerank)
        write_missed_positives_csv(all_results, positives_df)

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
