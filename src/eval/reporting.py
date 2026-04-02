"""
Reporting utilities for evaluation results.

Serializes evaluation results to JSON and CSV formats for analysis.
"""

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

import mlflow
from eval import eval_config, metrics, types

logger = logging.getLogger(__name__)


def _sanitize_metric_name(name: str) -> str:
    """
    Sanitize a string for use in MLflow metric names.

    Replaces spaces and slashes with underscores to avoid file path issues.
    """
    return name.replace(" ", "_").replace("/", "_")


def write_results_json(
    results: list[types.ResumeEvalResult],
    batch_metrics: metrics.BatchMetricsAtK,
    skip_rerank: bool,
    output_path: str | None = None,
) -> None:
    """
    Write evaluation results to JSON with miss analysis.

    Args:
        results: List of per-resume evaluation results
        batch_metrics: Batch metrics dictionary
        skip_rerank: Whether reranking was skipped
        output_path: Path to write JSON to (default: TUNE_RESULTS_JSON)
    """
    if output_path is None:
        output_path = eval_config.TUNE_RESULTS_JSON
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
            "seed": eval_config.SAMPLE_SEED,
            "k_precision": eval_config.K_PRECISION,
            "k_recall": eval_config.K_RECALL,
            "skip_rerank": skip_rerank,
            "num_resumes": len(results),
            "num_sampled_tune_jobs": eval_config.TUNE_SAMPLE_N,
        },
        "aggregate": {
            "mean_precision_at_5": batch_metrics["mean_precision"][eval_config.K_PRECISION],
            "mean_recall_at_10": batch_metrics["mean_recall"][eval_config.K_RECALL],
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

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Wrote results to %s", output_path)

    # MLflow logging
    if mlflow.active_run():
        # Artifacts
        mlflow.log_artifact(str(output_path), artifact_path="results")

        # Aggregate metrics
        mlflow.log_metrics({
            f"mean_precision_at_{eval_config.K_PRECISION}": batch_metrics["mean_precision"][eval_config.K_PRECISION],
            f"mean_recall_at_{eval_config.K_RECALL}": batch_metrics["mean_recall"][eval_config.K_RECALL],
            "num_queries": float(batch_metrics["num_queries"]),
            "total_positives":     float(total_positives),
            "total_hits":          float(hits),
            "embedding_misses":    float(embedding_misses),
            "reranker_misses":     float(reranker_misses),
            "embedding_miss_rate": embedding_misses / total_positives if total_positives else 0.0,
            "reranker_miss_rate":  reranker_misses / total_positives if total_positives else 0.0,
        })

        # Per-domain and per-seniority miss rates (flat metrics)
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

        # Per-resume table (browsable in MLflow UI Artifacts tab)
        cols = ["resume_id", "seniority", "domain",
                f"precision_at_{eval_config.K_PRECISION}",
                f"recall_at_{eval_config.K_RECALL}", "num_positives"]
        rows = [
            [r["resume_id"], r["seniority"], r["domain"],
             r["precision_at_5"], r["recall_at_10"], r["num_positives"]]
            for r in results
        ]
        mlflow.log_table(data={"columns": cols, "data": rows},
                         artifact_file="results/per_resume_metrics.json")


def write_missed_positives_csv(
    results: list[types.ResumeEvalResult],
    positives_df: pd.DataFrame,
    output_path: str | None = None,
) -> None:
    """
    Write missed positives to CSV for analysis.

    Args:
        results: List of per-resume evaluation results
        positives_df: DataFrame of all positives (for lookup)
        output_path: Path to write CSV to (default: TUNE_MISSED_CSV)
    """
    if output_path is None:
        output_path = eval_config.TUNE_MISSED_CSV
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
    missed_df.to_csv(output_path, index=False)
    logger.info("Wrote %d missed positives to %s", len(missed_df), output_path)

    # MLflow logging
    if mlflow.active_run():
        mlflow.log_artifact(str(output_path), artifact_path="results")
        mlflow.log_metric("num_missed_positives", float(len(missed_df)))
