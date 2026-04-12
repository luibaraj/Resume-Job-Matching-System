"""
Precision@k and recall@k metrics for retrieval evaluation.

Provides functions to compute ranking quality metrics for the resume-job
matching pipeline. All functions are pure and require no ML framework.
"""

from typing import TypedDict

__all__ = [
    "MetricsAtK",
    "BatchMetricsAtK",
    "precision_at_k",
    "recall_at_k",
    "compute_metrics_at_k",
    "batch_precision_at_k",
    "batch_recall_at_k",
    "batch_compute_metrics_at_k",
]


class MetricsAtK(TypedDict):
    """Precision and recall scores for a single query at multiple k values."""

    k_values: list[int]           # The k values that were evaluated
    precision: dict[int, float]   # precision@k for each k; key is k
    recall: dict[int, float]      # recall@k for each k; key is k


class BatchMetricsAtK(TypedDict):
    """Mean precision and recall scores across a batch of queries at multiple k values."""

    k_values: list[int]           # The k values that were evaluated
    mean_precision: dict[int, float]  # mean precision@k across all queries
    mean_recall: dict[int, float]     # mean recall@k across all queries
    num_queries: int              # Number of queries included in the batch


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Compute precision@k for a single query.

    Precision@k = |relevant ∩ top-k retrieved| / k.

    Args:
        retrieved_ids: Ranked list of retrieved item IDs (index 0 = highest rank).
        relevant_ids: Set of ground-truth relevant item IDs.
        k: Cut-off rank. Must be >= 1.

    Returns:
        Precision@k as a float in [0.0, 1.0].
        Returns 0.0 if retrieved_ids is empty or k == 0.
        If k > len(retrieved_ids), the actual retrieved list length is used as
        the denominator (no padding with non-relevant items is assumed).

    Raises:
        ValueError: If k < 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    if not retrieved_ids or not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0

    hits = sum(1 for id in top_k if id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Compute recall@k for a single query.

    Recall@k = |relevant ∩ top-k retrieved| / |relevant|.

    Args:
        retrieved_ids: Ranked list of retrieved item IDs (index 0 = highest rank).
        relevant_ids: Set of ground-truth relevant item IDs.
        k: Cut-off rank. Must be >= 1.

    Returns:
        Recall@k as a float in [0.0, 1.0].
        Returns 0.0 if retrieved_ids is empty or relevant_ids is empty.
        If k > len(retrieved_ids), all retrieved items are considered.

    Raises:
        ValueError: If k < 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    if not relevant_ids or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for id in top_k if id in relevant_ids)
    return hits / len(relevant_ids)


def compute_metrics_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k_values: list[int],
) -> MetricsAtK:
    """
    Compute precision@k and recall@k for a single query across multiple k values.

    Args:
        retrieved_ids: Ranked list of retrieved item IDs (index 0 = highest rank).
        relevant_ids: Set of ground-truth relevant item IDs.
        k_values: List of k cut-offs to evaluate (e.g., [1, 5, 10]).
                  Values must all be >= 1 and are evaluated independently.

    Returns:
        MetricsAtK with precision and recall dicts keyed by k.

    Raises:
        ValueError: If k_values is empty or any k < 1.
    """
    if not k_values:
        raise ValueError("k_values must not be empty")

    precision_scores = {}
    recall_scores = {}

    for k in k_values:
        precision_scores[k] = precision_at_k(retrieved_ids, relevant_ids, k)
        recall_scores[k] = recall_at_k(retrieved_ids, relevant_ids, k)

    return MetricsAtK(
        k_values=k_values,
        precision=precision_scores,
        recall=recall_scores,
    )


def batch_precision_at_k(
    batch_retrieved_ids: list[list[str]],
    batch_relevant_ids: list[set[str]],
    k: int,
) -> float:
    """
    Compute mean precision@k across a batch of queries.

    Args:
        batch_retrieved_ids: One ranked list of retrieved IDs per query.
        batch_relevant_ids: One set of relevant IDs per query.
            Must be the same length as batch_retrieved_ids.
        k: Cut-off rank. Must be >= 1.

    Returns:
        Mean precision@k as a float in [0.0, 1.0].
        Returns 0.0 if the batch is empty.

    Raises:
        ValueError: If k < 1 or lengths of the two batch lists differ.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    if len(batch_retrieved_ids) != len(batch_relevant_ids):
        raise ValueError(
            f"batch_retrieved_ids and batch_relevant_ids must have the same length, "
            f"got {len(batch_retrieved_ids)} and {len(batch_relevant_ids)}"
        )

    if not batch_retrieved_ids:
        return 0.0

    scores = [
        precision_at_k(retrieved, relevant, k)
        for retrieved, relevant in zip(batch_retrieved_ids, batch_relevant_ids)
    ]
    return sum(scores) / len(scores)


def batch_recall_at_k(
    batch_retrieved_ids: list[list[str]],
    batch_relevant_ids: list[set[str]],
    k: int,
) -> float:
    """
    Compute mean recall@k across a batch of queries.

    Args:
        batch_retrieved_ids: One ranked list of retrieved IDs per query.
        batch_relevant_ids: One set of relevant IDs per query.
            Must be the same length as batch_retrieved_ids.
        k: Cut-off rank. Must be >= 1.

    Returns:
        Mean recall@k as a float in [0.0, 1.0].
        Returns 0.0 if the batch is empty.

    Raises:
        ValueError: If k < 1 or lengths of the two batch lists differ.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    if len(batch_retrieved_ids) != len(batch_relevant_ids):
        raise ValueError(
            f"batch_retrieved_ids and batch_relevant_ids must have the same length, "
            f"got {len(batch_retrieved_ids)} and {len(batch_relevant_ids)}"
        )

    if not batch_retrieved_ids:
        return 0.0

    scores = [
        recall_at_k(retrieved, relevant, k)
        for retrieved, relevant in zip(batch_retrieved_ids, batch_relevant_ids)
    ]
    return sum(scores) / len(scores)


def batch_compute_metrics_at_k(
    batch_retrieved_ids: list[list[str]],
    batch_relevant_ids: list[set[str]],
    k_values: list[int],
) -> BatchMetricsAtK:
    """
    Compute mean precision@k and recall@k across a batch of queries for multiple k values.

    Args:
        batch_retrieved_ids: One ranked list of retrieved IDs per query.
        batch_relevant_ids: One set of relevant IDs per query.
            Must be the same length as batch_retrieved_ids.
        k_values: List of k cut-offs to evaluate (e.g., [1, 5, 10]).

    Returns:
        BatchMetricsAtK with mean_precision and mean_recall dicts keyed by k,
        and num_queries reflecting the batch size.

    Raises:
        ValueError: If k_values is empty, any k < 1, or batch lengths differ.
    """
    if not k_values:
        raise ValueError("k_values must not be empty")

    if len(batch_retrieved_ids) != len(batch_relevant_ids):
        raise ValueError(
            f"batch_retrieved_ids and batch_relevant_ids must have the same length, "
            f"got {len(batch_retrieved_ids)} and {len(batch_relevant_ids)}"
        )

    mean_precision_scores = {}
    mean_recall_scores = {}

    for k in k_values:
        mean_precision_scores[k] = batch_precision_at_k(
            batch_retrieved_ids, batch_relevant_ids, k
        )
        mean_recall_scores[k] = batch_recall_at_k(
            batch_retrieved_ids, batch_relevant_ids, k
        )

    return BatchMetricsAtK(
        k_values=k_values,
        mean_precision=mean_precision_scores,
        mean_recall=mean_recall_scores,
        num_queries=len(batch_retrieved_ids),
    )
