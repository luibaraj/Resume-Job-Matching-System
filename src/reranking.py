"""
Reranking module using Cohere Rerank 3 to reorder retrieved job results by relevance.
"""

import logging
import os
import random
import time

import cohere

from src.config import (
    COHERE_RERANK_MODEL,
    RERANK_TOP_N,
    RERANK_MAX_RETRIES,
    RERANK_RETRY_BASE_DELAY,
    RERANK_INTER_REQUEST_DELAY,
)
from src.retrieval import JobResult

logger = logging.getLogger(__name__)


def create_rerank_client(api_key: str) -> cohere.ClientV2:
    """
    Create and return a Cohere V2 client.

    Args:
        api_key: The Cohere API key (COHERE_API_KEY).

    Returns:
        An authenticated cohere.ClientV2 instance.

    Raises:
        ValueError: If api_key is empty or None.
    """
    if not api_key:
        raise ValueError("COHERE_API_KEY must be set and non-empty")
    return cohere.ClientV2(api_key=api_key)


_SENIORITY_LABEL: dict[int, str] = {1: "Entry-level", 2: "Mid-level", 3: "Senior", 4: "Staff"}


def _format_document(job: JobResult) -> str:
    """Format a JobResult into a single string for Cohere reranking."""
    seniority_label = _SENIORITY_LABEL.get(job["seniority_level"], "")
    header = f"{job['title']} | {job['location']}"
    if seniority_label:
        header = f"{header} | {seniority_label}"
    return f"{header}\n{job['cleaned_description']}"


def rerank_jobs(
    query: str,
    jobs: list[JobResult],
    top_n: int = RERANK_TOP_N,
    api_key: str | None = None,
    client: cohere.ClientV2 | None = None,
    max_retries: int = RERANK_MAX_RETRIES,
    retry_base_delay: float = RERANK_RETRY_BASE_DELAY,
    run_id: str | None = None,
) -> list[JobResult]:
    """
    Rerank a list of JobResult dicts using Cohere Rerank 3.

    Retries on transient errors (rate limits, network errors) with exponential
    back-off. Raises on permanent failure after all retries are exhausted.

    Args:
        query: The user's resume or query text.
        jobs: Retrieved job results from query_collection().
        top_n: Number of top results to return after reranking.
        api_key: Cohere API key. Falls back to COHERE_API_KEY env var.
        client: Pre-instantiated cohere.ClientV2. Takes precedence over api_key.
        max_retries: Maximum number of retry attempts on transient errors.
        retry_base_delay: Base delay in seconds for exponential back-off.
        run_id: Optional trace ID for request tracing.

    Returns:
        Reranked list of JobResult, length <= top_n, best first.

    Raises:
        Exception: If all retries fail.
    """
    if not jobs:
        return []

    co = client or cohere.ClientV2(
        api_key=api_key or os.environ["COHERE_API_KEY"]
    )
    documents = [_format_document(job) for job in jobs]

    attempt = 0
    while True:
        try:
            response = co.rerank(
                model=COHERE_RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=min(top_n, len(jobs)),
            )
            return [jobs[result.index] for result in response.results]
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                if run_id:
                    logger.error(
                        "rerank_jobs (run_id=%s) failed after %d attempts: %s (%s)",
                        run_id,
                        max_retries,
                        type(exc).__name__,
                        exc,
                    )
                else:
                    logger.error(
                        "rerank_jobs failed after %d attempts: %s (%s)",
                        max_retries,
                        type(exc).__name__,
                        exc,
                    )
                raise
            delay = retry_base_delay * (2 ** (attempt - 1))
            # Add jitter: ±10% of delay
            delay += random.uniform(0, delay * 0.1)
            is_429 = "429" in str(exc) or "Too Many Requests" in str(exc)
            level = "rate-limited (429)" if is_429 else type(exc).__name__
            logger.warning(
                "rerank_jobs attempt %d/%d (%s). Retrying in %.1fs...",
                attempt,
                max_retries,
                level,
                delay,
            )
            time.sleep(delay)


def batch_rerank_jobs(
    queries_and_jobs: list[tuple[str, list[JobResult]]],
    top_n: int = RERANK_TOP_N,
    api_key: str | None = None,
    client: cohere.ClientV2 | None = None,
    inter_request_delay: float = RERANK_INTER_REQUEST_DELAY,
    max_retries: int = RERANK_MAX_RETRIES,
    retry_base_delay: float = RERANK_RETRY_BASE_DELAY,
) -> list[list[JobResult]]:
    """
    Rerank multiple (query, jobs) pairs sequentially with throttling.

    Processes each pair with inter-request delays to avoid rate limiting.

    Args:
        queries_and_jobs: List of (query_string, job_list) pairs.
        top_n: Number of top results per rerank call.
        api_key: Cohere API key. Falls back to COHERE_API_KEY env var.
        client: Pre-instantiated cohere.ClientV2. Takes precedence over api_key.
        inter_request_delay: Seconds to sleep between successive API calls.
        max_retries: Max retry attempts per call.
        retry_base_delay: Base delay for exponential back-off.

    Returns:
        List of reranked JobResult lists, one per input pair, in the same order.
    """
    co = client or cohere.ClientV2(
        api_key=api_key or os.environ["COHERE_API_KEY"]
    )
    results = []
    for i, (query, jobs) in enumerate(queries_and_jobs):
        reranked = rerank_jobs(
            query=query,
            jobs=jobs,
            top_n=top_n,
            client=co,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
        )
        results.append(reranked)
        if i < len(queries_and_jobs) - 1:
            time.sleep(inter_request_delay)
    return results
