"""
Reranking module using Cohere Rerank 3 to reorder retrieved job results by relevance.
"""

import os
import cohere
from retrieval import JobResult
from config import COHERE_RERANK_MODEL, RERANK_TOP_N


def _format_document(job: JobResult) -> str:
    """Format a JobResult into a single string for Cohere reranking."""
    return f"{job['title']} | {job['location']}\n{job['cleaned_description']}"


def rerank_jobs(
    query: str,
    jobs: list[JobResult],
    top_n: int = RERANK_TOP_N,
    api_key: str | None = None,
) -> list[JobResult]:
    """
    Rerank a list of JobResult dicts using Cohere Rerank 3.

    Args:
        query: The user's resume or query text.
        jobs: Retrieved job results from query_collection().
        top_n: Number of top results to return after reranking.
        api_key: Cohere API key. Falls back to COHERE_API_KEY env var.

    Returns:
        Reranked list of JobResult, length <= top_n, best first.
    """
    if not jobs:
        return []

    key = api_key or os.environ["COHERE_API_KEY"]
    co = cohere.ClientV2(api_key=key)

    documents = [_format_document(job) for job in jobs]

    response = co.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=min(top_n, len(jobs)),
    )

    return [jobs[result.index] for result in response.results]
