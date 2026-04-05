"""
TypedDict definitions shared across evaluation modules.

Houses the core TypedDicts used by evaluation logic and reporting:
- PositiveRetrievalStatus: tracks how a single synthetic positive was retrieved/ranked
- ResumeEvalResult: evaluation metrics and retrieval details for a single resume
- JobSkeleton: structured job posting data used in generation and validation
"""

from typing import Optional, TypedDict


class PositiveRetrievalStatus(TypedDict):
    """Status of a synthetic positive in the retrieval results."""

    positive_id: str  # UUID
    resume_id: int
    resume_seniority: str
    resume_domain: str
    positive_title: str
    positive_seniority: str
    positive_domain: str
    primary_skills: list[str]
    embedding_rank: Optional[int]
    embedding_hit: bool
    rerank_rank: Optional[int]
    reranker_hit: Optional[bool]  # None if skip_rerank=True
    miss_type: str  # "hit" | "embedding_miss" | "reranker_miss"
    seniority_gap: bool
    domain_gap: bool


class ResumeEvalResult(TypedDict):
    """Evaluation result for a single resume."""

    resume_id: int
    seniority: str
    domain: str
    precision_at_5: float
    recall_at_10: float
    num_positives: int
    positives: list[PositiveRetrievalStatus]


class JobSkeleton(TypedDict):
    title: str
    seniority: str
    years_required: str
    domain: str
    primary_skills: list[str]
    secondary_skills: list[str]
    responsibilities: list[str]
