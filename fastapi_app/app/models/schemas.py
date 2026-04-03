"""
Pydantic schemas for request/response validation.
"""
from typing import List, Optional
from pydantic import BaseModel, HttpUrl


class ResumeRequest(BaseModel):
    """Request body for resume matching."""
    resume_text: str
    top_k: Optional[int] = 10
    use_filters: Optional[bool] = True
    include_explanations: Optional[bool] = True


class JobResult(BaseModel):
    """Single job match result."""
    id: int
    title: str
    location: Optional[str]
    company_name: Optional[str]
    board_token: str
    source_url: Optional[HttpUrl]
    min_years_experience: Optional[int] = 0
    distance: Optional[float] = None
    rerank_score: Optional[float] = None
    explanation: Optional[str] = None


class MatchResponse(BaseModel):
    """Response containing matched jobs."""
    matches: List[JobResult]
    total_candidates: int
    total_reranked: int
    filters_applied: Optional[dict] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ollama_available: bool
    database_available: bool
    chroma_collection_count: int
