"""
Pydantic schemas for request/response validation.
"""
from typing import List, Optional
from pydantic import BaseModel, HttpUrl, field_validator


class ResumeRequest(BaseModel):
    """Request body for resume matching."""
    resume_text: str
    top_k: Optional[int] = 10
    use_filters: Optional[bool] = True
    include_explanations: Optional[bool] = True

    @field_validator('resume_text')
    @classmethod
    def resume_text_not_empty(cls, v):
        """Validate that resume_text is not empty."""
        if not v or not v.strip():
            raise ValueError('resume_text cannot be empty')
        return v


class JobResult(BaseModel):
    """Single job match result."""
    id: int
    title: str
    location: Optional[str] = None
    company_name: Optional[str] = None
    board_token: str
    source_url: Optional[HttpUrl] = None
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
