from pydantic import BaseModel, Field

class MatchRequest(BaseModel):
    resume: str = Field(..., min_length=50)
    top_k: int = Field(default=10, ge=1, le=50)
    min_years_experience: int | None = Field(default=None, ge=0)
    seniority_level: int | None = Field(default=None, ge=0, le=3)
    required_degree: int | None = Field(default=None, ge=0, le=3)

class JobMatch(BaseModel):
    job_id: int
    title: str
    company_name: str = ""
    score: float
    explanation: str | None = None
    absolute_url: str | None = None

class MatchResponse(BaseModel):
    matches: list[JobMatch]
    resume_id: str | None = None  # hash of resume for caching later
    corpus_warning: str | None = None  # NEW: warning about corpus limitations
