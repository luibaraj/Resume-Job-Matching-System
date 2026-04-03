from pydantic import BaseModel, Field

class MatchRequest(BaseModel):
    resume: str = Field(..., min_length=50)
    top_k: int = Field(default=10, ge=1, le=50)

class JobMatch(BaseModel):
    job_id: int
    title: str
    score: float
    explanation: str | None = None

class MatchResponse(BaseModel):
    matches: list[JobMatch]
    resume_id: str | None = None  # hash of resume for caching later
