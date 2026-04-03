"""
Tests for Pydantic schemas.
"""
import pytest
from pydantic import ValidationError
from fastapi_app.app.models.schemas import ResumeRequest, JobResult, MatchResponse, HealthResponse

def test_resume_request_validation():
    """resume_text required"""
    # Valid request
    valid = ResumeRequest(resume_text="Valid resume text")
    assert valid.resume_text == "Valid resume text"
    assert valid.top_k == 10  # default
    assert valid.use_filters == True  # default
    assert valid.include_explanations == True  # default
    
    # Missing resume_text
    with pytest.raises(ValidationError):
        ResumeRequest()
    
    # Empty resume_text
    with pytest.raises(ValidationError):
        ResumeRequest(resume_text="")

def test_job_result_validation():
    """id, title, board_token required"""
    # Valid job result
    valid = JobResult(
        id=123,
        title="Software Engineer",
        board_token="example-board"
    )
    assert valid.id == 123
    assert valid.title == "Software Engineer"
    assert valid.board_token == "example-board"
    
    # With optional fields
    valid_with_optional = JobResult(
        id=123,
        title="Software Engineer",
        board_token="example-board",
        location="Remote",
        company_name="Tech Corp",
        source_url="https://example.com/job",
        min_years_experience=3,
        distance=0.1,
        rerank_score=0.9,
        explanation="Good match"
    )
    assert valid_with_optional.location == "Remote"
    assert valid_with_optional.explanation == "Good match"
    
    # Missing required fields
    with pytest.raises(ValidationError):
        JobResult(title="Job", board_token="board")  # Missing id
    
    with pytest.raises(ValidationError):
        JobResult(id=123, board_token="board")  # Missing title
    
    with pytest.raises(ValidationError):
        JobResult(id=123, title="Job")  # Missing board_token

def test_match_response_validation():
    """matches list, total_candidates, total_reranked required"""
    # Valid response
    job = JobResult(
        id=123,
        title="Job",
        board_token="board"
    )
    
    valid = MatchResponse(
        matches=[job],
        total_candidates=10,
        total_reranked=5
    )
    
    assert len(valid.matches) == 1
    assert valid.total_candidates == 10
    assert valid.total_reranked == 5
    
    # With filters_applied
    with_filters = MatchResponse(
        matches=[job],
        total_candidates=10,
        total_reranked=5,
        filters_applied={"degree": 1, "seniority": 2}
    )
    assert with_filters.filters_applied == {"degree": 1, "seniority": 2}
    
    # Empty matches list is valid
    empty = MatchResponse(matches=[], total_candidates=0, total_reranked=0)
    assert empty.matches == []
    
    # Missing required fields
    with pytest.raises(ValidationError):
        MatchResponse(matches=[], total_reranked=0)  # Missing total_candidates
    
    with pytest.raises(ValidationError):
        MatchResponse(matches=[], total_candidates=0)  # Missing total_reranked

def test_health_response_validation():
    """status, ollama_available, database_available, chroma_collection_count required"""
    valid = HealthResponse(
        status="healthy",
        ollama_available=True,
        database_available=True,
        chroma_collection_count=5
    )
    
    assert valid.status == "healthy"
    assert valid.ollama_available == True
    assert valid.database_available == True
    assert valid.chroma_collection_count == 5
    
    # Missing required fields
    with pytest.raises(ValidationError):
        HealthResponse(
            ollama_available=True,
            database_available=True,
            chroma_collection_count=5
        )  # Missing status
