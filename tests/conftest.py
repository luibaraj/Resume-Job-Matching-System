"""Shared pytest fixtures for test suite."""

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from fastapi_app.app.main import create_app
from fastapi_app.app.services.matching_service import MatchingService

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def tmp_db():
    """
    In-memory SQLite database with jobs table schema.

    Creates the jobs table with columns matching the pipeline:
    - id (PRIMARY KEY)
    - title, description, location (job metadata)
    - url, greenhouse_id (source tracking)
    - board_token (job board identifier)
    - created_at (timestamp)
    - embedding, embedded (optional columns added by pipeline)
    - cleaned_description, preprocessed (optional columns added by pipeline)

    Yields:
        sqlite3.Connection: in-memory connection with schema initialized
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            location TEXT,
            url TEXT NOT NULL,
            greenhouse_id TEXT,
            board_token TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    yield conn
    conn.close()


@pytest.fixture
def sample_job_row():
    """
    Pre-built job row dict for testing.

    Returns a minimal valid job record that can be inserted into the jobs table.
    Includes all required fields plus optional metadata.

    Returns:
        dict: job record with keys matching jobs table columns
    """
    return {
        "id": 1,
        "title": "Senior Software Engineer",
        "description": (
            "We are looking for a Senior Software Engineer to join our team. "
            "You will work on backend services and APIs. "
            "Requirements: 5+ years Python, experience with AWS, familiarity with Docker."
        ),
        "location": "San Francisco, CA",
        "url": "https://example.com/jobs/1",
        "greenhouse_id": "12345",
        "board_token": "example-board",
        "created_at": "2026-04-01T10:00:00Z",
    }


@pytest.fixture
def multiple_job_rows():
    """
    Multiple job rows for testing batch operations.

    Returns:
        list[dict]: 3 job records with varying titles, descriptions, and seniority levels
    """
    return [
        {
            "id": 1,
            "title": "Senior Software Engineer",
            "description": "5+ years Python, AWS, Docker. Backend services.",
            "location": "San Francisco, CA",
            "url": "https://example.com/jobs/1",
            "greenhouse_id": "12345",
            "board_token": "board-a",
            "created_at": "2026-04-01T10:00:00Z",
        },
        {
            "id": 2,
            "title": "Junior Data Analyst",
            "description": "SQL, Python basics, Excel. Data visualization with Tableau.",
            "location": "New York, NY",
            "url": "https://example.com/jobs/2",
            "greenhouse_id": "12346",
            "board_token": "board-b",
            "created_at": "2026-04-01T11:00:00Z",
        },
        {
            "id": 3,
            "title": "Mid-level DevOps Engineer",
            "description": "Kubernetes, Terraform, CI/CD pipelines. 3+ years infrastructure.",
            "location": "Seattle, WA",
            "url": "https://example.com/jobs/3",
            "greenhouse_id": "12347",
            "board_token": "board-c",
            "created_at": "2026-04-01T12:00:00Z",
        },
    ]


# FastAPI testing fixtures
@pytest.fixture
def mock_matching_service():
    """MagicMock of MatchingService."""
    mock = MagicMock(spec=MatchingService)
    mock.match.return_value = {
        "matches": [],
        "total_candidates": 0,
        "total_reranked": 0,
        "filters_applied": None,
        "run_id": "test-run-id"
    }
    return mock

@pytest.fixture
def fastapi_test_client(mock_matching_service):
    """FastAPI TestClient with overridden dependencies."""
    from fastapi_app.app.dependencies import get_matching_service
    
    def override_get_matching_service():
        yield mock_matching_service
    
    app = create_app()
    app.dependency_overrides[get_matching_service] = override_get_matching_service
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()

@pytest.fixture
def sample_resume_text():
    """Short resume string."""
    return "Senior Software Engineer with 5+ years experience in Python, Django, and AWS. BS in Computer Science."

@pytest.fixture
def sample_job_result():
    """Dict matching JobResult schema."""
    return {
        "id": 123,
        "title": "Senior Backend Engineer",
        "location": "Remote",
        "company_name": "Tech Corp",
        "board_token": "example-board",
        "source_url": "https://example.com/job/123",
        "min_years_experience": 5,
        "distance": 0.1,
        "rerank_score": 0.95,
        "explanation": "Strong match with Python and backend experience."
    }

@pytest.fixture
def sample_match_response(sample_job_result):
    """Dict matching MatchResponse schema."""
    return {
        "matches": [sample_job_result],
        "total_candidates": 10,
        "total_reranked": 5,
        "filters_applied": {"degree": 1, "seniority": 2, "years": 5},
        "run_id": "test-run-id"
    }
