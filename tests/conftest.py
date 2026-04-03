"""Shared pytest fixtures for test suite."""

import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from fastapi_app.app.main import create_app
from fastapi_app.app.services.matching_service import MatchingService


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
def mock_embedding_service():
    """Mock EmbeddingService that returns dummy embeddings."""
    mock_service = MagicMock()
    # Create a dummy embedding vector (1024-dim zeros)
    dummy_embedding = np.zeros(1024, dtype=np.float32)
    mock_service.load_or_embed_resume.return_value = [dummy_embedding]
    return mock_service


@pytest.fixture
def mock_matching_service(mock_embedding_service):
    """MagicMock of MatchingService with mocked dependencies."""
    mock = MagicMock(spec=MatchingService)
    mock.match.return_value = {
        "matches": [],
        "total_candidates": 0,
        "total_reranked": 0,
        "filters_applied": None,
        "run_id": "test-run-id"
    }
    # Set the mocked embedding service
    mock.embedding_service = mock_embedding_service
    # Mock other services to avoid API calls
    mock.retrieval_service = MagicMock()
    mock.reranking_service = MagicMock()
    mock.generation_service = MagicMock()
    return mock

@pytest.fixture
def fastapi_test_client(mock_matching_service):
    """FastAPI TestClient with mocked MatchingService."""
    import os
    from fastapi_app.app.main import create_app
    # Import get_matching_service from the SAME location that routes.py uses
    from fastapi_app.app.api.routes import get_matching_service

    # Set dummy API keys to avoid authentication errors
    os.environ["VOYAGE_API_KEY"] = "dummy_key"
    os.environ["COHERE_API_KEY"] = "dummy_key"

    # Create app first with the real function
    app = create_app()

    # NOW replace the dependency using FastAPI's override mechanism
    # Use the SAME function reference that routes.py imported
    def mock_get_matching_service():
        yield mock_matching_service

    app.dependency_overrides[get_matching_service] = mock_get_matching_service

    try:
        # Create test client
        client = TestClient(app)
        client._mock_matching_service = mock_matching_service

        yield client
    finally:
        # Clean up overrides
        app.dependency_overrides.clear()

    # Clean up environment variables
    if "VOYAGE_API_KEY" in os.environ:
        del os.environ["VOYAGE_API_KEY"]
    if "COHERE_API_KEY" in os.environ:
        del os.environ["COHERE_API_KEY"]

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
