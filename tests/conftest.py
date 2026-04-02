"""Shared pytest fixtures for test suite."""

import sqlite3
import sys
from pathlib import Path

import pytest

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
