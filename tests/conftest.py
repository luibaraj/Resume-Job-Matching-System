import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.database import DatabaseManager


@pytest.fixture
def sample_job_description() -> str:
    """Load a realistic HTML job description from data/sample_jobs.json.

    Returns the 'content' field from the first job in the sample file.
    """
    sample_file = Path(__file__).parent.parent / "data" / "sample_jobs.json"
    with open(sample_file) as f:
        data = json.load(f)
    return data["jobs"][0]["content"]


@pytest.fixture
def temp_db() -> str:
    """Create a temporary SQLite database for testing.

    Yields the path to the temp database, cleans up after test.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    try:
        Path(db_path).unlink()
    except Exception:
        pass


@pytest.fixture
def db_manager(temp_db: str) -> DatabaseManager:
    """Initialize and return a DatabaseManager with a temp database.

    Initializes the schema so tables exist.
    """
    manager = DatabaseManager(temp_db)
    manager.initialize_schema()
    return manager


@pytest.fixture
def raw_greenhouse_job() -> dict:
    """Return a raw Greenhouse API job dict (matches sample_jobs.json shape).

    Used by collection tests that need a realistic raw API payload.
    """
    return {
        "id": 4001234,
        "title": "Software Engineer, Backend",
        "updated_at": "2026-02-28T18:00:00-05:00",
        "location": {"name": "San Francisco, CA or Remote"},
        "content": "<div><p>We&apos;re hiring.</p><ul><li>Python</li></ul></div>",
        "absolute_url": "https://boards.greenhouse.io/example-company/jobs/4001234",
        "departments": [{"id": 10, "name": "Engineering"}],
        "offices": [{"id": 22, "name": "San Francisco"}],
        "metadata": [],
        "board_token": "example-company",
    }


@pytest.fixture
def normalized_job() -> dict:
    """Return a normalized job dict in DatabaseManager.insert_job() format.

    Used by database tests to insert rows without depending on collection.
    """
    return {
        "greenhouse_id": 4001234,
        "board_token": "example-company",
        "title": "Software Engineer, Backend",
        "company": "example-company",
        "location": "San Francisco, CA or Remote",
        "raw_description": "<div><p>We&apos;re hiring.</p><ul><li>Python</li></ul></div>",
        "absolute_url": "https://boards.greenhouse.io/example-company/jobs/4001234",
        "updated_at_source": "2026-02-28T18:00:00-05:00",
        "departments": '["Engineering"]',
        "offices": '["San Francisco"]',
        "collected_at": "2026-03-05T12:00:00Z",
    }


@pytest.fixture
def duplicate_normalized_job(normalized_job) -> dict:
    """Return a second job dict that produces the same job_hash as normalized_job.

    Used to assert that insert_job() returns False on duplicate.
    """
    return dict(normalized_job)
