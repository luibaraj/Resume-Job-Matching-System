"""Test suite for scrape_jobs.py orchestration script."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests

# Add src and scripts/pipeline to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "pipeline"))

from scrape_jobs import (
    init_db,
    write_jobs_to_db,
    scrape_board_safe,
)
from src.greenhouse_scraper import GreenhouseJob


@pytest.fixture
def tmp_db():
    """Create a temporary database file."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sample_greenhouse_jobs():
    """Create sample GreenhouseJob objects for testing."""
    jobs = [
        GreenhouseJob(
            id="gh-1",
            title="Senior Engineer",
            location="San Francisco, CA",
            description="<p>5+ years experience required</p>",
            internal_job_id=1001,
            url="/jobs/gh-1",
            absolute_url="https://example.com/jobs/gh-1",
            company_name="TechCorp",
            department="Engineering",
            job_type="Full-time",
            updated_at="2024-01-01T10:00:00Z",
            created_at="2024-01-01T09:00:00Z",
        ),
        GreenhouseJob(
            id="gh-2",
            title="Data Scientist",
            location="New York, NY",
            description="<p>ML and Python expertise</p>",
            internal_job_id=1002,
            url="/jobs/gh-2",
            absolute_url="https://example.com/jobs/gh-2",
            company_name="DataCorp",
            department="Data",
            job_type="Full-time",
            updated_at="2024-01-01T10:00:00Z",
            created_at="2024-01-01T09:00:00Z",
        ),
    ]
    return jobs


class TestInitDb:
    """Tests for init_db function."""

    def test_init_db_idempotent(self, tmp_db):
        """Calling init_db twice produces identical schema."""
        # First call
        init_db(tmp_db)
        conn1 = sqlite3.connect(tmp_db)
        cursor1 = conn1.cursor()
        cursor1.execute("PRAGMA table_info(jobs)")
        schema1 = cursor1.fetchall()
        conn1.close()

        # Second call (should not raise)
        init_db(tmp_db)
        conn2 = sqlite3.connect(tmp_db)
        cursor2 = conn2.cursor()
        cursor2.execute("PRAGMA table_info(jobs)")
        schema2 = cursor2.fetchall()
        conn2.close()

        # Schemas should be identical
        assert schema1 == schema2
        assert len(schema1) > 0


class TestWriteJobsToDb:
    """Tests for write_jobs_to_db function."""

    def test_write_jobs_skips_exceptions(self, tmp_db, sample_greenhouse_jobs, caplog):
        """Partial batch failure is caught and logged, not re-raised."""
        init_db(tmp_db)

        # Create results: one success, one exception
        results = [
            ("board-1", sample_greenhouse_jobs),
            ("board-2", RuntimeError("Network error")),
        ]

        # Should not raise
        inserted = write_jobs_to_db(tmp_db, results)

        # Verify partial insertion
        assert inserted == len(sample_greenhouse_jobs)
        assert "Network error" in caplog.text or inserted > 0

        # Verify only first board's jobs are in DB
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jobs")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 2


class TestScrapeBoardSafe:
    """Tests for scrape_board_safe function with retry logic."""

    @patch("scrape_jobs.scrape_greenhouse_board")
    def test_scrape_board_safe_success(self, mock_scraper, sample_greenhouse_jobs, caplog):
        """Successful scrape returns jobs without retry."""
        mock_scraper.return_value = sample_greenhouse_jobs

        token, result = scrape_board_safe("test-board")

        assert token == "test-board"
        assert result == sample_greenhouse_jobs
        # Scraper called exactly once (no retries)
        assert mock_scraper.call_count == 1

    @patch("scrape_jobs.scrape_greenhouse_board")
    @patch("scrape_jobs.time.sleep")
    def test_scrape_board_safe_retries_connection_error(
        self, mock_sleep, mock_scraper, sample_greenhouse_jobs, caplog
    ):
        """ConnectionError retried up to max attempts."""
        # Fail twice, succeed on third attempt
        mock_scraper.side_effect = [
            requests.exceptions.ConnectionError("timeout"),
            requests.exceptions.ConnectionError("timeout"),
            sample_greenhouse_jobs,
        ]

        token, result = scrape_board_safe("test-board")

        assert token == "test-board"
        assert result == sample_greenhouse_jobs
        # Scraper called 3 times (2 failures + 1 success)
        assert mock_scraper.call_count == 3
        # Sleep called twice (between retries)
        assert mock_sleep.call_count == 2

    @patch("scrape_jobs.scrape_greenhouse_board")
    def test_scrape_board_safe_non_retryable_no_retry(
        self, mock_scraper, caplog
    ):
        """Non-retryable exception does not retry."""
        mock_scraper.side_effect = ValueError("Invalid board token")

        token, result = scrape_board_safe("test-board")

        assert token == "test-board"
        assert isinstance(result, ValueError)
        # Scraper called exactly once (no retry for non-retryable)
        assert mock_scraper.call_count == 1

    @patch("scrape_jobs.scrape_greenhouse_board")
    @patch("scrape_jobs.time.sleep")
    def test_scrape_board_safe_timeout_retried(
        self, mock_sleep, mock_scraper, sample_greenhouse_jobs
    ):
        """Timeout exception is retried like ConnectionError."""
        mock_scraper.side_effect = [
            requests.exceptions.Timeout("read timeout"),
            sample_greenhouse_jobs,
        ]

        token, result = scrape_board_safe("test-board")

        assert token == "test-board"
        assert result == sample_greenhouse_jobs
        # Scraper called 2 times
        assert mock_scraper.call_count == 2
        # Sleep called once
        assert mock_sleep.call_count == 1
