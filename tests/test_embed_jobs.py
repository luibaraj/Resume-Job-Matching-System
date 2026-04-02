"""Test suite for embed_jobs.py orchestration script."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

# Add src and scripts/pipeline to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "pipeline"))

from embed_jobs import run_embedding
from src.config import EMBEDDING_DIM
from src.embedding import serialize_embedding


@pytest.fixture
def tmp_db():
    """Create a temporary database with jobs schema."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT NOT NULL,
            board_token TEXT NOT NULL,
            title TEXT,
            location TEXT,
            description TEXT,
            source TEXT,
            source_url TEXT,
            company_name TEXT,
            department TEXT,
            job_type TEXT,
            scraped_at TEXT,
            updated_at TEXT,
            cleaned_description TEXT,
            preprocessed INTEGER DEFAULT 0,
            embedding BLOB,
            embedded INTEGER DEFAULT 0,
            UNIQUE(external_id, board_token)
        )
    """)
    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def mock_voyage_client():
    """Create a mock Voyage AI client."""
    return MagicMock()


def _insert_job(db_path, external_id, cleaned_description, embedded=0, embedding=None):
    """Helper to insert a test job into the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO jobs
        (external_id, board_token, title, cleaned_description, preprocessed, embedded, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (external_id, "test-board", "Test Job", cleaned_description, 1, embedded, embedding),
    )
    conn.commit()
    conn.close()


class TestRunEmbedding:
    """Tests for run_embedding function."""

    @patch("embed_jobs.create_client")
    @patch("embed_jobs.embed_batch")
    def test_run_embedding_adds_columns(self, mock_embed_batch, mock_create_client, tmp_db, caplog):
        """run_embedding adds embedding and embedded columns to jobs table."""
        # Insert unembedded jobs
        _insert_job(tmp_db, "job-1", "Software engineer position")
        _insert_job(tmp_db, "job-2", "Data analyst role")

        # Mock embeddings
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_embed_batch.return_value = [
            np.ones(EMBEDDING_DIM, dtype=np.float32),
            np.ones(EMBEDDING_DIM, dtype=np.float32),
        ]

        run_embedding(tmp_db, "mock-api-key")

        # Verify columns exist
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(jobs)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert "embedding" in columns
        assert "embedded" in columns

    @patch("embed_jobs.create_client")
    @patch("embed_jobs.embed_batch")
    def test_run_embedding_skips_empty_descriptions(
        self, mock_embed_batch, mock_create_client, tmp_db
    ):
        """Jobs with empty descriptions are not embedded (NULL or empty string)."""
        # Insert one with description, one empty, one NULL
        _insert_job(tmp_db, "job-1", "Valid job description")
        _insert_job(tmp_db, "job-2", "")  # Empty string
        _insert_job(tmp_db, "job-3", None)  # NULL

        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        # Only job-1 should be embedded
        mock_embed_batch.return_value = [np.ones(EMBEDDING_DIM, dtype=np.float32)]

        run_embedding(tmp_db, "mock-api-key")

        # Verify embed_batch was called with only 1 text (job-1)
        mock_embed_batch.assert_called()
        # Check that only 1 job is embedded
        call_args = mock_embed_batch.call_args
        texts = call_args[0][1]  # Second arg is the texts list
        assert len(texts) == 1

    @patch("embed_jobs.VOYAGE_BATCH_SIZE", 2)
    @patch("embed_jobs.create_client")
    @patch("embed_jobs.embed_batch")
    def test_run_embedding_bad_batch_skipped(
        self, mock_embed_batch, mock_create_client, tmp_db, caplog
    ):
        """When a batch fails, exception is caught and subsequent batches continue."""
        # Insert 3 jobs; with VOYAGE_BATCH_SIZE=2 (mocked), they split into:
        # - Batch 1: jobs 1–2 (raises)
        # - Batch 2: job 3 (succeeds with 1 embedding)
        _insert_job(tmp_db, "job-1", "First job")
        _insert_job(tmp_db, "job-2", "Second job")
        _insert_job(tmp_db, "job-3", "Third job")

        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        # First call raises, second call succeeds
        mock_embed_batch.side_effect = [
            RuntimeError("API error"),
            [np.ones(EMBEDDING_DIM, dtype=np.float32)],  # 1 embedding for batch 2
        ]

        run_embedding(tmp_db, "mock-api-key")

        # Verify error was logged
        assert "embed_batch raised" in caplog.text or "error" in caplog.text.lower()

        # Verify batch 2 succeeded (exactly 1 job embedded from batch 2)
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE embedded=1")
        embedded_count = cursor.fetchone()[0]
        conn.close()

        assert embedded_count == 1

    @patch("embed_jobs.create_client")
    @patch("embed_jobs.embed_batch")
    def test_run_embedding_idempotent(self, mock_embed_batch, mock_create_client, tmp_db):
        """Calling run_embedding twice doesn't duplicate rows or embeddings."""
        _insert_job(tmp_db, "job-1", "Test job")

        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        test_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)
        mock_embed_batch.return_value = [test_embedding]

        # First run
        run_embedding(tmp_db, "mock-api-key")

        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jobs")
        count_after_first = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE embedded=1")
        embedded_after_first = cursor.fetchone()[0]
        conn.close()

        # Reset mock for second call
        mock_embed_batch.reset_mock()
        mock_embed_batch.return_value = []

        # Second run (no unembedded jobs, so mock won't be called)
        run_embedding(tmp_db, "mock-api-key")

        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jobs")
        count_after_second = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE embedded=1")
        embedded_after_second = cursor.fetchone()[0]
        conn.close()

        # Row count unchanged, embedding count unchanged
        assert count_after_first == count_after_second
        assert embedded_after_first == embedded_after_second
        assert embedded_after_first == 1


class TestMainFunction:
    """Tests for main() entry point."""

    @patch("embed_jobs.load_dotenv")
    @patch("embed_jobs.run_embedding")
    @patch("sys.argv", ["embed_jobs.py", "--db-path", "/tmp/test.db"])
    def test_main_voyage_key_missing(self, mock_run_embedding, mock_load_dotenv):
        """main() exits with error if VOYAGE_API_KEY is missing."""
        with patch("os.getenv") as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                "DB_PATH": "/tmp/test.db",
                "VOYAGE_API_KEY": None,
            }.get(key, default)

            with patch("sys.exit", side_effect=SystemExit) as mock_exit:
                from embed_jobs import main

                with pytest.raises(SystemExit):
                    main()
                # sys.exit(1) should be called
                mock_exit.assert_called_with(1)
                # run_embedding should NOT be called
                mock_run_embedding.assert_not_called()
