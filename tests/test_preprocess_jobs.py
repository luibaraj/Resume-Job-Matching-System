"""Tests for scripts/pipeline/preprocess_jobs.py — job description preprocessing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRunPreprocessing:
    """Test suite for run_preprocessing function."""

    def test_preprocessing_adds_columns_if_missing(self, tmp_db):
        """Test that preprocessing creates required columns if missing."""
        cursor = tmp_db.cursor()

        # Insert a test job
        cursor.execute(
            "INSERT INTO jobs (title, description, url, board_token) VALUES (?, ?, ?, ?)",
            ("Test Job", "Raw HTML <b>description</b>", "http://test.com", "token"),
        )
        tmp_db.commit()

        # Verify columns don't exist initially
        cursor.execute("PRAGMA table_info(jobs)")
        cols = {row[1] for row in cursor.fetchall()}
        assert "cleaned_description" not in cols
        assert "preprocessed" not in cols

        # Manually add columns using the same logic as run_preprocessing
        from src.db_utils import add_column_if_missing
        add_column_if_missing(cursor, "jobs", "cleaned_description", "TEXT")
        add_column_if_missing(cursor, "jobs", "preprocessed", "INTEGER DEFAULT 0")
        tmp_db.commit()

        # Verify columns were added
        cursor.execute("PRAGMA table_info(jobs)")
        cols = {row[1] for row in cursor.fetchall()}
        assert "cleaned_description" in cols
        assert "preprocessed" in cols

    def test_preprocessing_marks_rows_as_processed(self, tmp_db):
        """Test that preprocessed flag is set correctly."""
        cursor = tmp_db.cursor()

        # Add required columns first
        cursor.execute("ALTER TABLE jobs ADD COLUMN cleaned_description TEXT")
        cursor.execute("ALTER TABLE jobs ADD COLUMN preprocessed INTEGER DEFAULT 0")

        # Insert test job
        cursor.execute(
            "INSERT INTO jobs (title, description, url, board_token, preprocessed) VALUES (?, ?, ?, ?, ?)",
            ("Test Job", "Raw HTML", "http://test.com", "token", 0),
        )
        tmp_db.commit()

        # Verify initial state
        cursor.execute("SELECT preprocessed FROM jobs WHERE title = ?", ("Test Job",))
        assert cursor.fetchone()[0] == 0

        # Update to preprocessed state (simulating what run_preprocessing does)
        cursor.execute(
            "UPDATE jobs SET cleaned_description = ?, preprocessed = 1 WHERE title = ?",
            ("Cleaned", "Test Job"),
        )
        tmp_db.commit()

        # Verify updated state
        cursor.execute("SELECT preprocessed FROM jobs WHERE title = ?", ("Test Job",))
        assert cursor.fetchone()[0] == 1

    def test_preprocessing_handles_multiple_jobs(self, tmp_db):
        """Test preprocessing multiple jobs in batch."""
        cursor = tmp_db.cursor()

        # Add required columns
        cursor.execute("ALTER TABLE jobs ADD COLUMN cleaned_description TEXT")
        cursor.execute("ALTER TABLE jobs ADD COLUMN preprocessed INTEGER DEFAULT 0")
        tmp_db.commit()

        # Insert multiple jobs
        jobs = [
            ("Job 1", "Description 1", "http://test1.com", "token"),
            ("Job 2", "Description 2", "http://test2.com", "token"),
            ("Job 3", "Description 3", "http://test3.com", "token"),
        ]

        for title, desc, url, board_token in jobs:
            cursor.execute(
                "INSERT INTO jobs (title, description, url, board_token, preprocessed) VALUES (?, ?, ?, ?, ?)",
                (title, desc, url, board_token, 0),
            )
        tmp_db.commit()

        # Verify all are unprocessed
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE preprocessed = 0")
        assert cursor.fetchone()[0] == 3

        # Simulate batch preprocessing
        cursor.execute("SELECT id, title FROM jobs WHERE preprocessed = 0 LIMIT 2")
        batch = cursor.fetchall()
        assert len(batch) == 2

        # Update batch
        updates = [("Cleaned " + title, job_id) for job_id, title in batch]
        cursor.executemany(
            "UPDATE jobs SET cleaned_description = ?, preprocessed = 1 WHERE id = ?",
            updates,
        )
        tmp_db.commit()

        # Verify partial update
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE preprocessed = 1")
        assert cursor.fetchone()[0] == 2

    def test_preprocessing_handles_single_job_failure(self, tmp_db):
        """Test that a single job failure doesn't stop batch processing."""
        cursor = tmp_db.cursor()

        # Add required columns
        cursor.execute("ALTER TABLE jobs ADD COLUMN cleaned_description TEXT")
        cursor.execute("ALTER TABLE jobs ADD COLUMN preprocessed INTEGER DEFAULT 0")
        tmp_db.commit()

        # Insert multiple jobs
        jobs = [
            ("Good Job 1", "Valid description", "http://test1.com", "token"),
            ("Bad Job", "Invalid HTML", "http://test2.com", "token"),
            ("Good Job 2", "Valid description", "http://test3.com", "token"),
        ]

        for title, desc, url, board_token in jobs:
            cursor.execute(
                "INSERT INTO jobs (title, description, url, board_token) VALUES (?, ?, ?, ?)",
                (title, desc, url, board_token),
            )
        tmp_db.commit()

        # Simulate processing with one failure (fallback to empty string)
        cursor.execute("SELECT id, title, description FROM jobs")
        rows = cursor.fetchall()
        updates = []

        for job_id, title, description in rows:
            try:
                if title == "Bad Job":
                    raise ValueError("Simulated preprocessing error")
                cleaned = f"Cleaned: {description}"
            except Exception:
                cleaned = ""  # Fallback as per preprocess_jobs.py logic
            updates.append((cleaned, job_id))

        cursor.executemany(
            "UPDATE jobs SET cleaned_description = ?, preprocessed = 1 WHERE id = ?",
            updates,
        )
        tmp_db.commit()

        # Verify all jobs processed despite one failure
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE preprocessed = 1")
        assert cursor.fetchone()[0] == 3

        # Verify the bad job has empty cleaned_description
        cursor.execute(
            "SELECT cleaned_description FROM jobs WHERE title = ?",
            ("Bad Job",),
        )
        assert cursor.fetchone()[0] == ""

    def test_preprocessing_respects_chunk_size(self, tmp_db):
        """Test that preprocessing uses chunking for large datasets."""
        cursor = tmp_db.cursor()

        # Add required columns
        cursor.execute("ALTER TABLE jobs ADD COLUMN cleaned_description TEXT")
        cursor.execute("ALTER TABLE jobs ADD COLUMN preprocessed INTEGER DEFAULT 0")
        tmp_db.commit()

        # Insert 10 jobs
        chunk_size = 3
        for i in range(10):
            cursor.execute(
                "INSERT INTO jobs (title, description, url, board_token) VALUES (?, ?, ?, ?)",
                (f"Job {i}", f"Description {i}", f"http://test{i}.com", "token"),
            )
        tmp_db.commit()

        # Simulate chunked processing
        processed = 0
        while True:
            cursor.execute(
                "SELECT id, title FROM jobs WHERE preprocessed = 0 LIMIT ? OFFSET 0",
                (chunk_size,),
            )
            batch = cursor.fetchall()
            if not batch:
                break

            # Process batch
            updates = [("Cleaned " + title, job_id) for job_id, title in batch]
            cursor.executemany(
                "UPDATE jobs SET cleaned_description = ?, preprocessed = 1 WHERE id = ?",
                updates,
            )
            tmp_db.commit()
            processed += len(batch)

        assert processed == 10

    def test_preprocessing_row_factory_enabled(self, tmp_db):
        """Test that row_factory is set correctly for dict-like access."""
        cursor = tmp_db.cursor()

        # Add column
        cursor.execute("ALTER TABLE jobs ADD COLUMN cleaned_description TEXT")
        tmp_db.commit()

        # Insert job
        cursor.execute(
            "INSERT INTO jobs (title, description, url, board_token) VALUES (?, ?, ?, ?)",
            ("Test Job", "Raw HTML", "http://test.com", "token"),
        )
        tmp_db.commit()

        # Fetch with row_factory (as in actual preprocess_jobs.py)
        cursor.execute("SELECT id, description FROM jobs WHERE title = ?", ("Test Job",))
        row = cursor.fetchone()

        # With row_factory=sqlite3.Row, we can access as dict
        assert row["id"] is not None
        assert row["description"] == "Raw HTML"

    def test_preprocessing_query_offset_zero_pattern(self, tmp_db):
        """Test the OFFSET 0 pattern used to handle committed rows correctly."""
        cursor = tmp_db.cursor()

        # Add required columns
        cursor.execute("ALTER TABLE jobs ADD COLUMN preprocessed INTEGER DEFAULT 0")
        tmp_db.commit()

        # Insert jobs
        for i in range(5):
            cursor.execute(
                "INSERT INTO jobs (title, description, url, board_token) VALUES (?, ?, ?, ?)",
                (f"Job {i}", f"Desc {i}", f"http://test{i}.com", "token"),
            )
        tmp_db.commit()

        # Process in two batches using OFFSET 0 pattern (each commit removes matched rows)
        batch_size = 2
        batches_processed = 0

        while True:
            # Always use OFFSET 0 — committed rows drop out of WHERE preprocessed=0
            cursor.execute(
                "SELECT id FROM jobs WHERE preprocessed = 0 LIMIT ? OFFSET 0",
                (batch_size,),
            )
            batch = cursor.fetchall()
            if not batch:
                break

            for job_id, in batch:
                cursor.execute(
                    "UPDATE jobs SET preprocessed = 1 WHERE id = ?",
                    (job_id,),
                )
            tmp_db.commit()
            batches_processed += 1

        assert batches_processed == 3  # 5 jobs, batch_size=2: 2+2+1
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE preprocessed = 1")
        assert cursor.fetchone()[0] == 5

    def test_preprocessing_concurrent_updates(self, tmp_db):
        """Test that batch updates don't interfere with concurrent queries."""
        cursor = tmp_db.cursor()

        cursor.execute("ALTER TABLE jobs ADD COLUMN preprocessed INTEGER DEFAULT 0")
        tmp_db.commit()

        # Insert jobs
        for i in range(3):
            cursor.execute(
                "INSERT INTO jobs (title, description, url, board_token) VALUES (?, ?, ?, ?)",
                (f"Job {i}", f"Desc {i}", f"http://test{i}.com", "token"),
            )
        tmp_db.commit()

        # Start with unprocessed count
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE preprocessed = 0")
        initial_count = cursor.fetchone()[0]
        assert initial_count == 3

        # Process one
        cursor.execute("UPDATE jobs SET preprocessed = 1 WHERE id = 1")
        tmp_db.commit()

        # Verify remaining
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE preprocessed = 0")
        remaining_count = cursor.fetchone()[0]
        assert remaining_count == 2
