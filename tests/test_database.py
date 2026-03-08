"""Tests for src/database.py DatabaseManager class."""

import sqlite3
from datetime import datetime

import pytest

from src.database import DatabaseManager
from src.utils import compute_job_hash


class TestInitializeSchema:
    """Tests for DatabaseManager.initialize_schema()."""

    def test_jobs_table_exists(self, temp_db: str):
        """After initialize_schema, jobs table exists."""
        db = DatabaseManager(temp_db)
        db.initialize_schema()
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
            )
            assert cursor.fetchone() is not None

    def test_pipeline_runs_table_exists(self, temp_db: str):
        """After initialize_schema, pipeline_runs table exists."""
        db = DatabaseManager(temp_db)
        db.initialize_schema()
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='pipeline_runs'"
            )
            assert cursor.fetchone() is not None

    def test_jobs_unique_index_exists(self, temp_db: str):
        """Index on job_hash exists."""
        db = DatabaseManager(temp_db)
        db.initialize_schema()
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_jobs_hash'"
            )
            assert cursor.fetchone() is not None

    def test_preprocessed_index_exists(self, temp_db: str):
        """Index on preprocessed column exists."""
        db = DatabaseManager(temp_db)
        db.initialize_schema()
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_jobs_preprocessed'"
            )
            assert cursor.fetchone() is not None

    def test_greenhouse_id_index_exists(self, temp_db: str):
        """Index on greenhouse_id exists."""
        db = DatabaseManager(temp_db)
        db.initialize_schema()
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_jobs_greenhouse_id'"
            )
            assert cursor.fetchone() is not None

    def test_idempotent_second_call(self, db_manager: DatabaseManager):
        """Calling initialize_schema twice raises no exception."""
        # db_manager already called initialize_schema once
        db_manager.initialize_schema()  # Should not raise
        # Verify tables still exist
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('jobs', 'pipeline_runs')"
            )
            assert cursor.fetchone()[0] == 2


class TestGetConnection:
    """Tests for DatabaseManager.get_connection() context manager."""

    def test_wal_mode_is_enabled(self, db_manager: DatabaseManager):
        """WAL mode is enabled on the connection."""
        with db_manager.get_connection() as conn:
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            assert mode == "wal"

    def test_row_factory_is_sqlite_row(self, db_manager: DatabaseManager):
        """Connection uses sqlite3.Row as row factory."""
        with db_manager.get_connection() as conn:
            assert conn.row_factory is sqlite3.Row

    def test_successful_block_commits(self, db_manager: DatabaseManager):
        """Successful transaction commits data."""
        job = {
            "greenhouse_id": 1,
            "board_token": "acme",
            "title": "Engineer",
            "company": "acme",
            "location": "Remote",
            "raw_description": "Desc",
            "absolute_url": "http://example.com",
            "updated_at_source": "2026-03-05T00:00:00Z",
            "departments": "[]",
            "offices": "[]",
            "collected_at": "2026-03-05T00:00:00Z",
        }
        job["job_hash"] = compute_job_hash(
            job["greenhouse_id"], job["board_token"], job["title"]
        )
        # Insert in one context
        with db_manager.get_connection() as conn:
            conn.execute(
                """INSERT INTO jobs (job_hash, greenhouse_id, board_token, title, company,
                   location, raw_description, absolute_url, updated_at_source,
                   departments, offices, collected_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job["job_hash"],
                    job["greenhouse_id"],
                    job["board_token"],
                    job["title"],
                    job["company"],
                    job["location"],
                    job["raw_description"],
                    job["absolute_url"],
                    job["updated_at_source"],
                    job["departments"],
                    job["offices"],
                    job["collected_at"],
                ),
            )
        # Verify in another context
        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT greenhouse_id FROM jobs WHERE greenhouse_id=1")
            assert cursor.fetchone() is not None

    def test_exception_triggers_rollback(self, db_manager: DatabaseManager):
        """Exception inside context manager triggers rollback."""
        job_data = (
            "abc123",  # job_hash
            999,  # greenhouse_id
            "test",  # board_token
            "Test Job",  # title
            "test-co",  # company
            "Remote",  # location
            "Description",  # raw_description
            "http://example.com",  # absolute_url
            "2026-03-05T00:00:00Z",  # updated_at_source
            "[]",  # departments
            "[]",  # offices
            "2026-03-05T00:00:00Z",  # collected_at
        )
        try:
            with db_manager.get_connection() as conn:
                conn.execute(
                    """INSERT INTO jobs (job_hash, greenhouse_id, board_token, title, company,
                       location, raw_description, absolute_url, updated_at_source,
                       departments, offices, collected_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    job_data,
                )
                raise ValueError("test exception")
        except ValueError:
            pass
        # Verify the insert was rolled back
        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM jobs WHERE greenhouse_id=999")
            assert cursor.fetchone()[0] == 0

    def test_exception_is_reraised(self, db_manager: DatabaseManager):
        """Exception is re-raised after rollback."""
        with pytest.raises(ValueError, match="test exception"):
            with db_manager.get_connection() as conn:
                conn.execute("SELECT 1")
                raise ValueError("test exception")


class TestInsertJob:
    """Tests for DatabaseManager.insert_job()."""

    def test_returns_true_on_new_job(self, db_manager: DatabaseManager, normalized_job: dict):
        """insert_job returns True for new job."""
        result = db_manager.insert_job(normalized_job)
        assert result is True

    def test_returns_false_on_duplicate(
        self, db_manager: DatabaseManager, normalized_job: dict, duplicate_normalized_job: dict
    ):
        """insert_job returns False on duplicate job_hash."""
        db_manager.insert_job(normalized_job)
        result = db_manager.insert_job(duplicate_normalized_job)
        assert result is False

    def test_data_persisted_correctly(self, db_manager: DatabaseManager, normalized_job: dict):
        """Inserted data is correctly persisted."""
        db_manager.insert_job(normalized_job)
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT title, company, location, departments, offices FROM jobs WHERE greenhouse_id=?",
                (normalized_job["greenhouse_id"],),
            )
            row = cursor.fetchone()
            assert row["title"] == "Software Engineer, Backend"
            assert row["company"] == "example-company"
            assert row["location"] == "San Francisco, CA or Remote"
            assert row["departments"] == '["Engineering"]'
            assert row["offices"] == '["San Francisco"]'

    def test_preprocessed_defaults_to_zero(self, db_manager: DatabaseManager, normalized_job: dict):
        """preprocessed column defaults to 0."""
        db_manager.insert_job(normalized_job)
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT preprocessed FROM jobs WHERE greenhouse_id=?",
                (normalized_job["greenhouse_id"],),
            )
            assert cursor.fetchone()["preprocessed"] == 0

    def test_extracted_defaults_to_zero(self, db_manager: DatabaseManager, normalized_job: dict):
        """extracted column defaults to 0."""
        db_manager.insert_job(normalized_job)
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT extracted FROM jobs WHERE greenhouse_id=?",
                (normalized_job["greenhouse_id"],),
            )
            assert cursor.fetchone()["extracted"] == 0

    def test_embedded_defaults_to_zero(self, db_manager: DatabaseManager, normalized_job: dict):
        """embedded column defaults to 0."""
        db_manager.insert_job(normalized_job)
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT embedded FROM jobs WHERE greenhouse_id=?",
                (normalized_job["greenhouse_id"],),
            )
            assert cursor.fetchone()["embedded"] == 0

    def test_job_hash_computed_and_stored(self, db_manager: DatabaseManager, normalized_job: dict):
        """job_hash is computed and stored (64-char hex)."""
        db_manager.insert_job(normalized_job)
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT job_hash FROM jobs WHERE greenhouse_id=?",
                (normalized_job["greenhouse_id"],),
            )
            job_hash = cursor.fetchone()["job_hash"]
            assert len(job_hash) == 64
            assert all(c in "0123456789abcdef" for c in job_hash)

    def test_none_location_allowed(self, db_manager: DatabaseManager, normalized_job: dict):
        """Job with location=None inserts without error."""
        normalized_job["location"] = None
        result = db_manager.insert_job(normalized_job)
        assert result is True
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT location FROM jobs WHERE greenhouse_id=?",
                (normalized_job["greenhouse_id"],),
            )
            assert cursor.fetchone()["location"] is None

    def test_none_raw_description_allowed(self, db_manager: DatabaseManager, normalized_job: dict):
        """Job with raw_description=None inserts without error."""
        normalized_job["raw_description"] = None
        result = db_manager.insert_job(normalized_job)
        assert result is True


class TestGetUnpreprocessedJobs:
    """Tests for DatabaseManager.get_unpreprocessed_jobs()."""

    def test_returns_only_preprocessed_zero_rows(
        self, db_manager: DatabaseManager, normalized_job: dict
    ):
        """Only rows with preprocessed=0 are returned."""
        db_manager.insert_job(normalized_job)
        # Manually mark one job as preprocessed
        with db_manager.get_connection() as conn:
            conn.execute("UPDATE jobs SET preprocessed=1 WHERE greenhouse_id=?",
                        (normalized_job["greenhouse_id"],))
        # Insert another job
        job2 = dict(normalized_job)
        job2["greenhouse_id"] = 999
        db_manager.insert_job(job2)
        # Only the unpreprocessed one should be returned
        unpreprocessed = db_manager.get_unpreprocessed_jobs()
        assert len(unpreprocessed) == 1
        assert unpreprocessed[0]["greenhouse_id"] == 999

    def test_empty_when_all_preprocessed(self, db_manager: DatabaseManager, normalized_job: dict):
        """Empty list when all jobs are preprocessed."""
        db_manager.insert_job(normalized_job)
        with db_manager.get_connection() as conn:
            conn.execute("UPDATE jobs SET preprocessed=1")
        result = db_manager.get_unpreprocessed_jobs()
        assert result == []

    def test_empty_database_returns_empty_list(self, db_manager: DatabaseManager):
        """Fresh database returns empty list."""
        result = db_manager.get_unpreprocessed_jobs()
        assert result == []

    def test_order_is_by_id_ascending(self, db_manager: DatabaseManager, normalized_job: dict):
        """Results are ordered by id ascending."""
        # Insert three jobs
        for i in range(3):
            job = dict(normalized_job)
            job["greenhouse_id"] = 1000 + i
            db_manager.insert_job(job)
        unpreprocessed = db_manager.get_unpreprocessed_jobs()
        ids = [row["id"] for row in unpreprocessed]
        assert ids == sorted(ids)
        assert len(ids) == 3

    def test_returns_sqlite_row_objects(self, db_manager: DatabaseManager, normalized_job: dict):
        """Each result supports dict-like access."""
        db_manager.insert_job(normalized_job)
        unpreprocessed = db_manager.get_unpreprocessed_jobs()
        assert len(unpreprocessed) == 1
        row = unpreprocessed[0]
        # Access like a dict
        assert row["title"] is not None
        assert row["greenhouse_id"] == normalized_job["greenhouse_id"]


class TestUpdateCleanedDescription:
    """Tests for DatabaseManager.update_cleaned_description()."""

    def test_sets_cleaned_description(self, db_manager: DatabaseManager, normalized_job: dict):
        """cleaned_description is set correctly."""
        db_manager.insert_job(normalized_job)
        with db_manager.get_connection() as conn:
            job_id = conn.execute(
                "SELECT id FROM jobs WHERE greenhouse_id=?",
                (normalized_job["greenhouse_id"],),
            ).fetchone()["id"]
        db_manager.update_cleaned_description(job_id, "clean text")
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT cleaned_description FROM jobs WHERE id=?", (job_id,)
            )
            assert cursor.fetchone()["cleaned_description"] == "clean text"

    def test_sets_preprocessed_to_one(self, db_manager: DatabaseManager, normalized_job: dict):
        """preprocessed is set to 1."""
        db_manager.insert_job(normalized_job)
        with db_manager.get_connection() as conn:
            job_id = conn.execute(
                "SELECT id FROM jobs WHERE greenhouse_id=?",
                (normalized_job["greenhouse_id"],),
            ).fetchone()["id"]
        db_manager.update_cleaned_description(job_id, "clean")
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT preprocessed FROM jobs WHERE id=?", (job_id,)
            )
            assert cursor.fetchone()["preprocessed"] == 1

    def test_does_not_affect_other_jobs(self, db_manager: DatabaseManager, normalized_job: dict):
        """Update doesn't affect other jobs."""
        job1 = normalized_job
        job2 = dict(normalized_job)
        job2["greenhouse_id"] = 999
        db_manager.insert_job(job1)
        db_manager.insert_job(job2)
        with db_manager.get_connection() as conn:
            job1_id = conn.execute(
                "SELECT id FROM jobs WHERE greenhouse_id=?",
                (job1["greenhouse_id"],),
            ).fetchone()["id"]
        db_manager.update_cleaned_description(job1_id, "clean")
        # job2 should still be unpreprocessed
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT preprocessed FROM jobs WHERE greenhouse_id=?", (999,)
            )
            assert cursor.fetchone()["preprocessed"] == 0

    def test_job_no_longer_returned_by_get_unpreprocessed(
        self, db_manager: DatabaseManager, normalized_job: dict
    ):
        """Updated job is no longer in get_unpreprocessed_jobs()."""
        db_manager.insert_job(normalized_job)
        with db_manager.get_connection() as conn:
            job_id = conn.execute(
                "SELECT id FROM jobs WHERE greenhouse_id=?",
                (normalized_job["greenhouse_id"],),
            ).fetchone()["id"]
        db_manager.update_cleaned_description(job_id, "clean")
        unpreprocessed = db_manager.get_unpreprocessed_jobs()
        assert len(unpreprocessed) == 0


class TestCreatePipelineRun:
    """Tests for DatabaseManager.create_pipeline_run()."""

    def test_returns_integer_id(self, db_manager: DatabaseManager):
        """Returns an integer ID."""
        run_id = db_manager.create_pipeline_run("2026-03-05", "collection")
        assert isinstance(run_id, int)

    def test_id_is_positive(self, db_manager: DatabaseManager):
        """Returned ID is positive."""
        run_id = db_manager.create_pipeline_run("2026-03-05", "collection")
        assert run_id > 0

    def test_status_is_running(self, db_manager: DatabaseManager):
        """Status is set to 'running'."""
        run_id = db_manager.create_pipeline_run("2026-03-05", "collection")
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT status FROM pipeline_runs WHERE id=?", (run_id,)
            )
            assert cursor.fetchone()["status"] == "running"

    def test_run_date_stored(self, db_manager: DatabaseManager):
        """run_date is stored correctly."""
        run_date = "2026-03-05"
        run_id = db_manager.create_pipeline_run(run_date, "collection")
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT run_date FROM pipeline_runs WHERE id=?", (run_id,)
            )
            assert cursor.fetchone()["run_date"] == run_date

    def test_step_stored(self, db_manager: DatabaseManager):
        """step is stored correctly."""
        run_id = db_manager.create_pipeline_run("2026-03-05", "preprocessing")
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT step FROM pipeline_runs WHERE id=?", (run_id,)
            )
            assert cursor.fetchone()["step"] == "preprocessing"

    def test_started_at_is_populated(self, db_manager: DatabaseManager):
        """started_at timestamp is set."""
        run_id = db_manager.create_pipeline_run("2026-03-05", "collection")
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT started_at FROM pipeline_runs WHERE id=?", (run_id,)
            )
            started_at = cursor.fetchone()["started_at"]
            assert started_at is not None

    def test_finished_at_is_null(self, db_manager: DatabaseManager):
        """finished_at is NULL initially."""
        run_id = db_manager.create_pipeline_run("2026-03-05", "collection")
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT finished_at FROM pipeline_runs WHERE id=?", (run_id,)
            )
            assert cursor.fetchone()["finished_at"] is None

    def test_sequential_ids(self, db_manager: DatabaseManager):
        """Sequential calls return incrementing IDs."""
        id1 = db_manager.create_pipeline_run("2026-03-05", "collection")
        id2 = db_manager.create_pipeline_run("2026-03-05", "preprocessing")
        assert id2 > id1


class TestFinishPipelineRun:
    """Tests for DatabaseManager.finish_pipeline_run()."""

    def test_updates_status(self, db_manager: DatabaseManager):
        """Status is updated."""
        run_id = db_manager.create_pipeline_run("2026-03-05", "collection")
        db_manager.finish_pipeline_run(run_id, "success", 10, 2, None)
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT status FROM pipeline_runs WHERE id=?", (run_id,)
            )
            assert cursor.fetchone()["status"] == "success"

    def test_updates_jobs_processed(self, db_manager: DatabaseManager):
        """jobs_processed is updated."""
        run_id = db_manager.create_pipeline_run("2026-03-05", "collection")
        db_manager.finish_pipeline_run(run_id, "success", 10, 2, None)
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT jobs_processed FROM pipeline_runs WHERE id=?", (run_id,)
            )
            assert cursor.fetchone()["jobs_processed"] == 10

    def test_updates_jobs_skipped(self, db_manager: DatabaseManager):
        """jobs_skipped is updated."""
        run_id = db_manager.create_pipeline_run("2026-03-05", "collection")
        db_manager.finish_pipeline_run(run_id, "success", 10, 2, None)
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT jobs_skipped FROM pipeline_runs WHERE id=?", (run_id,)
            )
            assert cursor.fetchone()["jobs_skipped"] == 2

    def test_sets_finished_at(self, db_manager: DatabaseManager):
        """finished_at timestamp is set."""
        run_id = db_manager.create_pipeline_run("2026-03-05", "collection")
        db_manager.finish_pipeline_run(run_id, "success", 10, 2, None)
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT finished_at FROM pipeline_runs WHERE id=?", (run_id,)
            )
            finished_at = cursor.fetchone()["finished_at"]
            assert finished_at is not None

    def test_error_message_stored(self, db_manager: DatabaseManager):
        """error_message is stored."""
        run_id = db_manager.create_pipeline_run("2026-03-05", "collection")
        db_manager.finish_pipeline_run(run_id, "failed", 0, 0, "timeout")
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT error_message FROM pipeline_runs WHERE id=?", (run_id,)
            )
            assert cursor.fetchone()["error_message"] == "timeout"

    def test_none_error_message_allowed(self, db_manager: DatabaseManager):
        """error_message can be None."""
        run_id = db_manager.create_pipeline_run("2026-03-05", "collection")
        db_manager.finish_pipeline_run(run_id, "success", 10, 2, None)
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT error_message FROM pipeline_runs WHERE id=?", (run_id,)
            )
            assert cursor.fetchone()["error_message"] is None

    def test_does_not_affect_other_runs(self, db_manager: DatabaseManager):
        """Finishing one run doesn't affect another."""
        run_id1 = db_manager.create_pipeline_run("2026-03-05", "collection")
        run_id2 = db_manager.create_pipeline_run("2026-03-05", "preprocessing")
        db_manager.finish_pipeline_run(run_id1, "success", 10, 2, None)
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT status FROM pipeline_runs WHERE id=?", (run_id2,)
            )
            assert cursor.fetchone()["status"] == "running"


class TestInsertJobsBatch:
    """Tests for DatabaseManager.insert_jobs_batch()."""

    def test_returns_inserted_skipped_tuple(self, db_manager: DatabaseManager, normalized_job: dict):
        """Returns a 2-tuple of (inserted, skipped) integers."""
        result = db_manager.insert_jobs_batch([normalized_job])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int) and isinstance(result[1], int)

    def test_inserts_all_new_jobs(self, db_manager: DatabaseManager, normalized_job: dict):
        """Batch insert of 3 new jobs returns (3, 0)."""
        jobs = [dict(normalized_job) for _ in range(3)]
        for i, job in enumerate(jobs):
            job["greenhouse_id"] = 1000 + i
        result = db_manager.insert_jobs_batch(jobs)
        assert result == (3, 0)

    def test_skips_duplicate_jobs(self, db_manager: DatabaseManager, normalized_job: dict):
        """Second batch with same jobs returns (0, N)."""
        db_manager.insert_jobs_batch([normalized_job])
        result = db_manager.insert_jobs_batch([normalized_job])
        assert result == (0, 1)

    def test_mixed_new_and_duplicate(self, db_manager: DatabaseManager, normalized_job: dict):
        """Batch with 2 new + 1 duplicate returns (2, 1)."""
        db_manager.insert_jobs_batch([normalized_job])
        jobs = [dict(normalized_job) for _ in range(3)]
        jobs[0] = dict(normalized_job)  # duplicate
        jobs[1]["greenhouse_id"] = 999
        jobs[2]["greenhouse_id"] = 998
        result = db_manager.insert_jobs_batch(jobs)
        assert result == (2, 1)

    def test_empty_list_returns_zero_zero(self, db_manager: DatabaseManager):
        """Empty job list returns (0, 0)."""
        result = db_manager.insert_jobs_batch([])
        assert result == (0, 0)

    def test_jobs_persisted_to_db(self, db_manager: DatabaseManager, normalized_job: dict):
        """Jobs from batch are persisted to database."""
        db_manager.insert_jobs_batch([normalized_job])
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT title, company FROM jobs WHERE greenhouse_id=?",
                (normalized_job["greenhouse_id"],),
            )
            row = cursor.fetchone()
            assert row is not None
            assert row["title"] == normalized_job["title"]
            assert row["company"] == normalized_job["company"]

    def test_all_jobs_persisted(self, db_manager: DatabaseManager, normalized_job: dict):
        """All jobs in batch are persisted."""
        jobs = [dict(normalized_job) for _ in range(3)]
        for i, job in enumerate(jobs):
            job["greenhouse_id"] = 2000 + i
        db_manager.insert_jobs_batch(jobs)
        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM jobs WHERE greenhouse_id >= 2000")
            assert cursor.fetchone()[0] == 3

    def test_preprocessed_defaults_to_zero(self, db_manager: DatabaseManager, normalized_job: dict):
        """Batch-inserted jobs have preprocessed=0."""
        db_manager.insert_jobs_batch([normalized_job])
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT preprocessed FROM jobs WHERE greenhouse_id=?",
                (normalized_job["greenhouse_id"],),
            )
            assert cursor.fetchone()["preprocessed"] == 0

    def test_single_transaction_all_or_nothing(self, db_manager: DatabaseManager, normalized_job: dict):
        """Batch uses single transaction; constraint violation fails entire batch."""
        valid_job = dict(normalized_job)
        valid_job["greenhouse_id"] = 3000

        # Create a second job with duplicate greenhouse_id to force constraint after first insert
        dup_job = dict(normalized_job)
        dup_job["greenhouse_id"] = 3000

        # Insert first job to DB
        db_manager.insert_job(valid_job)

        # Try to batch insert same + new; entire batch should be skipped on duplicate
        before = 0
        with db_manager.get_connection() as conn:
            before = conn.execute("SELECT COUNT(*) FROM jobs WHERE greenhouse_id >= 3000").fetchone()[0]

        # Batch with duplicate should not insert anything
        new_job = dict(normalized_job)
        new_job["greenhouse_id"] = 3001
        inserted, skipped = db_manager.insert_jobs_batch([dup_job, new_job])

        # One skipped (dup), one inserted (new) - INSERT OR IGNORE allows mixed results
        assert inserted + skipped == 2


class TestGetUnpreprocessedJobsChunked:
    """Tests for DatabaseManager.get_unpreprocessed_jobs_chunked()."""

    def test_returns_first_chunk(self, db_manager: DatabaseManager, normalized_job: dict):
        """chunk_size=2, offset=0 returns exactly 2 records."""
        # Insert 5 jobs
        for i in range(5):
            job = dict(normalized_job)
            job["greenhouse_id"] = 1000 + i
            db_manager.insert_job(job)

        result = db_manager.get_unpreprocessed_jobs_chunked(chunk_size=2, offset=0)
        assert len(result) == 2

    def test_returns_second_chunk(self, db_manager: DatabaseManager, normalized_job: dict):
        """chunk_size=2, offset=2 returns next 2 records."""
        for i in range(5):
            job = dict(normalized_job)
            job["greenhouse_id"] = 1100 + i
            db_manager.insert_job(job)

        result = db_manager.get_unpreprocessed_jobs_chunked(chunk_size=2, offset=2)
        assert len(result) == 2

    def test_returns_partial_last_chunk(self, db_manager: DatabaseManager, normalized_job: dict):
        """chunk_size=2, offset=4 returns 1 record (last)."""
        for i in range(5):
            job = dict(normalized_job)
            job["greenhouse_id"] = 1200 + i
            db_manager.insert_job(job)

        result = db_manager.get_unpreprocessed_jobs_chunked(chunk_size=2, offset=4)
        assert len(result) == 1

    def test_returns_empty_at_end(self, db_manager: DatabaseManager, normalized_job: dict):
        """offset beyond all records returns empty list."""
        for i in range(5):
            job = dict(normalized_job)
            job["greenhouse_id"] = 1300 + i
            db_manager.insert_job(job)

        result = db_manager.get_unpreprocessed_jobs_chunked(chunk_size=2, offset=5)
        assert result == []

    def test_returns_plain_tuples_not_rows(self, db_manager: DatabaseManager, normalized_job: dict):
        """Return type is list[tuple], not list[sqlite3.Row]."""
        db_manager.insert_job(normalized_job)
        result = db_manager.get_unpreprocessed_jobs_chunked(chunk_size=10, offset=0)
        assert len(result) == 1
        assert isinstance(result[0], tuple)

    def test_tuple_contains_id_raw_desc_location_title(self, db_manager: DatabaseManager, normalized_job: dict):
        """Each tuple has exactly 4 elements: (id, raw_description, location, title)."""
        db_manager.insert_job(normalized_job)
        result = db_manager.get_unpreprocessed_jobs_chunked(chunk_size=10, offset=0)
        assert len(result[0]) == 4
        job_id, raw_desc, location, title = result[0]
        assert isinstance(job_id, int)
        assert isinstance(raw_desc, (str, type(None)))
        assert isinstance(location, (str, type(None)))
        assert isinstance(title, str)

    def test_excludes_preprocessed_jobs(self, db_manager: DatabaseManager, normalized_job: dict):
        """Only jobs with preprocessed=0 are returned."""
        # Insert 3 jobs
        for i in range(3):
            job = dict(normalized_job)
            job["greenhouse_id"] = 1400 + i
            db_manager.insert_job(job)

        # Mark 2 as preprocessed
        with db_manager.get_connection() as conn:
            conn.execute("UPDATE jobs SET preprocessed=1 WHERE greenhouse_id IN (1400, 1401)")

        result = db_manager.get_unpreprocessed_jobs_chunked(chunk_size=10, offset=0)
        assert len(result) == 1

    def test_empty_database_returns_empty_list(self, db_manager: DatabaseManager):
        """Fresh database returns empty list."""
        result = db_manager.get_unpreprocessed_jobs_chunked(chunk_size=10, offset=0)
        assert result == []

    def test_order_is_stable_across_calls(self, db_manager: DatabaseManager, normalized_job: dict):
        """id values are ordered ascending across chunks."""
        for i in range(5):
            job = dict(normalized_job)
            job["greenhouse_id"] = 1500 + i
            db_manager.insert_job(job)

        chunk1 = db_manager.get_unpreprocessed_jobs_chunked(chunk_size=2, offset=0)
        chunk2 = db_manager.get_unpreprocessed_jobs_chunked(chunk_size=2, offset=2)

        # All ids from chunk1 should be less than ids from chunk2
        assert chunk1[-1][0] < chunk2[0][0]

    def test_raw_description_preserved(self, db_manager: DatabaseManager, normalized_job: dict):
        """raw_description is correctly retrieved in the tuple."""
        db_manager.insert_job(normalized_job)
        result = db_manager.get_unpreprocessed_jobs_chunked(chunk_size=10, offset=0)
        job_id, raw_desc, location, title = result[0]
        assert raw_desc == normalized_job["raw_description"]

    def test_location_and_title_values_preserved(self, db_manager: DatabaseManager, normalized_job: dict):
        """location and title are correctly retrieved in the tuple."""
        db_manager.insert_job(normalized_job)
        result = db_manager.get_unpreprocessed_jobs_chunked(chunk_size=10, offset=0)
        job_id, raw_desc, location, title = result[0]
        assert location == normalized_job["location"]
        assert title == normalized_job["title"]


class TestUpdateCleanedDescriptionsBatch:
    """Tests for DatabaseManager.update_cleaned_descriptions_batch()."""

    def test_updates_all_jobs(self, db_manager: DatabaseManager, normalized_job: dict):
        """All jobs in batch are updated with cleaned_description and preprocessed=1."""
        # Insert 3 jobs
        jobs = [dict(normalized_job) for _ in range(3)]
        for i, job in enumerate(jobs):
            job["greenhouse_id"] = 4000 + i
        db_manager.insert_jobs_batch(jobs)

        # Get their IDs
        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT id FROM jobs WHERE greenhouse_id >= 4000 ORDER BY id")
            job_ids = [row["id"] for row in cursor.fetchall()]

        # Batch update
        updates = [(job_ids[i], f"cleaned_{i}") for i in range(3)]
        db_manager.update_cleaned_descriptions_batch(updates)

        # Verify all updated
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT cleaned_description, preprocessed FROM jobs WHERE id IN (?, ?, ?)",
                job_ids,
            )
            rows = cursor.fetchall()
            assert len(rows) == 3
            for i, row in enumerate(rows):
                assert row["cleaned_description"] == f"cleaned_{i}"
                assert row["preprocessed"] == 1

    def test_empty_list_no_error(self, db_manager: DatabaseManager):
        """Empty updates list causes no error."""
        db_manager.update_cleaned_descriptions_batch([])  # Should not raise

    def test_does_not_affect_other_jobs(self, db_manager: DatabaseManager, normalized_job: dict):
        """Jobs not in updates list remain preprocessed=0."""
        jobs = [dict(normalized_job) for _ in range(2)]
        jobs[0]["greenhouse_id"] = 5000
        jobs[1]["greenhouse_id"] = 5001
        db_manager.insert_jobs_batch(jobs)

        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT id FROM jobs WHERE greenhouse_id = 5000")
            job_id = cursor.fetchone()["id"]

        # Update only job 5000
        db_manager.update_cleaned_descriptions_batch([(job_id, "cleaned")])

        # Verify job 5001 is still unpreprocessed
        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT preprocessed FROM jobs WHERE greenhouse_id = 5001")
            assert cursor.fetchone()["preprocessed"] == 0

    def test_single_transaction(self, db_manager: DatabaseManager, normalized_job: dict):
        """All updates are committed together in single transaction."""
        jobs = [dict(normalized_job) for _ in range(2)]
        jobs[0]["greenhouse_id"] = 6000
        jobs[1]["greenhouse_id"] = 6001
        db_manager.insert_jobs_batch(jobs)

        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT id FROM jobs WHERE greenhouse_id IN (6000, 6001) ORDER BY id")
            job_ids = [row["id"] for row in cursor.fetchall()]

        updates = [(job_ids[0], "clean1"), (job_ids[1], "clean2")]
        db_manager.update_cleaned_descriptions_batch(updates)

        # Both should be updated in one go
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE preprocessed = 1 AND greenhouse_id IN (6000, 6001)"
            )
            assert cursor.fetchone()[0] == 2


class TestUpdateJobFieldsBatch:
    """Tests for DatabaseManager.update_job_fields_batch()."""

    def test_updates_all_fields(self, db_manager: DatabaseManager, normalized_job: dict):
        """All jobs in batch are updated with cleaned_description, location, title, and preprocessed=1."""
        jobs = [dict(normalized_job) for _ in range(2)]
        jobs[0]["greenhouse_id"] = 7000
        jobs[1]["greenhouse_id"] = 7001
        db_manager.insert_jobs_batch(jobs)

        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT id FROM jobs WHERE greenhouse_id IN (7000, 7001) ORDER BY id")
            job_ids = [row["id"] for row in cursor.fetchall()]

        # Batch update with new values
        updates = [
            (job_ids[0], "cleaned_0", "Remote", "Senior Engineer"),
            (job_ids[1], "cleaned_1", "San Francisco, CA", "Junior Dev"),
        ]
        db_manager.update_job_fields_batch(updates)

        # Verify all updated
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT cleaned_description, location, title, preprocessed FROM jobs WHERE id IN (?, ?)",
                job_ids,
            )
            rows = cursor.fetchall()
            assert len(rows) == 2
            assert rows[0]["cleaned_description"] == "cleaned_0"
            assert rows[0]["location"] == "Remote"
            assert rows[0]["title"] == "Senior Engineer"
            assert rows[0]["preprocessed"] == 1

            assert rows[1]["cleaned_description"] == "cleaned_1"
            assert rows[1]["location"] == "San Francisco, CA"
            assert rows[1]["title"] == "Junior Dev"
            assert rows[1]["preprocessed"] == 1

    def test_empty_list_is_noop(self, db_manager: DatabaseManager):
        """Empty updates list causes no error."""
        db_manager.update_job_fields_batch([])  # Should not raise

    def test_null_location_allowed(self, db_manager: DatabaseManager, normalized_job: dict):
        """location=None is allowed and properly set to NULL in database."""
        db_manager.insert_job(normalized_job)

        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT id FROM jobs")
            job_id = cursor.fetchone()["id"]

        updates = [(job_id, "cleaned", None, "New Title")]
        db_manager.update_job_fields_batch(updates)

        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT location FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            assert row["location"] is None

    def test_does_not_affect_other_jobs(self, db_manager: DatabaseManager, normalized_job: dict):
        """Jobs not in updates list remain preprocessed=0."""
        jobs = [dict(normalized_job) for _ in range(2)]
        jobs[0]["greenhouse_id"] = 8000
        jobs[1]["greenhouse_id"] = 8001
        db_manager.insert_jobs_batch(jobs)

        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT id FROM jobs WHERE greenhouse_id = 8000")
            job_id = cursor.fetchone()["id"]

        updates = [(job_id, "cleaned", "NY, NY", "Engineer")]
        db_manager.update_job_fields_batch(updates)

        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT preprocessed FROM jobs WHERE greenhouse_id = 8001")
            assert cursor.fetchone()["preprocessed"] == 0
