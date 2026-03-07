import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Optional

from src.utils import serialize_list, deserialize_list


class DatabaseManager:
    """Manages SQLite database connections and schema for the pipeline."""

    def __init__(self, db_path: str):
        """Initialize the database manager.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path

    def initialize_schema(self) -> None:
        """Create all tables and indexes if they do not exist.

        Safe to call on every startup (idempotent via IF NOT EXISTS).
        """
        with self.get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_hash            TEXT    NOT NULL UNIQUE,
                    greenhouse_id       INTEGER NOT NULL,
                    board_token         TEXT    NOT NULL,
                    title               TEXT    NOT NULL,
                    company             TEXT    NOT NULL,
                    location            TEXT,
                    raw_description     TEXT,
                    cleaned_description TEXT,
                    employment_type     TEXT,
                    departments         TEXT,
                    offices             TEXT,
                    absolute_url        TEXT,
                    updated_at_source   TEXT,
                    collected_at        TEXT    NOT NULL,
                    preprocessed        INTEGER NOT NULL DEFAULT 0,
                    extracted           INTEGER NOT NULL DEFAULT 0,
                    embedded            INTEGER NOT NULL DEFAULT 0,
                    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_hash ON jobs(job_hash);
                CREATE INDEX IF NOT EXISTS idx_jobs_preprocessed ON jobs(preprocessed);
                CREATE INDEX IF NOT EXISTS idx_jobs_greenhouse_id ON jobs(greenhouse_id);

                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date        TEXT    NOT NULL,
                    step            TEXT    NOT NULL,
                    jobs_processed  INTEGER NOT NULL DEFAULT 0,
                    jobs_skipped    INTEGER NOT NULL DEFAULT 0,
                    status          TEXT    NOT NULL,
                    error_message   TEXT,
                    started_at      TEXT    NOT NULL DEFAULT (datetime('now')),
                    finished_at     TEXT
                );
                """
            )

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding a sqlite3.Connection with WAL mode enabled.

        Features:
        - WAL (Write-Ahead Log) journal mode for concurrent read/write
        - row_factory = sqlite3.Row for dict-like access
        - Auto-commits on clean exit, auto-rollsback on exception

        Yields:
            sqlite3.Connection instance
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def insert_job(self, job: dict) -> bool:
        """INSERT OR IGNORE a job row.

        Uses UNIQUE constraint on job_hash to prevent duplicates.

        Args:
            job: Dictionary with keys: greenhouse_id, board_token, title, company,
                 location, raw_description, absolute_url, updated_at_source,
                 departments, offices, collected_at

        Returns:
            True if inserted, False if skipped (already exists)
        """
        from src.utils import compute_job_hash

        job_hash = compute_job_hash(
            job["greenhouse_id"], job["board_token"], job["title"]
        )

        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO jobs (
                    job_hash, greenhouse_id, board_token, title, company, location,
                    raw_description, absolute_url, updated_at_source,
                    departments, offices, collected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_hash,
                    job["greenhouse_id"],
                    job["board_token"],
                    job["title"],
                    job["company"],
                    job.get("location"),
                    job.get("raw_description"),
                    job.get("absolute_url"),
                    job.get("updated_at_source"),
                    job.get("departments"),
                    job.get("offices"),
                    job.get("collected_at"),
                ),
            )
            return cursor.rowcount > 0

    def get_unpreprocessed_jobs(self) -> list[sqlite3.Row]:
        """Fetch all jobs WHERE preprocessed = 0 ORDER BY id.

        Returns:
            List of sqlite3.Row objects (dict-like access to columns)
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE preprocessed = 0 ORDER BY id"
            )
            return cursor.fetchall()

    def get_unpreprocessed_jobs_chunked(self, chunk_size: int, offset: int) -> list[tuple[int, str | None]]:
        """Fetch a chunk of unpreprocessed jobs by id and raw_description only.

        Returns plain tuples (id, raw_description) instead of sqlite3.Row objects
        for compatibility with multiprocessing.Pool (rows are not picklable).

        Args:
            chunk_size: Number of records to fetch
            offset: Number of records to skip

        Returns:
            List of (id, raw_description) tuples
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, raw_description FROM jobs WHERE preprocessed = 0 ORDER BY id LIMIT ? OFFSET ?",
                (chunk_size, offset),
            )
            return [(row[0], row[1]) for row in cursor.fetchall()]

    def update_cleaned_description(self, job_id: int, cleaned: str) -> None:
        """Update cleaned_description and set preprocessed = 1 for a job.

        Args:
            job_id: Primary key ID of the job row
            cleaned: Cleaned plain-text description
        """
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET cleaned_description = ?, preprocessed = 1
                WHERE id = ?
                """,
                (cleaned, job_id),
            )

    def insert_jobs_batch(self, jobs: list[dict]) -> tuple[int, int]:
        """INSERT OR IGNORE a batch of job rows in a single transaction.

        Args:
            jobs: List of dicts with keys matching insert_job() expectations

        Returns:
            Tuple of (inserted_count, skipped_count)
        """
        if not jobs:
            return 0, 0

        from src.utils import compute_job_hash

        rows = [
            (
                compute_job_hash(j["greenhouse_id"], j["board_token"], j["title"]),
                j["greenhouse_id"],
                j["board_token"],
                j["title"],
                j["company"],
                j.get("location"),
                j.get("raw_description"),
                j.get("absolute_url"),
                j.get("updated_at_source"),
                j.get("departments"),
                j.get("offices"),
                j.get("collected_at"),
            )
            for j in jobs
        ]

        with self.get_connection() as conn:
            before = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            conn.executemany(
                """
                INSERT OR IGNORE INTO jobs (
                    job_hash, greenhouse_id, board_token, title, company, location,
                    raw_description, absolute_url, updated_at_source,
                    departments, offices, collected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            after = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

        inserted = after - before
        return inserted, len(jobs) - inserted

    def update_cleaned_descriptions_batch(self, updates: list[tuple[int, str]]) -> None:
        """Update cleaned_description and set preprocessed=1 for multiple jobs in one transaction.

        Args:
            updates: List of (job_id, cleaned_description) tuples
        """
        if not updates:
            return

        rows = [(cleaned, job_id) for job_id, cleaned in updates]
        with self.get_connection() as conn:
            conn.executemany(
                """
                UPDATE jobs
                SET cleaned_description = ?, preprocessed = 1
                WHERE id = ?
                """,
                rows,
            )

    def create_pipeline_run(self, run_date: str, step: str) -> int:
        """Create a pipeline_runs audit log entry.

        Args:
            run_date: Date in YYYY-MM-DD format
            step: Step name (collection, preprocessing, extraction, embedding)

        Returns:
            ID of the created run record
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO pipeline_runs (run_date, step, status)
                VALUES (?, ?, 'running')
                """,
                (run_date, step),
            )
            return cursor.lastrowid

    def finish_pipeline_run(
        self,
        run_id: int,
        status: str,
        jobs_processed: int,
        jobs_skipped: int,
        error_message: Optional[str] = None,
    ) -> None:
        """Update pipeline_runs with final status and metrics.

        Args:
            run_id: ID of the pipeline_runs record to update
            status: Final status (success, failed)
            jobs_processed: Number of jobs processed
            jobs_skipped: Number of jobs skipped
            error_message: Optional error message if status is failed
        """
        finished_at = datetime.utcnow().isoformat() + "Z"
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE pipeline_runs
                SET status = ?, jobs_processed = ?, jobs_skipped = ?,
                    error_message = ?, finished_at = ?
                WHERE id = ?
                """,
                (status, jobs_processed, jobs_skipped, error_message, finished_at, run_id),
            )
