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
                    is_us               INTEGER,
                    is_target_role      INTEGER,
                    preprocessed        INTEGER NOT NULL DEFAULT 0,
                    extracted           INTEGER NOT NULL DEFAULT 0,
                    embedded            INTEGER NOT NULL DEFAULT 0,
                    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_hash ON jobs(job_hash);
                CREATE INDEX IF NOT EXISTS idx_jobs_preprocessed ON jobs(preprocessed);
                CREATE INDEX IF NOT EXISTS idx_jobs_greenhouse_id ON jobs(greenhouse_id);

                CREATE TABLE IF NOT EXISTS job_extractions (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id                  INTEGER NOT NULL UNIQUE REFERENCES jobs(id),
                    job_title               TEXT,
                    responsibilities        TEXT,
                    skills                  TEXT,
                    tools_and_platforms     TEXT,
                    education               TEXT,
                    experience_min_years    INTEGER,
                    experience_is_inferred  INTEGER,
                    extracted_at            TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_job_extractions_job_id ON job_extractions(job_id);

                CREATE TABLE IF NOT EXISTS job_embeddings (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id      INTEGER NOT NULL UNIQUE REFERENCES jobs(id),
                    embedding   BLOB    NOT NULL,
                    model_id    TEXT    NOT NULL,
                    embedded_at TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_job_embeddings_job_id ON job_embeddings(job_id);

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

                CREATE TABLE IF NOT EXISTS job_matches (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id          INTEGER NOT NULL UNIQUE REFERENCES jobs(id),
                    score           REAL    NOT NULL,
                    rank            INTEGER NOT NULL,
                    model_id        TEXT    NOT NULL,
                    retrieved_at    TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_job_matches_rank ON job_matches(rank);
                CREATE INDEX IF NOT EXISTS idx_job_matches_job_id ON job_matches(job_id);

                CREATE TABLE IF NOT EXISTS job_reranked (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id      INTEGER NOT NULL UNIQUE REFERENCES jobs(id),
                    score       REAL    NOT NULL,
                    rank        INTEGER NOT NULL,
                    model_id    TEXT    NOT NULL,
                    reranked_at TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_job_reranked_rank ON job_reranked(rank);
                CREATE INDEX IF NOT EXISTS idx_job_reranked_job_id ON job_reranked(job_id);

                CREATE TABLE IF NOT EXISTS job_summaries (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id          INTEGER NOT NULL UNIQUE REFERENCES jobs(id),
                    rank            INTEGER NOT NULL,
                    summary         TEXT    NOT NULL,
                    citations_json  TEXT    NOT NULL,
                    evaluation_json TEXT    NOT NULL,
                    passed_eval     INTEGER NOT NULL DEFAULT 0,
                    model_id        TEXT    NOT NULL,
                    generated_at    TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_job_summaries_job_id ON job_summaries(job_id);
                CREATE INDEX IF NOT EXISTS idx_job_summaries_passed_eval ON job_summaries(passed_eval);

                CREATE TABLE IF NOT EXISTS eval_needles (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id           TEXT    NOT NULL UNIQUE,
                    resume_text         TEXT    NOT NULL,
                    golden_title        TEXT    NOT NULL,
                    golden_company      TEXT    NOT NULL,
                    golden_description  TEXT    NOT NULL,
                    adversarial_title   TEXT    NOT NULL,
                    adversarial_company TEXT    NOT NULL,
                    adversarial_description TEXT NOT NULL,
                    deal_breaker        TEXT    NOT NULL,
                    generator_model_id  TEXT    NOT NULL,
                    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );

                CREATE TABLE IF NOT EXISTS eval_results (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id      TEXT    NOT NULL,
                    resume_id   TEXT    NOT NULL,
                    metric      TEXT    NOT NULL,
                    value       REAL    NOT NULL,
                    computed_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                """
            )
            # Add is_us column for existing databases
            try:
                with self.get_connection() as conn:
                    conn.execute("ALTER TABLE jobs ADD COLUMN is_us INTEGER")
            except sqlite3.OperationalError:
                pass  # column already exists
            # Add is_target_role column for existing databases
            try:
                with self.get_connection() as conn:
                    conn.execute("ALTER TABLE jobs ADD COLUMN is_target_role INTEGER")
            except sqlite3.OperationalError:
                pass  # column already exists
            # Create job_summaries table for existing databases (if schema was updated)
            try:
                with self.get_connection() as conn:
                    conn.executescript(
                        """
                        CREATE TABLE IF NOT EXISTS job_summaries (
                            id              INTEGER PRIMARY KEY AUTOINCREMENT,
                            job_id          INTEGER NOT NULL UNIQUE REFERENCES jobs(id),
                            rank            INTEGER NOT NULL,
                            summary         TEXT    NOT NULL,
                            citations_json  TEXT    NOT NULL,
                            evaluation_json TEXT    NOT NULL,
                            passed_eval     INTEGER NOT NULL DEFAULT 0,
                            model_id        TEXT    NOT NULL,
                            generated_at    TEXT    NOT NULL DEFAULT (datetime('now'))
                        );
                        CREATE INDEX IF NOT EXISTS idx_job_summaries_job_id ON job_summaries(job_id);
                        CREATE INDEX IF NOT EXISTS idx_job_summaries_passed_eval ON job_summaries(passed_eval);
                        """
                    )
            except sqlite3.OperationalError:
                pass  # table already exists

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

    def get_unpreprocessed_jobs_chunked(self, chunk_size: int, offset: int) -> list[tuple[int, str | None, str | None, str]]:
        """Fetch a chunk of unpreprocessed jobs with id, raw_description, location, and title.

        Returns plain tuples instead of sqlite3.Row objects for compatibility with
        multiprocessing.Pool (rows are not picklable).

        Args:
            chunk_size: Number of records to fetch
            offset: Number of records to skip

        Returns:
            List of (id, raw_description, location, title) tuples
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, raw_description, location, title FROM jobs WHERE preprocessed = 0 ORDER BY id LIMIT ? OFFSET ?",
                (chunk_size, offset),
            )
            return [(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()]

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

    def update_job_fields_batch(self, updates: list[tuple[int, str, str | None, str, int | None]]) -> None:
        """Update cleaned_description, location, title, is_us, and set preprocessed=1 in one transaction.

        Args:
            updates: List of (job_id, cleaned_description, cleaned_location, cleaned_title, is_us) tuples
        """
        if not updates:
            return

        rows = [
            (cleaned_desc, cleaned_loc, cleaned_title, is_us, job_id)
            for job_id, cleaned_desc, cleaned_loc, cleaned_title, is_us in updates
        ]
        with self.get_connection() as conn:
            conn.executemany(
                """
                UPDATE jobs
                SET cleaned_description = ?, location = ?, title = ?, is_us = ?, preprocessed = 1
                WHERE id = ?
                """,
                rows,
            )

    def get_unclassified_roles_chunked(self, chunk_size: int, offset: int) -> list[tuple[int, str]]:
        """Fetch a chunk of preprocessed jobs that have not yet been role-classified.

        Args:
            chunk_size: Number of records to fetch
            offset: Number of records to skip

        Returns:
            List of (id, title) tuples
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, title FROM jobs WHERE preprocessed=1 AND is_target_role IS NULL ORDER BY id LIMIT ? OFFSET ?",
                (chunk_size, offset),
            )
            return [(row[0], row[1]) for row in cursor.fetchall()]

    def update_target_role_batch(self, updates: list[tuple[int, int]]) -> None:
        """Set is_target_role for a batch of jobs.

        Args:
            updates: List of (job_id, is_target_role) tuples where is_target_role is 1 or 0
        """
        if not updates:
            return
        rows = [(is_target, job_id) for job_id, is_target in updates]
        with self.get_connection() as conn:
            conn.executemany(
                "UPDATE jobs SET is_target_role = ? WHERE id = ?",
                rows,
            )

    def get_unextracted_jobs_chunked(self, chunk_size: int, offset: int) -> list[tuple[int, str | None, str]]:
        """Fetch a chunk of preprocessed but unextracted jobs.

        Returns plain tuples instead of sqlite3.Row objects for compatibility with
        multiprocessing (rows are not picklable).

        Args:
            chunk_size: Number of records to fetch
            offset: Number of records to skip

        Returns:
            List of (id, cleaned_description, title) tuples
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, cleaned_description, title FROM jobs WHERE preprocessed=1 AND extracted=0 AND is_target_role=1 ORDER BY id LIMIT ? OFFSET ?",
                (chunk_size, offset),
            )
            return [(row[0], row[1], row[2]) for row in cursor.fetchall()]

    def update_extraction_batch(self, updates: list[tuple[int, dict]]) -> None:
        """Write extraction results to job_extractions and set extracted=1 on jobs.

        Args:
            updates: List of (job_id, extracted_dict) tuples where extracted_dict
                     matches EXTRACTION_JSON_SCHEMA structure.
        """
        import json

        if not updates:
            return

        extraction_rows = [
            (
                job_id,
                data.get("job_title"),
                json.dumps(data.get("responsibilities", [])),
                json.dumps(data.get("skills", [])),
                json.dumps(data.get("tools_and_platforms", [])),
                data.get("education"),
                data.get("experience", {}).get("min_years"),
                int(data.get("experience", {}).get("is_inferred", False)),
            )
            for job_id, data in updates
        ]
        job_ids = [job_id for job_id, _ in updates]

        with self.get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO job_extractions
                    (job_id, job_title, responsibilities, skills, tools_and_platforms,
                     education, experience_min_years, experience_is_inferred)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                extraction_rows,
            )
            conn.executemany(
                "UPDATE jobs SET extracted=1 WHERE id=?",
                [(job_id,) for job_id in job_ids],
            )

    def mark_jobs_extracted(self, job_ids: list[int]) -> None:
        """Set extracted=1 on jobs by ID without writing to job_extractions.

        Used by the extraction pipeline to optimistically mark jobs as done
        before the async writer thread flushes the full extraction data.
        """
        if not job_ids:
            return
        with self.get_connection() as conn:
            conn.executemany(
                "UPDATE jobs SET extracted=1 WHERE id=?",
                [(job_id,) for job_id in job_ids],
            )

    def get_unembedded_jobs_chunked(self, chunk_size: int, offset: int) -> list[tuple[int, str | None, str | None, str | None, str | None]]:
        """Fetch a chunk of extracted but unembedded jobs with extraction fields.

        Returns plain tuples for multiprocessing compatibility (sqlite3.Row is not picklable).

        Args:
            chunk_size: Number of records to fetch
            offset: Number of records to skip

        Returns:
            List of (job_id, job_title, responsibilities_json, skills_json, tools_json) tuples
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT j.id, e.job_title, e.responsibilities, e.skills, e.tools_and_platforms
                FROM jobs j
                JOIN job_extractions e ON e.job_id = j.id
                WHERE j.extracted = 1 AND j.embedded = 0 AND j.is_target_role = 1
                ORDER BY j.id
                LIMIT ? OFFSET ?
                """,
                (chunk_size, offset),
            )
            return [(row[0], row[1], row[2], row[3], row[4]) for row in cursor.fetchall()]

    def insert_embeddings_batch(self, updates: list[tuple[int, bytes, str]]) -> None:
        """Write embedding blobs to job_embeddings and set embedded=1 on jobs.

        Args:
            updates: List of (job_id, embedding_blob, model_id) tuples
        """
        if not updates:
            return

        job_ids = [job_id for job_id, _, _ in updates]
        with self.get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO job_embeddings (job_id, embedding, model_id)
                VALUES (?, ?, ?)
                """,
                updates,
            )
            conn.executemany(
                "UPDATE jobs SET embedded=1 WHERE id=?",
                [(job_id,) for job_id in job_ids],
            )

    def mark_jobs_embedded(self, job_ids: list[int]) -> None:
        """Set embedded=1 on jobs by ID without writing to job_embeddings.

        Used by the embedding pipeline to optimistically mark jobs as done
        before the async writer thread flushes the full embedding data.
        """
        if not job_ids:
            return
        with self.get_connection() as conn:
            conn.executemany(
                "UPDATE jobs SET embedded=1 WHERE id=?",
                [(job_id,) for job_id in job_ids],
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

    def get_all_embeddings(self, model_id: str) -> list[tuple[int, bytes]]:
        """Fetch all job embeddings produced by a given model.

        Args:
            model_id: Model identifier to filter by

        Returns:
            List of (job_id, embedding_blob) tuples ordered by job_id
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT e.job_id, e.embedding FROM job_embeddings e JOIN jobs j ON j.id = e.job_id WHERE e.model_id = ? AND j.is_target_role = 1 ORDER BY e.job_id",
                (model_id,),
            )
            return [(row[0], bytes(row[1])) for row in cursor.fetchall()]

    def get_all_cleaned_descriptions(self) -> list[tuple[int, str]]:
        """Fetch cleaned descriptions for all embedded jobs.

        Only returns jobs that have been embedded so both corpora remain aligned.

        Returns:
            List of (job_id, cleaned_description) tuples ordered by job_id
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, cleaned_description
                FROM jobs
                WHERE embedded = 1 AND cleaned_description IS NOT NULL AND is_target_role = 1
                ORDER BY id
                """,
            )
            return [(row[0], row[1]) for row in cursor.fetchall()]

    def insert_job_matches(self, matches: list[tuple[int, float, int, str]]) -> None:
        """Replace all job_matches rows with new retrieval results.

        Deletes all prior results and inserts the new batch in one transaction,
        so re-runs never leave stale rows from a prior run's larger top-k.

        Args:
            matches: List of (job_id, score, rank, model_id) tuples
        """
        if not matches:
            return
        with self.get_connection() as conn:
            conn.execute("DELETE FROM job_matches")
            conn.executemany(
                """
                INSERT INTO job_matches (job_id, score, rank, model_id)
                VALUES (?, ?, ?, ?)
                """,
                matches,
            )

    def get_job_matches_with_text(
        self, limit: Optional[int] = None
    ) -> list[tuple[int, str, str, str]]:
        """Fetch job_matches joined with job text fields for reranking.

        Args:
            limit: Optional cap on number of results

        Returns:
            List of (job_id, title, cleaned_description, company) ordered by rank ASC
        """
        sql = """
            SELECT m.job_id, j.title, j.cleaned_description, j.company
            FROM job_matches m
            JOIN jobs j ON j.id = m.job_id
            ORDER BY m.rank ASC
        """
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            return [(row[0], row[1] or "", row[2] or "", row[3] or "") for row in cursor.fetchall()]

    def insert_reranked(self, matches: list[tuple[int, float, int, str]]) -> None:
        """Replace all job_reranked rows with new results (DELETE + INSERT).

        Args:
            matches: List of (job_id, score, rank, model_id) tuples
        """
        if not matches:
            return
        with self.get_connection() as conn:
            conn.execute("DELETE FROM job_reranked")
            conn.executemany(
                """
                INSERT INTO job_reranked (job_id, score, rank, model_id)
                VALUES (?, ?, ?, ?)
                """,
                matches,
            )

    def get_job_matches(self, limit: Optional[int] = None) -> list[tuple[int, float, int]]:
        """Read job_matches ordered by rank ascending.

        Called by reranking.py to consume retrieval results.

        Args:
            limit: Optional cap on number of results

        Returns:
            List of (job_id, score, rank) tuples
        """
        sql = "SELECT job_id, score, rank FROM job_matches ORDER BY rank ASC"
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            return [(row[0], row[1], row[2]) for row in cursor.fetchall()]

    def get_reranked_with_full_text(self, limit: int) -> list[tuple]:
        """Fetch reranked jobs with full text data for generation.

        Joins reranked scores with job and extraction details. LEFT JOIN on
        job_extractions is defensive against missing extraction records.

        Args:
            limit: Maximum number of results to return

        Returns:
            List of tuples: (job_id, rank, score, title, company, location,
                            absolute_url, cleaned_description, responsibilities,
                            skills, tools_and_platforms, experience_min_years)
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT r.job_id, r.rank, r.score,
                       j.title, j.company, j.location, j.absolute_url, j.cleaned_description,
                       e.responsibilities, e.skills, e.tools_and_platforms, e.experience_min_years
                FROM job_reranked r
                JOIN jobs j ON j.id = r.job_id
                LEFT JOIN job_extractions e ON e.job_id = r.job_id
                ORDER BY r.rank ASC
                LIMIT ?
                """,
                (limit,),
            )
            return cursor.fetchall()

    def insert_summaries(self, summaries: list[dict]) -> None:
        """Replace all job_summaries rows with new results (DELETE + INSERT).

        Idempotent: deletes all rows then inserts new batch in one transaction.

        Args:
            summaries: List of dicts with keys: job_id, rank, summary, citations_json,
                      evaluation_json, passed_eval, model_id
        """
        if not summaries:
            return

        rows = [
            (
                s["job_id"],
                s["rank"],
                s["summary"],
                s["citations_json"],
                s["evaluation_json"],
                s["passed_eval"],
                s["model_id"],
            )
            for s in summaries
        ]

        with self.get_connection() as conn:
            conn.execute("DELETE FROM job_summaries")
            conn.executemany(
                """
                INSERT INTO job_summaries
                    (job_id, rank, summary, citations_json, evaluation_json, passed_eval, model_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
