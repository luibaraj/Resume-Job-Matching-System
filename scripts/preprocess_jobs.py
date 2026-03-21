"""Orchestration script: preprocess all job descriptions in the database."""

import logging
import os
import sqlite3
import sys
import time

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db_utils import add_column_if_missing
from src.preprocess import preprocess_description

CHUNK_SIZE = 500
logger = logging.getLogger(__name__)


def run_preprocessing(db_path: str) -> None:
    """
    Preprocess all unprocessed job descriptions in the database.

    Adds cleaned_description and preprocessed columns if missing, then processes
    all jobs where preprocessed=0 in batches, updating the database.

    Args:
        db_path: Path to the SQLite database.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        # Add columns if missing
        add_column_if_missing(cur, "jobs", "cleaned_description", "TEXT")
        add_column_if_missing(cur, "jobs", "preprocessed", "INTEGER DEFAULT 0")
        conn.commit()

        # Count total jobs to preprocess
        cur.execute("SELECT COUNT(*) FROM jobs WHERE preprocessed=0")
        total = cur.fetchone()[0]
        logger.info(f"Jobs to preprocess: {total}")

        processed = 0
        start = time.monotonic()

        while True:
            # Always query at OFFSET 0; committed rows drop out of WHERE preprocessed=0
            cur.execute(
                "SELECT id, description FROM jobs WHERE preprocessed=0 LIMIT ? OFFSET 0",
                (CHUNK_SIZE,),
            )
            batch = cur.fetchall()
            if not batch:
                break

            # Preprocess each job
            updates = [
                (preprocess_description(row["description"]), row["id"])
                for row in batch
            ]

            # Update database
            cur.executemany(
                "UPDATE jobs SET cleaned_description=?, preprocessed=1 WHERE id=?",
                updates,
            )
            conn.commit()

            # Log progress
            processed += len(batch)
            elapsed = time.monotonic() - start
            logger.info(f"Processed {processed}/{total} ({elapsed:.1f}s elapsed)")

        logger.info(f"Done. {processed} jobs preprocessed.")
    finally:
        conn.close()


def main():
    load_dotenv()
    db_path = os.getenv("DB_PATH", "data/jobs.db")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_preprocessing(db_path)


if __name__ == "__main__":
    main()
