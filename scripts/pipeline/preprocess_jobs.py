"""Orchestration script: preprocess all job descriptions in the database."""

import argparse
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import DB_DEFAULT_PATH, DB_CHUNK_SIZE
from src.db_utils import add_column_if_missing
from src.preprocess import preprocess_description
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
        logger.info("Jobs to preprocess: %d", total)

        processed = 0
        start = time.monotonic()

        while True:
            # Always query at OFFSET 0; committed rows drop out of WHERE preprocessed=0
            cur.execute(
                "SELECT id, description FROM jobs WHERE preprocessed=0 LIMIT ? OFFSET 0",
                (DB_CHUNK_SIZE,),
            )
            batch = cur.fetchall()
            if not batch:
                break

            # Preprocess each job — isolate failures to prevent batch abort
            updates = []
            for row in batch:
                try:
                    cleaned = preprocess_description(row["description"])
                except Exception as e:
                    logger.warning(
                        "Failed to preprocess job id=%s: %s. Falling back to empty string.",
                        row["id"],
                        e,
                    )
                    cleaned = ""

                updates.append((cleaned, row["id"]))

            # Update database
            cur.executemany(
                "UPDATE jobs SET cleaned_description=?, preprocessed=1 WHERE id=?",
                updates,
            )
            conn.commit()

            # Log progress with throughput
            processed += len(batch)
            elapsed = time.monotonic() - start
            throughput = processed / elapsed if elapsed > 0 else 0
            logger.info(
                "Processed %d/%d (%.1fs elapsed, %.1f jobs/sec)",
                processed,
                total,
                elapsed,
                throughput,
            )

        elapsed_total = time.monotonic() - start
        avg_throughput = processed / elapsed_total if elapsed_total > 0 else 0
        logger.info(
            "Done. %d jobs preprocessed in %.1fs (%.1f jobs/sec avg)",
            processed,
            elapsed_total,
            avg_throughput,
        )
    finally:
        conn.close()


def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Preprocess all job descriptions in the database."
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite database path (default: DB_PATH env var or config default)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Configure logging (after argparse)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # Load environment variables (after argparse, after logging)
    load_dotenv()

    # Resolve database path: arg → env var → default
    db_path = args.db_path or os.getenv("DB_PATH", DB_DEFAULT_PATH)

    run_preprocessing(db_path)


if __name__ == "__main__":
    main()
