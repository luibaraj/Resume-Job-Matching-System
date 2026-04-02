"""Orchestration script: embed all preprocessed job descriptions using Voyage AI."""

import argparse
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import DB_DEFAULT_PATH, VOYAGE_BATCH_SIZE
from src.db_utils import add_column_if_missing
from src.embedding import create_client, embed_batch, serialize_embedding

# DB_CHUNK_SIZE: rows fetched from SQLite per outer loop iteration.
# Power-of-two alignment (512) improves memory efficiency and aligns with
# typical batch processing patterns across embedding and preprocessing.
DB_CHUNK_SIZE = 512

logger = logging.getLogger(__name__)


def run_embedding(db_path: str, voyage_api_key: str) -> None:
    """
    Embed all unembedded, preprocessed job descriptions in the database.

    Adds embedding and embedded columns if missing, then processes all jobs
    where embedded=0 and cleaned_description is non-empty, in batches.

    Args:
        db_path: Path to the SQLite database.
        voyage_api_key: Voyage AI API key.
    """
    client = create_client(voyage_api_key)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        add_column_if_missing(cur, "jobs", "embedding", "BLOB")
        add_column_if_missing(cur, "jobs", "embedded", "INTEGER DEFAULT 0")
        conn.commit()

        cur.execute(
            "SELECT COUNT(*) FROM jobs "
            "WHERE embedded=0 AND cleaned_description IS NOT NULL AND cleaned_description != ''"
        )
        total = cur.fetchone()[0]
        logger.info("Jobs to embed: %d", total)

        embedded_count = 0
        skipped_count = 0
        start = time.monotonic()

        while True:
            cur.execute(
                "SELECT id, cleaned_description FROM jobs "
                "WHERE embedded=0 AND cleaned_description IS NOT NULL AND cleaned_description != '' "
                "LIMIT ? OFFSET 0",
                (DB_CHUNK_SIZE,),
            )
            chunk = cur.fetchall()
            if not chunk:
                break

            # Split into Voyage AI sub-batches
            for batch_start in range(0, len(chunk), VOYAGE_BATCH_SIZE):
                sub_batch = chunk[batch_start : batch_start + VOYAGE_BATCH_SIZE]
                ids = [row["id"] for row in sub_batch]
                texts = [row["cleaned_description"] for row in sub_batch]
                min_id = min(ids)
                max_id = max(ids)

                # Time sub-batch embedding
                sub_batch_start = time.monotonic()
                try:
                    embeddings = embed_batch(client, texts)
                except Exception as exc:
                    logger.error(
                        "embed_batch raised after exhausting retries — %s: %s",
                        type(exc).__name__,
                        exc,
                    )
                    skipped_count += len(sub_batch)
                    continue

                sub_batch_elapsed = time.monotonic() - sub_batch_start
                # Log at DEBUG level with per-sub-batch timing
                logger.debug(
                    "Embedded sub-batch of %d jobs (id range %d-%d) in %.2fs",
                    len(sub_batch),
                    min_id,
                    max_id,
                    sub_batch_elapsed,
                )

                # Serialize and write back
                updates = [
                    (serialize_embedding(emb), row_id)
                    for emb, row_id in zip(embeddings, ids)
                ]
                cur.executemany(
                    "UPDATE jobs SET embedding=?, embedded=1 WHERE id=?",
                    updates,
                )
                conn.commit()
                embedded_count += len(sub_batch)

            elapsed = time.monotonic() - start
            logger.info(
                "Progress: %d embedded, %d skipped (%.1fs elapsed)",
                embedded_count,
                skipped_count,
                elapsed,
            )

        elapsed_total = time.monotonic() - start
        throughput = embedded_count / elapsed_total if elapsed_total > 0 else 0
        logger.info(
            "Done. %d jobs embedded, %d skipped in %.1fs (%.1f jobs/sec avg)",
            embedded_count,
            skipped_count,
            elapsed_total,
            throughput,
        )
    finally:
        conn.close()


def main() -> None:
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Embed all preprocessed job descriptions using Voyage AI."
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
    voyage_api_key = os.getenv("VOYAGE_API_KEY", "")

    if not voyage_api_key:
        logger.error("VOYAGE_API_KEY is not set in .env")
        sys.exit(1)

    run_embedding(db_path, voyage_api_key)


if __name__ == "__main__":
    main()
