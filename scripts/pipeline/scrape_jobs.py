#!/usr/bin/env python3
"""
Concurrent Greenhouse job scraper.

Reads job board tokens from .env, scrapes all boards concurrently using asyncio,
and stores results in data/jobs.db (SQLite).
"""

import argparse
import asyncio
import logging
import os
import sqlite3
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv

# Add project root to path so we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DB_DEFAULT_PATH
from src.greenhouse_scraper import scrape_greenhouse_board, GreenhouseJob

logger = logging.getLogger(__name__)

# Retry configuration for scraping
_SCRAPE_MAX_RETRIES: int = 3
_SCRAPE_RETRY_BASE_DELAY: float = 2.0


def init_db(db_path: str) -> None:
    """Create the jobs table if it doesn't exist."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                external_id  TEXT NOT NULL,
                board_token  TEXT NOT NULL,
                title        TEXT,
                location     TEXT,
                description  TEXT,
                source       TEXT,
                source_url   TEXT,
                company_name TEXT,
                department   TEXT,
                job_type     TEXT,
                scraped_at   TEXT,
                updated_at   TEXT,
                created_at   TEXT,
                UNIQUE(external_id, board_token)
            )
        """)
        conn.commit()
    finally:
        conn.close()


def scrape_board_safe(token: str) -> tuple[str, list[GreenhouseJob] | Exception]:
    """
    Wrap scrape_greenhouse_board to handle errors gracefully with retries.

    Retries transient network errors (ConnectionError, Timeout) with exponential backoff.
    Non-retryable exceptions fail immediately. Returns (board_token, jobs_list_or_exception).
    Never raises — exceptions are returned in the tuple.
    """
    logger.info("[%s] Starting scrape...", token)

    for attempt in range(1, _SCRAPE_MAX_RETRIES + 1):
        try:
            jobs = scrape_greenhouse_board(token)
            logger.info("[%s] Done — %d jobs.", token, len(jobs))
            return (token, jobs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < _SCRAPE_MAX_RETRIES:
                # Exponential backoff: 2.0s, 4.0s, 8.0s
                delay = _SCRAPE_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "[%s] Transient error (attempt %d/%d): %s. Retrying in %.1fs...",
                    token,
                    attempt,
                    _SCRAPE_MAX_RETRIES,
                    type(e).__name__,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "[%s] Transient error after %d attempts: %s",
                    token,
                    _SCRAPE_MAX_RETRIES,
                    e,
                )
                return (token, e)
        except Exception as e:
            # Non-retryable exception — fail immediately
            logger.error("[%s] Non-retryable error: %s", token, e)
            return (token, e)

    # Should not reach here, but safeguard
    return (token, Exception("Unknown scrape error"))


async def scrape_all_boards(tokens: list[str], max_workers: int) -> list[tuple]:
    """
    Scrape all boards concurrently using ThreadPoolExecutor.

    Returns a list of (board_token, jobs_list_or_exception) tuples.
    """
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(executor, scrape_board_safe, token)
            for token in tokens
        ]
        results = await asyncio.gather(*tasks)
    return results


def write_jobs_to_db(db_path: str, results: list[tuple]) -> int:
    """
    Write scraped jobs to SQLite.

    Skips boards where the result is an Exception.
    Uses INSERT OR IGNORE to make re-runs idempotent.

    Returns total number of rows inserted.
    """
    total_inserted = 0
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        for token, result in results:
            if isinstance(result, Exception):
                # Already logged during scrape; skip this board
                continue

            rows = []
            for job in result:
                d = job.to_dict()
                rows.append((
                    d['external_id'],
                    token,
                    d['title'],
                    d['location'],
                    d['description'],
                    d['source'],
                    d['source_url'],
                    d['company_name'],
                    d['department'],
                    d['job_type'],
                    d['scraped_at'],
                    job.updated_at,  # Not in to_dict(), pull directly
                    job.created_at,  # Not in to_dict(), pull directly
                ))

            if rows:
                cursor.executemany(
                    """INSERT OR IGNORE INTO jobs
                       (external_id, board_token, title, location, description,
                        source, source_url, company_name, department, job_type, scraped_at, updated_at, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows,
                )
                inserted = cursor.rowcount
                total_inserted += inserted

            conn.commit()

        return total_inserted
    finally:
        conn.close()


async def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Scrape job listings from Greenhouse job boards and store in SQLite."
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

    # Parse board tokens from environment variable
    board_tokens = [t.strip() for t in os.getenv("GREENHOUSE_BOARD_TOKENS", "").split(",") if t.strip()]

    # Generate unique run ID for tracing this execution
    run_id = uuid.uuid4().hex[:8]

    # Resolve database path: arg → env var → default
    db_path = args.db_path or os.getenv("DB_PATH", DB_DEFAULT_PATH)

    if not board_tokens:
        logger.error("ERROR: GREENHOUSE_BOARD_TOKENS is not set or empty in .env")
        sys.exit(1)

    logger.info("[%s] Boards to scrape: %s", run_id, board_tokens)
    logger.info("[%s] Database: %s\n", run_id, db_path)

    # Initialize DB and time it
    start_init = time.monotonic()
    init_db(db_path)
    elapsed_init = time.monotonic() - start_init
    logger.info("[%s] init_db completed in %.2fs", run_id, elapsed_init)

    # Scrape all boards concurrently and time it
    max_workers = min(len(board_tokens), 10)
    start_scrape = time.monotonic()
    results = await scrape_all_boards(board_tokens, max_workers)
    elapsed_scrape = time.monotonic() - start_scrape
    logger.info("[%s] scrape_all_boards completed in %.2fs", run_id, elapsed_scrape)

    logger.info("")  # Blank line before summary

    # Write results to DB and time it
    start_write = time.monotonic()
    total = write_jobs_to_db(db_path, results)
    elapsed_write = time.monotonic() - start_write
    logger.info("[%s] write_jobs_to_db completed in %.2fs", run_id, elapsed_write)

    # Print summary
    failed = [t for t, r in results if isinstance(r, Exception)]
    logger.info("[%s] Summary: %d jobs inserted into %s", run_id, total, db_path)
    if failed:
        logger.info("[%s] Failed boards (%d): %s", run_id, len(failed), ", ".join(failed))
    else:
        logger.info("[%s] All boards scraped successfully.", run_id)


if __name__ == "__main__":
    asyncio.run(main())
