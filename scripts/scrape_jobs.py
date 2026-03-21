#!/usr/bin/env python3
"""
Concurrent Greenhouse job scraper.

Reads job board tokens from .env, scrapes all boards concurrently using asyncio,
and stores results in data/jobs.db (SQLite).
"""

import asyncio
import logging
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path so we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.greenhouse_scraper import scrape_greenhouse_board, GreenhouseJob

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


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
                UNIQUE(external_id, board_token)
            )
        """)
        conn.commit()
    finally:
        conn.close()


def scrape_board_safe(token: str) -> tuple[str, list[GreenhouseJob] | Exception]:
    """
    Wrap scrape_greenhouse_board to handle errors gracefully.

    Returns a tuple of (board_token, jobs_list_or_exception).
    Never raises — exceptions are returned in the tuple.
    """
    logger.info(f"[{token}] Starting scrape...")
    try:
        jobs = scrape_greenhouse_board(token)
        logger.info(f"[{token}] Done — {len(jobs)} jobs fetched.")
        return (token, jobs)
    except Exception as e:
        logger.error(f"[{token}] ERROR: {e}")
        return (token, e)


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
                ))

            if rows:
                cursor.executemany(
                    """INSERT OR IGNORE INTO jobs
                       (external_id, board_token, title, location, description,
                        source, source_url, company_name, department, job_type, scraped_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
    load_dotenv()

    # Parse env vars
    raw_tokens = os.getenv("GREENHOUSE_BOARD_TOKENS", "")
    board_tokens = [t.strip() for t in raw_tokens.split(",") if t.strip()]
    db_path = os.getenv("DB_PATH", "data/jobs.db")

    if not board_tokens:
        logger.error("ERROR: GREENHOUSE_BOARD_TOKENS is not set or empty in .env")
        sys.exit(1)

    logger.info(f"Boards to scrape: {board_tokens}")
    logger.info(f"Database: {db_path}\n")

    # Initialize DB
    init_db(db_path)

    # Scrape all boards concurrently
    max_workers = min(len(board_tokens), 10)
    results = await scrape_all_boards(board_tokens, max_workers)

    logger.info("")  # Blank line before summary

    # Write results to DB
    total = write_jobs_to_db(db_path, results)

    # Print summary
    failed = [t for t, r in results if isinstance(r, Exception)]
    logger.info(f"Summary: {total} jobs inserted into {db_path}")
    if failed:
        logger.info(f"Failed boards ({len(failed)}): {', '.join(failed)}")
    else:
        logger.info("All boards scraped successfully.")


if __name__ == "__main__":
    asyncio.run(main())
