"""
Preprocessing module for job descriptions.

Converts raw HTML-encoded job descriptions into clean plain text through a 5-step pipeline:
1. Unescape HTML entities (handling double-encoding)
2. Strip iframes and images
3. Extract plain text from HTML
4. Normalize whitespace
5. Normalize Unicode punctuation to ASCII equivalents
"""

import html
import logging
import re
import sqlite3
import time
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500

# Unicode replacements for Step 5
_UNICODE_TABLE = str.maketrans({
    '\u2018': "'",   # left single quotation mark
    '\u2019': "'",   # right single quotation mark
    '\u201c': '"',   # left double quotation mark
    '\u201d': '"',   # right double quotation mark
    '\u2014': '-',   # em dash
    '\u2013': '-',   # en dash
    '\u2026': '...',  # ellipsis
})

# Regex for collapsing whitespace (Step 4)
_WHITESPACE_RE = re.compile(r'[\s\xa0]+')


def _step1_unescape(raw: str) -> str:
    """Step 1: Unescape HTML entities, handling double-encoding."""
    for _ in range(5):
        decoded = html.unescape(raw)
        if decoded == raw:
            break
        raw = decoded
    return raw


def _step2_strip_iframes_and_images(html_text: str) -> str:
    """Step 2: Remove iframe and img tags using BeautifulSoup."""
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup.find_all(["iframe", "img"]):
        tag.decompose()
    return str(soup)


def _step3_extract_text(html_text: str) -> str:
    """Step 3: Extract plain text from HTML, using space as separator between blocks."""
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(separator=" ")


def _step4_normalize_whitespace(text: str) -> str:
    """Step 4: Collapse all whitespace sequences to single space and strip."""
    return _WHITESPACE_RE.sub(' ', text).strip()


def _step5_normalize_unicode_punctuation(text: str) -> str:
    """Step 5: Replace curly quotes, em-dashes, and ellipses with ASCII equivalents."""
    return text.translate(_UNICODE_TABLE)


def preprocess_description(raw: str) -> str:
    """
    Apply all 5 preprocessing steps to a raw job description.

    Args:
        raw: Raw HTML-encoded job description (may be None or empty).

    Returns:
        Clean plain text description, or empty string if input is None/empty.
    """
    if not raw:
        return ''

    text = _step1_unescape(raw)
    text = _step2_strip_iframes_and_images(text)
    text = _step3_extract_text(text)
    text = _step4_normalize_whitespace(text)
    text = _step5_normalize_unicode_punctuation(text)
    return text


def _add_column_if_missing(cursor: sqlite3.Cursor, table: str, column: str, col_type: str) -> None:
    """Add a column to a table if it doesn't already exist."""
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except sqlite3.OperationalError:
        # Column already exists
        pass


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
        _add_column_if_missing(cur, "jobs", "cleaned_description", "TEXT")
        _add_column_if_missing(cur, "jobs", "preprocessed", "INTEGER DEFAULT 0")
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_preprocessing("data/jobs.db")
