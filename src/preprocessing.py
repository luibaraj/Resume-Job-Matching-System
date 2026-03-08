import html
import multiprocessing
import re
import traceback
from typing import Optional

from src.config import load_config
from src.database import DatabaseManager
from src.utils import setup_logging


# Regex patterns for preprocessing
LI_TAG_PATTERN = re.compile(r"<li[^>]*>", re.IGNORECASE)
LI_CLOSE_PATTERN = re.compile(r"</li>", re.IGNORECASE)
BULLET_PATTERN = re.compile(r"[•·]\s*")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
MULTI_SPACE_PATTERN = re.compile(r"[ \t\xa0]+")
MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")


def decode_html_entities(text: str) -> str:
    """Decode HTML entities to their Unicode equivalents.

    Uses html.unescape() from the standard library.

    Examples:
        "&amp;" -> "&"
        "&lt;" -> "<"
        "&gt;" -> ">"
        "&#39;" -> "'"
        "&nbsp;" -> "\xa0" (non-breaking space, handled in whitespace step)

    Args:
        text: String potentially containing HTML entities

    Returns:
        String with HTML entities decoded
    """
    return html.unescape(text)


def normalize_list_items(text: str) -> str:
    """Replace HTML list item tags and bullet characters with "- " prefix.

    Handles:
        <li>content</li> -> "- content"
        <li>content -> "- content" (unclosed tags)
        • content -> "- content" (Unicode bullet U+2022)
        · content -> "- content" (Unicode middle dot U+00B7)

    Args:
        text: String potentially containing list items

    Returns:
        String with list items normalized to "- " prefix
    """
    # Replace opening <li> tags with "- " (handles attributes like <li class="...">)
    text = LI_TAG_PATTERN.sub("- ", text)
    # Remove closing </li> tags (content is already prefixed)
    text = LI_CLOSE_PATTERN.sub("", text)
    # Replace bullet characters with "- "
    text = BULLET_PATTERN.sub("- ", text)
    return text


def strip_html_tags(text: str) -> str:
    """Remove all remaining HTML tags, replacing each with a single space.

    Uses regex <[^>]+> to match any tag including attributes.

    Args:
        text: String potentially containing HTML tags

    Returns:
        String with all HTML tags removed/replaced with spaces
    """
    return HTML_TAG_PATTERN.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace sequences to single spaces.

    Operations:
        - Collapse spaces, tabs, non-breaking spaces (\xa0) to single space
        - Normalize Windows/Mac line endings to Unix (\n)
        - Collapse 3+ blank lines to 2 blank lines
        - Strip leading and trailing whitespace from final string

    Args:
        text: String potentially containing irregular whitespace

    Returns:
        String with normalized whitespace
    """
    # Normalize line endings: CRLF or CR to LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple spaces/tabs/nbsp to single space
    text = MULTI_SPACE_PATTERN.sub(" ", text)
    # Collapse 3+ blank lines to 2 blank lines (max 1 blank line)
    text = MULTI_NEWLINE_PATTERN.sub("\n\n", text)
    # Strip leading/trailing whitespace from entire string
    text = text.strip()
    return text


def clean_job_description(raw_description: Optional[str]) -> str:
    """Orchestrator: run all four cleaning steps in sequence.

    Handles None/empty input gracefully, returning empty string.
    This is the single public API for the preprocessing module.

    Step order is critical:
        1. Entity decoding (before tag stripping so &lt;div&gt; is handled)
        2. List item normalization (semantic replacement before generic removal)
        3. HTML tag stripping (generic tag removal)
        4. Whitespace normalization (after all other steps)

    Args:
        raw_description: Raw HTML job description or None

    Returns:
        Cleaned plain-text description
    """
    if not raw_description:
        return ""

    text = raw_description
    text = decode_html_entities(text)
    text = normalize_list_items(text)
    text = strip_html_tags(text)
    text = normalize_whitespace(text)
    return text


def _clean_record(record: tuple[int, str | None]) -> tuple[int, str] | None:
    """Worker function for multiprocessing.Pool.

    Must be module-level (not a closure) for pickle serialization to worker processes.
    Returns None on exception to enable per-record fault tolerance.
    Configures its own logger because worker processes do not inherit the
    parent process's logging configuration.

    Args:
        record: Tuple of (job_id, raw_description)

    Returns:
        Tuple of (job_id, cleaned_description) on success, None on exception
    """
    job_id, raw_description = record
    try:
        return (job_id, clean_job_description(raw_description))
    except Exception as e:
        logger = setup_logging(name="_clean_record")
        logger.exception(
            "Failed to clean record job_id=%s: %s: %s",
            job_id,
            type(e).__name__,
            e,
        )
        return None


def preprocess_jobs(db: DatabaseManager, run_id: int, chunk_size: int, num_workers: int) -> tuple[int, int]:
    """Read unpreprocessed jobs, clean descriptions in parallel, update DB in batches.

    Processes jobs in chunks using multiprocessing.Pool to parallelize CPU-bound
    text cleaning. Each chunk is processed independently and written to DB atomically.

    Args:
        db: DatabaseManager instance
        run_id: Pipeline run ID for audit logging
        chunk_size: Number of records to fetch per iteration
        num_workers: Number of worker processes in the pool

    Returns:
        Tuple of (processed_count, error_count)
    """
    import os
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger = setup_logging(log_level=log_level, name="preprocess_jobs")
    total_processed = 0
    total_errors = 0
    offset = 0

    while True:
        # Fetch one chunk
        records = db.get_unpreprocessed_jobs_chunked(chunk_size, offset)
        if not records:
            break

        logger.info(f"Processing chunk at offset {offset}: {len(records)} records")

        # Process chunk in parallel
        updates = []
        errors_in_chunk = 0

        with multiprocessing.Pool(processes=num_workers) as pool:
            for result in pool.imap(_clean_record, records, chunksize=50):
                if result is None:
                    errors_in_chunk += 1
                else:
                    updates.append(result)

        # Write results atomically, then free memory
        if updates:
            db.update_cleaned_descriptions_batch(updates)

        total_processed += len(updates)
        total_errors += errors_in_chunk

        if errors_in_chunk > 0:
            logger.warning(
                "Chunk at offset %d: %d record(s) failed (see _clean_record logs for details)",
                offset,
                errors_in_chunk,
            )
        logger.info(
            "Chunk complete at offset %d: %d processed, %d errors",
            offset,
            len(updates),
            errors_in_chunk,
        )

        # Advance offset
        offset += len(records)

    logger.info(f"Preprocessing complete: {total_processed} processed, {total_errors} errors")
    return total_processed, total_errors


def main() -> None:
    """CLI entrypoint called by Docker container.

    Reads config, initializes DB, runs preprocessing, updates pipeline_runs.
    Exits with code 0 on success, 1 on failure.
    """
    logger = setup_logging(name="preprocessing_main")
    try:
        config = load_config()
        logger.setLevel(config.log_level)

        db = DatabaseManager(config.db_path)
        db.initialize_schema()

        from datetime import datetime

        run_date = datetime.utcnow().strftime("%Y-%m-%d")
        run_id = db.create_pipeline_run(run_date, "preprocessing")

        processed, errors = preprocess_jobs(
            db, run_id,
            chunk_size=config.preprocessing_chunk_size,
            num_workers=config.preprocessing_workers,
        )
        db.finish_pipeline_run(
            run_id, "success", jobs_processed=processed, jobs_skipped=errors
        )
        logger.info("Preprocessing step completed successfully")

    except Exception as e:
        logger.exception("Preprocessing step failed")
        try:
            from datetime import datetime

            config = load_config()
            db = DatabaseManager(config.db_path)
            run_date = datetime.utcnow().strftime("%Y-%m-%d")
            run_id = db.create_pipeline_run(run_date, "preprocessing")
            db.finish_pipeline_run(run_id, "failed", 0, 0, str(e))
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
