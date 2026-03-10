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

# Location standardization constants
GENERIC_LOCATIONS: frozenset[str] = frozenset({
    "united states", "us", "usa", "u.s.", "u.s.a.",
    "hybrid", "remote", "unknown", "not specified"
})

STATE_ABBREVIATIONS: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}

# US state abbreviations for quick lookup
US_STATE_ABBREVS: frozenset[str] = frozenset(STATE_ABBREVIATIONS.values())

# Known US cities (without state) that are unambiguous
KNOWN_US_CITIES: frozenset[str] = frozenset({
    "san francisco", "new york city", "chicago", "seattle", "boston", "austin",
    "los angeles", "mountain view", "palo alto", "menlo park", "cupertino",
    "san jose", "hawthorne", "redmond", "starbase",
})

# International country and city keywords
INTERNATIONAL_KEYWORDS: frozenset[str] = frozenset({
    "india", "uk", "united kingdom", "england", "ireland", "japan", "canada",
    "australia", "singapore", "germany", "france", "netherlands", "spain",
    "brazil", "mexico", "israel", "south korea", "china", "poland", "hungary",
    "bulgaria", "philippines", "ind", "nz", "jpn", "ie",
})


def decode_html_entities(text: str) -> str:
    """Decode HTML entities to their Unicode equivalents.

    Applies html.unescape() repeatedly until the string no longer changes,
    handling double-encoded input like &amp;nbsp; → &nbsp; → \xa0.

    Examples:
        "&amp;" -> "&"
        "&amp;nbsp;" -> "\xa0"
        "&lt;" -> "<"
        "&gt;" -> ">"
        "&#39;" -> "'"
        "&nbsp;" -> "\xa0" (non-breaking space, handled in whitespace step)

    Args:
        text: String potentially containing HTML entities

    Returns:
        String with HTML entities decoded
    """
    for _ in range(5):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    return text


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


def normalize_location(location: str | None) -> str | None:
    """Standardize job location strings to a consistent format.

    Handles:
    - Remote variants: "Remote - USA", "Remote - US" → "Remote"
    - State abbreviation expansion: "City, California" → "City, CA"
    - Generic/useless locations: "United States", "Hybrid" → None
    - Whitespace normalization: leading/trailing/internal

    Args:
        location: Raw location string or None

    Returns:
        Standardized location string or None if input is None/empty/generic
    """
    if not location:
        return None

    # Normalize whitespace and casing
    cleaned = MULTI_SPACE_PATTERN.sub(" ", location).strip()
    if not cleaned:
        return None

    # Detect and standardize remote variants
    if "remote" in cleaned.lower():
        return "Remote"

    # Expand state abbreviations (only the last segment after comma)
    parts = cleaned.split(",")
    if len(parts) > 1:
        state_part = parts[-1].strip().lower()
        if state_part in STATE_ABBREVIATIONS:
            parts[-1] = STATE_ABBREVIATIONS[state_part]
            cleaned = ", ".join(parts)
            # Re-normalize whitespace after join
            cleaned = MULTI_SPACE_PATTERN.sub(" ", cleaned).strip()

    # Check for generic/useless locations (after state expansion)
    if cleaned.lower() in GENERIC_LOCATIONS:
        return None

    return cleaned


def classify_location_is_us(location: str | None) -> int | None:
    """Classify a job location as US (1), non-US (0), or unknown (None).

    Returns:
        1 for US jobs
        0 for non-US jobs
        None for unknown/Remote/ambiguous locations

    Classification logic (order matters):
    - None or "Remote" → None
    - Contains US state abbreviation (e.g., ", CA", ", TX") → 1
    - Contains US state full name → 1
    - Contains "United States", "USA", "DC" → 1
    - Contains international keyword → 0 (check before US cities to avoid false matches)
    - Contains known unambiguous US city name → 1
    - Otherwise → None (don't guess)
    """
    if not location:
        return None

    if location == "Remote":
        return None

    # Normalize for comparison
    loc_lower = location.lower()

    # Check for US state abbreviations (e.g., ", CA") — must have comma prefix and word boundary
    for abbrev in US_STATE_ABBREVS:
        # Match ", CA" or ", NY" with word boundary to avoid ", IN" matching in ", INDIA"
        pattern = rf', {re.escape(abbrev.lower())}\b'
        if re.search(pattern, loc_lower):
            return 1

    # Check for explicit "United States" / "USA" mentions
    if any(kw in loc_lower for kw in ["united states", "usa", ", dc", "washington, dc"]):
        return 1

    # Check for international keywords early (word boundary) to avoid state names like "indiana" matching in "india"
    for keyword in INTERNATIONAL_KEYWORDS:
        if re.search(rf'\b{re.escape(keyword)}\b', loc_lower):
            return 0

    # Check for US state full names (with word boundary)
    for state_name in STATE_ABBREVIATIONS.keys():
        # Use word boundary to match whole words only
        if re.search(rf'\b{re.escape(state_name)}\b', loc_lower):
            return 1

    # Check for known unambiguous US cities (with word boundary)
    for city in KNOWN_US_CITIES:
        if re.search(rf'\b{re.escape(city)}\b', loc_lower):
            return 1

    # Default: unknown/ambiguous (don't classify as US or non-US)
    return None


def clean_title(title: str) -> str:
    """Normalize job title whitespace: collapse multiple spaces to single space and trim.

    Args:
        title: Raw title string

    Returns:
        Title with normalized whitespace
    """
    return MULTI_SPACE_PATTERN.sub(" ", title).strip()


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


def _clean_record(record: tuple[int, str | None, str | None, str]) -> tuple[int, str] | None:
    """Worker function for multiprocessing.Pool.

    Must be module-level (not a closure) for pickle serialization to worker processes.
    Returns None on exception to enable per-record fault tolerance.
    Configures its own logger because worker processes do not inherit the
    parent process's logging configuration.

    Args:
        record: Tuple of (job_id, raw_description, location, title)

    Returns:
        Tuple of (job_id, cleaned_description) on success, None on exception
    """
    job_id, raw_description, *_ = record
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


def preprocess_jobs(db: DatabaseManager, run_id: int, chunk_size: int, num_workers: int, max_retries: int = 2) -> tuple[int, int]:
    """Read unpreprocessed jobs, clean descriptions in parallel, update DB in batches.

    Processes jobs in chunks using multiprocessing.Pool to parallelize CPU-bound
    text cleaning. Each chunk is processed independently and written to DB atomically.
    Detects missing IDs (silent worker crashes) and retries them; fails the pipeline
    if records still missing after retries.

    Args:
        db: DatabaseManager instance
        run_id: Pipeline run ID for audit logging
        chunk_size: Number of records to fetch per iteration
        num_workers: Number of worker processes in the pool
        max_retries: Maximum retries for missing IDs; defaults to 2

    Returns:
        Tuple of (processed_count, error_count)
    """
    import os
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger = setup_logging(log_level=log_level, name="preprocess_jobs")
    total_processed = 0
    total_errors = 0

    while True:
        # Fetch one chunk (always from offset 0 — committed rows drop out of WHERE preprocessed=0)
        records = db.get_unpreprocessed_jobs_chunked(chunk_size, 0)
        if not records:
            break

        logger.info(f"Processing chunk: {len(records)} records")

        # Process chunk in parallel
        updates = []
        errors_in_chunk = 0

        with multiprocessing.Pool(processes=num_workers) as pool:
            for result in pool.imap(_clean_record, records, chunksize=50):
                if result is None:
                    errors_in_chunk += 1
                else:
                    updates.append(result)

        # Track which IDs were returned vs what went in
        input_ids = {record[0] for record in records}
        output_ids = {result[0] for result in updates}
        missing_ids = input_ids - output_ids

        # Retry missing records up to max_retries times
        for attempt in range(1, max_retries + 1):
            if not missing_ids:
                break

            logger.warning(
                "%d record(s) missing after pool run "
                "(attempt %d/%d), retrying IDs: %s",
                len(missing_ids), attempt, max_retries, sorted(missing_ids),
            )

            records_to_retry = [r for r in records if r[0] in missing_ids]
            retry_updates = []

            with multiprocessing.Pool(processes=num_workers) as pool:
                for result in pool.imap(_clean_record, records_to_retry, chunksize=50):
                    if result is not None:
                        retry_updates.append(result)

            updates.extend(retry_updates)
            output_ids = {result[0] for result in updates}
            missing_ids = input_ids - output_ids

        # After all retries, if still missing — stop the pipeline
        if missing_ids:
            raise RuntimeError(
                f"{len(missing_ids)} record(s) still missing "
                f"after {max_retries} retries. Missing job IDs: {sorted(missing_ids)}"
            )

        # Build metadata lookup: id → (location, title) from original records
        id_to_meta = {record[0]: (record[2], record[3]) for record in records}

        # Apply location and title cleaning, then construct full updates
        full_updates = [
            (job_id, cleaned_desc, normalize_location(id_to_meta[job_id][0]), clean_title(id_to_meta[job_id][1]), classify_location_is_us(id_to_meta[job_id][0]))
            for job_id, cleaned_desc in updates
        ]

        # Write results atomically, then free memory
        if full_updates:
            db.update_job_fields_batch(full_updates)

        total_processed += len(updates)
        total_errors += errors_in_chunk

        if errors_in_chunk > 0:
            logger.warning(
                "%d record(s) failed in this chunk (see _clean_record logs for details)",
                errors_in_chunk,
            )
        logger.info(
            "Chunk complete: %d processed, %d errors",
            len(updates),
            errors_in_chunk,
        )

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
            max_retries=config.preprocessing_max_retries,
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
