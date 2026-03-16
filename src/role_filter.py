import logging
import re

from src.utils import setup_logging

TARGET_PATTERNS = [
    r'\bmachine\s+learning\b',
    r'\bml\b',
    r'\bdata\s+scientist?\b',
    r'\bdata\s+science\b',
    r'\bresearch\s+scientist?\b',
    r'\bapplied\s+scientist?\b',
    r'\bai\b',
    r'\bai\s+research\b',
    r'\bresearch\s+engineer\b',
    r'\bcomputational\b',
    r'\bnatural\s+language\s+processing\b',
    r'\bnlp\b',
    r'\bcomputer\s+vision\b',
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in TARGET_PATTERNS]


def classify_is_target_role(title: str | None) -> int:
    """Return 1 if the job title matches ML/Data Science/Research patterns, else 0.

    Args:
        title: Job title string (may be None)

    Returns:
        1 if target role, 0 otherwise
    """
    if not title:
        return 0
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(title):
            return 1
    return 0


def filter_roles(db, run_id: int, chunk_size: int = 500) -> tuple[int, int]:
    """Classify all preprocessed-but-unclassified jobs and write is_target_role.

    Processes in chunks from OFFSET 0; committed rows drop out of the
    WHERE is_target_role IS NULL set so re-querying at offset 0 is correct.

    Args:
        db: DatabaseManager instance
        run_id: pipeline_runs ID for audit logging
        chunk_size: Number of jobs to process per chunk

    Returns:
        Tuple of (classified_count, error_count)
    """
    logger = logging.getLogger(__name__)
    total_classified = 0
    total_target = 0
    error_count = 0

    while True:
        records = db.get_unclassified_roles_chunked(chunk_size=chunk_size, offset=0)
        if not records:
            break

        updates: list[tuple[int, int]] = []
        for job_id, title in records:
            try:
                is_target = classify_is_target_role(title)
                updates.append((job_id, is_target))
                if is_target:
                    total_target += 1
            except Exception as exc:
                logger.warning("Error classifying job_id=%d: %s", job_id, exc)
                error_count += 1

        if updates:
            db.update_target_role_batch(updates)
            total_classified += len(updates)
            logger.debug("Classified %d jobs in this chunk (%d target so far)", len(updates), total_target)

    logger.info(
        "Role filtering complete: %d jobs classified, %d target roles, %d non-target, %d errors",
        total_classified,
        total_target,
        total_classified - total_target,
        error_count,
    )
    return total_classified, error_count


def main() -> None:
    """Entry point for standalone role_filter execution."""
    from datetime import datetime

    from src.config import load_config
    from src.database import DatabaseManager

    setup_logging()
    logger = logging.getLogger(__name__)

    config = load_config()
    db = DatabaseManager(config.db_path)
    db.initialize_schema()

    run_date = datetime.utcnow().strftime("%Y-%m-%d")
    run_id = db.create_pipeline_run(run_date, "role_filter")

    try:
        classified, errors = filter_roles(db, run_id, chunk_size=500)
        db.finish_pipeline_run(run_id, "success", classified, 0)
        logger.info("Done. classified=%d errors=%d", classified, errors)
    except Exception as exc:
        db.finish_pipeline_run(run_id, "failed", 0, 0, str(exc))
        logger.exception("Role filtering failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
