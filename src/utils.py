import hashlib
import json
import logging
import sys
from typing import Optional


def compute_job_hash(greenhouse_id: int, board_token: str, title: str) -> str:
    """Compute a SHA-256 hash that uniquely identifies a job posting.

    Inputs are normalized (lowercased, stripped) before hashing.

    Args:
        greenhouse_id: Integer ID from Greenhouse API
        board_token: Company slug (board token)
        title: Job title

    Returns:
        64-character lowercase hex string (SHA-256 digest)
    """
    normalized = f"{greenhouse_id}:{board_token.lower().strip()}:{title.lower().strip()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def setup_logging(log_level: str = "INFO", name: Optional[str] = None) -> logging.Logger:
    """Configure a logger that writes to stdout with ISO-8601 timestamps.

    Format: "2026-03-05T14:00:00Z [INFO] collection: Fetched 42 jobs"

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        name: Logger name; if None, uses root logger

    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Remove any existing handlers to avoid duplicates
    logger.handlers = []

    # Create stdout handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # ISO-8601 format with Z suffix for UTC
    formatter = logging.Formatter(
        "%(asctime)sZ [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def serialize_list(items: list[str]) -> str:
    """Serialize a list of strings to a JSON string for SQLite storage.

    Args:
        items: List of strings

    Returns:
        JSON-encoded string (e.g., '["a", "b", "c"]')
    """
    return json.dumps(items)


def deserialize_list(json_str: Optional[str]) -> list[str]:
    """Deserialize a JSON string from SQLite back to a Python list.

    Args:
        json_str: JSON-encoded string or None

    Returns:
        List of strings, or empty list if json_str is None or empty
    """
    if not json_str:
        return []
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return []
