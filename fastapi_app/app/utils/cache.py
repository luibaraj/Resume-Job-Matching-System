"""
Utility functions for caching.
"""
import hashlib
import json
from typing import Any, Optional
from pathlib import Path

from app.config import settings


def get_cache_key(data: Any, prefix: str = "") -> str:
    """
    Generate a cache key from arbitrary data.

    Args:
        data: Serializable data.
        prefix: Optional prefix.

    Returns:
        Hexadecimal SHA256 hash string.
    """
    data_str = json.dumps(data, sort_keys=True, default=str)
    full_str = prefix + data_str
    return hashlib.sha256(full_str.encode()).hexdigest()


def ensure_cache_dir(file_path: str) -> Path:
    """
    Ensure the parent directory of a cache file exists.

    Args:
        file_path: Path to the cache file.

    Returns:
        Path object for the cache file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
