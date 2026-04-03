"""
FastAPI dependencies.
"""
from typing import Generator

from app.services.matching_service import MatchingService


def get_matching_service() -> Generator[MatchingService, None, None]:
    """
    Dependency that provides a MatchingService instance.

    Yields:
        MatchingService: An instance of the matching service.
    """
    service = MatchingService()
    try:
        yield service
    finally:
        # Cleanup if needed
        pass
