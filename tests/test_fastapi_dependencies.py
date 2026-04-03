"""
Tests for FastAPI dependencies.
"""
import pytest
from fastapi_app.app.dependencies import get_matching_service
from fastapi_app.app.services.matching_service import MatchingService

def test_get_matching_service_yields_instance():
    """Generator yields MatchingService"""
    generator = get_matching_service()
    service = next(generator)
    
    assert isinstance(service, MatchingService)
    
    # Cleanup
    try:
        next(generator)
    except StopIteration:
        pass

def test_get_matching_service_cleanup():
    """Any cleanup after yield"""
    generator = get_matching_service()
    service = next(generator)
    
    # Verify service is valid
    assert service is not None
    
    # Verify cleanup completes
    try:
        next(generator)
    except StopIteration:
        pass  # Expected - cleanup completed
