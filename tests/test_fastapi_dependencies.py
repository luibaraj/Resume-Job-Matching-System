"""
Tests for FastAPI dependencies.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi_app.app.dependencies import get_matching_service
from fastapi_app.app.services.matching_service import MatchingService

@pytest.fixture
def mock_services():
    """Mock all services to avoid API key initialization."""
    with patch('fastapi_app.app.services.embedding_service.EmbeddingService'), \
         patch('fastapi_app.app.services.retrieval_service.RetrievalService'), \
         patch('fastapi_app.app.services.reranking_service.RerankingService'), \
         patch('fastapi_app.app.services.generation_service.GenerationService'):
        yield

def test_get_matching_service_yields_instance(mock_services):
    """Generator yields MatchingService"""
    generator = get_matching_service()
    service = next(generator)

    # Just verify it's an object with the expected methods/attributes
    assert hasattr(service, 'match')
    assert hasattr(service, 'embedding_service')
    assert hasattr(service, 'retrieval_service')

    # Cleanup
    try:
        next(generator)
    except StopIteration:
        pass

def test_get_matching_service_cleanup(mock_services):
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
