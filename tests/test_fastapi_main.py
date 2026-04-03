"""
Tests for FastAPI main endpoints.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi import status
from fastapi.testclient import TestClient

def test_health_endpoint_healthy(fastapi_test_client):
    """GET /health → 200, {"status": "healthy"}"""
    response = fastapi_test_client.get("/api/v1/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert "ollama_available" in data
    assert "database_available" in data
    assert "chroma_collection_count" in data

def test_match_endpoint_success(fastapi_test_client, sample_resume_text, sample_match_response):
    """POST /match with resume_text → 200, correct schema"""
    # Use the mock_matching_service from the fixture
    fastapi_test_client._mock_matching_service.match.return_value = sample_match_response

    response = fastapi_test_client.post(
        "/api/v1/match",
        json={"resume_text": sample_resume_text}
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "matches" in data
    assert "total_candidates" in data
    assert "total_reranked" in data
    assert len(data["matches"]) == 1
    assert data["matches"][0]["id"] == 123

def test_match_endpoint_missing_resume(fastapi_test_client):
    """POST /match without resume_text → 422"""
    response = fastapi_test_client.post("/api/v1/match", json={})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_match_endpoint_with_params(fastapi_test_client, sample_resume_text):
    """POST with top_k, top_n, use_filters, include_explanations"""
    from fastapi_app.app.config import settings
    
    response = fastapi_test_client.post(
        "/api/v1/match",
        json={
            "resume_text": sample_resume_text,
            "top_k": 50,
            "use_filters": False,
            "include_explanations": False
        }
    )

    assert response.status_code == status.HTTP_200_OK
    fastapi_test_client._mock_matching_service.match.assert_called_once_with(
        resume_text=sample_resume_text,
        top_k=50,
        top_n=settings.RERANK_TOP_N,  # Use actual setting value
        use_filters=False,
        include_explanations=False
    )

def test_match_endpoint_service_error(fastapi_test_client, sample_resume_text):
    """When match() raises exception → 500"""
    fastapi_test_client._mock_matching_service.match.side_effect = Exception("Service error")
    
    response = fastapi_test_client.post(
        "/api/v1/match",
        json={"resume_text": sample_resume_text}
    )
    
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
