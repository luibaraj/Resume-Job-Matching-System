"""Tests for /health and /ready endpoints."""

import time
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from fastapi_app.api.main import app
import sqlite3


def test_health_endpoint_returns_200(api_client):
    """GET /health returns 200 with {"status": "ok"}."""
    response = api_client.get("/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_response_time_under_50ms(api_client):
    """GET /health response time < 50ms (no I/O)."""
    start_time = time.time()
    response = api_client.get("/health")
    end_time = time.time()
    
    elapsed_ms = (end_time - start_time) * 1000
    assert elapsed_ms < 50, f"Response time {elapsed_ms:.2f}ms exceeds 50ms limit"
    assert response.status_code == 200


def test_ready_all_healthy_returns_200(api_client, mock_db, mock_collection, mock_voyage, mock_ollama_health):
    """GET /ready returns 200 when all dependencies are healthy."""
    # Setup all mocks to succeed
    mock_db.cursor().fetchone.return_value = (1,)  # DB check passes
    mock_collection.count.return_value = 5  # Chroma has data
    mock_voyage.embed.return_value.embeddings = [[0.1] * 1024]  # Voyage works
    mock_ollama_health.return_value = (True, "Ollama is healthy")  # Ollama works
    
    response = api_client.get("/ready")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    # Check all 4 services are present and healthy
    assert "checks" in data
    checks = data["checks"]
    assert len(checks) == 4
    assert checks["database"]["healthy"] is True
    assert checks["chroma"]["healthy"] is True
    assert checks["voyage"]["healthy"] is True
    assert checks["ollama"]["healthy"] is True


def test_ready_db_unreachable_returns_503(api_client, mock_db, mock_ollama_health):
    """GET /ready returns 503 when DB is unreachable."""
    # Make DB check fail
    mock_db.cursor().execute.side_effect = Exception("Connection failed")
    mock_ollama_health.return_value = (True, "Ollama is healthy")
    
    response = api_client.get("/ready")
    
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not ready"
    checks = data["checks"]
    assert checks["database"]["healthy"] is False
    assert "error" in checks["database"]["message"].lower()
    assert checks["chroma"]["healthy"] is True
    assert checks["voyage"]["healthy"] is True
    assert checks["ollama"]["healthy"] is True


def test_ready_chroma_unreachable_returns_503(api_client, mock_collection, mock_ollama_health):
    """GET /ready returns 503 when ChromaDB is unreachable."""
    # Make Chroma check fail
    mock_collection.count.side_effect = Exception("Collection not found")
    mock_ollama_health.return_value = (True, "Ollama is healthy")
    
    response = api_client.get("/ready")
    
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not ready"
    checks = data["checks"]
    assert checks["database"]["healthy"] is True
    assert checks["chroma"]["healthy"] is False
    assert "error" in checks["chroma"]["message"].lower()
    assert checks["voyage"]["healthy"] is True
    assert checks["ollama"]["healthy"] is True


def test_ready_voyage_unreachable_returns_503(api_client, mock_voyage, mock_ollama_health):
    """GET /ready returns 503 when VoyageAI is unreachable."""
    # Make Voyage check fail
    mock_voyage.embed.side_effect = Exception("API key invalid")
    mock_ollama_health.return_value = (True, "Ollama is healthy")
    
    response = api_client.get("/ready")
    
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not ready"
    checks = data["checks"]
    assert checks["database"]["healthy"] is True
    assert checks["chroma"]["healthy"] is True
    assert checks["voyage"]["healthy"] is False
    assert "error" in checks["voyage"]["message"].lower()
    assert checks["ollama"]["healthy"] is True


def test_ready_ollama_unreachable_returns_503(api_client, mock_ollama_health):
    """GET /ready returns 503 when Ollama is unreachable."""
    # Make Ollama check fail
    mock_ollama_health.return_value = (False, "Connection failed")
    
    response = api_client.get("/ready")
    
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not ready"
    checks = data["checks"]
    assert checks["database"]["healthy"] is True
    assert checks["chroma"]["healthy"] is True
    assert checks["voyage"]["healthy"] is True
    assert checks["ollama"]["healthy"] is False
    assert "connection" in checks["ollama"]["message"].lower()


def test_ready_ollama_missing_model_returns_503(api_client, mock_ollama_health):
    """GET /ready returns 503 when Ollama missing required model."""
    # Make Ollama check fail due to missing model
    mock_ollama_health.return_value = (False, "Required model 'llama3.2:3b-instruct-q4_K_M' not found in Ollama")
    
    response = api_client.get("/ready")
    
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not ready"
    checks = data["checks"]
    assert checks["ollama"]["healthy"] is False
    assert "model" in checks["ollama"]["message"].lower()


def test_ready_partial_failure_shows_only_failing_checks(api_client, mock_db, mock_voyage, mock_ollama_health):
    """GET /ready with partial failure shows only failing checks as errors."""
    # Make DB and Voyage fail, Chroma and Ollama succeed
    mock_db.cursor().execute.side_effect = Exception("DB connection failed")
    mock_voyage.embed.side_effect = Exception("Voyage API error")
    mock_ollama_health.return_value = (True, "Ollama is healthy")
    
    response = api_client.get("/ready")
    
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not ready"
    checks = data["checks"]
    assert checks["database"]["healthy"] is False
    assert checks["chroma"]["healthy"] is True  # Only Chroma and Ollama are healthy
    assert checks["voyage"]["healthy"] is False
    assert checks["ollama"]["healthy"] is True


def test_ready_checks_always_has_four_keys(api_client):
    """GET /ready always returns checks with exactly four keys."""
    response = api_client.get("/ready")
    data = response.json()
    
    checks = data["checks"]
    assert set(checks.keys()) == {"database", "chroma", "voyage", "ollama"}
    assert len(checks) == 4
    # Each check should have healthy boolean and message string
    for service, check in checks.items():
        assert "healthy" in check
        assert isinstance(check["healthy"], bool)
        assert "message" in check
        assert isinstance(check["message"], str)


def test_ready_response_time_under_2s(api_client):
    """GET /ready response time < 2s (due to external service checks)."""
    start_time = time.time()
    response = api_client.get("/ready")
    end_time = time.time()
    
    elapsed_s = end_time - start_time
    assert elapsed_s < 2, f"Response time {elapsed_s:.2f}s exceeds 2s limit"
    assert response.status_code in [200, 503]  # Can be either depending on health
