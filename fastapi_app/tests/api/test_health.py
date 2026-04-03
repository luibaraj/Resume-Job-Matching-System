"""Tests for /health and /ready endpoints."""

import time
from unittest.mock import MagicMock


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


def test_ready_all_healthy_returns_200(api_client, mock_db, mock_collection, mock_voyage):
    """GET /ready returns 200 when all dependencies are healthy."""
    # Setup all mocks to succeed
    mock_db.cursor().fetchone.return_value = (1,)  # DB check passes
    mock_collection.count.return_value = 5  # Chroma has data
    mock_voyage.embed.return_value.embeddings = [[0.1] * 1024]  # Voyage works
    
    response = api_client.get("/ready")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["checks"] == {
        "db": "ok",
        "chroma": "ok", 
        "voyage": "ok"
    }


def test_ready_db_unreachable_returns_503(api_client, mock_db):
    """GET /ready returns 503 when DB is unreachable."""
    # Make DB check fail
    mock_db.cursor().execute.side_effect = Exception("Connection failed")
    
    response = api_client.get("/ready")
    
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["checks"]["db"].startswith("error:")
    assert data["checks"]["chroma"] == "ok"
    assert data["checks"]["voyage"] == "ok"


def test_ready_chroma_unreachable_returns_503(api_client, mock_collection):
    """GET /ready returns 503 when ChromaDB is unreachable."""
    # Make Chroma check fail
    mock_collection.count.side_effect = Exception("Collection not found")
    
    response = api_client.get("/ready")
    
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["checks"]["db"] == "ok"
    assert data["checks"]["chroma"].startswith("error:")
    assert data["checks"]["voyage"] == "ok"


def test_ready_voyage_unreachable_returns_503(api_client, mock_voyage):
    """GET /ready returns 503 when VoyageAI is unreachable."""
    # Make Voyage check fail
    mock_voyage.embed.side_effect = Exception("API key invalid")
    
    response = api_client.get("/ready")
    
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["checks"]["db"] == "ok"
    assert data["checks"]["chroma"] == "ok"
    assert data["checks"]["voyage"].startswith("error:")


def test_ready_partial_failure_shows_only_failing_checks(api_client, mock_db, mock_voyage):
    """GET /ready with partial failure shows only failing checks as errors."""
    # Make DB and Voyage fail, Chroma succeed
    mock_db.cursor().execute.side_effect = Exception("DB connection failed")
    mock_voyage.embed.side_effect = Exception("Voyage API error")
    
    response = api_client.get("/ready")
    
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["checks"]["db"].startswith("error:")
    assert data["checks"]["chroma"] == "ok"  # Only Chroma is healthy
    assert data["checks"]["voyage"].startswith("error:")


def test_ready_checks_always_has_three_keys(api_client):
    """GET /ready always returns checks with exactly three keys."""
    response = api_client.get("/ready")
    data = response.json()
    
    checks = data["checks"]
    assert set(checks.keys()) == {"db", "chroma", "voyage"}
    assert len(checks) == 3
