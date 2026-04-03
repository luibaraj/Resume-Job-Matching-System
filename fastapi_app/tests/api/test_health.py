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
"""Tests for health and readiness endpoints."""

from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pytest
from fastapi_app.api.dependencies import get_voyage_client, get_chroma_collection, get_db
import sqlite3

def test_health_endpoint_returns_200():
    """Test that GET /health always returns 200 with correct structure."""
    from fastapi_app.api.main import app
    client = TestClient(app)
    
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}

def test_health_response_time():
    """Test that /health response time is fast (no I/O)."""
    from fastapi_app.api.main import app
    client = TestClient(app)
    
    import time
    start = time.time()
    response = client.get("/health")
    end = time.time()
    
    assert response.status_code == 200
    # Should be very fast - less than 50ms as per contract
    assert (end - start) < 0.05  # 50ms

def test_ready_all_healthy(api_client):
    """Test /ready when all dependencies are healthy."""
    # api_client fixture already has all mocks
    response = api_client.get("/ready")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "checks" in data
    checks = data["checks"]
    assert checks["db"] == "ok"
    assert checks["chroma"] == "ok"
    assert checks["voyage"] == "ok"

def test_ready_db_unhealthy(api_client):
    """Test /ready when database is unreachable."""
    from fastapi_app.api.main import app
    
    # Create a mock that raises an exception
    mock_db = MagicMock()
    mock_db.cursor.side_effect = sqlite3.OperationalError("database not found")
    
    # Override the get_db dependency
    from fastapi_app.api.dependencies import get_db
    app.dependency_overrides[get_db] = lambda: mock_db
    
    client = TestClient(app)
    response = client.get("/ready")
    
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    checks = data["checks"]
    assert checks["db"].startswith("error:")
    assert checks["chroma"] == "ok"
    assert checks["voyage"] == "ok"
    
    # Clean up
    app.dependency_overrides.clear()

def test_ready_chroma_unhealthy(api_client):
    """Test /ready when ChromaDB is unreachable."""
    from fastapi_app.api.main import app
    
    # Create a mock that raises an exception
    mock_collection = MagicMock()
    mock_collection.count.side_effect = Exception("connection failed")
    
    # Override the get_chroma_collection dependency
    from fastapi_app.api.dependencies import get_chroma_collection
    app.dependency_overrides[get_chroma_collection] = lambda: mock_collection
    
    client = TestClient(app)
    response = client.get("/ready")
    
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    checks = data["checks"]
    assert checks["db"] == "ok"
    assert checks["chroma"].startswith("error:")
    assert checks["voyage"] == "ok"
    
    # Clean up
    app.dependency_overrides.clear()

def test_ready_voyage_unhealthy(api_client):
    """Test /ready when VoyageAI is unreachable."""
    from fastapi_app.api.main import app
    
    # Create a mock that raises an exception
    mock_voyage = MagicMock()
    mock_voyage.embed.side_effect = Exception("API key invalid")
    
    # Override the get_voyage_client dependency
    from fastapi_app.api.dependencies import get_voyage_client
    app.dependency_overrides[get_voyage_client] = lambda: mock_voyage
    
    client = TestClient(app)
    response = client.get("/ready")
    
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    checks = data["checks"]
    assert checks["db"] == "ok"
    assert checks["chroma"] == "ok"
    assert checks["voyage"].startswith("error:")
    
    # Clean up
    app.dependency_overrides.clear()

def test_ready_partial_failure(api_client):
    """Test /ready when multiple dependencies fail."""
    from fastapi_app.api.main import app
    
    # Create mocks that raise exceptions
    mock_db = MagicMock()
    mock_db.cursor.side_effect = sqlite3.OperationalError("database not found")
    
    mock_voyage = MagicMock()
    mock_voyage.embed.side_effect = Exception("API key invalid")
    
    # Override dependencies
    from fastapi_app.api.dependencies import get_db, get_voyage_client
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_voyage_client] = lambda: mock_voyage
    
    client = TestClient(app)
    response = client.get("/ready")
    
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    checks = data["checks"]
    assert checks["db"].startswith("error:")
    assert checks["chroma"] == "ok"  # This one is still healthy
    assert checks["voyage"].startswith("error:")
    
    # Clean up
    app.dependency_overrides.clear()

def test_ready_checks_keys(api_client):
    """Test that /ready always returns exactly db, chroma, voyage keys."""
    response = api_client.get("/ready")
    data = response.json()
    checks = data["checks"]
    
    # Must have exactly these three keys
    assert set(checks.keys()) == {"db", "chroma", "voyage"}
    # Each value must be a string
    for value in checks.values():
        assert isinstance(value, str)
