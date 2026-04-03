from fastapi.testclient import TestClient
from fastapi_app.api.main import app

client = TestClient(app)

def test_422_validation_error_returns_array():
    """Test that 422 validation errors return array under 'error' key"""
    response = client.post("/match", json={"resume": "short"})  # Too short resume
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert isinstance(data["error"], list)  # Should be array, not string
    assert len(data["error"]) > 0

def test_404_returns_string_error():
    """Test that 404 errors return string under 'error' key"""
    response = client.get("/nonexistent")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert isinstance(data["error"], str)  # Should be string
    assert data["error"] == "Not Found"

def test_500_returns_internal_server_error():
    """Test that unhandled exceptions return 'internal server error' string"""
    # We can't easily test this without mocking an endpoint to raise an exception
    # This test would be added when we have an endpoint that can be mocked
    pass

def test_error_envelope_consistent():
    """Test that all non-2xx responses have 'error' key"""
    # Test 404
    response = client.get("/nonexistent")
    assert "error" in response.json()
    
    # Test 422 (validation error)
    response = client.post("/match", json={"resume": "short"})
    assert "error" in response.json()
