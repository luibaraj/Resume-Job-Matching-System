"""Performance tests for API endpoints."""

import time
import pytest


def test_health_response_time_under_100ms(api_client):
    """GET /health response time < 100ms (contract requirement)."""
    start_time = time.time()
    response = api_client.get("/health")
    end_time = time.time()
    
    elapsed_ms = (end_time - start_time) * 1000
    assert elapsed_ms < 100, f"Response time {elapsed_ms:.2f}ms exceeds 100ms limit"
    assert response.status_code == 200


def test_ready_response_time_under_2s(api_client):
    """GET /ready response time < 2s (contract requirement)."""
    start_time = time.time()
    response = api_client.get("/ready")
    end_time = time.time()
    
    elapsed_s = end_time - start_time
    assert elapsed_s < 2, f"Response time {elapsed_s:.2f}s exceeds 2s limit"
    assert response.status_code in [200, 503]


