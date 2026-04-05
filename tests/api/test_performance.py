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


def test_match_response_time_under_10s(api_client):
    """POST /match response time < 10s (contract requirement)."""
    resume_text = "Experienced software engineer with 5+ years in Python, AWS, and Docker. " * 20
    
    start_time = time.time()
    response = api_client.post(
        "/match",
        json={"resume": resume_text, "top_k": 5}
    )
    end_time = time.time()
    
    elapsed_s = end_time - start_time
    assert elapsed_s < 10, f"Response time {elapsed_s:.2f}s exceeds 10s limit"
    # The endpoint should return either 200 or an error status
    assert response.status_code in [200, 400, 404, 500, 503]


def test_match_with_ollama_failure_still_under_10s(api_client, mock_ollama_health):
    """POST /match when Ollama fails should still be under 10s."""
    # Make Ollama health check fail
    mock_ollama_health.return_value = (False, "Ollama connection failed")
    
    resume_text = "Experienced software engineer with 5+ years in Python, AWS, and Docker. " * 20
    
    start_time = time.time()
    response = api_client.post(
        "/match",
        json={"resume": resume_text, "top_k": 5}
    )
    end_time = time.time()
    
    elapsed_s = end_time - start_time
    assert elapsed_s < 10, f"Response time {elapsed_s:.2f}s exceeds 10s limit"
    # Should still return 200 (explanations may be null)
    assert response.status_code == 200


def test_match_with_large_resume_under_10s(api_client):
    """POST /match with max length resume (10,000 chars) under 10s."""
    # Create a resume of 10,000 characters
    resume_text = "A" * 10000
    
    start_time = time.time()
    response = api_client.post(
        "/match",
        json={"resume": resume_text, "top_k": 10}
    )
    end_time = time.time()
    
    elapsed_s = end_time - start_time
    assert elapsed_s < 10, f"Response time {elapsed_s:.2f}s exceeds 10s limit"
    assert response.status_code in [200, 400, 404, 500, 503]
