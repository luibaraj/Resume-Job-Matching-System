"""Tests for /match endpoint."""

import json
from unittest.mock import MagicMock


def test_match_missing_resume_returns_422(api_client):
    """POST /match without resume returns 422 validation error."""
    response = api_client.post("/match", json={"top_k": 10})
    
    assert response.status_code == 422
    data = response.json()
    assert "error" in data


def test_match_resume_too_short_returns_422(api_client):
    """POST /match with resume < 50 chars returns 422."""
    short_resume = "a" * 49  # 49 chars
    response = api_client.post("/match", json={"resume": short_resume})
    
    assert response.status_code == 422
    data = response.json()
    assert "error" in data


def test_match_resume_exact_50_chars_returns_200(api_client):
    """POST /match with resume exactly 50 chars returns 200 (boundary)."""
    exact_resume = "a" * 50  # Exactly 50 chars
    response = api_client.post("/match", json={"resume": exact_resume})
    
    assert response.status_code == 200
    data = response.json()
    assert "matches" in data
    assert "resume_id" in data


def test_match_top_k_zero_returns_422(api_client):
    """POST /match with top_k = 0 returns 422."""
    response = api_client.post(
        "/match", 
        json={"resume": "a" * 50, "top_k": 0}
    )
    
    assert response.status_code == 422
    data = response.json()
    assert "error" in data


def test_match_top_k_51_returns_422(api_client):
    """POST /match with top_k = 51 returns 422."""
    response = api_client.post(
        "/match", 
        json={"resume": "a" * 50, "top_k": 51}
    )
    
    assert response.status_code == 422
    data = response.json()
    assert "error" in data


def test_match_top_k_50_returns_200(api_client):
    """POST /match with top_k = 50 returns 200 (boundary)."""
    response = api_client.post(
        "/match", 
        json={"resume": "a" * 50, "top_k": 50}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "matches" in data


def test_match_top_k_defaults_to_10(api_client):
    """POST /match without top_k defaults to 10."""
    response = api_client.post(
        "/match", 
        json={"resume": "a" * 50}  # No top_k specified
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["matches"]) <= 10


def test_match_happy_path_returns_correct_structure(api_client, mock_voyage, mock_collection, mock_cohere):
    """POST /match with valid resume returns correct response structure."""
    resume_text = "Experienced software engineer with 5+ years in Python, AWS, and Docker. " * 10
    
    response = api_client.post(
        "/match",
        json={"resume": resume_text, "top_k": 5}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "matches" in data
    assert "resume_id" in data
    assert len(data["matches"]) <= 5
    for match in data["matches"]:
        assert "job_id" in match
        assert "title" in match
        assert "score" in match
        assert "explanation" in match


def test_match_empty_chroma_collection_returns_404(api_client, mock_collection):
    """POST /match with empty ChromaDB collection returns 404."""
    mock_collection.count.return_value = 0
    mock_collection.query.return_value = mock_collection.empty_query_result
    
    response = api_client.post(
        "/match",
        json={"resume": "a" * 50, "top_k": 10}
    )
    
    assert response.status_code == 404
    data = response.json()
    assert data["error"] == "no jobs in index"


def test_match_voyage_exception_returns_503(api_client, mock_voyage):
    """POST /match when VoyageAI fails returns 503."""
    mock_voyage.embed.side_effect = Exception("Voyage API error")
    
    response = api_client.post(
        "/match",
        json={"resume": "a" * 50, "top_k": 10}
    )
    
    assert response.status_code == 503
    data = response.json()
    assert data["error"] == "embedding service unavailable"


def test_match_cohere_exception_returns_503(api_client, mock_cohere):
    """POST /match when Cohere fails returns 503."""
    mock_cohere.rerank.side_effect = Exception("Cohere API error")
    
    response = api_client.post(
        "/match",
        json={"resume": "a" * 50, "top_k": 10}
    )
    
    assert response.status_code == 503
    data = response.json()
    assert data["error"] == "reranking service unavailable"


def test_match_unhandled_exception_returns_500(api_client, mock_voyage):
    """POST /match with unhandled exception returns 500."""
    mock_voyage.embed.side_effect = ValueError("Unexpected error")
    
    response = api_client.post(
        "/match",
        json={"resume": "a" * 50, "top_k": 10}
    )
    
    assert response.status_code == 500
    data = response.json()
    assert data["error"] == "internal server error"
    # Ensure no stack trace
    assert "traceback" not in str(data).lower()
    assert "Unexpected error" not in str(data)


def test_error_envelope_format_consistent(api_client):
    """All non-2xx responses use {"error": "..."} format."""
    # Test 404 on unknown route
    response = api_client.get("/nonexistent")
    
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    
    # Test 422 validation error
    response = api_client.post("/match", json={})  # Empty body
    
    assert response.status_code == 422
    data = response.json()
    assert "error" in data


def test_matches_sorted_by_score_descending(api_client, mock_collection):
    """POST /match returns matches sorted by score descending."""
    response = api_client.post(
        "/match",
        json={"resume": "a" * 50, "top_k": 3}
    )
    
    assert response.status_code == 200
    data = response.json()
    matches = data["matches"]
    scores = [m["score"] for m in matches]
    assert scores == sorted(scores, reverse=True)


def test_explanation_field_nullable(api_client):
    """POST /match explanation field can be null."""
    response = api_client.post(
        "/match",
        json={"resume": "a" * 50, "top_k": 3}
    )
    
    assert response.status_code == 200
    data = response.json()
    for match in data["matches"]:
        assert "explanation" in match
        # Can be string or null
        assert match["explanation"] is None or isinstance(match["explanation"], str)
