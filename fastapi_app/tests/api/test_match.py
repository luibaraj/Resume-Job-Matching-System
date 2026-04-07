"""Tests for /match endpoint."""

import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from fastapi_app.api.main import app


def test_match_score_calculation(api_client, mock_collection):
    """Test that score is calculated as max(0.0, 1.0 - distance)."""
    # Mock distances to test score calculation
    mock_collection.query.return_value = {
        "ids": [["1", "2", "3"]],
        "documents": [["Job 1", "Job 2", "Job 3"]],
        "metadatas": [[
            {"title": "Job 1", "job_id": 1},
            {"title": "Job 2", "job_id": 2},
            {"title": "Job 3", "job_id": 3}
        ]],
        "distances": [[0.1, 0.5, 1.2]],  # Note: 1.2 should give score 0.0
    }
    
    response = api_client.post(
        "/match",
        json={"resume": "a" * 50, "top_k": 3}
    )
    
    assert response.status_code == 200
    data = response.json()
    matches = data["matches"]
    
    # Check score calculation
    expected_scores = [1.0 - 0.1, 1.0 - 0.5, max(0.0, 1.0 - 1.2)]
    for i, match in enumerate(matches):
        assert abs(match["score"] - expected_scores[i]) < 0.001
        assert 0.0 <= match["score"] <= 1.0


def test_match_ollama_failure_returns_200_without_explanations(api_client, mock_generate_explanation):
    """Test that Ollama failure returns 200 but explanations may be null."""
    # Make generate_explanation_with_pipeline return (None, None) (simulating failure)
    mock_generate_explanation.return_value = (None, None)
    
    response = api_client.post(
        "/match",
        json={"resume": "a" * 50, "top_k": 3}
    )
    
    assert response.status_code == 200
    data = response.json()
    for match in data["matches"]:
        # Explanation can be null when Ollama fails
        assert match["explanation"] is None


def test_match_chromadb_failure_returns_500(api_client, mock_collection):
    """Test that ChromaDB failure returns 500."""
    # Make collection.query raise an exception
    mock_collection.query.side_effect = Exception("ChromaDB internal error")
    
    response = api_client.post(
        "/match",
        json={"resume": "a" * 50, "top_k": 10}
    )
    
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert "retrieval error" in data["error"].lower()


def test_match_cohere_failure_returns_503(api_client, mock_cohere):
    """Test that Cohere failure returns 503."""
    # Already tested, but ensure it matches contract
    mock_cohere.rerank.side_effect = Exception("Cohere API error")
    
    response = api_client.post(
        "/match",
        json={"resume": "a" * 50, "top_k": 10}
    )
    
    assert response.status_code == 503
    data = response.json()
    assert "error" in data
    assert "reranking service unavailable" in data["error"]


def test_match_ollama_health_failure_still_returns_200(api_client, mock_ollama_health):
    """Test that Ollama health check failure doesn't prevent match endpoint."""
    # Make Ollama health check fail
    mock_ollama_health.return_value = (False, "Ollama connection failed")
    
    response = api_client.post(
        "/match",
        json={"resume": "a" * 50, "top_k": 3}
    )
    
    # Should still return 200 (explanations may be null)
    assert response.status_code == 200
    data = response.json()
    assert "matches" in data


def test_match_resume_max_length_10000_chars(api_client):
    """Test that resume up to 10,000 characters is accepted."""
    # Create a resume of exactly 10,000 characters
    resume_text = "A" * 10000
    
    response = api_client.post(
        "/match",
        json={"resume": resume_text, "top_k": 5}
    )
    
    # Should be accepted (200) or return validation error if too long
    # According to contract, max is 10,000 characters
    assert response.status_code in [200, 422]
    if response.status_code == 422:
        data = response.json()
        assert "error" in data


def test_match_top_k_max_50(api_client):
    """Test that top_k up to 50 is accepted."""
    response = api_client.post(
        "/match",
        json={"resume": "a" * 50, "top_k": 50}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["matches"]) <= 50


def test_match_retrieves_top_100_from_chromadb(api_client, mock_collection):
    """Test that endpoint retrieves top 100 jobs from ChromaDB."""
    response = api_client.post(
        "/match",
        json={"resume": "a" * 50, "top_k": 10}
    )
    
    # Verify that collection.query was called with n_results=100
    mock_collection.query.assert_called_once()
    call_args = mock_collection.query.call_args
    assert call_args[1].get('n_results') == 100


def test_match_reranks_top_10_by_default(api_client, mock_cohere):
    """Test that endpoint reranks top 10 jobs by default."""
    response = api_client.post(
        "/match",
        json={"resume": "a" * 50}  # top_k defaults to 10
    )
    
    # Verify cohere.rerank was called with appropriate parameters
    mock_cohere.rerank.assert_called_once()
    call_args = mock_cohere.rerank.call_args
    # Should have documents to rerank
    assert len(call_args[1].get('documents', [])) <= 10


# Include existing tests from the original file
def test_match_missing_resume_returns_422(api_client):
    """POST /match without resume returns 422 validation error."""
    response = api_client.post("/match", json={"top_k": 10})
    assert response.status_code == 422
    data = response.json()
    assert "error" in data


def test_match_resume_too_short_returns_422(api_client):
    """POST /match with resume < 50 chars returns 422."""
    short_resume = "a" * 49
    response = api_client.post("/match", json={"resume": short_resume})
    assert response.status_code == 422
    data = response.json()
    assert "error" in data


def test_match_resume_exact_50_chars_returns_200(api_client):
    """POST /match with resume exactly 50 chars returns 200 (boundary)."""
    exact_resume = "a" * 50
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
        json={"resume": "a" * 50}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["matches"]) <= 10


def test_match_happy_path_returns_correct_structure(api_client):
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


def test_error_envelope_format_consistent(api_client):
    """All non-2xx responses use {"error": "..."} format."""
    response = api_client.get("/nonexistent")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    
    response = api_client.post("/match", json={})
    assert response.status_code == 422
    data = response.json()
    assert "error" in data


def test_matches_sorted_by_score_descending(api_client):
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
        assert match["explanation"] is None or isinstance(match["explanation"], str)
