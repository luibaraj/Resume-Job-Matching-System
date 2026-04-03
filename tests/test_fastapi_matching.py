"""
Tests for MatchingService.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call
from fastapi_app.app.services.matching_service import MatchingService
from fastapi_app.app.services.embedding_service import EmbeddingService
from fastapi_app.app.services.retrieval_service import RetrievalService
from fastapi_app.app.services.reranking_service import RerankingService
from fastapi_app.app.services.generation_service import GenerationService

@pytest.fixture
def mock_embedding():
    mock = MagicMock(spec=EmbeddingService)
    mock.load_or_embed_resume.return_value = np.array([0.1] * 1024)
    return mock

@pytest.fixture
def mock_retrieval():
    mock = MagicMock(spec=RetrievalService)
    mock.query.return_value = [
        {"id": 1, "title": "Job 1", "description": "Desc 1", "cleaned_description": "Desc 1"},
        {"id": 2, "title": "Job 2", "description": "Desc 2", "cleaned_description": "Desc 2"}
    ]
    return mock

@pytest.fixture
def mock_reranking():
    mock = MagicMock(spec=RerankingService)
    mock.rerank.return_value = [
        {"id": 1, "title": "Job 1", "description": "Desc 1", "rerank_score": 0.9},
        {"id": 2, "title": "Job 2", "description": "Desc 2", "rerank_score": 0.8}
    ]
    return mock

@pytest.fixture
def mock_generation():
    mock = MagicMock(spec=GenerationService)
    mock.generate_explanations.return_value = [
        {"id": 1, "title": "Job 1", "explanation": "Good match"},
        {"id": 2, "title": "Job 2", "explanation": "Decent match"}
    ]
    return mock

def test_match_pipeline_flow(mock_embedding, mock_retrieval, mock_reranking, mock_generation):
    """Verify embedding → retrieval → reranking → generation sequence"""
    with patch('fastapi_app.app.services.matching_service.EmbeddingService', return_value=mock_embedding), \
         patch('fastapi_app.app.services.matching_service.RetrievalService', return_value=mock_retrieval), \
         patch('fastapi_app.app.services.matching_service.RerankingService', return_value=mock_reranking), \
         patch('fastapi_app.app.services.matching_service.GenerationService', return_value=mock_generation):
        
        service = MatchingService()
        resume_text = "Test resume"
        
        result = service.match(resume_text)
        
        # Verify call sequence
        mock_embedding.load_or_embed_resume.assert_called_once_with(resume_text)
        mock_retrieval.query.assert_called_once()
        mock_reranking.rerank.assert_called_once()
        mock_generation.generate_explanations.assert_called_once()
        
        assert "matches" in result
        assert "total_candidates" in result
        assert "total_reranked" in result

def test_match_without_filters(mock_embedding, mock_retrieval, mock_reranking, mock_generation):
    """use_filters=False → where_filter=None"""
    with patch('fastapi_app.app.services.matching_service.EmbeddingService', return_value=mock_embedding), \
         patch('fastapi_app.app.services.matching_service.RetrievalService', return_value=mock_retrieval), \
         patch('fastapi_app.app.services.matching_service.RerankingService', return_value=mock_reranking), \
         patch('fastapi_app.app.services.matching_service.GenerationService', return_value=mock_generation):
        
        service = MatchingService()
        resume_text = "Test resume"
        
        service.match(resume_text, use_filters=False)
        
        # Verify where_filter is None when use_filters=False
        mock_retrieval.query.assert_called_once()
        call_args = mock_retrieval.query.call_args
        assert call_args[1].get('where_filter') is None

def test_match_without_explanations(mock_embedding, mock_retrieval, mock_reranking, mock_generation):
    """include_explanations=False → generation skipped"""
    with patch('fastapi_app.app.services.matching_service.EmbeddingService', return_value=mock_embedding), \
         patch('fastapi_app.app.services.matching_service.RetrievalService', return_value=mock_retrieval), \
         patch('fastapi_app.app.services.matching_service.RerankingService', return_value=mock_reranking), \
         patch('fastapi_app.app.services.matching_service.GenerationService', return_value=mock_generation):

        service = MatchingService()
        resume_text = "Test resume"

        result = service.match(resume_text, include_explanations=False)

        # Generation should not be called
        mock_generation.generate_explanations.assert_not_called()

        # Matches should have explanation as None (not generated)
        for match in result["matches"]:
            assert match.get("explanation") is None

def test_match_empty_results(mock_embedding, mock_retrieval, mock_reranking, mock_generation):
    """Empty retrieval → empty matches list"""
    mock_retrieval.query.return_value = []
    mock_reranking.rerank.return_value = []  # Return empty when given empty candidates
    mock_generation.generate_explanations.return_value = []  # Return empty list for empty input

    with patch('fastapi_app.app.services.matching_service.EmbeddingService', return_value=mock_embedding), \
         patch('fastapi_app.app.services.matching_service.RetrievalService', return_value=mock_retrieval), \
         patch('fastapi_app.app.services.matching_service.RerankingService', return_value=mock_reranking), \
         patch('fastapi_app.app.services.matching_service.GenerationService', return_value=mock_generation):

        service = MatchingService()
        resume_text = "Test resume"

        result = service.match(resume_text)

        assert result["matches"] == []
        assert result["total_candidates"] == 0
        assert result["total_reranked"] == 0
        mock_reranking.rerank.assert_called_once()
        mock_generation.generate_explanations.assert_called_once()  # Always called even with empty results
