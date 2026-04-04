"""Tests for dependency functions."""

import pytest
from unittest.mock import Mock, patch
import requests
from fastapi_app.api.dependencies import check_ollama_health


def test_check_ollama_health_success():
    """Test Ollama health check when server is healthy and model exists."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "models": [
            {"name": "llama3.2:3b-instruct-q4_K_M"},
            {"name": "other-model"}
        ]
    }
    mock_response.raise_for_status.return_value = None
    
    with patch('requests.get', return_value=mock_response) as mock_get:
        healthy, message = check_ollama_health("http://localhost:11434")
        
        assert healthy is True
        assert message == "Ollama is healthy"
        mock_get.assert_called_once_with(
            "http://localhost:11434/api/tags",
            timeout=5
        )


def test_check_ollama_health_connection_error():
    """Test Ollama health check when connection fails."""
    with patch('requests.get', side_effect=requests.exceptions.ConnectionError("Connection refused")) as mock_get:
        healthy, message = check_ollama_health("http://localhost:11434")
        
        assert healthy is False
        assert "connection failed" in message.lower()
        mock_get.assert_called_once()


def test_check_ollama_health_timeout():
    """Test Ollama health check when request times out."""
    with patch('requests.get', side_effect=requests.exceptions.Timeout("Request timed out")) as mock_get:
        healthy, message = check_ollama_health("http://localhost:11434")
        
        assert healthy is False
        assert "timed out" in message.lower()
        mock_get.assert_called_once()


def test_check_ollama_health_http_error():
    """Test Ollama health check when HTTP error occurs."""
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
    
    with patch('requests.get', return_value=mock_response) as mock_get:
        healthy, message = check_ollama_health("http://localhost:11434")
        
        assert healthy is False
        assert "server error" in message.lower()
        mock_get.assert_called_once()


def test_check_ollama_health_missing_model():
    """Test Ollama health check when required model is missing."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "models": [
            {"name": "other-model"},
            {"name": "another-model"}
        ]
    }
    mock_response.raise_for_status.return_value = None
    
    with patch('requests.get', return_value=mock_response) as mock_get:
        healthy, message = check_ollama_health("http://localhost:11434")
        
        assert healthy is False
        assert "required model" in message.lower()
        assert "llama3.2:3b-instruct-q4_K_M" in message
        mock_get.assert_called_once()


def test_check_ollama_health_unexpected_error():
    """Test Ollama health check when unexpected error occurs."""
    with patch('requests.get', side_effect=ValueError("Unexpected error")) as mock_get:
        healthy, message = check_ollama_health("http://localhost:11434")
        
        assert healthy is False
        assert "unexpected error" in message.lower()
        mock_get.assert_called_once()


def test_check_ollama_health_empty_models():
    """Test Ollama health check when models list is empty."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "models": []
    }
    mock_response.raise_for_status.return_value = None
    
    with patch('requests.get', return_value=mock_response) as mock_get:
        healthy, message = check_ollama_health("http://localhost:11434")
        
        assert healthy is False
        assert "required model" in message.lower()
        mock_get.assert_called_once()
