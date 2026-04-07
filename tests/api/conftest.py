"""Shared pytest fixtures for FastAPI API tests."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from fastapi.testclient import TestClient

# Add the project root to sys.path to allow imports from fastapi_app
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi_app.api.main import app
from fastapi_app.api.dependencies import get_voyage_client, get_chroma_collection, get_db, get_cohere_client, get_ollama_base_url


@pytest.fixture
def mock_voyage():
    """Mock VoyageAI client."""
    client = Mock()
    client.embed.return_value = Mock(embeddings=[[0.1] * 1024])
    return client


@pytest.fixture
def mock_collection():
    """Mock ChromaDB collection."""
    collection = Mock()
    collection.count.return_value = 5
    collection.query.return_value = {
        "ids": [["1", "2", "3"]],
        "documents": [
            [
                "Senior Backend Engineer - Build scalable APIs with Python and AWS",
                "Data Scientist - Analyze large datasets with Python and SQL",
                "DevOps Engineer - Manage Kubernetes clusters and CI/CD pipelines"
            ]
        ],
        "metadatas": [[
            {"title": "Senior Backend Engineer", "job_id": 1},
            {"title": "Data Scientist", "job_id": 2},
            {"title": "DevOps Engineer", "job_id": 3}
        ]],
        "distances": [[0.1, 0.2, 0.3]],
    }
    return collection


@pytest.fixture
def mock_db():
    """Mock SQLite database connection."""
    conn = Mock()
    cursor = Mock()
    conn.cursor.return_value = cursor
    cursor.execute.return_value = None
    cursor.fetchone.return_value = (1,)
    return conn


@pytest.fixture
def mock_cohere():
    """Mock Cohere client for reranking."""
    client = Mock()
    mock_rerank_result = Mock()
    mock_rerank_result.results = [
        Mock(index=0, relevance_score=0.95),
        Mock(index=1, relevance_score=0.85),
        Mock(index=2, relevance_score=0.75),
    ]
    client.rerank.return_value = mock_rerank_result
    return client


@pytest.fixture
def mock_ollama_health():
    """Mock Ollama health check."""
    with patch('fastapi_app.api.routers.health.check_ollama_health') as mock:
        mock.return_value = (True, "Ollama is healthy")
        yield mock


@pytest.fixture
def mock_generate_explanation():
    """Mock generate_explanation function."""
    with patch('fastapi_app.api.routers.match.generate_explanation_with_pipeline') as mock:
        mock.return_value = ("This is a generated explanation", None)
        yield mock


@pytest.fixture
def api_client(mock_voyage, mock_collection, mock_db, mock_cohere, mock_ollama_health, mock_generate_explanation):
    """
    TestClient with all dependencies overridden.
    """
    # Set required environment variables
    import os
    os.environ["CHROMA_DIR"] = "/tmp/test_chroma"
    os.environ["CHROMA_COLLECTION"] = "test_collection"
    os.environ["VOYAGE_API_KEY"] = "test_key"
    os.environ["COHERE_API_KEY"] = "test_key"
    os.environ["DB_PATH"] = "/tmp/test.db"
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    
    # Override dependencies
    app.dependency_overrides[get_voyage_client] = lambda: mock_voyage
    app.dependency_overrides[get_chroma_collection] = lambda: mock_collection
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_cohere_client] = lambda: mock_cohere
    app.dependency_overrides[get_ollama_base_url] = lambda: "http://localhost:11434"
    
    yield TestClient(app)
    
    # Clear overrides after test
    app.dependency_overrides.clear()
