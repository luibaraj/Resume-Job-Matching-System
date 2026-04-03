"""Shared pytest fixtures for FastAPI API tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

# Add the project root to sys.path to allow imports from fastapi_app
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi_app.api.main import app
from fastapi_app.api.dependencies import get_voyage_client, get_chroma_collection, get_db, get_cohere_client


@pytest.fixture
def mock_voyage():
    """Mock VoyageAI client with embed method."""
    client = MagicMock()
    # Mock embed method to return embeddings
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1] * 1024]  # 1024-dim embedding
    client.embed.return_value = mock_result
    return client


@pytest.fixture
def mock_collection():
    """Mock ChromaDB collection with query and count methods."""
    collection = MagicMock()
    
    # Mock count method
    collection.count.return_value = 5
    
    # Mock query method for successful retrieval
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
    
    # Mock for empty collection scenario
    collection.empty_query_result = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    
    return collection


@pytest.fixture
def mock_db():
    """Mock SQLite database connection."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.execute.return_value = None
    cursor.fetchone.return_value = (1,)  # Simulate table exists
    return conn


@pytest.fixture
def mock_cohere():
    """Mock Cohere client for reranking."""
    client = MagicMock()
    
    # Mock rerank method
    mock_rerank_result = MagicMock()
    mock_rerank_result.results = [
        MagicMock(index=0, relevance_score=0.95),
        MagicMock(index=1, relevance_score=0.85),
        MagicMock(index=2, relevance_score=0.75),
    ]
    client.rerank.return_value = mock_rerank_result
    
    return client


@pytest.fixture
def api_client(mock_voyage, mock_collection, mock_db, mock_cohere):
    """
    TestClient with all dependencies overridden.
    
    Overrides:
    - get_voyage_client: returns mock_voyage
    - get_chroma_collection: returns mock_collection  
    - get_db: returns mock_db
    - get_cohere_client: returns mock_cohere
    """
    # Override dependencies
    app.dependency_overrides[get_voyage_client] = lambda: mock_voyage
    app.dependency_overrides[get_chroma_collection] = lambda: mock_collection
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_cohere_client] = lambda: mock_cohere
    
    yield TestClient(app)
    
    # Clear overrides after test
    app.dependency_overrides.clear()
