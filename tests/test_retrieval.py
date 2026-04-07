"""Test suite for the retrieval module."""

import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import chromadb
import numpy as np
import pytest

# Add src to path for imports
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from embedding import serialize_embedding
from retrieval import (
    EMBEDDING_DIM,
    DEFAULT_COLLECTION_NAME,
    JobResult,
    build_collection,
    query_collection,
)


@pytest.fixture
def tmp_db():
    """Create a temporary database with the jobs schema and sample embedded jobs."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT NOT NULL,
            board_token TEXT NOT NULL,
            title TEXT,
            location TEXT,
            description TEXT,
            source TEXT,
            source_url TEXT,
            company_name TEXT,
            department TEXT,
            job_type TEXT,
            scraped_at TEXT,
            updated_at TEXT,
            cleaned_description TEXT,
            preprocessed INTEGER DEFAULT 0,
            embedding BLOB,
            embedded INTEGER DEFAULT 0,
            UNIQUE(external_id, board_token)
        )
    """
    )

    # Insert 3 sample jobs with known embeddings
    # Job 1: all ones
    embedding_1 = np.ones(EMBEDDING_DIM, dtype=np.float32)
    # Job 2: all zeros
    embedding_2 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    # Job 3: all 0.5
    embedding_3 = np.full(EMBEDDING_DIM, 0.5, dtype=np.float32)

    conn.execute(
        """
        INSERT INTO jobs (external_id, board_token, title, location, source_url,
                          cleaned_description, embedding, embedded)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "job-1",
            "board-a",
            "Senior Engineer",
            "San Francisco, CA",
            "https://example.com/job-1",
            "We are looking for a senior engineer with 5+ years of experience.",
            serialize_embedding(embedding_1),
            1,
        ),
    )

    conn.execute(
        """
        INSERT INTO jobs (external_id, board_token, title, location, source_url,
                          cleaned_description, embedding, embedded)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "job-2",
            "board-b",
            "Data Scientist",
            "New York, NY",
            "https://example.com/job-2",
            "Join our data science team and work on cutting-edge ML projects.",
            serialize_embedding(embedding_2),
            1,
        ),
    )

    conn.execute(
        """
        INSERT INTO jobs (external_id, board_token, title, location, source_url,
                          cleaned_description, embedding, embedded)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "job-3",
            "board-c",
            "Product Manager",
            "Seattle, WA",
            "https://example.com/job-3",
            "We are seeking an experienced product manager to lead our next initiative.",
            serialize_embedding(embedding_3),
            1,
        ),
    )

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def tmp_chroma():
    """Create a temporary Chroma client directory."""
    tmp_dir = tempfile.mkdtemp()
    client = chromadb.PersistentClient(path=tmp_dir)
    yield client
    shutil.rmtree(tmp_dir)


class TestBuildCollection:
    """Tests for build_collection function."""

    def test_upserts_all_embedded_rows(self, tmp_db, tmp_chroma):
        """Verify that all embedded rows are upserted to the collection."""
        conn = sqlite3.connect(tmp_db)
        collection = build_collection(conn, tmp_chroma)

        assert collection.count() == 3
        conn.close()

    def test_skips_unembedded_rows(self, tmp_db, tmp_chroma):
        """Verify that rows with embedded=0 are skipped."""
        conn = sqlite3.connect(tmp_db)

        # Insert a 4th row with embedded=0
        embedding_4 = np.ones(EMBEDDING_DIM, dtype=np.float32)
        conn.execute(
            """
            INSERT INTO jobs (external_id, board_token, title, location, source_url,
                              cleaned_description, embedding, embedded)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-4",
                "board-d",
                "Intern",
                "Los Angeles, CA",
                "https://example.com/job-4",
                "Looking for a bright intern.",
                serialize_embedding(embedding_4),
                0,  # embedded=0
            ),
        )
        conn.commit()

        collection = build_collection(conn, tmp_chroma)

        assert collection.count() == 3  # Still only 3, not 4
        conn.close()

    def test_metadata_fields_stored(self, tmp_db, tmp_chroma):
        """Verify that metadata fields are correctly stored."""
        conn = sqlite3.connect(tmp_db)
        collection = build_collection(conn, tmp_chroma)

        result = collection.get(ids=["1"], include=["metadatas"])
        metadata = result["metadatas"][0]

        assert "title" in metadata
        assert "location" in metadata
        assert "source_url" in metadata
        assert "board_token" in metadata
        assert "cleaned_description" in metadata
        assert "required_degree" in metadata
        assert "seniority_level" in metadata
        assert "min_years_experience" in metadata

        assert metadata["title"] == "Senior Engineer"
        assert metadata["location"] == "San Francisco, CA"

        conn.close()

    def test_idempotent(self, tmp_db, tmp_chroma):
        """Verify that calling build_collection twice is idempotent."""
        conn = sqlite3.connect(tmp_db)

        build_collection(conn, tmp_chroma)
        count_after_first = tmp_chroma.get_or_create_collection(
            DEFAULT_COLLECTION_NAME
        ).count()

        build_collection(conn, tmp_chroma)
        count_after_second = tmp_chroma.get_or_create_collection(
            DEFAULT_COLLECTION_NAME
        ).count()

        assert count_after_first == 3
        assert count_after_second == 3

        conn.close()

    def test_returns_collection_object(self, tmp_db, tmp_chroma):
        """Verify that build_collection returns a chromadb.Collection object."""
        conn = sqlite3.connect(tmp_db)
        collection = build_collection(conn, tmp_chroma)

        assert isinstance(collection, chromadb.Collection)
        conn.close()

    def test_ef_construction_stored_in_metadata(self, tmp_db, tmp_chroma):
        """Verify that ef_construction parameter is stored in collection metadata."""
        conn = sqlite3.connect(tmp_db)
        collection = build_collection(conn, tmp_chroma, ef_construction=200)

        assert collection.metadata.get("hnsw_construction") == 200
        conn.close()

    def test_model_param_accepted(self, tmp_db, tmp_chroma):
        import sqlite3
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        collection = build_collection(conn, tmp_chroma, "test_model_param", model="test-model")
        conn.close()
        assert collection is not None

    def test_metadata_uses_fallback_functions(self, tmp_db, tmp_chroma):
        import sqlite3
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        with patch("retrieval.extract_degree_with_fallback", return_value=1) as mock_deg, \
             patch("retrieval.extract_seniority_with_fallback", return_value=2) as mock_sen, \
             patch("retrieval.extract_years_with_fallback", return_value=3) as mock_yrs:
            collection = build_collection(conn, tmp_chroma, "test_fallback_meta")
            embedded_count = collection.count()
            assert mock_deg.call_count == embedded_count
            assert mock_sen.call_count == embedded_count
            assert mock_yrs.call_count == embedded_count
        conn.close()


class TestQueryCollection:
    """Tests for query_collection function."""

    @pytest.fixture
    def populated_collection(self, tmp_db, tmp_chroma):
        """Build a collection and return it."""
        conn = sqlite3.connect(tmp_db)
        collection = build_collection(conn, tmp_chroma)
        conn.close()
        return collection

    def test_returns_top_k_results(self, populated_collection):
        """Verify that query_collection returns top_k results."""
        query_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)
        results = query_collection(populated_collection, query_embedding, top_k=3)

        assert len(results) == 3

    def test_results_ordered_by_distance(self, populated_collection):
        """Verify that results are ordered by ascending distance (most similar first)."""
        query_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)
        results = query_collection(populated_collection, query_embedding, top_k=3)

        # First result should be closest (distance ~0.0 for exact match)
        assert results[0]["distance"] < results[1]["distance"]
        assert results[1]["distance"] < results[2]["distance"]

    def test_result_has_expected_keys(self, populated_collection):
        """Verify that each result has all expected JobResult keys."""
        query_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)
        results = query_collection(populated_collection, query_embedding, top_k=1)

        assert len(results) == 1
        result = results[0]

        assert "id" in result
        assert "distance" in result
        assert "title" in result
        assert "location" in result
        assert "source_url" in result
        assert "board_token" in result
        assert "cleaned_description" in result
        assert "required_degree" in result
        assert "seniority_level" in result
        assert "min_years_experience" in result

    def test_invalid_embedding_shape_raises(self, populated_collection):
        """Verify that an invalid embedding shape raises ValueError."""
        invalid_embedding = np.ones(512, dtype=np.float32)  # Wrong shape

        with pytest.raises(ValueError, match="Expected query_embedding shape"):
            query_collection(populated_collection, invalid_embedding)

    def test_top_k_respected(self, populated_collection):
        """Verify that top_k parameter is respected."""
        query_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)

        results_1 = query_collection(populated_collection, query_embedding, top_k=1)
        results_2 = query_collection(populated_collection, query_embedding, top_k=2)

        assert len(results_1) == 1
        assert len(results_2) == 2

    def test_ef_set_on_collection(self, populated_collection):
        """Verify that ef parameter is set on the collection before querying."""
        query_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)
        query_collection(populated_collection, query_embedding, top_k=1, ef=50)

        assert populated_collection.metadata.get("hnsw:ef") == 50

    def test_where_filter_parameter_accepted(self, populated_collection):
        """Verify that where parameter is accepted without error."""
        query_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)
        where_filter = {"seniority_level": {"$eq": 1}}

        results = query_collection(populated_collection, query_embedding, top_k=3, where=where_filter)

        # Should return results (may be 0 if none match the filter, but no error)
        assert isinstance(results, list)

    def test_no_where_filter_returns_all(self, populated_collection):
        """Verify that omitting where parameter returns unfiltered results."""
        query_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)

        results = query_collection(populated_collection, query_embedding, top_k=3)

        # Should return all 3 results
        assert len(results) == 3
