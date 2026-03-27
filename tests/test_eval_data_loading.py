"""
Tests for eval.data_loading module.

Tests compute_hash, chunked_select, and job sampling functions.
"""

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.data_loading import chunked_select, compute_hash, sample_jobs


class TestComputeHash:
    """Tests for compute_hash function."""

    def test_compute_hash_returns_hexdigest(self) -> None:
        """Hash should return a valid hex string."""
        data = b"test data"
        result = compute_hash(data)

        assert isinstance(result, str)
        assert len(result) == 32  # MD5 produces 32 hex characters
        assert all(c in "0123456789abcdef" for c in result)

    def test_compute_hash_deterministic(self) -> None:
        """Same input should produce same hash."""
        data = b"test data"
        hash1 = compute_hash(data)
        hash2 = compute_hash(data)

        assert hash1 == hash2

    def test_compute_hash_different_inputs(self) -> None:
        """Different inputs should produce different hashes."""
        hash1 = compute_hash(b"data1")
        hash2 = compute_hash(b"data2")

        assert hash1 != hash2

    def test_compute_hash_empty_input(self) -> None:
        """Empty input should produce a valid hash."""
        result = compute_hash(b"")

        assert isinstance(result, str)
        assert len(result) == 32


class TestChunkedSelect:
    """Tests for chunked_select function."""

    def test_chunked_select_single_chunk(self, tmp_path) -> None:
        """Query with IDs fitting in one chunk."""
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Create test table
        cursor.execute(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)"
        )
        cursor.execute("INSERT INTO items (id, name) VALUES (1, 'item1')")
        cursor.execute("INSERT INTO items (id, name) VALUES (2, 'item2')")
        conn.commit()

        # Query with chunk size > num IDs
        rows = chunked_select(
            conn, "SELECT id, name FROM items WHERE id IN ({})", [1, 2], chunk_size=10
        )

        assert len(rows) == 2
        assert (1, "item1") in rows
        assert (2, "item2") in rows
        conn.close()

    def test_chunked_select_multiple_chunks(self, tmp_path) -> None:
        """Query with IDs spanning multiple chunks."""
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Create test table with 5 rows
        cursor.execute(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)"
        )
        for i in range(1, 6):
            cursor.execute(f"INSERT INTO items (id, name) VALUES ({i}, 'item{i}')")
        conn.commit()

        # Query with small chunk size
        rows = chunked_select(
            conn,
            "SELECT id, name FROM items WHERE id IN ({})",
            [1, 2, 3, 4, 5],
            chunk_size=2,
        )

        assert len(rows) == 5
        conn.close()

    def test_chunked_select_empty_ids(self, tmp_path) -> None:
        """Query with empty ID list."""
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        conn.commit()

        rows = chunked_select(
            conn, "SELECT id, name FROM items WHERE id IN ({})", []
        )

        assert len(rows) == 0
        conn.close()


class TestSampleJobs:
    """Tests for sample_jobs function."""

    @patch("eval.data_loading.sqlite3.connect")
    @patch("eval.data_loading.pd.read_csv")
    @patch("eval.data_loading.pd.DataFrame.to_csv")
    @patch("eval.data_loading.Path.exists")
    def test_sample_jobs_cache_hit(
        self, mock_exists, mock_to_csv, mock_read_csv, mock_connect
    ) -> None:
        """Should load from cache if CSVs exist and force=False."""
        # Mock CSV files exist
        mock_exists.side_effect = lambda: True

        # Mock read_csv returns dataframes
        import pandas as pd

        mock_df = pd.DataFrame({"job_id": [1, 2], "cleaned_description": ["desc1", "desc2"]})
        mock_read_csv.return_value = mock_df

        result_tune, result_test = sample_jobs("dummy.db", force=False)

        assert len(result_tune) == 2
        # Should not have called to_csv (no new sampling)
        mock_to_csv.assert_not_called()

    @patch("eval.data_loading._fetch_jobs_by_id")
    @patch("eval.data_loading.sqlite3.connect")
    @patch("eval.data_loading.Path.exists")
    def test_sample_jobs_cache_miss(
        self, mock_exists, mock_connect, mock_fetch
    ) -> None:
        """Should sample and write to CSV if cache does not exist."""
        # Mock CSV files don't exist
        mock_exists.side_effect = lambda: False

        # Mock database query results
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [(i,) for i in range(1, 101)]
        mock_connect.return_value = mock_conn

        # Mock _fetch_jobs_by_id
        import pandas as pd

        mock_df = pd.DataFrame(
            {"job_id": list(range(1, 51)), "cleaned_description": [f"desc{i}" for i in range(1, 51)]}
        )
        mock_fetch.return_value = mock_df

        with patch("eval.data_loading.Path.mkdir"):
            with patch.object(mock_df, "to_csv"):
                result_tune, result_test = sample_jobs("dummy.db", tune_n=50, test_n=50, force=False)

                assert len(result_tune) == 50
                assert len(result_test) == 50
