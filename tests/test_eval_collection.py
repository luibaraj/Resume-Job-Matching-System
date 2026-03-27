"""
Tests for eval.collection module.

Tests ChromaDB collection building and positive swapping.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.collection import get_or_build_tune_collection, swap_positives


class TestGetOrBuildTuneCollection:
    """Tests for get_or_build_tune_collection function."""

    @patch("eval.collection.chromadb.PersistentClient")
    @patch("eval.collection.data_loading.compute_hash")
    @patch("eval.collection.Path.exists")
    def test_reuse_existing_collection_on_hash_match(
        self, mock_exists, mock_hash, mock_chroma
    ) -> None:
        """Should reuse collection if hash matches."""
        # Setup mocks
        mock_exists.return_value = True
        mock_hash.return_value = "matching_hash"

        mock_client = MagicMock()
        mock_chroma.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        # Create test dataframe
        df = pd.DataFrame({"job_id": [1, 2, 3]})

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "matching_hash"

            result = get_or_build_tune_collection(df, "dummy.db", force_rebuild=False)

            # Should have called get_collection (reuse)
            mock_client.get_collection.assert_called_once()
            assert result == mock_collection

    @patch("eval.collection.chromadb.PersistentClient")
    @patch("eval.collection.data_loading.chunked_select")
    @patch("eval.collection.data_loading.compute_hash")
    @patch("eval.collection.Path.exists")
    def test_rebuild_collection_on_hash_mismatch(
        self, mock_exists, mock_hash, mock_chunked, mock_chroma
    ) -> None:
        """Should rebuild collection if hash doesn't match."""
        # Setup mocks
        mock_exists.return_value = True
        mock_hash.return_value = "new_hash"

        mock_client = MagicMock()
        mock_chroma.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.delete_collection.return_value = None

        # Mock chunked_select to return empty rows
        mock_chunked.return_value = []

        # Create test dataframe
        df = pd.DataFrame({"job_id": [1, 2, 3]})

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "old_hash"

            result = get_or_build_tune_collection(df, "dummy.db", force_rebuild=False)

            # Should have called get_or_create_collection (rebuild)
            mock_client.get_or_create_collection.assert_called_once()
            assert result == mock_collection

    @patch("eval.collection.chromadb.PersistentClient")
    @patch("eval.collection.data_loading.chunked_select")
    @patch("eval.collection.data_loading.compute_hash")
    def test_force_rebuild_creates_new_collection(
        self, mock_hash, mock_chunked, mock_chroma
    ) -> None:
        """Should rebuild collection if force_rebuild=True."""
        mock_hash.return_value = "current_hash"
        mock_client = MagicMock()
        mock_chroma.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.delete_collection.return_value = None

        # Mock chunked_select to return empty rows
        mock_chunked.return_value = []

        df = pd.DataFrame({"job_id": [1, 2, 3]})

        with patch("builtins.open", create=True):
            result = get_or_build_tune_collection(
                df, "dummy.db", force_rebuild=True
            )

            # Should have deleted and recreated
            mock_client.delete_collection.assert_called_once()
            mock_client.get_or_create_collection.assert_called_once()


class TestSwapPositives:
    """Tests for swap_positives function."""

    def test_swap_positives_deletes_previous(self) -> None:
        """Should delete previous positive IDs from collection."""
        mock_collection = MagicMock()

        # Create test data
        positives_df = pd.DataFrame(
            {
                "id": ["uuid-1", "uuid-2"],
                "title": ["Title 1", "Title 2"],
                "job_description": ["Desc 1", "Desc 2"],
            }
        )
        positive_embeddings = {
            "uuid-1": np.array([1.0, 2.0]),
            "uuid-2": np.array([3.0, 4.0]),
        }

        prev_ids = ["pos_old-uuid-1", "pos_old-uuid-2"]

        result = swap_positives(
            mock_collection, prev_ids, positives_df, positive_embeddings
        )

        # Should have called delete with previous IDs
        mock_collection.delete.assert_called_once_with(ids=prev_ids)

    def test_swap_positives_upserts_current(self) -> None:
        """Should upsert current positive IDs to collection."""
        mock_collection = MagicMock()

        # Create test data
        positives_df = pd.DataFrame(
            {
                "id": ["uuid-1"],
                "title": ["Software Engineer"],
                "job_description": ["Build systems..."],
            }
        )
        positive_embeddings = {
            "uuid-1": np.array([1.0, 2.0, 3.0]),
        }

        result = swap_positives(
            mock_collection, [], positives_df, positive_embeddings
        )

        # Should have called upsert
        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args[1]

        # Verify upsert arguments
        assert "pos_uuid-1" in call_kwargs["ids"]
        assert len(call_kwargs["embeddings"]) == 1
        assert len(call_kwargs["metadatas"]) == 1

    def test_swap_positives_returns_current_ids(self) -> None:
        """Should return current positive ChromaDB IDs."""
        mock_collection = MagicMock()

        positives_df = pd.DataFrame(
            {
                "id": ["uuid-1", "uuid-2"],
                "title": ["Title 1", "Title 2"],
                "job_description": ["Desc 1", "Desc 2"],
            }
        )
        positive_embeddings = {
            "uuid-1": np.array([1.0, 2.0]),
            "uuid-2": np.array([3.0, 4.0]),
        }

        result = swap_positives(mock_collection, [], positives_df, positive_embeddings)

        # Should return list of current ChromaDB IDs
        assert isinstance(result, list)
        assert "pos_uuid-1" in result
        assert "pos_uuid-2" in result
        assert len(result) == 2

    def test_swap_positives_skips_missing_embeddings(self) -> None:
        """Should skip positives without embeddings."""
        mock_collection = MagicMock()

        positives_df = pd.DataFrame(
            {
                "id": ["uuid-1", "uuid-2"],
                "title": ["Title 1", "Title 2"],
                "job_description": ["Desc 1", "Desc 2"],
            }
        )
        # Only uuid-1 has embedding
        positive_embeddings = {
            "uuid-1": np.array([1.0, 2.0]),
        }

        result = swap_positives(mock_collection, [], positives_df, positive_embeddings)

        # Should only include uuid-1
        assert len(result) == 1
        assert "pos_uuid-1" in result
        assert "pos_uuid-2" not in result
