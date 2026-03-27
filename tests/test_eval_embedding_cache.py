"""
Tests for eval.embedding_cache module.

Tests embed_with_cache and wrapper functions for caching behavior.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.embedding_cache import embed_positives, embed_resumes, embed_with_cache


class TestEmbedWithCache:
    """Tests for embed_with_cache generic function."""

    def test_embed_with_cache_hit(self, tmp_path) -> None:
        """Should load from cache if hash matches."""
        cache_path = tmp_path / "embeddings.npz"
        hash_path = tmp_path / "embeddings.hash"

        # Create test data
        test_data = {"id1": np.array([1.0, 2.0, 3.0])}
        np.savez(cache_path, **test_data)
        hash_path.write_text("test_hash")

        # Create dataframe
        df = pd.DataFrame({"id": ["id1"], "text": ["sample text"]})

        # Mock compute_hash to return matching hash
        with patch("eval.embedding_cache.data_loading.compute_hash") as mock_hash:
            mock_hash.return_value = "test_hash"

            # Mock voyage client
            voyage_client = MagicMock()

            result = embed_with_cache(
                voyage_client,
                df,
                id_col="id",
                text_col="text",
                cache_path=str(cache_path),
                hash_path=str(hash_path),
            )

            # Should return cached data, not call embed_batch
            assert "id1" in result
            np.testing.assert_array_equal(result["id1"], test_data["id1"])

    def test_embed_with_cache_miss_hash_mismatch(self, tmp_path) -> None:
        """Should re-embed if hash doesn't match."""
        cache_path = tmp_path / "embeddings.npz"
        hash_path = tmp_path / "embeddings.hash"

        # Create stale cache with different hash
        test_data = {"id1": np.array([1.0, 2.0, 3.0])}
        np.savez(cache_path, **test_data)
        hash_path.write_text("old_hash")

        # Create dataframe
        df = pd.DataFrame({"id": ["id1"], "text": ["new text"]})

        # Mock compute_hash to return different hash
        with patch("eval.embedding_cache.data_loading.compute_hash") as mock_hash:
            mock_hash.return_value = "new_hash"

            # Mock embed_batch
            with patch("eval.embedding_cache.embedding.embed_batch") as mock_embed:
                mock_embed.return_value = [np.array([4.0, 5.0, 6.0])]

                voyage_client = MagicMock()
                result = embed_with_cache(
                    voyage_client,
                    df,
                    id_col="id",
                    text_col="text",
                    cache_path=str(cache_path),
                    hash_path=str(hash_path),
                )

                # Should have called embed_batch
                mock_embed.assert_called_once()
                assert "id1" in result
                # Result should be new embedding, not cached
                np.testing.assert_array_equal(
                    result["id1"], np.array([4.0, 5.0, 6.0])
                )

    def test_embed_with_cache_skip_empty(self, tmp_path) -> None:
        """Should skip empty texts when skip_empty=True."""
        cache_path = tmp_path / "embeddings.npz"
        hash_path = tmp_path / "embeddings.hash"

        # Create dataframe with empty text
        df = pd.DataFrame(
            {
                "id": ["id1", "id2"],
                "text": ["valid text", ""],
            }
        )

        with patch("eval.embedding_cache.data_loading.compute_hash") as mock_hash:
            mock_hash.return_value = "test_hash"

            with patch("eval.embedding_cache.embedding.embed_batch") as mock_embed:
                mock_embed.return_value = [np.array([1.0, 2.0])]

                voyage_client = MagicMock()
                result = embed_with_cache(
                    voyage_client,
                    df,
                    id_col="id",
                    text_col="text",
                    cache_path=str(cache_path),
                    hash_path=str(hash_path),
                    skip_empty=True,
                )

                # Only one non-empty text should be embedded
                mock_embed.assert_called_once()
                call_args = mock_embed.call_args
                assert len(call_args[0][1]) == 1  # Only 1 text passed
                assert "id1" in result
                assert "id2" not in result


class TestEmbedPositives:
    """Tests for embed_positives wrapper."""

    def test_embed_positives_returns_string_keys(self, tmp_path) -> None:
        """embed_positives should return dict with string keys (UUIDs)."""
        cache_path = tmp_path / "positives.npz"
        hash_path = tmp_path / "positives.hash"

        df = pd.DataFrame(
            {
                "id": ["uuid-1", "uuid-2"],
                "job_description": ["desc1", "desc2"],
            }
        )

        with patch("eval.embedding_cache.embed_with_cache") as mock_cache:
            mock_cache.return_value = {"uuid-1": np.array([1.0, 2.0])}

            voyage_client = MagicMock()
            result = embed_positives(
                voyage_client,
                df,
                cache_path=str(cache_path),
                hash_path=str(hash_path),
            )

            # Keys should be strings
            assert all(isinstance(k, str) for k in result.keys())


class TestEmbedResumes:
    """Tests for embed_resumes wrapper."""

    def test_embed_resumes_returns_int_keys(self, tmp_path) -> None:
        """embed_resumes should return dict with int keys (resume IDs)."""
        cache_path = tmp_path / "resumes.npz"
        hash_path = tmp_path / "resumes.hash"

        df = pd.DataFrame(
            {
                "id": [1, 2],
                "resume": ["resume1", "resume2"],
            }
        )

        with patch("eval.embedding_cache.embed_with_cache") as mock_cache:
            mock_cache.return_value = {"1": np.array([1.0, 2.0]), "2": np.array([3.0, 4.0])}

            voyage_client = MagicMock()
            result = embed_resumes(
                voyage_client,
                df,
                cache_path=str(cache_path),
                hash_path=str(hash_path),
            )

            # Keys should be integers
            assert all(isinstance(k, int) for k in result.keys())
            assert 1 in result
            assert 2 in result
