"""Test suite for match_jobs.py orchestration script."""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

# Add src and scripts/pipeline to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "pipeline"))

from src.config import EMBEDDING_DIM, CORPUS_LIMITATION_MESSAGE
from match_jobs import (
    load_resume,
    load_or_embed_resume,
    write_results_markdown,
)


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test files."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp)


@pytest.fixture
def mock_voyage_client():
    """Create a mock Voyage AI client."""
    return MagicMock()


class TestLoadResume:
    """Tests for load_resume function."""

    def test_load_resume_resolved_from_project_root(self, tmp_dir):
        """load_resume resolves paths relative to project root, not CWD."""
        # Create a temp resume file
        resume_path = os.path.join(tmp_dir, "resume.txt")
        with open(resume_path, "w") as f:
            f.write("5 years Python experience\nBachelor's degree")

        # Change to a different directory (simulating non-project-root CWD)
        original_cwd = os.getcwd()
        try:
            os.chdir("/tmp")
            # Call with absolute path should work
            resume_text = load_resume(resume_path)
            assert "Python" in resume_text
        finally:
            os.chdir(original_cwd)

    def test_load_resume_empty_file(self, tmp_dir):
        """load_resume exits with error if resume is empty."""
        # Create an empty file
        resume_path = os.path.join(tmp_dir, "empty_resume.txt")
        with open(resume_path, "w") as f:
            f.write("")

        with pytest.raises(SystemExit):
            load_resume(resume_path)

    def test_load_resume_whitespace_only(self, tmp_dir):
        """load_resume exits if resume contains only whitespace."""
        resume_path = os.path.join(tmp_dir, "whitespace_resume.txt")
        with open(resume_path, "w") as f:
            f.write("   \n\t\n   ")

        with pytest.raises(SystemExit):
            load_resume(resume_path)

    def test_load_resume_file_not_found(self):
        """load_resume exits if file doesn't exist."""
        with pytest.raises(SystemExit):
            load_resume("/nonexistent/path/resume.txt")

    def test_load_resume_valid_content(self, tmp_dir):
        """load_resume returns content when file is valid."""
        resume_path = os.path.join(tmp_dir, "valid_resume.txt")
        content = "Senior Software Engineer with 10 years experience"
        with open(resume_path, "w") as f:
            f.write(content)

        result = load_resume(resume_path)
        assert result == content


class TestLoadOrEmbedResume:
    """Tests for load_or_embed_resume caching logic."""

    @patch("match_jobs.embed_batch")
    def test_load_or_embed_resume_cache_hit(
        self, mock_embed_batch, tmp_dir, mock_voyage_client
    ):
        """Cache hit: existing valid cache is loaded without calling API."""
        cache_path = os.path.join(tmp_dir, "embedding.npy")
        hash_path = os.path.join(tmp_dir, "hash.txt")

        resume_text = "Test resume content"

        # Pre-populate cache with valid data
        test_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)
        np.save(cache_path, test_embedding)

        # Pre-populate hash
        import hashlib
        hash_value = hashlib.md5(resume_text.encode()).hexdigest()
        with open(hash_path, "w") as f:
            f.write(hash_value)

        # Call load_or_embed_resume
        result = load_or_embed_resume(
            mock_voyage_client, resume_text, cache_path, hash_path
        )

        # Verify embedding loaded from cache
        assert np.allclose(result, test_embedding)
        # API should NOT be called
        mock_embed_batch.assert_not_called()

    @patch("match_jobs.embed_batch")
    def test_load_or_embed_resume_corrupted_cache(
        self, mock_embed_batch, tmp_dir, mock_voyage_client
    ):
        """Corrupted cache falls back to re-embedding."""
        cache_path = os.path.join(tmp_dir, "embedding.npy")
        hash_path = os.path.join(tmp_dir, "hash.txt")

        resume_text = "Test resume content"

        # Create corrupted cache (invalid numpy data)
        with open(cache_path, "w") as f:
            f.write("This is not a valid numpy file")

        # Pre-populate hash (so it looks valid)
        import hashlib
        hash_value = hashlib.md5(resume_text.encode()).hexdigest()
        with open(hash_path, "w") as f:
            f.write(hash_value)

        # Mock embed_batch to return valid embedding
        test_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)
        mock_embed_batch.return_value = [test_embedding]

        # Call load_or_embed_resume
        result = load_or_embed_resume(
            mock_voyage_client, resume_text, cache_path, hash_path
        )

        # Verify re-embedding happened
        assert np.allclose(result, test_embedding)
        # API should be called to re-embed
        mock_embed_batch.assert_called_once()

    @patch("match_jobs.embed_batch")
    def test_load_or_embed_resume_hash_mismatch(
        self, mock_embed_batch, tmp_dir, mock_voyage_client
    ):
        """Cache invalidation: mismatched hash triggers re-embed."""
        cache_path = os.path.join(tmp_dir, "embedding.npy")
        hash_path = os.path.join(tmp_dir, "hash.txt")

        resume_text = "New resume content"

        # Pre-populate cache and hash with OLD data
        old_embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        np.save(cache_path, old_embedding)
        with open(hash_path, "w") as f:
            f.write("old_hash_value")

        # Mock embed_batch to return NEW embedding
        new_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)
        mock_embed_batch.return_value = [new_embedding]

        # Call load_or_embed_resume with different resume text
        result = load_or_embed_resume(
            mock_voyage_client, resume_text, cache_path, hash_path
        )

        # Verify new embedding returned (not old cached one)
        assert np.allclose(result, new_embedding)
        # API should be called due to hash mismatch
        mock_embed_batch.assert_called_once()


class TestApiKeyValidation:
    """Tests for API key validation before expensive operations."""

    @patch("match_jobs.load_dotenv")
    @patch("sys.argv", ["match_jobs.py", "--db-path", "/tmp/test.db"])
    def test_api_keys_validated_before_build_collection(self, mock_load_dotenv):
        """main() validates API keys before calling build_collection()."""
        with patch("os.getenv") as mock_getenv:
            # Return None for API keys
            mock_getenv.side_effect = lambda key, default=None: {
                "VOYAGE_API_KEY": None,
                "COHERE_API_KEY": "valid",
                "DB_PATH": "/tmp/test.db",
            }.get(key, default)

            with patch("match_jobs.build_collection") as mock_build:
                with patch("sys.exit", side_effect=SystemExit) as mock_exit:
                    from match_jobs import main

                    with pytest.raises(SystemExit):
                        main()

                    # API key validation should fail
                    mock_exit.assert_called_with(1)
                    # build_collection should NOT be called
                    mock_build.assert_not_called()


class TestWriteResultsMarkdown:
    """Tests for write_results_markdown function."""

    def test_write_results_markdown_no_explanations(self, tmp_dir):
        """Gracefully handle case where no explanations are provided."""
        output_path = os.path.join(tmp_dir, "results.md")

        results = [
            {
                "id": 1,
                "title": "Job A",
                "board_token": "board-1",
                "source_url": "https://example.com/job-a",
                "cleaned_description": "Some description",
                "explanation": None,  # No explanation
            },
            {
                "id": 2,
                "title": "Job B",
                "board_token": "board-2",
                "source_url": "https://example.com/job-b",
                "cleaned_description": "Another description",
                "explanation": None,  # No explanation
            },
        ]

        write_results_markdown(results, output_path)

        # Verify file was created
        assert os.path.exists(output_path)

        # Verify it contains the limitation message
        with open(output_path, "r") as f:
            content = f.read()
            assert CORPUS_LIMITATION_MESSAGE in content

    def test_write_results_markdown_with_explanations(self, tmp_dir):
        """Successfully write results with explanations."""
        output_path = os.path.join(tmp_dir, "results.md")

        results = [
            {
                "id": 1,
                "title": "Senior Engineer",
                "board_token": "board-1",
                "source_url": "https://example.com/job-1",
                "cleaned_description": "5+ years Python required",
                "explanation": "Great fit: you have 10 years experience.",
            },
            {
                "id": 2,
                "title": "Data Scientist",
                "board_token": "board-2",
                "source_url": "https://example.com/job-2",
                "cleaned_description": "ML expertise needed",
                "explanation": "Good fit: ML background matches.",
            },
        ]

        write_results_markdown(results, output_path)

        # Verify file exists
        assert os.path.exists(output_path)

        # Verify content
        with open(output_path, "r") as f:
            content = f.read()
            assert "Senior Engineer" in content
            assert "Data Scientist" in content
            assert "Great fit" in content
            assert "Good fit" in content
            # Should NOT have limitation message when explanations exist
            assert CORPUS_LIMITATION_MESSAGE not in content

    def test_write_results_markdown_empty_results(self, tmp_dir):
        """Handle empty results list gracefully."""
        output_path = os.path.join(tmp_dir, "results.md")

        write_results_markdown([], output_path)

        # Verify file was created
        assert os.path.exists(output_path)

        # Verify it contains the limitation message (no jobs matched)
        with open(output_path, "r") as f:
            content = f.read()
            assert CORPUS_LIMITATION_MESSAGE in content
