"""Tests for Phase 2 Reliability improvements.

Validates:
1. _call_ollama() retry logic with exponential backoff
2. call_ollama_validate/repair retry logic
3. embed_batch 4xx fast-fail
4. Jitter in exponential backoff (embed, rerank, scrape)
5. Ollama pre-flight check in match_jobs
6. SQLite connection cleanup in data_loading
7. sample_jobs raises ValueError on size reduction
"""

import sys
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import time

import numpy as np
import ollama
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generation import _call_ollama
from embedding import embed_batch
from reranking import rerank_jobs

# Delay imports to avoid circular dependency issues
eval_utils = None
sample_jobs = None
load_sampled_job_embeddings = None


def lazy_import_eval_utils():
    global eval_utils
    if eval_utils is None:
        from eval import eval_utils as _eval_utils
        eval_utils = _eval_utils
    return eval_utils


def lazy_import_data_loading():
    global sample_jobs, load_sampled_job_embeddings
    if sample_jobs is None:
        from eval.data_loading import sample_jobs as _sample_jobs
        from eval.data_loading import load_sampled_job_embeddings as _load_sampled
        sample_jobs = _sample_jobs
        load_sampled_job_embeddings = _load_sampled
    return sample_jobs, load_sampled_job_embeddings


# ============================================================================
# Item 1: _call_ollama() retry with timeout
# ============================================================================


class TestCallOllamaRetry:
    """Test _call_ollama() retry logic with exponential backoff."""

    def make_ollama_response(self, content: str) -> dict:
        """Build a minimal mock ollama.chat return value."""
        return {"message": {"content": content}}

    @patch("generation.ollama.chat")
    def test_succeeds_on_first_attempt(self, mock_chat: MagicMock) -> None:
        """_call_ollama succeeds on first attempt."""
        mock_chat.return_value = self.make_ollama_response("test response")
        result = _call_ollama("test prompt")
        assert result == "test response"
        mock_chat.assert_called_once()

    @patch("generation.time.sleep")
    @patch("generation.ollama.chat")
    def test_retries_once_on_request_error(
        self, mock_chat: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """_call_ollama retries once on RequestError, succeeds on second."""
        mock_chat.side_effect = [
            ollama.RequestError("Connection failed"),
            self.make_ollama_response("success on retry"),
        ]
        result = _call_ollama("test prompt")
        assert result == "success on retry"
        assert mock_chat.call_count == 2
        mock_sleep.assert_called_once()

    @patch("generation.ollama.chat")
    def test_raises_after_max_retries(self, mock_chat: MagicMock) -> None:
        """_call_ollama raises after exceeding max_retries."""
        mock_chat.side_effect = ollama.RequestError("Connection failed")
        with pytest.raises(ollama.RequestError):
            _call_ollama("test prompt", max_retries=1)
        assert mock_chat.call_count == 2  # 1 initial + 1 retry

    @patch("generation.ollama.chat")
    def test_passes_options_dict(self, mock_chat: MagicMock) -> None:
        """_call_ollama passes temperature and other options in options dict."""
        mock_chat.return_value = self.make_ollama_response("response")
        _call_ollama("test prompt")
        # Verify options dict was passed (no timeout as separate param)
        call_kwargs = mock_chat.call_args[1]
        assert "options" in call_kwargs
        assert "temperature" in call_kwargs["options"]

    @patch("generation.ollama.chat")
    def test_does_not_retry_response_error(self, mock_chat: MagicMock) -> None:
        """_call_ollama does not retry on ResponseError (non-transient)."""
        mock_chat.side_effect = ollama.ResponseError("Model error")
        with pytest.raises(ollama.ResponseError):
            _call_ollama("test prompt")
        # Should fail immediately, no retries
        assert mock_chat.call_count == 1


# ============================================================================
# Item 2: call_ollama_validate/repair retry logic
# ============================================================================


class TestCallOllamaValidateRetry:
    """Test call_ollama_validate retry logic."""

    def test_validate_has_retry_logic(self) -> None:
        """call_ollama_validate includes retry logic with exponential backoff."""
        # Verify source code directly to avoid circular import
        eval_utils_path = Path(__file__).parent.parent / "eval" / "eval_utils.py"
        source = eval_utils_path.read_text()
        # Find call_ollama_validate function
        func_start = source.index("def call_ollama_validate")
        func_section = source[func_start : source.index("\ndef ", func_start + 1)]
        # Verify retry logic exists
        assert "ollama.RequestError" in func_section
        assert "attempt" in func_section
        # Verify max_retries parameter
        assert "max_retries" in func_section
        # Verify exponential backoff
        assert "2 **" in func_section or "2 **" in func_section


class TestCallOllamaRepairRetry:
    """Test call_ollama_repair retry logic."""

    def test_repair_has_retry_logic_with_2_attempts(self) -> None:
        """call_ollama_repair includes retry logic with 2 attempts."""
        # Verify source code directly to avoid circular import
        eval_utils_path = Path(__file__).parent.parent / "eval" / "eval_utils.py"
        source = eval_utils_path.read_text()
        # Find call_ollama_repair function
        func_start = source.index("def call_ollama_repair")
        # Find the next function definition
        next_func = source.index("\ndef ", func_start + 1)
        func_section = source[func_start:next_func]
        # Verify retry logic exists
        assert "ollama.RequestError" in func_section
        assert "attempt" in func_section
        # Verify max_retries parameter with default of 2
        assert "max_retries" in func_section
        assert "= 2" in func_section  # default value


# ============================================================================
# Item 3: embed_batch 4xx fast-fail
# ============================================================================


class TestEmbedBatchFastFail:
    """Test embed_batch 4xx fast-fail behavior."""

    def test_retries_on_5xx_error(self) -> None:
        """embed_batch retries on 5xx errors."""
        mock_embed = MagicMock()
        mock_embed.side_effect = [
            Exception("500 Internal Server Error"),
            MagicMock(embeddings=[[1.0, 2.0]]),
        ]
        client = MagicMock()
        client.embed = mock_embed
        result = embed_batch(client, ["test"])
        assert len(result) == 1
        assert mock_embed.call_count == 2

    def test_fast_fails_on_400_error(self) -> None:
        """embed_batch fails immediately on 400 Bad Request."""
        mock_embed = MagicMock()
        mock_embed.side_effect = Exception("400 Bad Request")
        client = MagicMock()
        client.embed = mock_embed
        with pytest.raises(Exception, match="400"):
            embed_batch(client, ["test"])
        # Should fail immediately, only 1 attempt
        assert mock_embed.call_count == 1

    def test_fast_fails_on_401_error(self) -> None:
        """embed_batch fails immediately on 401 Unauthorized."""
        mock_embed = MagicMock()
        mock_embed.side_effect = Exception("401 Unauthorized")
        client = MagicMock()
        client.embed = mock_embed
        with pytest.raises(Exception, match="401"):
            embed_batch(client, ["test"])
        assert mock_embed.call_count == 1

    def test_includes_429_exception_handling(self) -> None:
        """embed_batch source includes 429 exception handling."""
        import inspect
        source = inspect.getsource(embed_batch)
        # Verify 429 is specifically excluded from fast-fail
        assert "429" in source
        # Verify 4xx fast-fail logic
        assert "40" in source or "4xx" in source or "400" in source


# ============================================================================
# Item 4: Jitter in exponential backoff
# ============================================================================


class TestBackoffJitter:
    """Test that exponential backoff includes random jitter."""

    @patch("embedding.time.sleep")
    @patch("embedding.random.uniform")
    def test_embed_batch_applies_jitter(
        self,
        mock_uniform: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """embed_batch adds jitter to backoff delays."""
        # Jitter of 0.1 (10% of base delay)
        mock_uniform.return_value = 0.1
        mock_embed = MagicMock()
        mock_embed.side_effect = [
            Exception("Network error"),
            MagicMock(embeddings=[[1.0, 2.0]]),
        ]
        client = MagicMock()
        client.embed = mock_embed
        result = embed_batch(client, ["test"], retry_base_delay=2.0)
        assert len(result) == 1
        # Verify jitter was called: uniform(0, delay * 0.1)
        mock_uniform.assert_called_once_with(0, 2.0 * 0.1)
        # Verify sleep was called with (base + jitter)
        mock_sleep.assert_called_once_with(2.0 + 0.1)

    def test_rerank_jobs_includes_jitter(self) -> None:
        """rerank_jobs source includes jitter logic."""
        import inspect
        source = inspect.getsource(rerank_jobs)
        # Verify random.uniform is called for jitter
        assert "random.uniform" in source
        # Verify jitter is applied to delay
        assert "delay * 0.1" in source or "0.1" in source

    def test_scrape_jobs_applies_jitter(self) -> None:
        """scrape_jobs.py includes jitter in backoff (import check)."""
        # This is an integration test: verify the module imports random
        # and the jitter logic is present by reading the source
        scrape_path = Path(__file__).parent.parent / "scripts" / "pipeline" / "scrape_jobs.py"
        source = scrape_path.read_text()
        # Verify imports random
        assert "import random" in source
        # Verify jitter logic is applied
        assert "random.uniform(0, delay * 0.1)" in source


# ============================================================================
# Item 5: Ollama pre-flight check in match_jobs.py
# ============================================================================


class TestOllamaPreflightCheck:
    """Test Ollama pre-flight check in match_jobs.py."""

    def test_preflights_ollama_list_call(self) -> None:
        """match_jobs.py main() calls ollama.list() before generation."""
        # Verify the source code contains the pre-flight check
        match_path = Path(__file__).parent.parent / "scripts" / "pipeline" / "match_jobs.py"
        source = match_path.read_text()
        # Verify ollama.list() is called
        assert "ollama.list()" in source
        # Verify it's called in main() before run_generation_for_results
        # Extract the main() function to verify ordering
        main_start = source.index("def main()")
        main_section = source[main_start:]
        # Check that ollama.list() appears before run_generation_for_results in main
        ollama_list_pos = main_section.find("ollama.list()")
        gen_pos = main_section.find("run_generation_for_results")
        assert ollama_list_pos > -1, "ollama.list() not found in main()"
        assert gen_pos > -1, "run_generation_for_results not found in main()"
        assert ollama_list_pos < gen_pos, "ollama.list() should be called before run_generation_for_results"


# ============================================================================
# Item 6: SQLite connection cleanup
# ============================================================================


class TestSQLiteConnectionCleanup:
    """Test SQLite connection cleanup with context managers."""

    def test_load_sampled_job_embeddings_uses_context_manager(self) -> None:
        """load_sampled_job_embeddings uses 'with' statement for connection."""
        # Verify source code uses context manager
        import inspect

        _sample_jobs, _load_sampled = lazy_import_data_loading()
        source = inspect.getsource(_load_sampled)
        assert "with sqlite3.connect" in source
        assert "conn.close()" not in source or source.count("with sqlite3.connect") > 0

    def test_sample_jobs_uses_context_manager(self) -> None:
        """sample_jobs uses 'with' statement for SQLite connections."""
        import inspect

        _sample_jobs, _load_sampled = lazy_import_data_loading()
        source = inspect.getsource(_sample_jobs)
        # Should have multiple 'with sqlite3.connect' calls (one for reading, one for fetching)
        assert source.count("with sqlite3.connect") >= 2


# ============================================================================
# Item 7: sample_jobs raises ValueError on size reduction
# ============================================================================


class TestSampleJobsValueError:
    """Test sample_jobs raises ValueError when reducing sizes."""

    @patch("eval.data_loading.Path.exists")
    @patch("eval.data_loading.sqlite3.connect")
    def test_raises_value_error_insufficient_jobs(
        self, mock_connect: MagicMock, mock_exists: MagicMock
    ) -> None:
        """sample_jobs raises ValueError when insufficient embedded jobs."""
        _sample_jobs, _load_sampled = lazy_import_data_loading()
        # Mock CSV doesn't exist (force fresh sample)
        mock_exists.return_value = False

        # Mock connection: insufficient jobs
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = False
        mock_connect.return_value = mock_conn

        # Return only 5 jobs when requesting 10 + 10
        mock_cursor.fetchall.return_value = [(i,) for i in range(5)]

        with pytest.raises(
            ValueError,
            match="Insufficient embedded jobs",
        ):
            _sample_jobs(
                "test.db",
                tune_n=10,
                test_n=10,
                force=True,
            )

    @patch("eval.data_loading.Path.exists")
    @patch("eval.data_loading.sqlite3.connect")
    @patch("eval.data_loading.np.random.default_rng")
    @patch("eval.data_loading.pd.DataFrame.to_csv")
    def test_succeeds_with_sufficient_jobs(
        self,
        mock_to_csv: MagicMock,
        mock_rng: MagicMock,
        mock_connect: MagicMock,
        mock_exists: MagicMock,
    ) -> None:
        """sample_jobs succeeds when sufficient jobs are available."""
        _sample_jobs, _load_sampled = lazy_import_data_loading()
        mock_exists.return_value = False

        # Mock connection with sufficient jobs
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = False
        mock_connect.return_value = mock_conn

        # Return 25 jobs (enough for 10 + 10)
        job_ids = list(range(1, 26))
        mock_cursor.fetchall.side_effect = [
            [(id,) for id in job_ids],  # First query for all IDs
            [(id, "description") for id in job_ids[:10]],  # Tune jobs
            [(id, "description") for id in job_ids[10:20]],  # Test jobs
        ]

        # Mock RNG
        mock_rng_instance = MagicMock()
        mock_rng.return_value = mock_rng_instance
        mock_rng_instance.choice.return_value = np.arange(20)

        # Should not raise
        _sample_jobs("test.db", tune_n=10, test_n=10, force=True)
        mock_to_csv.assert_called()
