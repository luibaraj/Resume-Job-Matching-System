"""Tests for src/config.py Config dataclass and load_config function."""

import pytest

from src.config import Config, load_config


class TestLoadConfig:
    """Tests for load_config() function."""

    def test_defaults_when_no_env_vars(self, monkeypatch):
        """Returns default Config when no environment variables are set."""
        # Clear all relevant env vars
        monkeypatch.delenv("DB_PATH", raising=False)
        monkeypatch.delenv("GREENHOUSE_BOARD_TOKENS", raising=False)
        monkeypatch.delenv("REQUEST_TIMEOUT", raising=False)
        monkeypatch.delenv("MAX_RETRIES", raising=False)
        monkeypatch.delenv("RETRY_BACKOFF", raising=False)
        monkeypatch.delenv("LOG_LEVEL", raising=False)

        config = load_config()

        assert config.db_path == "/data/jobs.db"
        assert config.request_timeout_seconds == 30
        assert config.max_retries == 3
        assert config.retry_backoff_seconds == 2.0
        assert config.log_level == "INFO"
        assert config.greenhouse_board_tokens == []

    def test_greenhouse_board_tokens_defaults_empty(self, monkeypatch):
        """Default greenhouse_board_tokens is an empty list."""
        monkeypatch.delenv("GREENHOUSE_BOARD_TOKENS", raising=False)
        config = load_config()
        assert config.greenhouse_board_tokens == []
        assert isinstance(config.greenhouse_board_tokens, list)

    def test_reads_db_path_from_env(self, monkeypatch):
        """Reads DB_PATH from environment variable."""
        monkeypatch.setenv("DB_PATH", "/tmp/test.db")
        config = load_config()
        assert config.db_path == "/tmp/test.db"

    def test_reads_log_level_from_env(self, monkeypatch):
        """Reads LOG_LEVEL from environment variable."""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        config = load_config()
        assert config.log_level == "DEBUG"

    def test_reads_timeout_from_env(self, monkeypatch):
        """Reads REQUEST_TIMEOUT from environment variable as integer."""
        monkeypatch.setenv("REQUEST_TIMEOUT", "60")
        config = load_config()
        assert config.request_timeout_seconds == 60
        assert isinstance(config.request_timeout_seconds, int)

    def test_reads_max_retries_from_env(self, monkeypatch):
        """Reads MAX_RETRIES from environment variable."""
        monkeypatch.setenv("MAX_RETRIES", "5")
        config = load_config()
        assert config.max_retries == 5

    def test_reads_retry_backoff_from_env(self, monkeypatch):
        """Reads RETRY_BACKOFF from environment variable as float."""
        monkeypatch.setenv("RETRY_BACKOFF", "1.5")
        config = load_config()
        assert config.retry_backoff_seconds == 1.5
        assert isinstance(config.retry_backoff_seconds, float)

    def test_parses_multiple_board_tokens(self, monkeypatch):
        """Parses comma-separated board tokens."""
        monkeypatch.setenv("GREENHOUSE_BOARD_TOKENS", "openai,anthropic")
        config = load_config()
        assert config.greenhouse_board_tokens == ["openai", "anthropic"]

    def test_parses_single_board_token(self, monkeypatch):
        """Single board token is parsed into a list."""
        monkeypatch.setenv("GREENHOUSE_BOARD_TOKENS", "openai")
        config = load_config()
        assert config.greenhouse_board_tokens == ["openai"]

    def test_strips_whitespace_from_tokens(self, monkeypatch):
        """Whitespace is stripped from each token."""
        monkeypatch.setenv("GREENHOUSE_BOARD_TOKENS", " openai , anthropic ")
        config = load_config()
        assert config.greenhouse_board_tokens == ["openai", "anthropic"]

    def test_filters_empty_tokens(self, monkeypatch):
        """Empty segments in token list are filtered out."""
        monkeypatch.setenv("GREENHOUSE_BOARD_TOKENS", "openai,,anthropic")
        config = load_config()
        assert config.greenhouse_board_tokens == ["openai", "anthropic"]

    def test_returns_config_instance(self, monkeypatch):
        """load_config returns a Config instance."""
        monkeypatch.delenv("DB_PATH", raising=False)
        config = load_config()
        assert isinstance(config, Config)


class TestPreprocessingConfig:
    """Tests for preprocessing_workers and preprocessing_chunk_size config."""

    def test_preprocessing_workers_default(self, monkeypatch):
        """preprocessing_workers defaults to os.cpu_count() or 1."""
        monkeypatch.delenv("PREPROCESSING_WORKERS", raising=False)
        config = load_config()
        import os
        assert config.preprocessing_workers == (os.cpu_count() or 1)

    def test_preprocessing_chunk_size_default(self, monkeypatch):
        """preprocessing_chunk_size defaults to 500."""
        monkeypatch.delenv("PREPROCESSING_CHUNK_SIZE", raising=False)
        config = load_config()
        assert config.preprocessing_chunk_size == 500

    def test_reads_preprocessing_workers_from_env(self, monkeypatch):
        """Reads PREPROCESSING_WORKERS from environment."""
        monkeypatch.setenv("PREPROCESSING_WORKERS", "4")
        config = load_config()
        assert config.preprocessing_workers == 4

    def test_reads_preprocessing_chunk_size_from_env(self, monkeypatch):
        """Reads PREPROCESSING_CHUNK_SIZE from environment."""
        monkeypatch.setenv("PREPROCESSING_CHUNK_SIZE", "1000")
        config = load_config()
        assert config.preprocessing_chunk_size == 1000

    def test_preprocessing_workers_zero_raises(self, monkeypatch):
        """PREPROCESSING_WORKERS=0 raises ValueError."""
        monkeypatch.setenv("PREPROCESSING_WORKERS", "0")
        with pytest.raises(ValueError, match="PREPROCESSING_WORKERS must be >= 1"):
            load_config()

    def test_preprocessing_chunk_size_zero_raises(self, monkeypatch):
        """PREPROCESSING_CHUNK_SIZE=0 raises ValueError."""
        monkeypatch.setenv("PREPROCESSING_CHUNK_SIZE", "0")
        with pytest.raises(ValueError, match="PREPROCESSING_CHUNK_SIZE must be >= 1"):
            load_config()

    def test_preprocessing_workers_negative_raises(self, monkeypatch):
        """Negative PREPROCESSING_WORKERS raises ValueError."""
        monkeypatch.setenv("PREPROCESSING_WORKERS", "-1")
        with pytest.raises(ValueError, match="PREPROCESSING_WORKERS must be >= 1"):
            load_config()

    def test_preprocessing_chunk_size_negative_raises(self, monkeypatch):
        """Negative PREPROCESSING_CHUNK_SIZE raises ValueError."""
        monkeypatch.setenv("PREPROCESSING_CHUNK_SIZE", "-1")
        with pytest.raises(ValueError, match="PREPROCESSING_CHUNK_SIZE must be >= 1"):
            load_config()

    def test_preprocessing_max_retries_default(self, monkeypatch):
        """preprocessing_max_retries defaults to 2."""
        monkeypatch.delenv("PREPROCESSING_MAX_RETRIES", raising=False)
        config = load_config()
        assert config.preprocessing_max_retries == 2

    def test_preprocessing_max_retries_from_env(self, monkeypatch):
        """Reads PREPROCESSING_MAX_RETRIES from environment."""
        monkeypatch.setenv("PREPROCESSING_MAX_RETRIES", "5")
        config = load_config()
        assert config.preprocessing_max_retries == 5

    def test_preprocessing_max_retries_zero_allowed(self, monkeypatch):
        """PREPROCESSING_MAX_RETRIES=0 is valid (no retries)."""
        monkeypatch.setenv("PREPROCESSING_MAX_RETRIES", "0")
        config = load_config()
        assert config.preprocessing_max_retries == 0

    def test_preprocessing_max_retries_negative_raises(self, monkeypatch):
        """Negative PREPROCESSING_MAX_RETRIES raises ValueError."""
        monkeypatch.setenv("PREPROCESSING_MAX_RETRIES", "-1")
        with pytest.raises(ValueError, match="PREPROCESSING_MAX_RETRIES must be >= 0"):
            load_config()
