"""Tests for src/utils.py utility functions."""

import hashlib
import json
import logging
import sys

import pytest

from src.utils import (
    compute_job_hash,
    deserialize_list,
    serialize_list,
    setup_logging,
)


class TestComputeJobHash:
    """Tests for compute_job_hash function."""

    def test_returns_64_char_hex_string(self):
        """Result is a 64-character hex string."""
        result = compute_job_hash(1, "acme", "Engineer")
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_known_value(self):
        """Pre-computed hash value for known inputs."""
        # Computed externally: SHA256("{4001234:example-company:software engineer, backend}".lower())
        result = compute_job_hash(4001234, "example-company", "Software Engineer, Backend")
        expected = "feb4426c645d3e52cdbd32f4010c0cd5c237d016eaedb13d03ba9f5f405ad2f1"
        assert result == expected

    def test_is_deterministic(self):
        """Calling twice with same args returns same value."""
        result1 = compute_job_hash(1, "acme", "Engineer")
        result2 = compute_job_hash(1, "acme", "Engineer")
        assert result1 == result2

    def test_case_insensitive_board_token(self):
        """Board token is lowercased before hashing."""
        result1 = compute_job_hash(1, "ACME", "Engineer")
        result2 = compute_job_hash(1, "acme", "Engineer")
        assert result1 == result2

    def test_case_insensitive_title(self):
        """Title is lowercased before hashing."""
        result1 = compute_job_hash(1, "acme", "ENGINEER")
        result2 = compute_job_hash(1, "acme", "engineer")
        assert result1 == result2

    def test_strips_whitespace_from_board_token(self):
        """Leading/trailing whitespace on board token is stripped."""
        result1 = compute_job_hash(1, "  acme  ", "title")
        result2 = compute_job_hash(1, "acme", "title")
        assert result1 == result2

    def test_strips_whitespace_from_title(self):
        """Leading/trailing whitespace on title is stripped."""
        result1 = compute_job_hash(1, "acme", "  title  ")
        result2 = compute_job_hash(1, "acme", "title")
        assert result1 == result2

    def test_different_ids_produce_different_hashes(self):
        """Different greenhouse_id produces different hash."""
        result1 = compute_job_hash(1, "acme", "x")
        result2 = compute_job_hash(2, "acme", "x")
        assert result1 != result2

    def test_different_tokens_produce_different_hashes(self):
        """Different board_token produces different hash."""
        result1 = compute_job_hash(1, "acme", "x")
        result2 = compute_job_hash(1, "beta", "x")
        assert result1 != result2

    def test_different_titles_produce_different_hashes(self):
        """Different title produces different hash."""
        result1 = compute_job_hash(1, "acme", "x")
        result2 = compute_job_hash(1, "acme", "y")
        assert result1 != result2


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_returns_logger_instance(self):
        """setup_logging returns a Logger instance."""
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)

    def test_named_logger(self):
        """Logger name matches the provided name."""
        logger = setup_logging(name="myapp")
        assert logger.name == "myapp"

    def test_default_name_is_root(self):
        """Logger with no name uses root logger."""
        logger = setup_logging()
        assert logger.name == "root"

    def test_log_level_debug(self):
        """Log level can be set to DEBUG."""
        logger = setup_logging(log_level="DEBUG")
        assert logger.getEffectiveLevel() == logging.DEBUG

    def test_log_level_warning(self):
        """Log level can be set to WARNING."""
        logger = setup_logging(log_level="WARNING")
        assert logger.getEffectiveLevel() == logging.WARNING

    def test_has_exactly_one_handler(self):
        """Logger has exactly one handler."""
        logger = setup_logging()
        assert len(logger.handlers) == 1

    def test_handler_is_stream_handler(self):
        """Handler is a StreamHandler."""
        logger = setup_logging()
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_handler_writes_to_stdout(self):
        """StreamHandler writes to stdout."""
        logger = setup_logging()
        assert logger.handlers[0].stream is sys.stdout

    def test_formatter_has_iso8601_datefmt(self):
        """Formatter uses ISO-8601 date format."""
        logger = setup_logging()
        formatter = logger.handlers[0].formatter
        assert formatter is not None
        assert formatter.datefmt == "%Y-%m-%dT%H:%M:%S"


class TestSerializeList:
    """Tests for serialize_list function."""

    def test_empty_list_returns_empty_json_array(self):
        """Empty list serializes to '[]'."""
        result = serialize_list([])
        assert result == "[]"

    def test_single_item(self):
        """Single item list serializes correctly."""
        result = serialize_list(["Engineering"])
        assert result == '["Engineering"]'

    def test_multiple_items(self):
        """Multiple items serialize correctly."""
        result = serialize_list(["a", "b", "c"])
        # Verify it's valid JSON and round-trips
        assert json.loads(result) == ["a", "b", "c"]

    def test_output_is_valid_json(self):
        """Output is valid JSON that json.loads can parse."""
        result = serialize_list(["x"])
        parsed = json.loads(result)
        assert parsed == ["x"]

    def test_items_with_special_characters(self):
        """Items with special characters serialize correctly."""
        items = ["San Francisco", "R&D"]
        result = serialize_list(items)
        assert json.loads(result) == items


class TestDeserializeList:
    """Tests for deserialize_list function."""

    def test_valid_json_string(self):
        """Valid JSON string deserializes correctly."""
        result = deserialize_list('["a","b"]')
        assert result == ["a", "b"]

    def test_empty_json_array(self):
        """Empty JSON array deserializes to empty list."""
        result = deserialize_list("[]")
        assert result == []

    def test_none_returns_empty_list(self):
        """None input returns empty list."""
        result = deserialize_list(None)
        assert result == []

    def test_empty_string_returns_empty_list(self):
        """Empty string returns empty list."""
        result = deserialize_list("")
        assert result == []

    def test_malformed_json_returns_empty_list(self):
        """Malformed JSON returns empty list (no exception)."""
        result = deserialize_list("not-json")
        assert result == []

    def test_partial_json_returns_empty_list(self):
        """Incomplete JSON returns empty list."""
        result = deserialize_list('["unclosed')
        assert result == []

    def test_round_trip_with_serialize(self):
        """Data round-trips through serialize and deserialize."""
        original = ["Eng", "Sales"]
        serialized = serialize_list(original)
        deserialized = deserialize_list(serialized)
        assert deserialized == original
