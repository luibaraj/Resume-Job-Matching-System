"""Tests for src/db_utils.py — database utility functions."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db_utils import add_column_if_missing


class TestAddColumnIfMissing:
    """Test suite for add_column_if_missing function."""

    def test_add_valid_blob_column(self, tmp_db):
        """Test adding a BLOB column (e.g., embeddings)."""
        cursor = tmp_db.cursor()

        add_column_if_missing(cursor, "jobs", "embedding", "BLOB")
        tmp_db.commit()

        # Verify column exists and can store binary data
        cursor.execute("PRAGMA table_info(jobs)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        assert "embedding" in columns

    def test_add_valid_text_column(self, tmp_db):
        """Test adding a TEXT column (e.g., cleaned descriptions)."""
        cursor = tmp_db.cursor()

        add_column_if_missing(cursor, "jobs", "cleaned_description", "TEXT")
        tmp_db.commit()

        cursor.execute("PRAGMA table_info(jobs)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "cleaned_description" in columns

    def test_add_valid_integer_default_column(self, tmp_db):
        """Test adding INTEGER DEFAULT 0 column (e.g., preprocessed flag)."""
        cursor = tmp_db.cursor()

        add_column_if_missing(cursor, "jobs", "preprocessed", "INTEGER DEFAULT 0")
        tmp_db.commit()

        cursor.execute("PRAGMA table_info(jobs)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "preprocessed" in columns

    def test_add_column_idempotent_when_exists(self, tmp_db):
        """Test that adding an existing column doesn't raise an error."""
        cursor = tmp_db.cursor()

        # Add column first time
        add_column_if_missing(cursor, "jobs", "embedded", "INTEGER DEFAULT 0")
        tmp_db.commit()

        # Add same column again — should not raise OperationalError
        add_column_if_missing(cursor, "jobs", "embedded", "INTEGER DEFAULT 0")
        tmp_db.commit()

        cursor.execute("PRAGMA table_info(jobs)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "embedded" in columns

    def test_add_multiple_columns_sequential(self, tmp_db):
        """Test adding multiple distinct columns in sequence."""
        cursor = tmp_db.cursor()

        add_column_if_missing(cursor, "jobs", "embedding", "BLOB")
        add_column_if_missing(cursor, "jobs", "cleaned_description", "TEXT")
        add_column_if_missing(cursor, "jobs", "preprocessed", "INTEGER DEFAULT 0")
        add_column_if_missing(cursor, "jobs", "embedded", "INTEGER DEFAULT 0")
        tmp_db.commit()

        cursor.execute("PRAGMA table_info(jobs)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "embedding" in columns
        assert "cleaned_description" in columns
        assert "preprocessed" in columns
        assert "embedded" in columns

    def test_reject_invalid_table_name(self, tmp_db):
        """Test that invalid table names are rejected (SQL injection prevention)."""
        cursor = tmp_db.cursor()

        invalid_tables = [
            "users",  # Not in allowlist
            "jobs; DROP TABLE jobs;",  # SQL injection attempt
            "jobs' OR '1'='1",  # SQL injection attempt
        ]

        for invalid_table in invalid_tables:
            with pytest.raises(ValueError, match="Table.*not in allowlist"):
                add_column_if_missing(cursor, invalid_table, "embedding", "BLOB")

    def test_reject_invalid_column_name(self, tmp_db):
        """Test that invalid column names are rejected (SQL injection prevention)."""
        cursor = tmp_db.cursor()

        invalid_columns = [
            "bad_column",  # Not in allowlist
            "embedding; DROP TABLE",  # SQL injection attempt
            "embedded' OR '1'='1",  # SQL injection attempt
        ]

        for invalid_column in invalid_columns:
            with pytest.raises(ValueError, match="Column.*not in allowlist"):
                add_column_if_missing(cursor, "jobs", invalid_column, "BLOB")

    def test_reject_invalid_col_type(self, tmp_db):
        """Test that invalid column types are rejected (SQL injection prevention)."""
        cursor = tmp_db.cursor()

        invalid_types = [
            "VARCHAR(100)",  # Not in allowlist
            "BLOB; DROP TABLE",  # SQL injection attempt
            "TEXT' OR '1'='1",  # SQL injection attempt
        ]

        for invalid_type in invalid_types:
            with pytest.raises(ValueError, match="Column type.*not in allowlist"):
                add_column_if_missing(cursor, "jobs", "embedding", invalid_type)

    def test_allowlist_validation_enforced(self, tmp_db):
        """Test that all three allowlist checks are properly enforced."""
        cursor = tmp_db.cursor()

        # Invalid table check
        with pytest.raises(ValueError, match="Table.*not in allowlist"):
            add_column_if_missing(cursor, "invalid_table", "embedding", "BLOB")

        # Invalid column check
        with pytest.raises(ValueError, match="Column.*not in allowlist"):
            add_column_if_missing(cursor, "jobs", "invalid_column", "BLOB")

        # Invalid type check
        with pytest.raises(ValueError, match="Column type.*not in allowlist"):
            add_column_if_missing(cursor, "jobs", "embedding", "INVALID_TYPE")

    def test_embed_flow_with_data(self, tmp_db):
        """Test real flow: add column, insert data, verify storage."""
        cursor = tmp_db.cursor()

        # Add embedding column
        add_column_if_missing(cursor, "jobs", "embedding", "BLOB")
        tmp_db.commit()

        # Insert job with embedding
        import numpy as np
        embedding = np.ones(1024, dtype=np.float32).tobytes()
        cursor.execute(
            "INSERT INTO jobs (title, description, url, board_token, embedding) VALUES (?, ?, ?, ?, ?)",
            ("Test Job", "Test Desc", "http://test.com", "token", embedding),
        )
        tmp_db.commit()

        # Retrieve and verify
        cursor.execute("SELECT embedding FROM jobs WHERE title = ?", ("Test Job",))
        row = cursor.fetchone()
        assert row is not None
        assert isinstance(row[0], bytes)

    def test_preprocess_flow_with_data(self, tmp_db):
        """Test real flow: add preprocessed columns, update them."""
        cursor = tmp_db.cursor()

        # Add preprocessed columns
        add_column_if_missing(cursor, "jobs", "cleaned_description", "TEXT")
        add_column_if_missing(cursor, "jobs", "preprocessed", "INTEGER DEFAULT 0")
        tmp_db.commit()

        # Insert job
        cursor.execute(
            "INSERT INTO jobs (title, description, url, board_token) VALUES (?, ?, ?, ?)",
            ("Test Job", "Raw HTML", "http://test.com", "token"),
        )
        tmp_db.commit()

        # Update preprocessed columns
        cursor.execute(
            "UPDATE jobs SET cleaned_description = ?, preprocessed = 1 WHERE title = ?",
            ("Clean text", "Test Job"),
        )
        tmp_db.commit()

        # Verify update
        cursor.execute(
            "SELECT cleaned_description, preprocessed FROM jobs WHERE title = ?",
            ("Test Job",),
        )
        row = cursor.fetchone()
        assert row[0] == "Clean text"
        assert row[1] == 1
