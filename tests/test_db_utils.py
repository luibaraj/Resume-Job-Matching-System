"""Tests for src/db_utils.py."""

import sqlite3

import pytest

from src.db_utils import add_column_if_missing


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database with jobs table."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT
        )
    """)
    conn.commit()
    yield cursor
    conn.close()


class TestAddColumnIfMissing:
    """Tests for add_column_if_missing function."""

    def test_add_valid_column(self, in_memory_db):
        """Test adding a valid column."""
        add_column_if_missing(in_memory_db, "jobs", "embedding", "BLOB")
        # Verify column exists by inserting a row with it
        in_memory_db.execute(
            "INSERT INTO jobs (title, embedding) VALUES (?, ?)",
            ("Test Job", b"test_data"),
        )
        in_memory_db.connection.commit()

    def test_add_column_already_exists(self, in_memory_db):
        """Test adding a column that already exists (should not error)."""
        # First add the column
        add_column_if_missing(in_memory_db, "jobs", "embedding", "BLOB")
        # Adding it again should not raise an error
        add_column_if_missing(in_memory_db, "jobs", "embedding", "BLOB")

    def test_add_multiple_valid_columns(self, in_memory_db):
        """Test adding multiple valid columns."""
        add_column_if_missing(in_memory_db, "jobs", "embedding", "BLOB")
        add_column_if_missing(in_memory_db, "jobs", "embedded", "INTEGER DEFAULT 0")
        add_column_if_missing(in_memory_db, "jobs", "cleaned_description", "TEXT")
        add_column_if_missing(in_memory_db, "jobs", "preprocessed", "INTEGER DEFAULT 0")
        # All should succeed without error

    def test_reject_invalid_table(self, in_memory_db):
        """Test that invalid table names are rejected."""
        with pytest.raises(ValueError, match="Table .* not in allowlist"):
            add_column_if_missing(in_memory_db, "users", "embedding", "BLOB")

    def test_reject_invalid_column(self, in_memory_db):
        """Test that invalid column names are rejected."""
        with pytest.raises(ValueError, match="Column .* not in allowlist"):
            add_column_if_missing(in_memory_db, "jobs", "malicious_col", "BLOB")

    def test_reject_invalid_col_type(self, in_memory_db):
        """Test that invalid column types are rejected."""
        with pytest.raises(ValueError, match="Column type .* not in allowlist"):
            add_column_if_missing(in_memory_db, "jobs", "embedding", "VARCHAR(255)")

    def test_reject_sql_injection_in_table(self, in_memory_db):
        """Test that SQL injection in table name is prevented."""
        with pytest.raises(ValueError, match="Table .* not in allowlist"):
            add_column_if_missing(in_memory_db, "jobs; DROP TABLE jobs;", "embedding", "BLOB")

    def test_reject_sql_injection_in_column(self, in_memory_db):
        """Test that SQL injection in column name is prevented."""
        with pytest.raises(ValueError, match="Column .* not in allowlist"):
            add_column_if_missing(in_memory_db, "jobs", "embedding; --", "BLOB")

    def test_reject_sql_injection_in_type(self, in_memory_db):
        """Test that SQL injection in column type is prevented."""
        with pytest.raises(ValueError, match="Column type .* not in allowlist"):
            add_column_if_missing(in_memory_db, "jobs", "embedding", "BLOB; DROP TABLE jobs;")
