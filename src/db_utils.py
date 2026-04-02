"""Database utility functions."""

import sqlite3

# Allowlist for table names to prevent SQL injection
_ALLOWED_TABLES = {"jobs"}

# Allowlist for column names to prevent SQL injection
_ALLOWED_COLUMNS = {
    "embedding",
    "embedded",
    "cleaned_description",
    "preprocessed",
}

# Allowlist for column type definitions to prevent SQL injection
_ALLOWED_COL_TYPES = {
    "BLOB",
    "TEXT",
    "INTEGER DEFAULT 0",
}


def add_column_if_missing(cursor: sqlite3.Cursor, table: str, column: str, col_type: str) -> None:
    """
    Add a column to a table if it doesn't already exist.

    Validates table, column, and col_type against allowlists to prevent SQL injection.

    Args:
        cursor: SQLite cursor
        table: Table name (allowlist: jobs)
        column: Column name (allowlist: embedding, embedded, cleaned_description, preprocessed)
        col_type: Column type definition (allowlist: BLOB, TEXT, INTEGER DEFAULT 0)

    Raises:
        ValueError: If table, column, or col_type is not in the allowlist
    """
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Table {table!r} not in allowlist {_ALLOWED_TABLES}")
    if column not in _ALLOWED_COLUMNS:
        raise ValueError(f"Column {column!r} not in allowlist {_ALLOWED_COLUMNS}")
    if col_type not in _ALLOWED_COL_TYPES:
        raise ValueError(f"Column type {col_type!r} not in allowlist {_ALLOWED_COL_TYPES}")

    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except sqlite3.OperationalError:
        # Column already exists; safe to ignore
        pass
