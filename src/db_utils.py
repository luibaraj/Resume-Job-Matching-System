"""Database utility functions."""

import sqlite3


def add_column_if_missing(cursor: sqlite3.Cursor, table: str, column: str, col_type: str) -> None:
    """Add a column to a table if it doesn't already exist."""
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except sqlite3.OperationalError:
        # Column already exists
        pass
