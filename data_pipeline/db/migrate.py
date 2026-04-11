#!/usr/bin/env python3
"""
Idempotent migration script for jobs database.
Migrates from old schema to new multi-source schema.
"""
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_db_path():
    """Get database path, defaulting to data/jobs.db"""
    return Path(__file__).parent.parent.parent / "data" / "jobs.db"


def column_exists(conn, table, column):
    """Check if a column exists in a table"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    columns = {row[1] for row in cursor.fetchall()}
    return column in columns


def migrate_jobs_table(conn):
    """Migrate jobs table from old schema to new schema"""
    logger.info("Migrating jobs table...")

    # Check if table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
    if not cursor.fetchone():
        logger.info("jobs table does not exist, creating from schema.sql")
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path) as f:
            conn.executescript(f.read())
        return

    # Check if already migrated (source_system column exists)
    if column_exists(conn, 'jobs', 'source_system'):
        logger.info("jobs table already migrated, skipping")
        return

    logger.info("Performing in-place migration...")

    # Rename old table
    conn.execute("ALTER TABLE jobs RENAME TO jobs_old")

    # Create new table with new schema
    schema_path = Path(__file__).parent / "schema.sql"
    with open(schema_path) as f:
        schema = f.read()

    # Extract just the jobs table creation
    jobs_create = [line for line in schema.split('\n') if 'CREATE TABLE IF NOT EXISTS jobs' in line]
    for statement in schema.split(';'):
        if 'CREATE TABLE IF NOT EXISTS jobs' in statement:
            conn.execute(statement + ';')
            break

    # Migrate data from old to new schema
    # Map old columns to new columns
    try:
        conn.execute("""
            INSERT INTO jobs (
                external_id, source_system, source_board, title, location,
                description, company_name, department, source_url,
                scraped_at, updated_date, raw_data
            )
            SELECT
                external_id,
                COALESCE(source, 'greenhouse') as source_system,
                board_token as source_board,
                title,
                location,
                description,
                company_name,
                department,
                source_url,
                scraped_at,
                COALESCE(updated_at, created_at) as updated_date,
                NULL as raw_data
            FROM jobs_old
        """)
        logger.info("Data migrated successfully")
    except Exception as e:
        logger.error(f"Error during data migration: {e}")
        # Rollback
        conn.execute("DROP TABLE jobs")
        conn.execute("ALTER TABLE jobs_old RENAME TO jobs")
        raise

    # Drop old table
    conn.execute("DROP TABLE jobs_old")
    logger.info("Old table dropped")


def create_helper_tables(conn):
    """Create request_budget and collection_log tables if they don't exist"""
    cursor = conn.cursor()

    # request_budget
    if not cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='request_budget'"
    ).fetchone():
        logger.info("Creating request_budget table...")
        conn.execute("""
            CREATE TABLE request_budget (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source      TEXT NOT NULL,
                date        TEXT NOT NULL,
                requests    INTEGER DEFAULT 0,
                UNIQUE(source, date)
            )
        """)

    # collection_log
    if not cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='collection_log'"
    ).fetchone():
        logger.info("Creating collection_log table...")
        conn.execute("""
            CREATE TABLE collection_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at      TEXT NOT NULL,
                source      TEXT NOT NULL,
                jobs_found  INTEGER DEFAULT 0,
                jobs_new    INTEGER DEFAULT 0,
                jobs_error  INTEGER DEFAULT 0,
                requests    INTEGER DEFAULT 0,
                errors      TEXT
            )
        """)


def create_indexes(conn):
    """Create all indexes"""
    logger.info("Creating indexes...")
    indexes = [
        ("idx_jobs_source", "CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source_system)"),
        ("idx_jobs_title", "CREATE INDEX IF NOT EXISTS idx_jobs_title ON jobs(title)"),
        ("idx_jobs_company", "CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company_name)"),
        ("idx_jobs_posted", "CREATE INDEX IF NOT EXISTS idx_jobs_posted ON jobs(posted_date)"),
        ("idx_jobs_scraped", "CREATE INDEX IF NOT EXISTS idx_jobs_scraped ON jobs(scraped_at)"),
        ("idx_jobs_status", "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(job_status)"),
        ("idx_jobs_location", "CREATE INDEX IF NOT EXISTS idx_jobs_location ON jobs(location)"),
    ]

    for name, statement in indexes:
        try:
            conn.execute(statement)
            logger.info(f"Created index {name}")
        except Exception as e:
            logger.warning(f"Index {name} may already exist: {e}")


def main():
    """Run migration"""
    db_path = get_db_path()
    logger.info(f"Connecting to {db_path}")

    if not db_path.exists():
        logger.info(f"Database does not exist at {db_path}, will be created")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        migrate_jobs_table(conn)
        create_helper_tables(conn)
        create_indexes(conn)
        conn.commit()
        logger.info("Migration completed successfully")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
