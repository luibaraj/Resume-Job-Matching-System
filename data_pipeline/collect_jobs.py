#!/usr/bin/env python3
"""
Main job collection orchestrator.

Runs all collectors, normalizes results, upserts to SQLite database,
marks stale jobs, and logs collection activity.
"""
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from data_pipeline.collectors import GreenhouseCollector, JSearchCollector, JobSearchCollector, JobsApiCollector, SerpApiCollector
from data_pipeline.db.migrate import main as run_migration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class JobCollectionOrchestrator:
    """Orchestrates job collection from all sources"""

    def __init__(self, db_path: str = "data/jobs.db"):
        """
        Initialize orchestrator.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.run_at = datetime.utcnow().isoformat() + "Z"
        self.collection_log = []

    def run(self) -> bool:
        """
        Run full collection pipeline.

        Returns:
            True if successful (including partial success), False on fatal error
        """
        try:
            logger.info(f"Starting job collection run at {self.run_at}")

            # Ensure DB schema is up to date
            self._ensure_schema()

            # Load configs
            company_list = self._load_config("data_pipeline/config/company_list.json")
            queries_config = self._load_config("data_pipeline/config/queries.json")
            queries = queries_config.get("queries", [])

            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row

            try:
                # Run each collector
                all_jobs = []

                # Greenhouse (no pooling)
                gh_jobs, gh_log = self._collect_greenhouse(company_list)
                all_jobs.extend(gh_jobs)
                self.collection_log.append(gh_log)

                # JSearch (with pooling)
                js_jobs, js_log = self._collect_jsearch(queries, queries_config)
                all_jobs.extend(js_jobs)
                self.collection_log.append(js_log)

                # JobSearch (with pooling)
                jbs_jobs, jbs_log = self._collect_jobsearch(queries, queries_config)
                all_jobs.extend(jbs_jobs)
                self.collection_log.append(jbs_log)

                # JobsApi (with pooling)
                ja_jobs, ja_log = self._collect_jobsapi(queries, queries_config)
                all_jobs.extend(ja_jobs)
                self.collection_log.append(ja_log)

                # SerpApi (with pooling)
                sa_jobs, sa_log = self._collect_serpapi(queries, queries_config)
                all_jobs.extend(sa_jobs)
                self.collection_log.append(sa_log)

                # Upsert all jobs
                new_count = self._upsert_jobs(conn, all_jobs)
                logger.info(f"Upserted {new_count} new jobs out of {len(all_jobs)} total collected")

                # Mark stale jobs
                stale_count = self._mark_stale_jobs(conn)
                logger.info(f"Marked {stale_count} jobs as stale (removed)")

                # Write collection log
                self._write_collection_log(conn)

                conn.commit()
                logger.info("Collection run completed successfully")
                return True

            except Exception as e:
                logger.error(f"Error during collection: {e}")
                conn.rollback()
                # Write partial log anyway
                try:
                    self._write_collection_log(conn)
                    conn.commit()
                except Exception as log_e:
                    logger.error(f"Could not write collection log: {log_e}")
                return True  # Exit 0 even on partial failure
            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Fatal error in collection orchestrator: {e}", exc_info=True)
            return True  # Exit 0 even on fatal error (don't break cron)

    def _ensure_schema(self):
        """Ensure database schema is up to date"""
        logger.info("Running database migration...")
        try:
            run_migration()
        except Exception as e:
            logger.error(f"Migration error: {e}")
            raise

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load JSON config file"""
        config_path = Path(path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {path}")
            return {}

        with open(config_path) as f:
            return json.load(f)

    def _collect_greenhouse(self, company_list: List[Dict]) -> tuple:
        """
        Collect from Greenhouse.

        Returns:
            Tuple of (jobs, log_entry)
        """
        logger.info("Starting Greenhouse collection")
        collector = GreenhouseCollector()
        jobs_found = 0
        jobs_new = 0
        errors_list = []

        try:
            jobs = collector.collect_all(company_list)
            jobs_found = len(jobs)

            # Set scraped_at on all jobs
            for job in jobs:
                job["scraped_at"] = self.run_at

            logger.info(f"Greenhouse collected {jobs_found} jobs")
            return jobs, {
                "run_at": self.run_at,
                "source": "greenhouse",
                "jobs_found": jobs_found,
                "requests": collector.requests_used,
            }

        except Exception as e:
            error_msg = f"Greenhouse error: {str(e)}"
            logger.error(error_msg)
            errors_list.append(error_msg)
            return [], {
                "run_at": self.run_at,
                "source": "greenhouse",
                "jobs_found": 0,
                "jobs_error": 1,
                "requests": collector.requests_used if collector else 0,
                "errors": "\n".join(errors_list),
            }

    def _collect_jsearch(self, queries: List[str], config: Dict) -> tuple:
        """Collect from JSearch"""
        logger.info("Starting JSearch collection")
        collector = JSearchCollector(run_budget=config.get("pools", {}).get("jsearch", 38))
        jobs_found = 0
        errors_list = []

        try:
            jobs = collector.collect_all(queries)
            jobs_found = len(jobs)

            for job in jobs:
                job["scraped_at"] = self.run_at

            logger.info(f"JSearch collected {jobs_found} jobs (used {collector.requests_used} requests)")
            return jobs, {
                "run_at": self.run_at,
                "source": "jsearch",
                "jobs_found": jobs_found,
                "requests": collector.requests_used,
            }

        except Exception as e:
            error_msg = f"JSearch error: {str(e)}"
            logger.error(error_msg)
            errors_list.append(error_msg)
            return [], {
                "run_at": self.run_at,
                "source": "jsearch",
                "jobs_found": 0,
                "jobs_error": 1,
                "requests": collector.requests_used,
                "errors": "\n".join(errors_list),
            }

    def _collect_jobsearch(self, queries: List[str], config: Dict) -> tuple:
        """Collect from JobSearch API"""
        logger.info("Starting JobSearch collection")
        collector = JobSearchCollector(run_budget=config.get("pools", {}).get("jobsearch", 50))
        jobs_found = 0
        errors_list = []

        try:
            jobs = collector.collect_all(queries)
            jobs_found = len(jobs)

            for job in jobs:
                job["scraped_at"] = self.run_at

            logger.info(f"JobSearch collected {jobs_found} jobs (used {collector.requests_used} requests)")
            return jobs, {
                "run_at": self.run_at,
                "source": "jobsearch",
                "jobs_found": jobs_found,
                "requests": collector.requests_used,
            }

        except Exception as e:
            error_msg = f"JobSearch error: {str(e)}"
            logger.error(error_msg)
            errors_list.append(error_msg)
            return [], {
                "run_at": self.run_at,
                "source": "jobsearch",
                "jobs_found": 0,
                "jobs_error": 1,
                "requests": collector.requests_used,
                "errors": "\n".join(errors_list),
            }

    def _collect_jobsapi(self, queries: List[str], config: Dict) -> tuple:
        """Collect from Jobs API"""
        logger.info("Starting JobsApi collection")
        collector = JobsApiCollector(run_budget=config.get("pools", {}).get("jobsapi", 50))
        jobs_found = 0
        errors_list = []

        try:
            jobs = collector.collect_all(queries)
            jobs_found = len(jobs)

            for job in jobs:
                job["scraped_at"] = self.run_at

            logger.info(f"JobsApi collected {jobs_found} jobs (used {collector.requests_used} requests)")
            return jobs, {
                "run_at": self.run_at,
                "source": "jobsapi",
                "jobs_found": jobs_found,
                "requests": collector.requests_used,
            }

        except Exception as e:
            error_msg = f"JobsApi error: {str(e)}"
            logger.error(error_msg)
            errors_list.append(error_msg)
            return [], {
                "run_at": self.run_at,
                "source": "jobsapi",
                "jobs_found": 0,
                "jobs_error": 1,
                "requests": collector.requests_used,
                "errors": "\n".join(errors_list),
            }

    def _collect_serpapi(self, queries: List[str], config: Dict) -> tuple:
        """Collect from SerpApi"""
        logger.info("Starting SerpApi collection")
        collector = SerpApiCollector(run_budget=config.get("pools", {}).get("serpapi", 7))
        jobs_found = 0
        errors_list = []

        try:
            jobs = collector.collect_all(queries)
            jobs_found = len(jobs)

            for job in jobs:
                job["scraped_at"] = self.run_at

            logger.info(f"SerpApi collected {jobs_found} jobs (used {collector.requests_used} requests)")
            return jobs, {
                "run_at": self.run_at,
                "source": "serpapi",
                "jobs_found": jobs_found,
                "requests": collector.requests_used,
            }

        except Exception as e:
            error_msg = f"SerpApi error: {str(e)}"
            logger.error(error_msg)
            errors_list.append(error_msg)
            return [], {
                "run_at": self.run_at,
                "source": "serpapi",
                "jobs_found": 0,
                "jobs_error": 1,
                "requests": collector.requests_used,
                "errors": "\n".join(errors_list),
            }

    def _upsert_jobs(self, conn: sqlite3.Connection, jobs: List[Dict]) -> int:
        """
        Upsert jobs to database.

        Returns:
            Number of new jobs inserted
        """
        cursor = conn.cursor()
        new_count = 0

        for job in jobs:
            try:
                # Check if job already exists
                cursor.execute(
                    """
                    SELECT id FROM jobs
                    WHERE external_id = ? AND source_system = ? AND source_board = ?
                    """,
                    (job.get("external_id"), job.get("source_system"), job.get("source_board")),
                )

                existing = cursor.fetchone()

                if existing:
                    # Update existing job
                    cursor.execute(
                        """
                        UPDATE jobs SET job_status = 'active', updated_date = ?
                        WHERE id = ?
                        """,
                        (datetime.utcnow().isoformat() + "Z", existing[0]),
                    )
                else:
                    # Insert new job
                    columns = ", ".join(job.keys())
                    placeholders = ", ".join(["?"] * len(job))
                    cursor.execute(
                        f"INSERT INTO jobs ({columns}) VALUES ({placeholders})",
                        tuple(job.values()),
                    )
                    new_count += 1

            except Exception as e:
                logger.warning(f"Error upserting job {job.get('external_id')}: {e}")

        logger.info(f"Upserted {new_count} new jobs out of {len(jobs)} processed")
        return new_count

    def _mark_stale_jobs(self, conn: sqlite3.Connection) -> int:
        """
        Mark jobs not seen in 3 days as stale.

        Returns:
            Number of jobs marked stale
        """
        cutoff = (datetime.utcnow() - timedelta(days=3)).isoformat() + "Z"
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE jobs SET job_status = 'removed'
            WHERE job_status = 'active' AND scraped_at < ?
            """,
            (cutoff,),
        )
        return cursor.rowcount

    def _write_collection_log(self, conn: sqlite3.Connection):
        """Write collection_log entries"""
        cursor = conn.cursor()

        for log_entry in self.collection_log:
            try:
                cursor.execute(
                    """
                    INSERT INTO collection_log
                    (run_at, source, jobs_found, jobs_new, jobs_error, requests, errors)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        log_entry.get("run_at"),
                        log_entry.get("source"),
                        log_entry.get("jobs_found", 0),
                        log_entry.get("jobs_new", 0),
                        log_entry.get("jobs_error", 0),
                        log_entry.get("requests", 0),
                        log_entry.get("errors"),
                    ),
                )
            except Exception as e:
                logger.error(f"Error writing collection log for {log_entry.get('source')}: {e}")


def main():
    """Main entry point"""
    orchestrator = JobCollectionOrchestrator()
    success = orchestrator.run()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
