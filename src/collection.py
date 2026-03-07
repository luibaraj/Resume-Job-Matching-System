import concurrent.futures
import time
from datetime import datetime
from typing import Optional

import requests

from src.config import load_config
from src.database import DatabaseManager
from src.utils import setup_logging


class GreenhouseClient:
    """Client for the Greenhouse Job Board API."""

    BASE_URL = "https://boards-api.greenhouse.io/v1/boards"

    def __init__(
        self,
        board_tokens: list[str],
        request_timeout: int = 30,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        max_workers: int = 8,
        request_delay_seconds: float = 0.1,
    ):
        """Initialize the Greenhouse API client.

        Args:
            board_tokens: List of Greenhouse board tokens (company slugs)
            request_timeout: HTTP request timeout in seconds
            max_retries: Maximum number of retry attempts for failed requests
            retry_backoff: Backoff multiplier between retries
            max_workers: Maximum number of concurrent worker threads
            request_delay_seconds: Delay in seconds between board fetch submissions
        """
        self.board_tokens = board_tokens
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.max_workers = max_workers
        self.request_delay_seconds = request_delay_seconds
        self.logger = setup_logging(name="GreenhouseClient")

    def fetch_jobs(self, board_token: str) -> list[dict]:
        """Fetch all active jobs for a single board token.

        Args:
            board_token: Greenhouse board token (company slug)

        Returns:
            List of raw job dictionaries from Greenhouse API

        Raises:
            requests.HTTPError: On non-2xx responses after retries exhausted
        """
        url = f"{self.BASE_URL}/{board_token}/jobs"
        params = {"content": "true"}

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=self.request_timeout,
                )
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait_time = float(retry_after) if retry_after is not None else self.retry_backoff ** attempt
                    self.logger.warning(
                        f"Rate limited (429) on board {board_token}, "
                        f"waiting {wait_time}s (attempt {attempt + 1})"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    # Last attempt: fall through so raise_for_status() raises HTTPError
                response.raise_for_status()
                data = response.json()
                jobs = data.get("jobs", [])
                self.logger.info(f"Fetched {len(jobs)} jobs from board {board_token}")
                return jobs
            except requests.RequestException as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_backoff ** attempt
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed for {board_token}, "
                        f"retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(
                        f"All {self.max_retries} attempts failed for {board_token}: {e}"
                    )

        if last_exception:
            raise last_exception
        return []

    def fetch_all_boards(self) -> list[dict]:
        """Iterate all configured board tokens and aggregate job dicts.

        Fetches boards in parallel using ThreadPoolExecutor.
        Injects a 'board_token' key into each job dict.

        Returns:
            Flat list of raw job dicts from all boards
        """
        if not self.board_tokens:
            return []

        all_jobs = []

        def fetch_one_board(token: str) -> list[dict]:
            """Fetch jobs for a single board and inject board_token."""
            jobs = self.fetch_jobs(token)
            for job in jobs:
                job["board_token"] = token
            return jobs

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.max_workers, max(1, len(self.board_tokens)))
        ) as executor:
            futures = {}
            for i, token in enumerate(self.board_tokens):
                if i > 0 and self.request_delay_seconds > 0:
                    time.sleep(self.request_delay_seconds)
                futures[executor.submit(fetch_one_board, token)] = token
            for future in concurrent.futures.as_completed(futures):
                token = futures[future]
                try:
                    jobs = future.result()
                    all_jobs.extend(jobs)
                except requests.RequestException as e:
                    self.logger.error(f"Failed to fetch jobs from {token}: {e}")

        return all_jobs


def normalize_job_for_db(raw_job: dict) -> dict:
    """Transform a raw Greenhouse API job into database format.

    Args:
        raw_job: Raw job dict from Greenhouse API

    Returns:
        Dict with keys matching DatabaseManager.insert_job() expectations
    """
    from src.utils import serialize_list

    # Extract location name, handle nested structure
    location = None
    if isinstance(raw_job.get("location"), dict):
        location = raw_job["location"].get("name")
    elif isinstance(raw_job.get("location"), str):
        location = raw_job["location"]

    # Serialize departments and offices as JSON lists
    departments = raw_job.get("departments", [])
    department_names = [d.get("name") for d in departments if isinstance(d, dict)]

    offices = raw_job.get("offices", [])
    office_names = [o.get("name") for o in offices if isinstance(o, dict)]

    return {
        "greenhouse_id": raw_job["id"],
        "board_token": raw_job.get("board_token", "unknown"),
        "title": raw_job.get("title", ""),
        "company": raw_job.get("board_token", "unknown"),
        "location": location,
        "raw_description": raw_job.get("content"),
        "absolute_url": raw_job.get("absolute_url"),
        "updated_at_source": raw_job.get("updated_at"),
        "departments": serialize_list(department_names),
        "offices": serialize_list(office_names),
        "collected_at": datetime.utcnow().isoformat() + "Z",
    }


def collect_jobs(
    client: GreenhouseClient,
    db: DatabaseManager,
    run_id: int,
) -> tuple[int, int]:
    """Main entry point for the collection step.

    Fetches jobs, normalizes them, and inserts in a single batch transaction.

    Args:
        client: GreenhouseClient instance
        db: DatabaseManager instance
        run_id: Pipeline run ID for audit logging

    Returns:
        Tuple of (inserted_count, skipped_count)
    """
    logger = setup_logging(name="collect_jobs")
    all_jobs = client.fetch_all_boards()
    logger.info(f"Total jobs fetched from all boards: {len(all_jobs)}")

    if not all_jobs:
        return 0, 0

    normalized_jobs = [normalize_job_for_db(raw_job) for raw_job in all_jobs]
    inserted, skipped = db.insert_jobs_batch(normalized_jobs)

    logger.info(f"Collection complete: {inserted} inserted, {skipped} skipped")
    return inserted, skipped


def main() -> None:
    """CLI entrypoint called by Docker container.

    Reads config, initializes DB, runs collection, updates pipeline_runs.
    Exits with code 0 on success, 1 on failure.
    """
    logger = setup_logging(name="collection_main")
    try:
        config = load_config()
        logger.setLevel(config.log_level)

        db = DatabaseManager(config.db_path)
        db.initialize_schema()

        from datetime import datetime

        run_date = datetime.utcnow().strftime("%Y-%m-%d")
        run_id = db.create_pipeline_run(run_date, "collection")

        client = GreenhouseClient(
            board_tokens=config.greenhouse_board_tokens,
            request_timeout=config.request_timeout_seconds,
            max_retries=config.max_retries,
            retry_backoff=config.retry_backoff_seconds,
            max_workers=config.max_workers,
            request_delay_seconds=config.request_delay_seconds,
        )

        inserted, skipped = collect_jobs(client, db, run_id)
        db.finish_pipeline_run(
            run_id, "success", jobs_processed=inserted, jobs_skipped=skipped
        )
        logger.info("Collection step completed successfully")

    except Exception as e:
        logger.exception("Collection step failed")
        try:
            from datetime import datetime

            config = load_config()
            db = DatabaseManager(config.db_path)
            run_date = datetime.utcnow().strftime("%Y-%m-%d")
            run_id = db.create_pipeline_run(run_date, "collection")
            db.finish_pipeline_run(run_id, "failed", 0, 0, str(e))
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
