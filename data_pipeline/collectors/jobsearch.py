"""
Job Search API (job-search15) collector with adaptive pagination.
"""
import json
import logging
import os
from typing import Any, Dict, List

from data_pipeline.collectors.base import BaseCollector

logger = logging.getLogger(__name__)


class JobSearchCollector(BaseCollector):
    """
    Job Search API collector via RapidAPI.
    Uses shared request pool with adaptive pagination.
    """

    API_HOST = "job-search15.p.rapidapi.com"
    API_URL = "https://job-search15.p.rapidapi.com/"

    def __init__(self, run_budget: int = 50):
        """
        Initialize Job Search collector.

        Args:
            run_budget: Request budget for this run (default 50/month)
        """
        super().__init__("jobsearch", run_budget)
        self.api_key = os.getenv("X_RAPID_API")

        if not self.api_key:
            logger.warning("X_RAPID_API not set, Job Search will not be available")

    def fetch_page(self, query: str, page: int) -> List[Dict[str, Any]]:
        """
        Fetch a page of results from Job Search API.

        Args:
            query: Search query (e.g., "software engineer")
            page: Page number (1-indexed)

        Returns:
            List of job dicts
        """
        if not self.api_key:
            return []

        payload = {
            "api_type": "fetch_jobs",
            "search_terms": query,
            "location": "United States",
            "page": str(page),
        }

        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.API_HOST,
            "Content-Type": "application/json",
        }

        try:
            response = self.session.post(
                self.API_URL, json=payload, headers=headers, timeout=10
            )
            response.raise_for_status()
            data = response.json()

            jobs = []
            # Response can be a dict with "data" key or a direct list
            items = data.get("data", []) if isinstance(data, dict) else data if isinstance(data, list) else []
            for item in items:
                jobs.append(self._normalize_job(item))

            return jobs
        except Exception as e:
            logger.error(f"Error fetching Job Search page {page} for '{query}': {e}")
            return []

    def _normalize_job(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Job Search job to schema.

        Args:
            item: Raw Job Search job dict

        Returns:
            Normalized job dict
        """
        return {
            "external_id": item.get("job_id"),
            "source_system": "jobsearch",
            "source_board": None,
            "title": item.get("job_title"),
            "location": item.get("job_location"),
            "description": item.get("job_description"),
            "company_name": item.get("company_name"),
            "employment_type": item.get("job_employment_type"),
            "posted_date": item.get("job_posted_at_datetime_utc"),
            "source_url": None,
            "apply_url": item.get("job_apply_link"),
            "raw_data": json.dumps(item),
            "metadata": json.dumps({}),
            "scraped_at": None,
        }
