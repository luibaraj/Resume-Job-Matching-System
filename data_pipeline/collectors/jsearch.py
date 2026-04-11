"""
JSearch (RapidAPI) job collector with adaptive pagination.
"""
import json
import logging
import os
from typing import Any, Dict, List

from data_pipeline.collectors.base import BaseCollector

logger = logging.getLogger(__name__)


class JSearchCollector(BaseCollector):
    """
    JSearch collector via RapidAPI.
    Uses shared request pool with adaptive pagination.
    """

    API_HOST = "jsearch.p.rapidapi.com"
    API_URL = "https://jsearch.p.rapidapi.com/search"

    def __init__(self, run_budget: int = 38):
        """
        Initialize JSearch collector.

        Args:
            run_budget: Request budget for this run (default 38/month, ~3.8 per day)
        """
        super().__init__("jsearch", run_budget)
        self.api_key = os.getenv("X_RAPID_API")

        if not self.api_key:
            logger.warning("X_RAPID_API not set, JSearch will not be available")

    def fetch_page(self, query: str, page: int) -> List[Dict[str, Any]]:
        """
        Fetch a page of results from JSearch.

        Args:
            query: Search query (e.g., "machine learning engineer")
            page: Page number (1-indexed)

        Returns:
            List of job dicts
        """
        if not self.api_key:
            return []

        params = {
            "query": query,
            "date_posted": "3days",
            "page": page,
            "num_pages": 1,
            "country": "us",
        }

        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.API_HOST,
        }

        try:
            response = self._fetch_with_backoff(self.API_URL, headers=headers, params=params, timeout=10)
            data = response.json()

            jobs = []
            for item in data.get("data", []):
                jobs.append(self._normalize_job(item))

            return jobs
        except Exception as e:
            logger.error(f"Error fetching JSearch page {page} for '{query}': {e}")
            return []

    def _normalize_job(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize JSearch job to schema.

        Args:
            item: Raw JSearch job dict

        Returns:
            Normalized job dict
        """
        return {
            "external_id": item.get("job_id"),
            "source_system": "jsearch",
            "source_board": None,
            "title": item.get("job_title"),
            "location": item.get("job_location") or self._format_location(item),
            "city": item.get("job_city"),
            "state": item.get("job_state"),
            "country": item.get("job_country"),
            "description": item.get("job_description"),
            "company_name": item.get("employer_name"),
            "company_domain": item.get("employer_website"),
            "employment_type": item.get("job_employment_type"),
            "salary_min": item.get("job_min_salary"),
            "salary_max": item.get("job_max_salary"),
            "salary_currency": item.get("job_salary_currency", "USD"),
            "salary_period": item.get("job_salary_period"),
            "remote_status": "remote" if item.get("job_is_remote") else "onsite",
            "posted_date": item.get("job_posted_at_datetime_utc"),
            "expiry_date": item.get("job_offer_expiration_datetime_utc"),
            "source_url": item.get("job_google_link"),
            "apply_url": item.get("job_apply_link"),
            "raw_data": json.dumps(item),
            "metadata": json.dumps(
                {
                    "required_experience": item.get("job_required_experience"),
                    "highlights": item.get("job_highlights", {}),
                }
            ),
            "scraped_at": None,  # will be set by collect_jobs.py
        }

    def _format_location(self, item: Dict[str, Any]) -> str:
        """Format location from structured fields"""
        parts = []
        if item.get("job_city"):
            parts.append(item["job_city"])
        if item.get("job_state"):
            parts.append(item["job_state"])
        if item.get("job_country"):
            parts.append(item["job_country"])
        return ", ".join(parts) if parts else None
