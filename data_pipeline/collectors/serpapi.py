"""
SerpApi (Google Jobs) collector with adaptive pagination.
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from data_pipeline.collectors.base import BaseCollector

logger = logging.getLogger(__name__)


class SerpApiCollector(BaseCollector):
    """
    SerpApi collector for Google Jobs.
    Uses shared request pool with adaptive pagination.
    """

    API_URL = "https://serpapi.com/search"

    def __init__(self, run_budget: int = 19):
        """
        Initialize SerpApi collector.

        Args:
            run_budget: Request budget for this run (default 19/run, 250/month free tier)
        """
        super().__init__("serpapi", run_budget)
        self.api_key = os.getenv("SERPAPI_KEY")

        if not self.api_key:
            logger.warning("SERPAPI_KEY not set, SerpApi will not be available")

    def fetch_page(self, query: str, page: int) -> List[Dict[str, Any]]:
        """
        Fetch a page of results from SerpApi.

        Args:
            query: Search query
            page: Page number (1-indexed, SerpApi uses start offset)

        Returns:
            List of job dicts
        """
        if not self.api_key:
            return []

        start = (page - 1) * 10  # SerpApi uses 10 results per page

        params = {
            "engine": "google_jobs",
            "q": query,
            "chips": "date_posted:3days",
            "start": start,
            "api_key": self.api_key,
        }

        try:
            response = self._fetch_with_backoff(self.API_URL, params=params, timeout=10)
            data = response.json()

            jobs = []
            for item in data.get("jobs_results", []):
                jobs.append(self._normalize_job(item))

            return jobs
        except Exception as e:
            logger.error(f"Error fetching SerpApi page {page} for '{query}': {e}")
            return []

    def _normalize_job(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize SerpApi job to schema.

        Args:
            item: Raw SerpApi job dict

        Returns:
            Normalized job dict
        """
        detected = item.get("detected_extensions", {})
        salary = detected.get("salary")

        return {
            "external_id": item.get("job_id"),
            "source_system": "serpapi",
            "source_board": None,
            "title": item.get("title"),
            "location": item.get("location"),
            "description": item.get("description"),
            "company_name": item.get("company_name"),
            "employment_type": detected.get("schedule_type"),
            "remote_status": "remote" if detected.get("work_from_home") else "onsite",
            "posted_date": self._parse_relative_date(detected.get("posted_at")),
            "source_url": item.get("share_link"),
            "apply_url": (item.get("apply_options") or [{}])[0].get("link"),
            "raw_data": json.dumps(item),
            "metadata": json.dumps(
                {
                    "job_highlights": item.get("job_highlights", {}),
                    "via": item.get("via"),
                }
            ),
            "scraped_at": None,  # will be set by collect_jobs.py
        }

    def _parse_relative_date(self, relative_str: str) -> str:
        """
        Parse relative date string (e.g., '2 days ago') to ISO 8601.

        Args:
            relative_str: Relative date string

        Returns:
            ISO 8601 date string or None
        """
        if not relative_str:
            return None

        try:
            relative_str = relative_str.lower()

            if "just now" in relative_str or "minutes ago" in relative_str:
                from datetime import datetime, timedelta

                return datetime.utcnow().isoformat() + "Z"
            elif "hours ago" in relative_str:
                hours = int(relative_str.split()[0])
                from datetime import datetime, timedelta

                return (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
            elif "days ago" in relative_str:
                days = int(relative_str.split()[0])
                from datetime import datetime, timedelta

                return (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
            elif "weeks ago" in relative_str:
                weeks = int(relative_str.split()[0])
                from datetime import datetime, timedelta

                return (datetime.utcnow() - timedelta(weeks=weeks)).isoformat() + "Z"
            elif "months ago" in relative_str:
                months = int(relative_str.split()[0])
                from datetime import datetime, timedelta

                return (datetime.utcnow() - timedelta(days=months * 30)).isoformat() + "Z"

        except Exception as e:
            logger.debug(f"Could not parse relative date '{relative_str}': {e}")

        return None
