"""
Jobs API (jobs-api14) collector with adaptive pagination.
"""
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from data_pipeline.collectors.base import BaseCollector

logger = logging.getLogger(__name__)


class JobsApiCollector(BaseCollector):
    """
    Jobs API collector via RapidAPI (Bing Jobs aggregator).
    Uses shared request pool with adaptive pagination.
    """

    API_HOST = "jobs-api14.p.rapidapi.com"
    API_URL = "https://jobs-api14.p.rapidapi.com/v2/bing/search"

    def __init__(self, run_budget: int = 50):
        """
        Initialize Jobs API collector.

        Args:
            run_budget: Request budget for this run (default 50/month)
        """
        super().__init__("jobsapi", run_budget)
        self.api_key = os.getenv("X_RAPID_API")

        if not self.api_key:
            logger.warning("X_RAPID_API not set, Jobs API will not be available")

    def fetch_page(self, query: str, page: int) -> List[Dict[str, Any]]:
        """
        Fetch a page of results from Jobs API.

        Args:
            query: Search query (e.g., "software engineer")
            page: Page number (1-indexed, used for offset if supported)

        Returns:
            List of job dicts
        """
        if not self.api_key:
            return []

        params = {
            "query": query,
            "location": "United States",
            "countryCode": "us",
            "page": page,
        }

        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.API_HOST,
        }

        try:
            response = self._fetch_with_backoff(
                self.API_URL, headers=headers, params=params, timeout=10
            )
            data = response.json()

            jobs = []
            for item in data.get("data", []):
                jobs.append(self._normalize_job(item))

            return jobs
        except Exception as e:
            logger.error(f"Error fetching Jobs API page {page} for '{query}': {e}")
            return []

    def _normalize_job(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Jobs API job to schema.

        Args:
            item: Raw Jobs API job dict

        Returns:
            Normalized job dict
        """
        posted_date = None
        if item.get("postDate"):
            posted_date = self._parse_relative_date(item["postDate"])

        company = item.get("company")
        company_name = company if isinstance(company, str) else company.get("name") if isinstance(company, dict) else None

        return {
            "external_id": item.get("id"),
            "source_system": "jobsapi",
            "source_board": None,
            "title": item.get("title"),
            "location": item.get("location"),
            "description": item.get("description"),
            "company_name": company_name,
            "posted_date": posted_date,
            "source_url": item.get("link") or item.get("url"),
            "apply_url": item.get("link") or item.get("url"),
            "raw_data": json.dumps(item),
            "metadata": json.dumps({}),
            "scraped_at": None,
        }

    def _parse_relative_date(self, relative_str: str) -> Optional[str]:
        """
        Parse relative date string (e.g., '2d ago') to ISO 8601.

        Args:
            relative_str: Relative date string

        Returns:
            ISO 8601 date string or None
        """
        if not relative_str:
            return None

        try:
            relative_str = relative_str.lower().strip()

            if "just now" in relative_str or "now" in relative_str:
                return datetime.utcnow().isoformat() + "Z"
            elif "h ago" in relative_str or "hour" in relative_str:
                hours = int(relative_str.split()[0])
                return (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
            elif "d ago" in relative_str or "day" in relative_str:
                days = int(relative_str.split()[0])
                return (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
            elif "w ago" in relative_str or "week" in relative_str:
                weeks = int(relative_str.split()[0])
                return (datetime.utcnow() - timedelta(weeks=weeks)).isoformat() + "Z"
            elif "m ago" in relative_str or "month" in relative_str:
                months = int(relative_str.split()[0])
                return (datetime.utcnow() - timedelta(days=months * 30)).isoformat() + "Z"

        except Exception as e:
            logger.debug(f"Could not parse relative date '{relative_str}': {e}")

        return None
