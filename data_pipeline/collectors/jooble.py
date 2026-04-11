"""
Jooble job collector.
"""
import http.client
import json
import logging
import os
from typing import Any, Dict, List

from data_pipeline.collectors.base import BaseCollector

logger = logging.getLogger(__name__)


class JoobleCollector(BaseCollector):
    """Jooble collector via direct API."""

    HOST = "jooble.org"

    def __init__(self, run_budget: int = 38):
        """
        Initialize Jooble collector.

        Args:
            run_budget: Request budget for this run (default 38 = 500/month ÷ 13 runs)
        """
        super().__init__("jooble", run_budget)
        self.api_key = os.getenv("JOOBLE_API_KEY")

        if not self.api_key:
            logger.warning("JOOBLE_API_KEY not set, Jooble will not be available")

    def fetch_page(self, query: str, page: int) -> List[Dict[str, Any]]:
        """
        Fetch a page of results from Jooble.

        Args:
            query: Search query (e.g., "machine learning engineer")
            page: Page number (1-indexed)

        Returns:
            List of job dicts
        """
        if not self.api_key:
            return []

        try:
            conn = http.client.HTTPConnection(self.HOST)
            headers = {"Content-type": "application/json"}
            body = json.dumps({
                "keywords": query,
                "page": page,
                "ResultOnPage": 20,
            })

            conn.request("POST", f"/api/{self.api_key}", body, headers)
            response = conn.getresponse()

            if response.status == 200:
                data = json.loads(response.read().decode())
                jobs = [self._normalize_job(j) for j in data.get("jobs", [])]
                return jobs
            elif response.status == 403:
                logger.error("Jooble: Invalid API key (403)")
                return []
            else:
                logger.error(f"Jooble error {response.status}: {response.reason}")
                return []

        except Exception as e:
            logger.error(f"Error fetching Jooble page {page} for '{query}': {e}")
            return []
        finally:
            conn.close()

    def _normalize_job(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Jooble job to schema.

        Args:
            item: Raw Jooble job dict

        Returns:
            Normalized job dict
        """
        return {
            "external_id": str(item.get("id")),
            "source_system": "jooble",
            "source_board": None,
            "title": item.get("title"),
            "location": item.get("location"),
            "description": item.get("snippet"),
            "company_name": item.get("company"),
            "employment_type": item.get("type"),
            "remote_status": self._infer_remote_status(item),
            "posted_date": item.get("updated"),
            "source_url": item.get("link"),
            "apply_url": item.get("link"),
            "salary_min": None,
            "salary_max": None,
            "salary_currency": None,
            "salary_period": None,
            "raw_data": json.dumps(item),
            "scraped_at": None,
        }

    def _infer_remote_status(self, item: Dict[str, Any]) -> str:
        """Infer remote status from title, type, or location."""
        title_lower = (item.get("title") or "").lower()
        type_lower = (item.get("type") or "").lower()
        location_lower = (item.get("location") or "").lower()

        if any(word in title_lower or word in type_lower for word in ["remote", "work from home", "wfh"]):
            return "remote"
        if location_lower == "remote":
            return "remote"
        return "onsite"
