"""
Greenhouse job board collector.
Fetches from all configured board tokens (no pooling, single call per board).
"""
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from src.greenhouse_scraper import GreenhouseScraper

logger = logging.getLogger(__name__)


class GreenhouseCollector:
    """
    Collector for Greenhouse job board API.
    Iterates through all configured Greenhouse board tokens.
    """

    def __init__(self):
        """Initialize Greenhouse collector"""
        self.source_name = "greenhouse"
        self.requests_used = 0

    def collect_all(self, company_list: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Collect jobs from all Greenhouse boards.

        Args:
            company_list: List of dicts with "board_token" and "company_name"

        Returns:
            List of normalized job dicts
        """
        all_jobs = []

        for company in company_list:
            board_token = company.get("board_token")
            company_name = company.get("company_name")

            if not board_token:
                logger.warning(f"Skipping company with missing board_token: {company}")
                continue

            try:
                logger.info(f"Scraping Greenhouse board: {board_token} ({company_name})")
                jobs = self._scrape_board(board_token, company_name)
                all_jobs.extend(jobs)
                self.requests_used += 1
            except Exception as e:
                logger.error(f"Error scraping {board_token}: {e}")
                continue

        return all_jobs

    def _scrape_board(self, board_token: str, company_name: str) -> List[Dict[str, Any]]:
        """
        Scrape a single Greenhouse board and normalize to schema.

        Args:
            board_token: Greenhouse board token
            company_name: Company name for this board

        Returns:
            List of normalized job dicts
        """
        scraper = GreenhouseScraper(board_token)
        updated_after = datetime.now(timezone.utc) - timedelta(days=3)
        gh_jobs = scraper.fetch_jobs(status="published", updated_after=updated_after)

        normalized = []
        for job in gh_jobs:
            normalized.append(
                {
                    "external_id": str(job.id),
                    "source_system": "greenhouse",
                    "source_board": board_token,
                    "title": job.title,
                    "location": job.location,
                    "description": job.description,
                    "company_name": company_name,  # from config, not API
                    "department": job.department or None,
                    "updated_date": job.updated_at,
                    "source_url": job.absolute_url,
                    "raw_data": json.dumps(job.to_dict()),
                    "scraped_at": None,  # will be set by collect_jobs.py
                }
            )

        return normalized
