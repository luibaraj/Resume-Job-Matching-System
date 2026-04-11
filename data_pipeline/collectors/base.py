"""
Base collector class with adaptive pagination and shared budget pooling.
"""
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """
    Base class for job collectors with adaptive pagination and shared request budget.

    Implements exponential backoff retry logic, shared pool budget redistribution,
    and handles rate limiting gracefully.
    """

    def __init__(self, source_name: str, run_budget: int):
        """
        Initialize collector.

        Args:
            source_name: Name of the data source (greenhouse, jsearch, etc.)
            run_budget: Total request budget for this run (shared across all queries)
        """
        self.source_name = source_name
        self.pool = run_budget
        self.session = self._create_session()
        self.requests_used = 0

    def _create_session(self) -> requests.Session:
        """Create session with retry strategy"""
        session = requests.Session()
        # Retry on 5xx and connection errors, but not 429 (handle explicitly)
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def has_budget(self) -> bool:
        """Check if budget remains in the pool"""
        return self.pool > 0

    def collect_all(self, queries: List[str], page_size: int = 10) -> List[Dict[str, Any]]:
        """
        Collect jobs from all queries, reusing budget from exhausted queries.

        Args:
            queries: List of search queries in ascending order of expected volume
            page_size: Jobs per page (for pagination detection)

        Returns:
            List of job dictionaries
        """
        all_jobs = []
        for query in queries:
            if not self.has_budget():
                logger.warning(f"Budget exhausted before processing query: {query}")
                break

            logger.info(f"Processing query: {query} (budget remaining: {self.pool})")
            jobs = self._paginate(query, page_size)
            all_jobs.extend(jobs)
            logger.info(f"Query '{query}' collected {len(jobs)} jobs (budget now: {self.pool})")

        return all_jobs

    def _paginate(self, query: str, page_size: int = 10) -> List[Dict[str, Any]]:
        """
        Paginate through results until budget exhausted or results < page_size.

        Args:
            query: Search query
            page_size: Jobs per page

        Returns:
            List of job dictionaries
        """
        jobs = []
        page = 1

        while self.has_budget():
            try:
                results = self.fetch_page(query, page)
                self.pool -= 1
                self.requests_used += 1
                jobs.extend(results)

                # If partial page, we've exhausted results
                if len(results) < page_size:
                    logger.debug(f"Partial page ({len(results)} < {page_size}), stopping pagination for query: {query}")
                    break

                page += 1
            except Exception as e:
                logger.error(f"Error fetching page {page} for query '{query}': {e}")
                # Continue to next query rather than failing completely
                break

        return jobs

    def _fetch_with_backoff(
        self, url: str, headers: Optional[Dict] = None, params: Optional[Dict] = None, timeout: int = 10
    ) -> requests.Response:
        """
        Fetch URL with exponential backoff retry.

        Args:
            url: URL to fetch
            headers: Optional headers dict
            params: Optional query parameters
            timeout: Request timeout in seconds

        Returns:
            Response object

        Raises:
            Exception if all retries exhausted
        """
        backoff_times = [1, 2, 4]  # exponential backoff: 1s, 2s, 4s

        for attempt, backoff in enumerate(backoff_times):
            try:
                response = self.session.get(url, headers=headers, params=params, timeout=timeout)

                # Rate limited, stop this source
                if response.status_code == 429:
                    logger.warning(f"Rate limited (429) on {self.source_name}, stopping collection")
                    raise Exception(f"Rate limited: {response.status_code}")

                # Server error, retry
                if response.status_code >= 500:
                    if attempt < len(backoff_times) - 1:
                        logger.warning(f"Server error {response.status_code}, retrying in {backoff}s...")
                        time.sleep(backoff)
                        continue
                    else:
                        raise Exception(f"Server error {response.status_code} after retries")

                response.raise_for_status()
                return response

            except requests.Timeout:
                if attempt < len(backoff_times) - 1:
                    logger.warning(f"Timeout, retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    raise Exception("Timeout after retries")

            except requests.RequestException as e:
                if attempt < len(backoff_times) - 1:
                    logger.warning(f"Request error, retrying in {backoff}s: {e}")
                    time.sleep(backoff)
                else:
                    raise Exception(f"Request failed after retries: {e}")

        raise Exception("All retries exhausted")

    @abstractmethod
    def fetch_page(self, query: str, page: int) -> List[Dict[str, Any]]:
        """
        Fetch a single page of results. Subclass must implement.

        Args:
            query: Search query
            page: Page number (1-indexed)

        Returns:
            List of job dictionaries
        """
        pass
