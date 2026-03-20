"""
Greenhouse job board scraper module.

Scrapes job listings from Greenhouse job boards using their public API.
Handles rate limiting, pagination, and data transformation.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class GreenhouseJob:
    """Represents a job scraped from Greenhouse."""
    id: str
    title: str
    location: str
    description: str
    internal_job_id: int
    updated_at: str
    created_at: str
    url: str
    absolute_url: str
    company_name: Optional[str] = None
    department: Optional[str] = None
    job_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            'external_id': self.id,
            'title': self.title,
            'location': self.location,
            'description': self.description,
            'source': 'greenhouse',
            'source_url': self.absolute_url,
            'company_name': self.company_name,
            'department': self.department,
            'job_type': self.job_type,
            'scraped_at': datetime.utcnow().isoformat(),
        }


class GreenhouseScraper:
    """Scrapes job listings from Greenhouse job boards."""

    # Greenhouse API documentation: https://developers.greenhouse.io/job-board.html
    BASE_URL = "https://api.greenhouse.io/v1/public/jobs"

    def __init__(
        self,
        board_token: str,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        timeout: int = 10,
    ):
        """
        Initialize the Greenhouse scraper.

        Args:
            board_token: Greenhouse job board token (from job board URL)
            max_retries: Number of retries for failed requests
            backoff_factor: Backoff factor for retry strategy
            timeout: Request timeout in seconds
        """
        self.board_token = board_token
        self.timeout = timeout
        self.session = self._create_session(max_retries, backoff_factor)

    def _create_session(self, max_retries: int, backoff_factor: float) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _get_params(self, offset: int = 0) -> Dict[str, Any]:
        """Build query parameters for Greenhouse API."""
        return {
            'content': 'true',  # Include full job content
            'limit': 100,       # Max per page
            'offset': offset,
        }

    def fetch_jobs(
        self,
        status: str = "published",
        updated_after: Optional[datetime] = None,
    ) -> List[GreenhouseJob]:
        """
        Fetch all jobs from the Greenhouse board.

        Args:
            status: Job status filter ('published', 'draft', etc.)
            updated_after: Only fetch jobs updated after this datetime

        Returns:
            List of GreenhouseJob objects
        """
        jobs = []
        offset = 0
        has_more = True

        logger.info(f"Starting Greenhouse scrape for board: {self.board_token}")

        while has_more:
            try:
                params = self._get_params(offset)
                response = self.session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.timeout,
                )
                response.raise_for_status()

                data = response.json()
                batch = data.get('jobs', [])

                if not batch:
                    has_more = False
                    break

                for job_data in batch:
                    job = self._parse_job(job_data, status, updated_after)
                    if job:
                        jobs.append(job)

                # Check if there are more pages
                has_more = len(batch) == 100
                offset += 100

                logger.debug(f"Fetched {len(batch)} jobs, total so far: {len(jobs)}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching jobs from Greenhouse: {e}")
                raise

        logger.info(f"Successfully scraped {len(jobs)} jobs from Greenhouse")
        return jobs

    def _parse_job(
        self,
        job_data: Dict[str, Any],
        status_filter: str,
        updated_after: Optional[datetime],
    ) -> Optional[GreenhouseJob]:
        """
        Parse a job entry from Greenhouse API response.

        Args:
            job_data: Raw job data from API
            status_filter: Filter jobs by status
            updated_after: Only include jobs updated after this time

        Returns:
            GreenhouseJob if valid and passes filters, None otherwise
        """
        try:
            # Check status
            if job_data.get('status') != status_filter:
                return None

            # Check updated_at filter
            if updated_after:
                updated_at_str = job_data.get('updated_at', '')
                if updated_at_str:
                    updated_at = datetime.fromisoformat(
                        updated_at_str.replace('Z', '+00:00')
                    )
                    if updated_at < updated_after:
                        return None

            # Extract required fields
            job_id = str(job_data.get('id', ''))
            title = job_data.get('title', '').strip()
            internal_job_id = job_data.get('internal_job_id', 0)

            if not job_id or not title:
                logger.warning(f"Skipping job with missing required fields: {job_data}")
                return None

            # Extract location
            location = self._extract_location(job_data)

            # Extract description
            description = job_data.get('content', '').strip()
            if not description:
                logger.warning(f"Job {job_id} missing content/description")
                return None

            # Extract optional fields
            url = job_data.get('url', '')
            absolute_url = job_data.get('absolute_url', '')
            department = None
            if job_data.get('departments'):
                department = job_data['departments'][0].get('name')

            job_type = None
            if job_data.get('job_types'):
                job_type = ', '.join([jt.get('name', '') for jt in job_data['job_types']])

            return GreenhouseJob(
                id=job_id,
                title=title,
                location=location,
                description=description,
                internal_job_id=internal_job_id,
                updated_at=job_data.get('updated_at', ''),
                created_at=job_data.get('created_at', ''),
                url=url,
                absolute_url=absolute_url,
                department=department,
                job_type=job_type,
            )

        except Exception as e:
            logger.error(f"Error parsing job {job_data.get('id', 'unknown')}: {e}")
            return None

    def _extract_location(self, job_data: Dict[str, Any]) -> str:
        """
        Extract location from job data.

        Greenhouse can have multiple offices; prioritize primary office.
        """
        offices = job_data.get('offices', [])
        if offices:
            # Try to get primary office or first office
            primary = next((o for o in offices if o.get('primary')), offices[0])
            city = primary.get('city', '')
            state = primary.get('state', '')
            country = primary.get('country_code', '')

            parts = [p for p in [city, state, country] if p]
            return ', '.join(parts) if parts else 'Remote'

        return 'Remote'

    def close(self) -> None:
        """Close the session."""
        self.session.close()


class GreenhouseJobBoardDiscovery:
    """Discovers Greenhouse job board tokens from company URLs."""

    @staticmethod
    def extract_board_token_from_url(url: str) -> Optional[str]:
        """
        Extract Greenhouse board token from a job board URL.

        Greenhouse URLs typically follow: {company}.greenhouse.io/jobs

        Args:
            url: Job board URL

        Returns:
            Board token (company name) or None if not a Greenhouse board
        """
        if 'greenhouse.io' not in url:
            return None

        # Extract company name from URL like "mycompany.greenhouse.io"
        try:
            if 'greenhouse.io' in url:
                parts = url.split('.')
                if len(parts) >= 2:
                    return parts[0].replace('https://', '').replace('http://', '')
        except Exception as e:
            logger.warning(f"Could not extract board token from {url}: {e}")

        return None


def scrape_greenhouse_board(
    board_token: str,
    status: str = "published",
    updated_after_days: Optional[int] = None,
) -> List[GreenhouseJob]:
    """
    Convenience function to scrape a Greenhouse board.

    Args:
        board_token: Greenhouse board token
        status: Job status filter
        updated_after_days: Only fetch jobs updated in last N days

    Returns:
        List of GreenhouseJob objects
    """
    scraper = GreenhouseScraper(board_token)

    updated_after = None
    if updated_after_days:
        updated_after = datetime.utcnow() - timedelta(days=updated_after_days)

    try:
        jobs = scraper.fetch_jobs(status=status, updated_after=updated_after)
        return jobs
    finally:
        scraper.close()
