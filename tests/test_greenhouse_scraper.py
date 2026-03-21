"""Tests for Greenhouse job board scraper."""

import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.greenhouse_scraper import (
    GreenhouseScraper,
    GreenhouseJob,
    GreenhouseJobBoardDiscovery,
    scrape_greenhouse_board,
)


@pytest.fixture
def sample_job_data():
    """Sample job data from Greenhouse API (new boards-api format)."""
    return {
        'id': 12345,
        'title': 'Software Engineer',
        'location': {'name': 'San Francisco, CA'},
        'internal_job_id': 9999,
        'content': 'We are looking for a Software Engineer...',
        'url': '/jobs/12345',
        'absolute_url': 'https://example.greenhouse.io/jobs/12345',
        'updated_at': '2026-03-19T10:00:00Z',
        'created_at': '2026-03-01T10:00:00Z',
        'offices': [
            {
                'id': 1,
                'name': 'San Francisco',
                'location': 'San Francisco, CA, United States',
                'primary': True,
            }
        ],
        'departments': [
            {'id': 1, 'name': 'Engineering'}
        ],
        'job_types': [
            {'id': 1, 'name': 'Full-time'}
        ],
    }


@pytest.fixture
def scraper():
    """Create a Greenhouse scraper instance."""
    return GreenhouseScraper(board_token='example')


class TestGreenhouseJob:
    """Tests for GreenhouseJob dataclass."""

    def test_greenhouse_job_creation(self):
        """Test creating a GreenhouseJob instance."""
        job = GreenhouseJob(
            id='12345',
            title='Software Engineer',
            location='San Francisco, CA',
            description='Job description',
            internal_job_id=9999,
            updated_at='2026-03-19T10:00:00Z',
            created_at='2026-03-01T10:00:00Z',
            url='/jobs/12345',
            absolute_url='https://example.greenhouse.io/jobs/12345',
        )

        assert job.id == '12345'
        assert job.title == 'Software Engineer'
        assert job.location == 'San Francisco, CA'
        assert job.company_name is None

    def test_to_dict(self):
        """Test converting GreenhouseJob to dictionary."""
        job = GreenhouseJob(
            id='12345',
            title='Software Engineer',
            location='San Francisco, CA',
            description='Job description',
            internal_job_id=9999,
            updated_at='2026-03-19T10:00:00Z',
            created_at='2026-03-01T10:00:00Z',
            url='/jobs/12345',
            absolute_url='https://example.greenhouse.io/jobs/12345',
            company_name='Acme Corp',
        )

        job_dict = job.to_dict()

        assert job_dict['external_id'] == '12345'
        assert job_dict['title'] == 'Software Engineer'
        assert job_dict['source'] == 'greenhouse'
        assert job_dict['company_name'] == 'Acme Corp'
        assert 'scraped_at' in job_dict


class TestGreenhouseScraper:
    """Tests for GreenhouseScraper class."""

    def test_scraper_initialization(self):
        """Test scraper initialization."""
        scraper = GreenhouseScraper('test_board')

        assert scraper.board_token == 'test_board'
        assert scraper.timeout == 10
        assert scraper.session is not None

    def test_get_params(self, scraper):
        """Test building API parameters."""
        params = scraper._get_params()

        assert params['content'] == 'true'
        assert len(params) == 1  # Only content parameter

    def test_extract_location_with_office(self, scraper, sample_job_data):
        """Test location extraction with office data."""
        location = scraper._extract_location(sample_job_data)

        assert 'San Francisco' in location
        assert 'CA' in location

    def test_extract_location_no_office(self, scraper):
        """Test location extraction without office data."""
        job_data = {'offices': []}
        location = scraper._extract_location(job_data)

        assert location == 'Remote'

    def test_parse_job_success(self, scraper, sample_job_data):
        """Test successful job parsing."""
        job = scraper._parse_job(sample_job_data, 'published', None)

        assert job is not None
        assert job.id == '12345'
        assert job.title == 'Software Engineer'
        assert job.internal_job_id == 9999
        assert job.department == 'Engineering'
        assert job.job_type == 'Full-time'

    def test_parse_job_wrong_status(self, scraper, sample_job_data):
        """Test that jobs with wrong status are filtered."""
        sample_job_data['status'] = 'draft'
        job = scraper._parse_job(sample_job_data, 'published', None)

        assert job is None

    def test_parse_job_no_status_field(self, scraper, sample_job_data):
        """Test that jobs without status field (new API) are still parsed."""
        # Ensure no status field is present (new API omits it)
        sample_job_data.pop('status', None)
        job = scraper._parse_job(sample_job_data, 'published', None)

        assert job is not None
        assert job.id == '12345'

    def test_parse_job_missing_required_fields(self, scraper):
        """Test that jobs with missing required fields are skipped."""
        job_data = {'status': 'published'}
        job = scraper._parse_job(job_data, 'published', None)

        assert job is None

    def test_parse_job_missing_content(self, scraper, sample_job_data):
        """Test that jobs with missing content are skipped."""
        sample_job_data['content'] = ''
        job = scraper._parse_job(sample_job_data, 'published', None)

        assert job is None

    def test_parse_job_updated_after_filter(self, scraper, sample_job_data):
        """Test updated_after filtering."""
        old_date = datetime.fromisoformat('2026-03-01T00:00:00Z')
        job = scraper._parse_job(sample_job_data, 'published', old_date)

        assert job is not None

    def test_parse_job_updated_after_filter_excludes_old(self, scraper, sample_job_data):
        """Test that old jobs are excluded."""
        recent_date = datetime.fromisoformat('2026-03-20T00:00:00Z')
        job = scraper._parse_job(sample_job_data, 'published', recent_date)

        assert job is None

    def test_parse_job_without_departments(self, scraper, sample_job_data):
        """Test parsing job without departments."""
        sample_job_data['departments'] = []
        job = scraper._parse_job(sample_job_data, 'published', None)

        assert job is not None
        assert job.department is None

    def test_parse_job_without_job_types(self, scraper, sample_job_data):
        """Test parsing job without job types."""
        sample_job_data['job_types'] = []
        job = scraper._parse_job(sample_job_data, 'published', None)

        assert job is not None
        assert job.job_type is None

    @patch('src.greenhouse_scraper.requests.Session.get')
    def test_fetch_jobs_success(self, mock_get, scraper, sample_job_data):
        """Test successful job fetching."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'jobs': [sample_job_data]}
        mock_get.return_value = mock_response

        jobs = scraper.fetch_jobs()

        assert len(jobs) == 1
        assert jobs[0].id == '12345'
        assert jobs[0].title == 'Software Engineer'
        assert mock_get.call_count == 1  # Single API call, no pagination

    @patch('src.greenhouse_scraper.requests.Session.get')
    def test_fetch_jobs_api_error(self, mock_get, scraper):
        """Test handling of API errors."""
        mock_get.side_effect = requests.exceptions.RequestException("API Error")

        with pytest.raises(requests.exceptions.RequestException):
            scraper.fetch_jobs()

    @patch('src.greenhouse_scraper.requests.Session.get')
    def test_fetch_jobs_empty_response(self, mock_get, scraper):
        """Test handling of empty response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'jobs': []}
        mock_get.return_value = mock_response

        jobs = scraper.fetch_jobs()

        assert len(jobs) == 0

    def test_close_session(self, scraper):
        """Test closing the session."""
        scraper.close()
        # Should not raise any exception


class TestGreenhouseJobBoardDiscovery:
    """Tests for GreenhouseJobBoardDiscovery class."""

    def test_extract_board_token_standard_url(self):
        """Test extracting board token from standard URL."""
        url = 'https://acmecorp.greenhouse.io/jobs'
        token = GreenhouseJobBoardDiscovery.extract_board_token_from_url(url)

        assert token == 'acmecorp'

    def test_extract_board_token_with_job_id(self):
        """Test extracting token from URL with job ID."""
        url = 'https://mycompany.greenhouse.io/jobs/12345'
        token = GreenhouseJobBoardDiscovery.extract_board_token_from_url(url)

        assert token == 'mycompany'

    def test_extract_board_token_non_greenhouse_url(self):
        """Test that non-Greenhouse URLs return None."""
        url = 'https://example.com/careers'
        token = GreenhouseJobBoardDiscovery.extract_board_token_from_url(url)

        assert token is None

    def test_extract_board_token_http_url(self):
        """Test extracting token from HTTP (not HTTPS) URL."""
        url = 'http://startups.greenhouse.io/jobs'
        token = GreenhouseJobBoardDiscovery.extract_board_token_from_url(url)

        assert token == 'startups'


class TestConvenienceFunction:
    """Tests for convenience functions."""

    @patch('src.greenhouse_scraper.GreenhouseScraper.fetch_jobs')
    def test_scrape_greenhouse_board(self, mock_fetch):
        """Test convenience function."""
        mock_fetch.return_value = []

        jobs = scrape_greenhouse_board('test_board')

        assert jobs == []
        mock_fetch.assert_called_once()

    @patch('src.greenhouse_scraper.GreenhouseScraper.fetch_jobs')
    def test_scrape_greenhouse_board_with_days_filter(self, mock_fetch):
        """Test convenience function with days filter."""
        mock_fetch.return_value = []

        jobs = scrape_greenhouse_board('test_board', updated_after_days=7)

        assert jobs == []
        # Verify that updated_after was passed
        call_kwargs = mock_fetch.call_args[1]
        assert 'updated_after' in call_kwargs
        assert call_kwargs['updated_after'] is not None
