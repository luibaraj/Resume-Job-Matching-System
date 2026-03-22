"""
Tests for the reranking module (Cohere Rerank 3).
"""

import sys
import os
from unittest import TestCase
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from reranking import rerank_jobs, _format_document
from retrieval import JobResult


def make_job(
    id: str,
    title: str = "Software Engineer",
    location: str = "Remote",
    description: str = "Excellent opportunity",
) -> JobResult:
    """Helper to create a JobResult dict for testing."""
    return JobResult(
        id=id,
        distance=0.1,
        title=title,
        location=location,
        source_url="https://example.com/job",
        board_token="example",
        cleaned_description=description,
    )


class TestFormatDocument(TestCase):
    """Tests for the _format_document helper function."""

    def test_includes_title_location_description(self):
        """Formatted document includes all three fields."""
        job = make_job(id="j1", title="Engineer", location="SF", description="Great role")
        formatted = _format_document(job)

        self.assertIn("Engineer", formatted)
        self.assertIn("SF", formatted)
        self.assertIn("Great role", formatted)

    def test_format_structure(self):
        """Title and location on first line, description on second."""
        job = make_job(
            id="j1",
            title="Engineer",
            location="SF",
            description="A long description",
        )
        formatted = _format_document(job)

        lines = formatted.split("\n")
        self.assertEqual(len(lines), 2)
        self.assertIn("Engineer", lines[0])
        self.assertIn("SF", lines[0])
        self.assertIn("|", lines[0])
        self.assertEqual(lines[1], "A long description")


class TestRerankJobs(TestCase):
    """Tests for the rerank_jobs function."""

    def test_empty_list_returns_empty(self):
        """Empty input list returns empty output without API call."""
        with patch('reranking.cohere.ClientV2') as mock_client_cls:
            result = rerank_jobs(query="test", jobs=[])

        self.assertEqual(result, [])
        # Verify no API call was made
        mock_client_cls.assert_not_called()

    @patch('reranking.cohere.ClientV2')
    def test_rerank_reorders_results(self, mock_client_cls):
        """Cohere reorder is reflected in output."""
        job1 = make_job(id="j1", title="Job 1")
        job2 = make_job(id="j2", title="Job 2")
        jobs = [job1, job2]

        # Mock Cohere to return job2 first (index=1), then job1 (index=0)
        mock_co = mock_client_cls.return_value
        mock_co.rerank.return_value = MagicMock(
            results=[
                MagicMock(index=1, relevance_score=0.95),
                MagicMock(index=0, relevance_score=0.70),
            ]
        )

        result = rerank_jobs(query="test", jobs=jobs, api_key="test-key")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "j2")  # Reordered: job2 first
        self.assertEqual(result[1]["id"], "j1")  # Then job1

    @patch('reranking.cohere.ClientV2')
    def test_top_n_respected(self, mock_client_cls):
        """Output length respects top_n parameter."""
        jobs = [make_job(id=f"j{i}") for i in range(10)]

        mock_co = mock_client_cls.return_value
        # Simulate Cohere returning only 3 results
        mock_co.rerank.return_value = MagicMock(
            results=[
                MagicMock(index=0, relevance_score=0.9),
                MagicMock(index=2, relevance_score=0.8),
                MagicMock(index=1, relevance_score=0.7),
            ]
        )

        result = rerank_jobs(query="test", jobs=jobs, top_n=3, api_key="test-key")

        self.assertEqual(len(result), 3)

    @patch('reranking.cohere.ClientV2')
    def test_documents_passed_correctly(self, mock_client_cls):
        """Documents are formatted and passed to Cohere correctly."""
        job1 = make_job(id="j1", title="Engineer", location="NYC", description="desc1")
        job2 = make_job(id="j2", title="Manager", location="SF", description="desc2")
        jobs = [job1, job2]

        mock_co = mock_client_cls.return_value
        mock_co.rerank.return_value = MagicMock(results=[])

        rerank_jobs(query="test", jobs=jobs, api_key="test-key")

        # Verify rerank was called
        mock_co.rerank.assert_called_once()
        call_kwargs = mock_co.rerank.call_args.kwargs

        # Verify documents list
        documents = call_kwargs["documents"]
        self.assertEqual(len(documents), 2)
        self.assertIn("Engineer", documents[0])
        self.assertIn("NYC", documents[0])
        self.assertIn("desc1", documents[0])
        self.assertIn("Manager", documents[1])
        self.assertIn("SF", documents[1])
        self.assertIn("desc2", documents[1])

    @patch('reranking.cohere.ClientV2')
    def test_top_n_capped_at_list_length(self, mock_client_cls):
        """top_n > list length is capped to avoid API errors."""
        jobs = [make_job(id=f"j{i}") for i in range(3)]

        mock_co = mock_client_cls.return_value
        mock_co.rerank.return_value = MagicMock(results=[])

        rerank_jobs(query="test", jobs=jobs, top_n=100, api_key="test-key")

        # Verify Cohere was called with top_n=3, not 100
        call_kwargs = mock_co.rerank.call_args.kwargs
        self.assertEqual(call_kwargs["top_n"], 3)

    @patch('reranking.cohere.ClientV2')
    def test_api_key_env_fallback(self, mock_client_cls):
        """COHERE_API_KEY env var is used when no key passed."""
        jobs = [make_job(id="j1")]
        test_key = "test-key-from-env"

        with patch.dict(os.environ, {"COHERE_API_KEY": test_key}):
            mock_co = mock_client_cls.return_value
            mock_co.rerank.return_value = MagicMock(results=[])

            rerank_jobs(query="test", jobs=jobs)

        # Verify ClientV2 was instantiated with env var
        mock_client_cls.assert_called_once_with(api_key=test_key)

    @patch('reranking.cohere.ClientV2')
    def test_explicit_api_key_used(self, mock_client_cls):
        """Explicit api_key param takes precedence over env var."""
        jobs = [make_job(id="j1")]
        explicit_key = "explicit-test-key"
        env_key = "env-test-key"

        with patch.dict(os.environ, {"COHERE_API_KEY": env_key}):
            mock_co = mock_client_cls.return_value
            mock_co.rerank.return_value = MagicMock(results=[])

            rerank_jobs(query="test", jobs=jobs, api_key=explicit_key)

        # Verify ClientV2 was instantiated with explicit key, not env key
        mock_client_cls.assert_called_once_with(api_key=explicit_key)

    @patch('reranking.cohere.ClientV2')
    def test_api_error_propagates(self, mock_client_cls):
        """Cohere API errors are propagated to the caller."""
        jobs = [make_job(id="j1")]

        mock_co = mock_client_cls.return_value
        mock_co.rerank.side_effect = Exception("API Error: rate limited")

        with self.assertRaises(Exception) as cm:
            rerank_jobs(query="test", jobs=jobs, api_key="test-key")

        self.assertIn("rate limited", str(cm.exception))

    @patch('reranking.cohere.ClientV2')
    def test_returns_job_result_dicts(self, mock_client_cls):
        """Output contains JobResult dicts with all required keys."""
        job = make_job(id="j1", title="Job")
        jobs = [job]

        mock_co = mock_client_cls.return_value
        mock_co.rerank.return_value = MagicMock(
            results=[MagicMock(index=0, relevance_score=0.9)]
        )

        result = rerank_jobs(query="test", jobs=jobs, api_key="test-key")

        self.assertEqual(len(result), 1)
        result_dict = result[0]
        # Verify all JobResult keys are present
        required_keys = {"id", "distance", "title", "location", "source_url",
                        "board_token", "cleaned_description"}
        self.assertTrue(required_keys.issubset(set(result_dict.keys())))

    @patch('reranking.cohere.ClientV2')
    def test_model_and_query_passed_to_cohere(self, mock_client_cls):
        """Model name and query are passed correctly to Cohere."""
        jobs = [make_job(id="j1")]
        query_text = "Python developer with 5 years experience"

        mock_co = mock_client_cls.return_value
        mock_co.rerank.return_value = MagicMock(results=[])

        rerank_jobs(query=query_text, jobs=jobs, api_key="test-key")

        call_kwargs = mock_co.rerank.call_args.kwargs
        # Verify model and query
        self.assertEqual(call_kwargs["model"], "rerank-english-v3.0")
        self.assertEqual(call_kwargs["query"], query_text)
