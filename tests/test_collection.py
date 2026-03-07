"""Tests for src/collection.py collection functions and GreenhouseClient."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from src.collection import (
    GreenhouseClient,
    collect_jobs,
    main,
    normalize_job_for_db,
)


class TestGreenhouseClientFetchJobs:
    """Tests for GreenhouseClient.fetch_jobs()."""

    def test_success_returns_list_of_dicts(self):
        """Successful response returns job list."""
        with patch("src.collection.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"jobs": [{"id": 1, "title": "Engineer"}]}
            mock_get.return_value = mock_response

            client = GreenhouseClient(["acme"], request_timeout=30, max_retries=3, retry_backoff=2.0)
            result = client.fetch_jobs("acme")

            assert result == [{"id": 1, "title": "Engineer"}]

    def test_passes_content_param(self):
        """Request includes content=true parameter."""
        with patch("src.collection.requests.get") as mock_get:
            mock_response = Mock(status_code=200, json=lambda: {"jobs": []})
            mock_get.return_value = mock_response

            client = GreenhouseClient(["acme"], request_timeout=30, max_retries=3, retry_backoff=2.0)
            client.fetch_jobs("acme")

            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args[1]["params"]["content"] == "true"

    def test_passes_timeout(self):
        """Request timeout is passed."""
        with patch("src.collection.requests.get") as mock_get:
            mock_response = Mock(status_code=200, json=lambda: {"jobs": []})
            mock_get.return_value = mock_response

            client = GreenhouseClient(["acme"], request_timeout=15, max_retries=3, retry_backoff=2.0)
            client.fetch_jobs("acme")

            mock_get.assert_called_once()
            assert mock_get.call_args[1]["timeout"] == 15

    def test_raises_after_retries_exhausted(self):
        """Exception is raised after max_retries attempts."""
        with patch("src.collection.requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("Network error")

            client = GreenhouseClient(["acme"], request_timeout=30, max_retries=3, retry_backoff=2.0)

            with pytest.raises(requests.ConnectionError):
                client.fetch_jobs("acme")

            # Should be called max_retries times
            assert mock_get.call_count == 3

    def test_raises_on_non_200_after_retries(self):
        """HTTPError is raised after exhausting retries on non-200 response."""
        with patch("src.collection.requests.get") as mock_get:
            mock_response = Mock(status_code=500)
            mock_response.raise_for_status.side_effect = requests.HTTPError("Server error")
            mock_get.return_value = mock_response

            client = GreenhouseClient(["acme"], request_timeout=30, max_retries=3, retry_backoff=2.0)

            with pytest.raises(requests.HTTPError):
                client.fetch_jobs("acme")

            # Called max_retries times
            assert mock_get.call_count == 3

    def test_exponential_backoff_sleeps(self):
        """Exponential backoff sleep is used between retries."""
        with patch("src.collection.requests.get") as mock_get:
            with patch("src.collection.time.sleep") as mock_sleep:
                mock_get.side_effect = requests.ConnectionError("Network error")

                client = GreenhouseClient(["acme"], request_timeout=30, max_retries=3, retry_backoff=2.0)

                try:
                    client.fetch_jobs("acme")
                except requests.ConnectionError:
                    pass

                # Should sleep with backoff: 2^0, 2^1 (before 2nd and 3rd attempt)
                assert mock_sleep.call_count == 2
                # First call: 2.0 ** 0 = 1.0
                # Second call: 2.0 ** 1 = 2.0
                calls = [call[0][0] for call in mock_sleep.call_args_list]
                assert calls[0] == 1.0
                assert calls[1] == 2.0

    def test_no_sleep_on_last_attempt(self):
        """No sleep after the final attempt."""
        with patch("src.collection.requests.get") as mock_get:
            with patch("src.collection.time.sleep") as mock_sleep:
                mock_get.side_effect = requests.ConnectionError("Network error")

                client = GreenhouseClient(["acme"], request_timeout=30, max_retries=3, retry_backoff=2.0)

                try:
                    client.fetch_jobs("acme")
                except requests.ConnectionError:
                    pass

                # Sleep is called max_retries - 1 times (not after the last failure)
                assert mock_sleep.call_count == 2

    def test_returns_empty_list_on_empty_jobs_key(self):
        """Empty jobs array is handled."""
        with patch("src.collection.requests.get") as mock_get:
            mock_response = Mock(status_code=200, json=lambda: {"jobs": []})
            mock_get.return_value = mock_response

            client = GreenhouseClient(["acme"], request_timeout=30, max_retries=3, retry_backoff=2.0)
            result = client.fetch_jobs("acme")

            assert result == []


class TestGreenhouseClientFetchAllBoards:
    """Tests for GreenhouseClient.fetch_all_boards()."""

    def test_aggregates_jobs_from_multiple_boards(self):
        """Jobs from multiple boards are aggregated."""
        with patch.object(GreenhouseClient, "fetch_jobs") as mock_fetch:
            mock_fetch.side_effect = [
                [{"id": 1, "title": "Job 1"}],
                [{"id": 2, "title": "Job 2"}],
            ]

            client = GreenhouseClient(["acme", "beta"], request_timeout=30, max_retries=3, retry_backoff=2.0)
            result = client.fetch_all_boards()

            assert len(result) == 2

    def test_injects_board_token_key(self):
        """board_token key is injected into each job."""
        with patch.object(GreenhouseClient, "fetch_jobs") as mock_fetch:
            mock_fetch.side_effect = [
                [{"id": 1, "title": "Job 1"}],
                [{"id": 2, "title": "Job 2"}],
            ]

            client = GreenhouseClient(["acme", "beta"], request_timeout=30, max_retries=3, retry_backoff=2.0)
            result = client.fetch_all_boards()

            # Find job from each board
            acme_jobs = [j for j in result if j.get("board_token") == "acme"]
            beta_jobs = [j for j in result if j.get("board_token") == "beta"]
            assert len(acme_jobs) == 1
            assert len(beta_jobs) == 1

    def test_continues_when_one_board_fails(self):
        """If one board fails, others are still fetched."""
        with patch.object(GreenhouseClient, "fetch_jobs") as mock_fetch:
            mock_fetch.side_effect = [
                requests.ConnectionError("Network error"),
                [{"id": 2, "title": "Job 2"}],
            ]

            client = GreenhouseClient(["acme", "beta"], request_timeout=30, max_retries=3, retry_backoff=2.0)
            result = client.fetch_all_boards()

            # Should contain job from beta, not acme
            assert len(result) == 1
            assert result[0]["id"] == 2

    def test_returns_empty_list_if_all_boards_fail(self):
        """Empty list when all boards fail."""
        with patch.object(GreenhouseClient, "fetch_jobs") as mock_fetch:
            mock_fetch.side_effect = [
                requests.ConnectionError("Network error"),
                requests.ConnectionError("Network error"),
            ]

            client = GreenhouseClient(["acme", "beta"], request_timeout=30, max_retries=3, retry_backoff=2.0)
            result = client.fetch_all_boards()

            assert result == []

    def test_returns_empty_list_for_no_tokens(self):
        """Empty list for client with no tokens."""
        client = GreenhouseClient([], request_timeout=30, max_retries=3, retry_backoff=2.0)
        result = client.fetch_all_boards()

        assert result == []


class TestNormalizeJobForDb:
    """Tests for normalize_job_for_db()."""

    def test_nested_location_dict_flattened(self):
        """Nested location dict is flattened to string."""
        raw_job = {
            "id": 1,
            "title": "Engineer",
            "board_token": "acme",
            "location": {"name": "San Francisco"},
        }
        result = normalize_job_for_db(raw_job)
        assert result["location"] == "San Francisco"

    def test_string_location_preserved(self):
        """String location is preserved as-is."""
        raw_job = {
            "id": 1,
            "title": "Engineer",
            "board_token": "acme",
            "location": "Remote",
        }
        result = normalize_job_for_db(raw_job)
        assert result["location"] == "Remote"

    def test_missing_location_defaults_to_none(self):
        """Missing location defaults to None."""
        raw_job = {
            "id": 1,
            "title": "Engineer",
            "board_token": "acme",
        }
        result = normalize_job_for_db(raw_job)
        assert result["location"] is None

    def test_null_location_defaults_to_none(self):
        """Null location is None."""
        raw_job = {
            "id": 1,
            "title": "Engineer",
            "board_token": "acme",
            "location": None,
        }
        result = normalize_job_for_db(raw_job)
        assert result["location"] is None

    def test_departments_serialized_as_json(self):
        """Departments list is serialized to JSON string."""
        raw_job = {
            "id": 1,
            "title": "Engineer",
            "board_token": "acme",
            "departments": [{"id": 10, "name": "Engineering"}, {"id": 20, "name": "Sales"}],
        }
        result = normalize_job_for_db(raw_job)
        assert result["departments"] == '["Engineering", "Sales"]'

    def test_offices_serialized_as_json(self):
        """Offices list is serialized to JSON string."""
        raw_job = {
            "id": 1,
            "title": "Engineer",
            "board_token": "acme",
            "offices": [{"id": 22, "name": "San Francisco"}],
        }
        result = normalize_job_for_db(raw_job)
        assert result["offices"] == '["San Francisco"]'

    def test_empty_departments_serialized(self):
        """Empty departments list is serialized to empty JSON array."""
        raw_job = {
            "id": 1,
            "title": "Engineer",
            "board_token": "acme",
            "departments": [],
        }
        result = normalize_job_for_db(raw_job)
        assert result["departments"] == "[]"

    def test_missing_content_defaults_to_none(self):
        """Missing content defaults to None."""
        raw_job = {
            "id": 1,
            "title": "Engineer",
            "board_token": "acme",
        }
        result = normalize_job_for_db(raw_job)
        assert result["raw_description"] is None

    def test_missing_absolute_url_defaults_to_none(self):
        """Missing absolute_url defaults to None."""
        raw_job = {
            "id": 1,
            "title": "Engineer",
            "board_token": "acme",
        }
        result = normalize_job_for_db(raw_job)
        assert result["absolute_url"] is None

    def test_missing_updated_at_defaults_to_none(self):
        """Missing updated_at defaults to None."""
        raw_job = {
            "id": 1,
            "title": "Engineer",
            "board_token": "acme",
        }
        result = normalize_job_for_db(raw_job)
        assert result["updated_at_source"] is None

    def test_board_token_used_as_company(self):
        """board_token is used as company."""
        raw_job = {
            "id": 1,
            "title": "Engineer",
            "board_token": "acme",
        }
        result = normalize_job_for_db(raw_job)
        assert result["company"] == "acme"

    def test_collected_at_is_iso8601_utc_string(self):
        """collected_at is an ISO-8601 UTC string."""
        raw_job = {
            "id": 1,
            "title": "Engineer",
            "board_token": "acme",
        }
        result = normalize_job_for_db(raw_job)
        # Should end with Z (UTC indicator)
        assert result["collected_at"].endswith("Z")
        # Should be parseable as ISO-8601
        import datetime
        datetime.datetime.fromisoformat(result["collected_at"].replace("Z", "+00:00"))

    def test_uses_raw_greenhouse_job_fixture(self, raw_greenhouse_job: dict):
        """Normalizing the raw job fixture works."""
        result = normalize_job_for_db(raw_greenhouse_job)
        # Should have all required keys
        assert "greenhouse_id" in result
        assert "board_token" in result
        assert "title" in result
        assert "company" in result


class TestCollectJobs:
    """Tests for collect_jobs()."""

    def test_returns_inserted_skipped_tuple(self):
        """Returns a 2-tuple of (inserted, skipped)."""
        mock_client = Mock()
        mock_client.fetch_all_boards.return_value = []
        mock_db = Mock()

        result = collect_jobs(mock_client, mock_db, 1)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, int) for x in result)

    def test_counts_inserted_correctly(self):
        """Counts inserted jobs correctly."""
        raw_jobs = [
            {"id": 1, "title": "Job 1", "board_token": "acme"},
            {"id": 2, "title": "Job 2", "board_token": "acme"},
            {"id": 3, "title": "Job 3", "board_token": "acme"},
        ]
        mock_client = Mock()
        mock_client.fetch_all_boards.return_value = raw_jobs
        mock_db = Mock()
        mock_db.insert_jobs_batch.return_value = (3, 0)  # All succeed

        with patch("src.collection.normalize_job_for_db", return_value={}):
            inserted, skipped = collect_jobs(mock_client, mock_db, 1)

        assert inserted == 3
        assert skipped == 0

    def test_counts_skipped_correctly(self):
        """Counts skipped jobs (duplicates) correctly."""
        raw_jobs = [
            {"id": 1, "title": "Job 1", "board_token": "acme"},
            {"id": 2, "title": "Job 2", "board_token": "acme"},
            {"id": 3, "title": "Job 3", "board_token": "acme"},
        ]
        mock_client = Mock()
        mock_client.fetch_all_boards.return_value = raw_jobs
        mock_db = Mock()
        mock_db.insert_jobs_batch.return_value = (0, 3)  # All are duplicates

        with patch("src.collection.normalize_job_for_db", return_value={}):
            inserted, skipped = collect_jobs(mock_client, mock_db, 1)

        assert inserted == 0
        assert skipped == 3

    def test_mixed_insert_and_skip(self):
        """Counts correct mix of insertions and skips."""
        raw_jobs = [
            {"id": 1, "title": "Job 1", "board_token": "acme"},
            {"id": 2, "title": "Job 2", "board_token": "acme"},
            {"id": 3, "title": "Job 3", "board_token": "acme"},
            {"id": 4, "title": "Job 4", "board_token": "acme"},
        ]
        mock_client = Mock()
        mock_client.fetch_all_boards.return_value = raw_jobs
        mock_db = Mock()
        mock_db.insert_jobs_batch.return_value = (3, 1)  # 3 inserted, 1 skipped

        with patch("src.collection.normalize_job_for_db", return_value={}):
            inserted, skipped = collect_jobs(mock_client, mock_db, 1)

        assert inserted == 3
        assert skipped == 1

    def test_calls_insert_jobs_batch_once(self):
        """insert_jobs_batch is called once with all normalized jobs."""
        raw_jobs = [
            {"id": 1, "title": "Job 1", "board_token": "acme"},
            {"id": 2, "title": "Job 2", "board_token": "acme"},
        ]
        mock_client = Mock()
        mock_client.fetch_all_boards.return_value = raw_jobs
        mock_db = Mock()
        mock_db.insert_jobs_batch.return_value = (2, 0)

        with patch("src.collection.normalize_job_for_db", return_value={}):
            collect_jobs(mock_client, mock_db, 1)

        assert mock_db.insert_jobs_batch.call_count == 1

    def test_normalizes_before_batching(self):
        """normalize_job_for_db is called for each job before batching."""
        raw_jobs = [{"id": 1, "title": "Job 1", "board_token": "acme"}]
        mock_client = Mock()
        mock_client.fetch_all_boards.return_value = raw_jobs
        mock_db = Mock()
        mock_db.insert_jobs_batch.return_value = (1, 0)

        with patch("src.collection.normalize_job_for_db") as mock_normalize:
            mock_normalize.return_value = {"normalized": True}
            collect_jobs(mock_client, mock_db, 1)

            assert mock_normalize.call_count == 1
            assert mock_normalize.call_args[0][0] == raw_jobs[0]


class TestMain:
    """Tests for main() entrypoint."""

    def test_creates_pipeline_run_before_collecting(self):
        """create_pipeline_run is called before collect_jobs."""
        call_order = []

        def track_create(*args, **kwargs):
            call_order.append("create")
            return 1

        def track_collect(*args, **kwargs):
            call_order.append("collect")
            return (0, 0)

        mock_config = MagicMock()
        mock_config.log_level = "INFO"
        with patch("src.collection.load_config", return_value=mock_config):
            with patch("src.collection.DatabaseManager") as mock_db_class:
                with patch("src.collection.GreenhouseClient"):
                    with patch("src.collection.collect_jobs", side_effect=track_collect):
                        mock_db = MagicMock()
                        mock_db.create_pipeline_run.side_effect = track_create
                        mock_db.finish_pipeline_run.return_value = None
                        mock_db_class.return_value = mock_db

                        main()

                        assert call_order == ["create", "collect"]

    def test_calls_finish_with_success_on_clean_run(self):
        """finish_pipeline_run is called with status='success' on successful run."""
        mock_config = MagicMock()
        mock_config.log_level = "INFO"
        with patch("src.collection.load_config", return_value=mock_config):
            with patch("src.collection.DatabaseManager") as mock_db_class:
                with patch("src.collection.GreenhouseClient"):
                    with patch("src.collection.collect_jobs", return_value=(5, 2)):
                        mock_db = MagicMock()
                        mock_db.create_pipeline_run.return_value = 1
                        mock_db_class.return_value = mock_db

                        main()

                        # Verify finish was called with status='success'
                        mock_db.finish_pipeline_run.assert_called_once()
                        call_args = mock_db.finish_pipeline_run.call_args
                        assert call_args[0][1] == "success"

    def test_passes_inserted_skipped_to_finish(self):
        """finish_pipeline_run receives correct inserted/skipped counts."""
        mock_config = MagicMock()
        mock_config.log_level = "INFO"
        with patch("src.collection.load_config", return_value=mock_config):
            with patch("src.collection.DatabaseManager") as mock_db_class:
                with patch("src.collection.GreenhouseClient"):
                    with patch("src.collection.collect_jobs", return_value=(5, 2)):
                        mock_db = MagicMock()
                        mock_db.create_pipeline_run.return_value = 1
                        mock_db_class.return_value = mock_db

                        main()

                        mock_db.finish_pipeline_run.assert_called_once()
                        call_args = mock_db.finish_pipeline_run.call_args
                        assert call_args[1]["jobs_processed"] == 5
                        assert call_args[1]["jobs_skipped"] == 2

    def test_calls_finish_with_failed_on_exception(self):
        """finish_pipeline_run is called with status='failed' on exception."""
        mock_config = MagicMock()
        mock_config.log_level = "INFO"
        with patch("src.collection.load_config", return_value=mock_config):
            with patch("src.collection.DatabaseManager") as mock_db_class:
                with patch("src.collection.GreenhouseClient"):
                    with patch("src.collection.collect_jobs", side_effect=RuntimeError("test error")):
                        mock_db = MagicMock()
                        mock_db.create_pipeline_run.return_value = 1
                        mock_db_class.return_value = mock_db

                        with pytest.raises(RuntimeError):
                            main()

                        # Verify finish was called with status='failed'
                        mock_db.finish_pipeline_run.assert_called_once()
                        call_args = mock_db.finish_pipeline_run.call_args
                        assert call_args[0][1] == "failed"

    def test_exception_is_reraised(self):
        """Exception from collect_jobs is re-raised after audit update."""
        mock_config = MagicMock()
        mock_config.log_level = "INFO"
        with patch("src.collection.load_config", return_value=mock_config):
            with patch("src.collection.DatabaseManager") as mock_db_class:
                with patch("src.collection.GreenhouseClient"):
                    with patch("src.collection.collect_jobs", side_effect=RuntimeError("test error")):
                        mock_db = MagicMock()
                        mock_db.create_pipeline_run.return_value = 1
                        mock_db_class.return_value = mock_db

                        with pytest.raises(RuntimeError, match="test error"):
                            main()


class TestGreenhouseClientFetchAllBoardsParallel:
    """Tests for parallel execution in GreenhouseClient.fetch_all_boards()."""

    def test_fetches_all_boards_with_multiple_tokens(self):
        """Fetch with 3 tokens calls fetch_jobs for each token."""
        with patch.object(GreenhouseClient, "fetch_jobs") as mock_fetch:
            mock_fetch.side_effect = [
                [{"id": 1, "title": "Job 1"}],
                [{"id": 2, "title": "Job 2"}],
                [{"id": 3, "title": "Job 3"}],
            ]

            client = GreenhouseClient(
                ["token1", "token2", "token3"],
                request_timeout=30,
                max_retries=3,
                retry_backoff=2.0,
            )
            result = client.fetch_all_boards()

            assert mock_fetch.call_count == 3
            assert len(result) == 3

    def test_continues_when_one_board_fails_parallel(self):
        """If one board fails in parallel execution, others still complete."""
        with patch.object(GreenhouseClient, "fetch_jobs") as mock_fetch:
            mock_fetch.side_effect = [
                requests.ConnectionError("Network error"),
                [{"id": 2, "title": "Job 2"}],
                [{"id": 3, "title": "Job 3"}],
            ]

            client = GreenhouseClient(
                ["token1", "token2", "token3"],
                request_timeout=30,
                max_retries=3,
                retry_backoff=2.0,
            )
            result = client.fetch_all_boards()

            assert len(result) == 2
            ids = [job["id"] for job in result]
            assert 2 in ids and 3 in ids and 1 not in ids

    def test_all_boards_fail_returns_empty_parallel(self):
        """All boards fail in parallel returns empty list."""
        with patch.object(GreenhouseClient, "fetch_jobs") as mock_fetch:
            mock_fetch.side_effect = [
                requests.ConnectionError("Network error"),
                requests.ConnectionError("Network error"),
            ]

            client = GreenhouseClient(
                ["token1", "token2"],
                request_timeout=30,
                max_retries=3,
                retry_backoff=2.0,
            )
            result = client.fetch_all_boards()

            assert result == []

    def test_board_token_injected_in_parallel(self):
        """board_token is injected into each job even with parallel execution."""
        with patch.object(GreenhouseClient, "fetch_jobs") as mock_fetch:
            mock_fetch.side_effect = [
                [{"id": 1, "title": "Job 1"}],
                [{"id": 2, "title": "Job 2"}],
            ]

            client = GreenhouseClient(
                ["acme", "beta"],
                request_timeout=30,
                max_retries=3,
                retry_backoff=2.0,
            )
            result = client.fetch_all_boards()

            for job in result:
                assert "board_token" in job
                assert job["board_token"] in ["acme", "beta"]


class TestCollectJobsBatch:
    """Tests for updated collect_jobs() using batch insert."""

    def test_calls_insert_jobs_batch_not_insert_job(self):
        """collect_jobs() calls insert_jobs_batch(), not individual insert_job()."""
        raw_jobs = [
            {"id": 1, "title": "Job 1", "board_token": "acme"},
            {"id": 2, "title": "Job 2", "board_token": "acme"},
        ]
        mock_client = Mock()
        mock_client.fetch_all_boards.return_value = raw_jobs
        mock_db = Mock()
        mock_db.insert_jobs_batch.return_value = (2, 0)

        collect_jobs(mock_client, mock_db, 1)

        mock_db.insert_jobs_batch.assert_called_once()
        mock_db.insert_job.assert_not_called()

    def test_empty_boards_returns_zero_zero(self):
        """No jobs from boards returns (0, 0) without calling batch insert."""
        mock_client = Mock()
        mock_client.fetch_all_boards.return_value = []
        mock_db = Mock()

        result = collect_jobs(mock_client, mock_db, 1)

        assert result == (0, 0)
        mock_db.insert_jobs_batch.assert_not_called()

    def test_returns_batch_inserted_skipped(self):
        """collect_jobs() returns (inserted, skipped) from batch call."""
        mock_client = Mock()
        mock_client.fetch_all_boards.return_value = [
            {"id": 1, "title": "Job 1", "board_token": "acme"}
        ]
        mock_db = Mock()
        mock_db.insert_jobs_batch.return_value = (5, 2)

        with patch("src.collection.normalize_job_for_db", return_value={}):
            result = collect_jobs(mock_client, mock_db, 1)

        assert result == (5, 2)

    def test_normalizes_all_jobs_before_batch(self):
        """All jobs are normalized before being passed to batch insert."""
        raw_jobs = [
            {"id": 1, "title": "Job 1", "board_token": "acme"},
            {"id": 2, "title": "Job 2", "board_token": "acme"},
        ]
        mock_client = Mock()
        mock_client.fetch_all_boards.return_value = raw_jobs
        mock_db = Mock()
        mock_db.insert_jobs_batch.return_value = (2, 0)

        with patch("src.collection.normalize_job_for_db") as mock_normalize:
            mock_normalize.return_value = {"normalized": True}
            collect_jobs(mock_client, mock_db, 1)

        assert mock_normalize.call_count == 2
        # Verify batch was called with normalized dicts
        batch_args = mock_db.insert_jobs_batch.call_args[0][0]
        assert all("normalized" in job for job in batch_args)


class TestGreenhouseClientRateLimiting429:
    """Tests for 429 rate limit handling in GreenhouseClient.fetch_jobs()."""

    def test_429_uses_retry_after_header_for_sleep(self):
        """429 response with Retry-After header uses that value for sleep."""
        with patch("src.collection.requests.get") as mock_get:
            with patch("src.collection.time.sleep") as mock_sleep:
                # First two attempts: 429 with Retry-After header
                resp_429 = Mock(status_code=429)
                resp_429.headers.get.return_value = "5"
                # Third attempt: 200 success
                resp_200 = Mock(status_code=200)
                resp_200.json.return_value = {"jobs": [{"id": 1}]}

                mock_get.side_effect = [resp_429, resp_429, resp_200]

                client = GreenhouseClient(["acme"], request_timeout=30, max_retries=3, retry_backoff=2.0)
                result = client.fetch_jobs("acme")

                # Should have slept twice with value from Retry-After header
                assert mock_sleep.call_count == 2
                calls = [call[0][0] for call in mock_sleep.call_args_list]
                assert calls[0] == 5.0
                assert calls[1] == 5.0
                assert result == [{"id": 1}]

    def test_429_falls_back_to_exponential_backoff_when_no_retry_after(self):
        """429 response without Retry-After header falls back to exponential backoff."""
        with patch("src.collection.requests.get") as mock_get:
            with patch("src.collection.time.sleep") as mock_sleep:
                # First attempt: 429 with no Retry-After header
                resp_429 = Mock(status_code=429)
                resp_429.headers.get.return_value = None
                # Second attempt: 200 success
                resp_200 = Mock(status_code=200)
                resp_200.json.return_value = {"jobs": [{"id": 1}]}

                mock_get.side_effect = [resp_429, resp_200]

                client = GreenhouseClient(["acme"], request_timeout=30, max_retries=3, retry_backoff=2.0)
                result = client.fetch_jobs("acme")

                # Should sleep with exponential backoff: 2^0 = 1.0
                assert mock_sleep.call_count == 1
                assert mock_sleep.call_args[0][0] == 1.0
                assert result == [{"id": 1}]

    def test_429_exhausted_raises_http_error(self):
        """429 on all attempts raises HTTPError after retries exhausted."""
        with patch("src.collection.requests.get") as mock_get:
            resp_429 = Mock(status_code=429)
            resp_429.headers.get.return_value = None
            resp_429.raise_for_status.side_effect = requests.HTTPError("Too many requests")

            mock_get.return_value = resp_429

            client = GreenhouseClient(["acme"], request_timeout=30, max_retries=3, retry_backoff=2.0)

            with pytest.raises(requests.HTTPError):
                client.fetch_jobs("acme")

            # Should be called max_retries times
            assert mock_get.call_count == 3

    def test_non_429_http_error_still_retries(self):
        """500 error path still retries with exponential backoff."""
        with patch("src.collection.requests.get") as mock_get:
            resp_500 = Mock(status_code=500)
            resp_500.raise_for_status.side_effect = requests.HTTPError("Server error")

            mock_get.return_value = resp_500

            client = GreenhouseClient(["acme"], request_timeout=30, max_retries=3, retry_backoff=2.0)

            with pytest.raises(requests.HTTPError):
                client.fetch_jobs("acme")

            # Should be called max_retries times
            assert mock_get.call_count == 3


class TestGreenhouseClientConcurrencyControl:
    """Tests for concurrency limiting in GreenhouseClient.fetch_all_boards()."""

    def test_request_delay_sleeps_between_submissions(self):
        """request_delay_seconds causes sleep between board fetch submissions."""
        with patch("src.collection.time.sleep") as mock_sleep:
            with patch.object(GreenhouseClient, "fetch_jobs", return_value=[]):
                client = GreenhouseClient(
                    ["board1", "board2", "board3"],
                    request_timeout=30,
                    max_retries=3,
                    retry_backoff=2.0,
                    request_delay_seconds=0.5
                )
                client.fetch_all_boards()

            # Sleep should be called 2 times (n-1 for 3 boards)
            assert mock_sleep.call_count == 2
            # Both sleeps should be 0.5 seconds
            calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert calls == [0.5, 0.5]

    def test_no_sleep_for_single_board(self):
        """Single board token does not trigger sleep."""
        with patch("src.collection.time.sleep") as mock_sleep:
            with patch.object(GreenhouseClient, "fetch_jobs", return_value=[]):
                client = GreenhouseClient(
                    ["board1"],
                    request_timeout=30,
                    max_retries=3,
                    retry_backoff=2.0,
                    request_delay_seconds=0.5
                )
                client.fetch_all_boards()

            # No sleep should occur with single board
            mock_sleep.assert_not_called()

    def test_zero_delay_does_not_sleep(self):
        """Zero request_delay_seconds skips sleep between submissions."""
        with patch("src.collection.time.sleep") as mock_sleep:
            with patch.object(GreenhouseClient, "fetch_jobs", return_value=[]):
                client = GreenhouseClient(
                    ["board1", "board2", "board3"],
                    request_timeout=30,
                    max_retries=3,
                    retry_backoff=2.0,
                    request_delay_seconds=0
                )
                client.fetch_all_boards()

            # No sleep should occur with zero delay
            mock_sleep.assert_not_called()

    def test_max_workers_parameter_stored(self):
        """max_workers parameter is stored in client instance."""
        client = GreenhouseClient(
            ["t1", "t2"],
            request_timeout=30,
            max_retries=3,
            retry_backoff=2.0,
            max_workers=5
        )
        assert client.max_workers == 5

    def test_request_delay_parameter_stored(self):
        """request_delay_seconds parameter is stored in client instance."""
        client = GreenhouseClient(
            ["t1", "t2"],
            request_timeout=30,
            max_retries=3,
            retry_backoff=2.0,
            request_delay_seconds=1.5
        )
        assert client.request_delay_seconds == 1.5
