import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import jsonschema
import pytest

from src.database import DatabaseManager
from src.extraction import (
    EXTRACTION_JSON_SCHEMA,
    _extract_job_async,
    _repair_json,
    extract_jobs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_EXTRACTION = {
    "job_title": "Software Engineer",
    "responsibilities": ["Design scalable systems", "Lead code reviews"],
    "skills": ["Python", "Distributed systems"],
    "tools_and_platforms": ["AWS", "Kubernetes"],
    "education": "BS",
    "experience": {"min_years": 3, "is_inferred": False},
}


def _insert_preprocessed_job(db: DatabaseManager, job_id_override=None) -> int:
    """Insert a preprocessed job and return its DB id."""
    greenhouse_id = job_id_override or 9001
    base = {
        "greenhouse_id": greenhouse_id,
        "board_token": "test-co",
        "title": "Software Engineer",
        "company": "test-co",
        "location": "San Francisco, CA",
        "raw_description": "<p>Raw</p>",
        "absolute_url": f"https://example.com/jobs/{greenhouse_id}",
        "updated_at_source": "2026-01-01T00:00:00Z",
        "departments": '["Engineering"]',
        "offices": '["SF"]',
        "collected_at": "2026-01-01T00:00:00Z",
    }
    db.insert_job(base)
    with db.get_connection() as conn:
        conn.execute(
            "UPDATE jobs SET cleaned_description=?, preprocessed=1, is_target_role=1 WHERE greenhouse_id=?",
            ("Machine learning engineer role requiring Python.", greenhouse_id),
        )
        row = conn.execute(
            "SELECT id FROM jobs WHERE greenhouse_id=?", (greenhouse_id,)
        ).fetchone()
        return row[0]


def _make_gemini_response(content: str, finish_reason: str = "STOP") -> MagicMock:
    response = MagicMock()
    response.text = content
    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    response.candidates = [candidate]
    return response


# ---------------------------------------------------------------------------
# DB methods
# ---------------------------------------------------------------------------


class TestGetUnextractedJobsChunked:
    def test_excludes_unpreprocessed_jobs(self, db_manager):
        raw = {
            "greenhouse_id": 1001,
            "board_token": "co",
            "title": "Analyst",
            "company": "co",
            "location": None,
            "raw_description": "<p>raw</p>",
            "absolute_url": "https://example.com",
            "updated_at_source": None,
            "departments": None,
            "offices": None,
            "collected_at": "2026-01-01T00:00:00Z",
        }
        db_manager.insert_job(raw)
        result = db_manager.get_unextracted_jobs_chunked(10, 0)
        assert result == []

    def test_returns_preprocessed_unextracted_jobs(self, db_manager):
        job_id = _insert_preprocessed_job(db_manager)
        result = db_manager.get_unextracted_jobs_chunked(10, 0)
        assert len(result) == 1
        assert result[0][0] == job_id
        assert result[0][1] is not None  # cleaned_description

    def test_excludes_already_extracted(self, db_manager):
        job_id = _insert_preprocessed_job(db_manager)
        with db_manager.get_connection() as conn:
            conn.execute("UPDATE jobs SET extracted=1 WHERE id=?", (job_id,))
        result = db_manager.get_unextracted_jobs_chunked(10, 0)
        assert result == []

    def test_returns_plain_tuples(self, db_manager):
        _insert_preprocessed_job(db_manager)
        result = db_manager.get_unextracted_jobs_chunked(10, 0)
        assert isinstance(result[0], tuple)

    def test_chunk_size_limits_results(self, db_manager):
        for i in range(5):
            _insert_preprocessed_job(db_manager, job_id_override=2000 + i)
        result = db_manager.get_unextracted_jobs_chunked(2, 0)
        assert len(result) == 2


class TestUpdateExtractionBatch:
    def test_writes_all_fields_to_job_extractions(self, db_manager):
        job_id = _insert_preprocessed_job(db_manager)
        db_manager.update_extraction_batch([(job_id, VALID_EXTRACTION)])

        with db_manager.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM job_extractions WHERE job_id=?", (job_id,)
            ).fetchone()

        assert row is not None
        assert row["job_title"] == "Software Engineer"
        assert json.loads(row["responsibilities"]) == ["Design scalable systems", "Lead code reviews"]
        assert json.loads(row["skills"]) == ["Python", "Distributed systems"]
        assert json.loads(row["tools_and_platforms"]) == ["AWS", "Kubernetes"]
        assert row["education"] == "BS"
        assert row["experience_min_years"] == 3
        assert row["experience_is_inferred"] == 0

    def test_sets_extracted_flag_on_jobs(self, db_manager):
        job_id = _insert_preprocessed_job(db_manager)
        db_manager.update_extraction_batch([(job_id, VALID_EXTRACTION)])

        with db_manager.get_connection() as conn:
            row = conn.execute("SELECT extracted FROM jobs WHERE id=?", (job_id,)).fetchone()
        assert row["extracted"] == 1

    def test_empty_updates_is_noop(self, db_manager):
        db_manager.update_extraction_batch([])
        with db_manager.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM job_extractions").fetchone()[0]
        assert count == 0

    def test_upsert_on_duplicate_job_id(self, db_manager):
        job_id = _insert_preprocessed_job(db_manager)
        db_manager.update_extraction_batch([(job_id, VALID_EXTRACTION)])
        updated = dict(VALID_EXTRACTION)
        updated["job_title"] = "Senior Engineer"
        db_manager.update_extraction_batch([(job_id, updated)])

        with db_manager.get_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM job_extractions WHERE job_id=?", (job_id,)
            ).fetchone()[0]
            title = conn.execute(
                "SELECT job_title FROM job_extractions WHERE job_id=?", (job_id,)
            ).fetchone()[0]
        assert count == 1
        assert title == "Senior Engineer"


# ---------------------------------------------------------------------------
# JSON schema validation
# ---------------------------------------------------------------------------


class TestJsonSchemaValidation:
    def test_valid_extraction_passes(self):
        jsonschema.validate(VALID_EXTRACTION, EXTRACTION_JSON_SCHEMA)

    def test_missing_required_field_raises(self):
        bad = {k: v for k, v in VALID_EXTRACTION.items() if k != "skills"}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, EXTRACTION_JSON_SCHEMA)

    def test_wrong_type_for_array_field_raises(self):
        bad = dict(VALID_EXTRACTION)
        bad["skills"] = "Python, ML"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, EXTRACTION_JSON_SCHEMA)

    def test_experience_missing_min_years_raises(self):
        bad = dict(VALID_EXTRACTION)
        bad["experience"] = {"is_inferred": False}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, EXTRACTION_JSON_SCHEMA)

    def test_empty_arrays_are_valid(self):
        minimal = dict(VALID_EXTRACTION)
        minimal["responsibilities"] = []
        minimal["skills"] = []
        minimal["tools_and_platforms"] = []
        jsonschema.validate(minimal, EXTRACTION_JSON_SCHEMA)


# ---------------------------------------------------------------------------
# _repair_json function
# ---------------------------------------------------------------------------


class TestRepairJson:
    def test_repair_valid_json_passthrough(self):
        """Well-formed JSON should parse without modification."""
        raw = json.dumps(VALID_EXTRACTION)
        result = _repair_json(raw)
        assert result == VALID_EXTRACTION

    def test_repair_missing_comma_between_fields(self):
        """Missing commas between fields should be repaired."""
        # Simulate missing comma between "job_title" and "responsibilities"
        malformed = '{"job_title":"Engineer""responsibilities":[],"skills":[],"tools_and_platforms":[],"education":"unknown","experience":{"min_years":-1,"is_inferred":true}}'
        result = _repair_json(malformed)
        assert result is not None
        assert result["job_title"] == "Engineer"
        assert result["responsibilities"] == []

    def test_repair_trailing_garbage(self):
        """JSON with trailing garbage should extract valid object."""
        valid = json.dumps(VALID_EXTRACTION)
        malformed = valid + "\nSome trailing text that is not JSON"
        result = _repair_json(malformed)
        assert result == VALID_EXTRACTION

    def test_repair_markdown_fences(self):
        """JSON wrapped in markdown code fences should be extracted."""
        valid = json.dumps(VALID_EXTRACTION)
        malformed = f"```json\n{valid}\n```"
        result = _repair_json(malformed)
        assert result == VALID_EXTRACTION

    def test_repair_markdown_fences_plain(self):
        """JSON wrapped in plain backticks should be extracted."""
        valid = json.dumps(VALID_EXTRACTION)
        malformed = f"```\n{valid}\n```"
        result = _repair_json(malformed)
        assert result == VALID_EXTRACTION

    def test_repair_truncated_object_returns_none(self):
        """Genuinely truncated JSON (incomplete braces) should return None."""
        # Missing closing braces and content cut off mid-string
        truncated = '{"job_title":"Engineer","responsibilities":["Design scalable","Lead code'
        result = _repair_json(truncated)
        assert result is None

    def test_repair_completely_invalid_returns_none(self):
        """Completely invalid input should return None."""
        result = _repair_json("not json {{")
        assert result is None

    def test_repair_missing_comma_after_closing_brace(self):
        """Repair pattern should match closing braces followed by strings."""
        # This has a missing comma after the experience object, before the next field
        malformed = '{"job_title":"Eng","responsibilities":[],"skills":[],"tools_and_platforms":[],"education":"BS","experience":{"min_years":3,"is_inferred":false}"required_field":true}'
        result = _repair_json(malformed)
        # The comma repair should fix this
        assert result is not None
        # After repair, should have the comma and parse successfully
        assert "job_title" in result
        assert result["job_title"] == "Eng"

    def test_repair_missing_comma_after_closing_bracket(self):
        """Repair pattern should match closing brackets followed by strings."""
        # Missing comma after array
        malformed = '{"job_title":"Eng","responsibilities":[]"skills":[],"tools_and_platforms":[],"education":"BS","experience":{"min_years":3,"is_inferred":false}}'
        result = _repair_json(malformed)
        assert result is not None
        assert result["job_title"] == "Eng"


# ---------------------------------------------------------------------------
# _extract_job_async with mocked Gemini client
# ---------------------------------------------------------------------------


class TestExtractJobAsync:
    MODEL_ID = "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_success_returns_tuple(self):
        client = MagicMock()
        client.aio = MagicMock()
        client.aio.models = MagicMock()
        client.aio.models.generate_content = AsyncMock(
            return_value=_make_gemini_response(json.dumps(VALID_EXTRACTION))
        )
        semaphore = asyncio.Semaphore(10)
        result = await _extract_job_async((42, "Some job description text", "Software Engineer"), semaphore, client, self.MODEL_ID)
        assert result is not None
        job_id, data = result
        assert job_id == 42
        assert data["job_title"] == "Software Engineer"
        assert client.aio.models.generate_content.call_count == 1

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error_tuple(self):
        client = MagicMock()
        client.aio = MagicMock()
        client.aio.models = MagicMock()
        client.aio.models.generate_content = AsyncMock(return_value=_make_gemini_response("not json {{"))
        semaphore = asyncio.Semaphore(10)
        result = await _extract_job_async((1, "desc", "title"), semaphore, client, self.MODEL_ID)
        assert result is not None
        assert len(result) == 4
        assert result[0] == 1
        assert result[1] is None
        assert result[2] == "json_parse_error"
        assert client.aio.models.generate_content.call_count == 1

    @pytest.mark.asyncio
    async def test_schema_mismatch_returns_error_tuple(self):
        client = MagicMock()
        client.aio = MagicMock()
        client.aio.models = MagicMock()
        bad = {k: v for k, v in VALID_EXTRACTION.items() if k != "job_title"}
        client.aio.models.generate_content = AsyncMock(return_value=_make_gemini_response(json.dumps(bad)))
        semaphore = asyncio.Semaphore(10)
        result = await _extract_job_async((1, "desc", "title"), semaphore, client, self.MODEL_ID)
        assert result is not None
        assert len(result) == 4
        assert result[0] == 1
        assert result[1] is None
        assert result[2] == "schema_validation_error"

    @pytest.mark.asyncio
    async def test_model_exception_returns_error_tuple(self):
        client = MagicMock()
        client.aio = MagicMock()
        client.aio.models = MagicMock()
        client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("API error"))
        semaphore = asyncio.Semaphore(10)
        result = await _extract_job_async((1, "desc", "title"), semaphore, client, self.MODEL_ID)
        assert result is not None
        assert len(result) == 4
        assert result[0] == 1
        assert result[1] is None
        assert result[2] == "api_error"

    @pytest.mark.asyncio
    async def test_none_description_returns_error_tuple(self):
        client = MagicMock()
        client.aio = MagicMock()
        client.aio.models = MagicMock()
        semaphore = asyncio.Semaphore(10)
        result = await _extract_job_async((1, None, "title"), semaphore, client, self.MODEL_ID)
        assert result is not None
        assert len(result) == 4
        assert result[0] == 1
        assert result[1] is None
        assert result[2] == "missing_description"
        client.aio.models.generate_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_tokens_finish_reason_logged(self, caplog):
        """MAX_TOKENS finish reason should be logged as warning, but still return valid result."""
        client = MagicMock()
        client.aio = MagicMock()
        client.aio.models = MagicMock()
        # Valid JSON but with MAX_TOKENS finish reason
        client.aio.models.generate_content = AsyncMock(
            return_value=_make_gemini_response(json.dumps(VALID_EXTRACTION), finish_reason="MAX_TOKENS")
        )
        semaphore = asyncio.Semaphore(10)
        with caplog.at_level("WARNING"):
            result = await _extract_job_async((42, "Some description", "Engineer"), semaphore, client, self.MODEL_ID)
        assert result is not None
        job_id, data = result
        assert job_id == 42
        assert data["job_title"] == "Software Engineer"
        assert "MAX_TOKENS" in caplog.text


# ---------------------------------------------------------------------------
# extract_jobs integration loop
# ---------------------------------------------------------------------------


class TestExtractJobsLoop:
    def _make_mock_client(self):
        client = MagicMock()
        client.aio = MagicMock()
        client.aio.models = MagicMock()
        client.aio.models.generate_content = AsyncMock(
            return_value=_make_gemini_response(json.dumps(VALID_EXTRACTION))
        )
        return client

    def test_processes_all_jobs(self, db_manager):
        for i in range(3):
            _insert_preprocessed_job(db_manager, job_id_override=3000 + i)

        mock_client = self._make_mock_client()

        def mock_chunk_run(coro):
            """Execute the async chunk coroutine synchronously."""
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        with patch("src.extraction.genai.Client", return_value=mock_client):
            with patch("src.extraction.asyncio.run", side_effect=mock_chunk_run):
                processed, errors = extract_jobs(
                    db_manager, run_id=1, chunk_size=10,
                    api_key="fake-key", model_id="gemini-2.5-flash",
                )

        assert processed == 3
        assert errors == 0
        with db_manager.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM jobs WHERE extracted=1").fetchone()[0]
        assert count == 3

    def test_idempotent_second_run(self, db_manager):
        _insert_preprocessed_job(db_manager, job_id_override=4000)

        mock_client = self._make_mock_client()

        def mock_chunk_run(coro):
            """Execute the async chunk coroutine synchronously."""
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        with patch("src.extraction.genai.Client", return_value=mock_client):
            with patch("src.extraction.asyncio.run", side_effect=mock_chunk_run):
                extract_jobs(
                    db_manager, run_id=1, chunk_size=10,
                    api_key="fake-key", model_id="gemini-2.5-flash",
                )
                processed2, errors2 = extract_jobs(
                    db_manager, run_id=2, chunk_size=10,
                    api_key="fake-key", model_id="gemini-2.5-flash",
                )

        assert processed2 == 0
        assert errors2 == 0

    def test_errors_counted_per_chunk(self, db_manager):
        """A failed extraction is counted as an error for that chunk.

        The failed record stays unextracted and will be retried on the next
        chunk iteration (offset-0 pattern), so total_errors reflects errors
        per-chunk, not permanently-failed records.
        """
        for i in range(3):
            _insert_preprocessed_job(db_manager, job_id_override=5000 + i)

        # Always fail — client raises on every call
        mock_client = MagicMock()
        mock_client.aio = MagicMock()
        mock_client.aio.models = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("API error"))

        def mock_chunk_run(coro):
            """Execute the async chunk coroutine synchronously."""
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        with patch("src.extraction.genai.Client", return_value=mock_client):
            with patch("src.extraction.asyncio.run", side_effect=mock_chunk_run):
                processed, errors = extract_jobs(
                    db_manager, run_id=1, chunk_size=10,
                    api_key="fake-key", model_id="gemini-2.5-flash",
                    max_retries=0,
                )

        # All 3 jobs fail → 0 extracted, 3 errors counted, loop breaks to avoid
        # infinite re-processing of the same unextracted records.
        assert processed == 0
        assert errors == 3
        with db_manager.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM jobs WHERE extracted=1").fetchone()[0]
        assert count == 0

    def test_retry_uses_exponential_backoff_sleep(self, db_manager):
        """Verify that retries use exponential backoff with time.sleep."""
        # Insert a job that will fail
        job_id = _insert_preprocessed_job(db_manager, job_id_override=6000)

        # Create a client that always fails
        client = MagicMock()
        client.aio = MagicMock()
        client.aio.models = MagicMock()
        client.aio.models.generate_content = AsyncMock(
            side_effect=RuntimeError("Transient failure")
        )

        def mock_chunk_run(coro):
            """Execute the async chunk coroutine synchronously."""
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        with patch("src.extraction.genai.Client", return_value=client):
            with patch("src.extraction.asyncio.run", side_effect=mock_chunk_run):
                with patch("src.extraction.asyncio.sleep") as mock_async_sleep:
                    processed, errors = extract_jobs(
                        db_manager, run_id=1, chunk_size=10,
                        api_key="fake-key", model_id="gemini-2.5-flash",
                        max_retries=2,
                    )

        # Should have slept with exponential backoff on both retries: 2^1 = 2 seconds, 2^2 = 4 seconds
        assert mock_async_sleep.call_count == 2
        calls = [call[0][0] for call in mock_async_sleep.call_args_list]
        assert calls == [2.0, 4.0]  # Exponential backoff: 2s then 4s


# ---------------------------------------------------------------------------
# Error persistence tests
# ---------------------------------------------------------------------------


class TestWriteExtractionErrorsBatch:
    """Test the write_extraction_errors_batch() method on DatabaseManager."""

    def test_writes_error_row_to_db(self, db_manager):
        """Verify that error records are persisted to job_extraction_errors."""
        job_id = _insert_preprocessed_job(db_manager)
        db_manager.write_extraction_errors_batch([
            (job_id, "api_error", "Connection timeout", 1)
        ])

        with db_manager.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM job_extraction_errors WHERE job_id=?", (job_id,)
            ).fetchone()

        assert row is not None
        assert row["job_id"] == job_id
        assert row["error_type"] == "api_error"
        assert row["error_message"] == "Connection timeout"
        assert row["attempt_count"] == 1

    def test_upsert_replaces_stale_error(self, db_manager):
        """Re-running for the same job_id should replace the old error row."""
        job_id = _insert_preprocessed_job(db_manager)
        db_manager.write_extraction_errors_batch([
            (job_id, "json_parse_error", "Invalid JSON", 1)
        ])
        # Second write for the same job_id with different error
        db_manager.write_extraction_errors_batch([
            (job_id, "schema_validation_error", "Missing field", 2)
        ])

        with db_manager.get_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM job_extraction_errors WHERE job_id=?", (job_id,)
            ).fetchone()[0]
            row = conn.execute(
                "SELECT error_type, error_message FROM job_extraction_errors WHERE job_id=?",
                (job_id,)
            ).fetchone()

        assert count == 1
        assert row["error_type"] == "schema_validation_error"
        assert row["error_message"] == "Missing field"

    def test_empty_list_is_noop(self, db_manager):
        """Calling with an empty list should not create any rows."""
        db_manager.write_extraction_errors_batch([])

        with db_manager.get_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM job_extraction_errors"
            ).fetchone()[0]
        assert count == 0

    def test_success_clears_error_row(self, db_manager):
        """When a job is successfully extracted, its error row should be deleted."""
        job_id = _insert_preprocessed_job(db_manager)
        # Write an error for this job
        db_manager.write_extraction_errors_batch([
            (job_id, "api_error", "Temporary failure", 1)
        ])

        # Verify error row exists
        with db_manager.get_connection() as conn:
            error_count_before = conn.execute(
                "SELECT COUNT(*) FROM job_extraction_errors WHERE job_id=?", (job_id,)
            ).fetchone()[0]
        assert error_count_before == 1

        # Now extract the job successfully
        db_manager.update_extraction_batch([(job_id, VALID_EXTRACTION)])

        # Verify error row is gone and extracted=1
        with db_manager.get_connection() as conn:
            error_count_after = conn.execute(
                "SELECT COUNT(*) FROM job_extraction_errors WHERE job_id=?", (job_id,)
            ).fetchone()[0]
            extracted = conn.execute(
                "SELECT extracted FROM jobs WHERE id=?", (job_id,)
            ).fetchone()[0]

        assert error_count_after == 0
        assert extracted == 1


class TestExtractionErrorPersistence:
    """Integration tests for error persistence in the extraction loop."""

    MODEL_ID = "gemini-2.5-flash"

    def _make_mock_client(self, side_effect=None, return_value=None):
        """Helper to create a mocked Gemini client."""
        client = MagicMock()
        client.aio = MagicMock()
        client.aio.models = MagicMock()
        if side_effect:
            client.aio.models.generate_content = AsyncMock(side_effect=side_effect)
        else:
            client.aio.models.generate_content = AsyncMock(
                return_value=return_value or _make_gemini_response(json.dumps(VALID_EXTRACTION))
            )
        return client

    def _mock_chunk_run(self, coro):
        """Execute async chunk coroutine synchronously."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_permanent_failure_writes_error_record(self, db_manager):
        """When extraction fails after retries, an error record should be persisted."""
        job_id = _insert_preprocessed_job(db_manager)

        # Create a client that always fails with RuntimeError
        mock_client = self._make_mock_client(side_effect=RuntimeError("API error"))

        with patch("src.extraction.genai.Client", return_value=mock_client):
            with patch("src.extraction.asyncio.run", side_effect=self._mock_chunk_run):
                extract_jobs(
                    db_manager, run_id=1, chunk_size=10,
                    api_key="fake-key", model_id=self.MODEL_ID,
                    max_retries=0,  # No retries; fail immediately and persist
                )

        # Verify error row was created
        with db_manager.get_connection() as conn:
            row = conn.execute(
                "SELECT error_type FROM job_extraction_errors WHERE job_id=?",
                (job_id,)
            ).fetchone()

        assert row is not None
        assert row["error_type"] == "api_error"

    def test_no_error_record_on_success(self, db_manager):
        """When extraction succeeds, no error record should exist."""
        job_id = _insert_preprocessed_job(db_manager)

        mock_client = self._make_mock_client()

        with patch("src.extraction.genai.Client", return_value=mock_client):
            with patch("src.extraction.asyncio.run", side_effect=self._mock_chunk_run):
                extract_jobs(
                    db_manager, run_id=1, chunk_size=10,
                    api_key="fake-key", model_id=self.MODEL_ID,
                )

        # Verify no error row exists
        with db_manager.get_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM job_extraction_errors"
            ).fetchone()[0]

        assert count == 0

    def test_retry_exhausted_error_count(self, db_manager):
        """Error record should have attempt_count = max_retries + 1."""
        job_id = _insert_preprocessed_job(db_manager)

        mock_client = self._make_mock_client(side_effect=RuntimeError("Persistent failure"))

        with patch("src.extraction.genai.Client", return_value=mock_client):
            with patch("src.extraction.asyncio.run", side_effect=self._mock_chunk_run):
                extract_jobs(
                    db_manager, run_id=1, chunk_size=10,
                    api_key="fake-key", model_id=self.MODEL_ID,
                    max_retries=2,
                )

        # Verify attempt_count = 3 (initial attempt + 2 retries)
        with db_manager.get_connection() as conn:
            row = conn.execute(
                "SELECT attempt_count FROM job_extraction_errors WHERE job_id=?",
                (job_id,)
            ).fetchone()

        assert row is not None
        assert row["attempt_count"] == 3
