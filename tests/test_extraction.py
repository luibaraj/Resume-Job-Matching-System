import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import jsonschema
import pytest

from src.database import DatabaseManager
from src.extraction import (
    EXTRACTION_JSON_SCHEMA,
    _extract_job_async,
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


def _make_gemini_response(content: str) -> MagicMock:
    response = MagicMock()
    response.text = content
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
    async def test_invalid_json_returns_none(self):
        client = MagicMock()
        client.aio = MagicMock()
        client.aio.models = MagicMock()
        client.aio.models.generate_content = AsyncMock(return_value=_make_gemini_response("not json {{"))
        semaphore = asyncio.Semaphore(10)
        result = await _extract_job_async((1, "desc", "title"), semaphore, client, self.MODEL_ID)
        assert result is None
        assert client.aio.models.generate_content.call_count == 1

    @pytest.mark.asyncio
    async def test_schema_mismatch_returns_none(self):
        client = MagicMock()
        client.aio = MagicMock()
        client.aio.models = MagicMock()
        bad = {k: v for k, v in VALID_EXTRACTION.items() if k != "job_title"}
        client.aio.models.generate_content = AsyncMock(return_value=_make_gemini_response(json.dumps(bad)))
        semaphore = asyncio.Semaphore(10)
        result = await _extract_job_async((1, "desc", "title"), semaphore, client, self.MODEL_ID)
        assert result is None

    @pytest.mark.asyncio
    async def test_model_exception_returns_none(self):
        client = MagicMock()
        client.aio = MagicMock()
        client.aio.models = MagicMock()
        client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("API error"))
        semaphore = asyncio.Semaphore(10)
        result = await _extract_job_async((1, "desc", "title"), semaphore, client, self.MODEL_ID)
        assert result is None

    @pytest.mark.asyncio
    async def test_none_description_returns_none(self):
        client = MagicMock()
        client.aio = MagicMock()
        client.aio.models = MagicMock()
        semaphore = asyncio.Semaphore(10)
        result = await _extract_job_async((1, None, "title"), semaphore, client, self.MODEL_ID)
        assert result is None
        client.aio.models.generate_content.assert_not_called()


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
