import json
from unittest.mock import MagicMock, patch

import jsonschema
import pytest

from src.database import DatabaseManager
from src.extraction import (
    EXTRACTION_JSON_SCHEMA,
    extract_job,
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
            "UPDATE jobs SET cleaned_description=?, preprocessed=1 WHERE greenhouse_id=?",
            ("Machine learning engineer role requiring Python.", greenhouse_id),
        )
        row = conn.execute(
            "SELECT id FROM jobs WHERE greenhouse_id=?", (greenhouse_id,)
        ).fetchone()
        return row[0]


def _make_model_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


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
# extract_job with mocked Llama
# ---------------------------------------------------------------------------


class TestExtractJob:
    def test_success_returns_tuple(self):
        model = MagicMock()
        model.create_chat_completion.return_value = _make_model_response(
            json.dumps(VALID_EXTRACTION)
        )
        result = extract_job((42, "Some job description text", "Software Engineer"), model)
        assert result is not None
        job_id, data = result
        assert job_id == 42
        assert data["job_title"] == "Software Engineer"

    def test_invalid_json_returns_none(self):
        model = MagicMock()
        model.create_chat_completion.return_value = _make_model_response("not json {{")
        result = extract_job((1, "desc", "title"), model)
        assert result is None

    def test_schema_mismatch_returns_none(self):
        model = MagicMock()
        bad = {k: v for k, v in VALID_EXTRACTION.items() if k != "job_title"}
        model.create_chat_completion.return_value = _make_model_response(json.dumps(bad))
        result = extract_job((1, "desc", "title"), model)
        assert result is None

    def test_model_exception_returns_none(self):
        model = MagicMock()
        model.create_chat_completion.side_effect = RuntimeError("GPU OOM")
        result = extract_job((1, "desc", "title"), model)
        assert result is None

    def test_none_description_returns_none(self):
        model = MagicMock()
        result = extract_job((1, None, "title"), model)
        assert result is None
        model.create_chat_completion.assert_not_called()


# ---------------------------------------------------------------------------
# extract_jobs integration loop
# ---------------------------------------------------------------------------


class TestExtractJobsLoop:
    def _make_mock_model(self):
        model = MagicMock()
        model.create_chat_completion.return_value = _make_model_response(
            json.dumps(VALID_EXTRACTION)
        )
        return model

    def test_processes_all_jobs(self, db_manager):
        for i in range(3):
            _insert_preprocessed_job(db_manager, job_id_override=3000 + i)

        with patch("src.extraction.load_model", return_value=self._make_mock_model()):
            processed, errors = extract_jobs(
                db_manager, run_id=1, chunk_size=10,
                model_path="/fake/model.gguf", n_ctx=2048, n_gpu_layers=-1,
            )

        assert processed == 3
        assert errors == 0
        with db_manager.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM jobs WHERE extracted=1").fetchone()[0]
        assert count == 3

    def test_idempotent_second_run(self, db_manager):
        _insert_preprocessed_job(db_manager, job_id_override=4000)

        with patch("src.extraction.load_model", return_value=self._make_mock_model()):
            extract_jobs(
                db_manager, run_id=1, chunk_size=10,
                model_path="/fake/model.gguf", n_ctx=2048, n_gpu_layers=-1,
            )
            processed2, errors2 = extract_jobs(
                db_manager, run_id=2, chunk_size=10,
                model_path="/fake/model.gguf", n_ctx=2048, n_gpu_layers=-1,
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

        # Always fail — model raises on every call
        model = MagicMock()
        model.create_chat_completion.side_effect = RuntimeError("GPU OOM")

        with patch("src.extraction.load_model", return_value=model):
            processed, errors = extract_jobs(
                db_manager, run_id=1, chunk_size=10,
                model_path="/fake/model.gguf", n_ctx=2048, n_gpu_layers=-1,
                max_retries=0,
            )

        # All 3 jobs fail → 0 extracted, 3 errors counted, loop breaks to avoid
        # infinite re-processing of the same unextracted records.
        assert processed == 0
        assert errors == 3
        with db_manager.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM jobs WHERE extracted=1").fetchone()[0]
        assert count == 0
