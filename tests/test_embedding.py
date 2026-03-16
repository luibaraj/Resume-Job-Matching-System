import json
import queue
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.database import DatabaseManager
from src.embedding import (
    _db_writer_thread,
    build_embedding_string,
    embed_jobs,
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

DIM = 8  # Small vector dimension used in all mocked encode() calls


def _make_fake_vectors(n: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(42)
    vecs = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _insert_extracted_job(db: DatabaseManager, greenhouse_id: int = 9001) -> int:
    """Insert a preprocessed + extracted job and return its DB id."""
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
            "UPDATE jobs SET cleaned_description=?, preprocessed=1, extracted=1, is_target_role=1 WHERE greenhouse_id=?",
            ("Machine learning engineer role requiring Python.", greenhouse_id),
        )
        row = conn.execute(
            "SELECT id FROM jobs WHERE greenhouse_id=?", (greenhouse_id,)
        ).fetchone()
        job_id = row[0]
        conn.execute(
            """
            INSERT OR REPLACE INTO job_extractions
                (job_id, job_title, responsibilities, skills, tools_and_platforms,
                 education, experience_min_years, experience_is_inferred)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                VALID_EXTRACTION["job_title"],
                json.dumps(VALID_EXTRACTION["responsibilities"]),
                json.dumps(VALID_EXTRACTION["skills"]),
                json.dumps(VALID_EXTRACTION["tools_and_platforms"]),
                VALID_EXTRACTION["education"],
                VALID_EXTRACTION["experience"]["min_years"],
                int(VALID_EXTRACTION["experience"]["is_inferred"]),
            ),
        )
    return job_id


# ---------------------------------------------------------------------------
# DB methods
# ---------------------------------------------------------------------------


class TestGetUnembeddedJobsChunked:
    def test_excludes_unextracted_jobs(self, db_manager):
        base = {
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
        db_manager.insert_job(base)
        result = db_manager.get_unembedded_jobs_chunked(10, 0)
        assert result == []

    def test_returns_extracted_unembedded_jobs(self, db_manager):
        job_id = _insert_extracted_job(db_manager)
        result = db_manager.get_unembedded_jobs_chunked(10, 0)
        assert len(result) == 1
        assert result[0][0] == job_id

    def test_excludes_already_embedded(self, db_manager):
        job_id = _insert_extracted_job(db_manager)
        with db_manager.get_connection() as conn:
            conn.execute("UPDATE jobs SET embedded=1 WHERE id=?", (job_id,))
        result = db_manager.get_unembedded_jobs_chunked(10, 0)
        assert result == []

    def test_returns_plain_tuples(self, db_manager):
        _insert_extracted_job(db_manager)
        result = db_manager.get_unembedded_jobs_chunked(10, 0)
        assert isinstance(result[0], tuple)

    def test_chunk_size_limits_results(self, db_manager):
        for i in range(5):
            _insert_extracted_job(db_manager, greenhouse_id=2000 + i)
        result = db_manager.get_unembedded_jobs_chunked(2, 0)
        assert len(result) == 2

    def test_returns_all_extracted_fields(self, db_manager):
        _insert_extracted_job(db_manager)
        result = db_manager.get_unembedded_jobs_chunked(10, 0)
        job_id, job_title, responsibilities, skills, tools = result[0]
        assert job_title == "Software Engineer"
        assert json.loads(responsibilities) == VALID_EXTRACTION["responsibilities"]
        assert json.loads(skills) == VALID_EXTRACTION["skills"]
        assert json.loads(tools) == VALID_EXTRACTION["tools_and_platforms"]


class TestInsertEmbeddingsBatch:
    def test_writes_blob_to_job_embeddings(self, db_manager):
        job_id = _insert_extracted_job(db_manager)
        blob = _make_fake_vectors(1)[0].tobytes()
        db_manager.insert_embeddings_batch([(job_id, blob, "test-model")])

        with db_manager.get_connection() as conn:
            row = conn.execute(
                "SELECT embedding, model_id FROM job_embeddings WHERE job_id=?", (job_id,)
            ).fetchone()
        assert row is not None
        assert row["model_id"] == "test-model"
        assert len(row["embedding"]) == len(blob)

    def test_sets_embedded_flag_on_jobs(self, db_manager):
        job_id = _insert_extracted_job(db_manager)
        blob = _make_fake_vectors(1)[0].tobytes()
        db_manager.insert_embeddings_batch([(job_id, blob, "test-model")])

        with db_manager.get_connection() as conn:
            row = conn.execute("SELECT embedded FROM jobs WHERE id=?", (job_id,)).fetchone()
        assert row["embedded"] == 1

    def test_empty_updates_is_noop(self, db_manager):
        db_manager.insert_embeddings_batch([])
        with db_manager.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0]
        assert count == 0

    def test_upsert_on_duplicate_job_id(self, db_manager):
        job_id = _insert_extracted_job(db_manager)
        blob1 = _make_fake_vectors(1)[0].tobytes()
        blob2 = _make_fake_vectors(1, dim=4)[0].tobytes()
        db_manager.insert_embeddings_batch([(job_id, blob1, "model-v1")])
        db_manager.insert_embeddings_batch([(job_id, blob2, "model-v2")])

        with db_manager.get_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM job_embeddings WHERE job_id=?", (job_id,)
            ).fetchone()[0]
            row = conn.execute(
                "SELECT model_id FROM job_embeddings WHERE job_id=?", (job_id,)
            ).fetchone()
        assert count == 1
        assert row["model_id"] == "model-v2"


# ---------------------------------------------------------------------------
# build_embedding_string
# ---------------------------------------------------------------------------


class TestBuildEmbeddingString:
    def _make_record(self, job_id=1, title="SWE", resps=None, skills=None, tools=None):
        return (
            job_id,
            title,
            json.dumps(resps or []),
            json.dumps(skills or []),
            json.dumps(tools or []),
        )

    def test_joins_all_fields(self):
        record = self._make_record(
            title="Software Engineer",
            resps=["Design systems"],
            skills=["Python"],
            tools=["AWS"],
        )
        result = build_embedding_string(record)
        assert result is not None
        job_id, text = result
        assert job_id == 1
        assert "Software Engineer" in text
        assert "Design systems" in text
        assert "Python" in text
        assert "AWS" in text

    def test_empty_arrays_still_produces_title(self):
        record = self._make_record(title="Data Analyst")
        result = build_embedding_string(record)
        assert result is not None
        _, text = result
        assert "Data Analyst" in text

    def test_none_all_fields_returns_none(self):
        record = (42, None, json.dumps([]), json.dumps([]), json.dumps([]))
        result = build_embedding_string(record)
        assert result is None

    def test_returns_tuple_with_job_id(self):
        record = self._make_record(job_id=99, title="ML Engineer")
        result = build_embedding_string(record)
        assert result is not None
        assert result[0] == 99
        assert isinstance(result[1], str)


# ---------------------------------------------------------------------------
# _db_writer_thread
# ---------------------------------------------------------------------------


class TestDbWriterThread:
    def test_processes_batch_and_exits_on_sentinel(self, db_manager):
        job_id = _insert_extracted_job(db_manager, greenhouse_id=9001)
        blob = _make_fake_vectors(1)[0].tobytes()
        q = queue.Queue()
        error_event = threading.Event()
        q.put([(job_id, blob, "test-model")])
        q.put(None)

        t = threading.Thread(target=_db_writer_thread, args=(db_manager, q, error_event))
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        assert not error_event.is_set()
        with db_manager.get_connection() as conn:
            row = conn.execute("SELECT embedded FROM jobs WHERE id=?", (job_id,)).fetchone()
        assert row["embedded"] == 1

    def test_sets_error_event_on_db_exception(self, db_manager):
        q = queue.Queue()
        error_event = threading.Event()

        bad_db = MagicMock()
        bad_db.insert_embeddings_batch.side_effect = Exception("DB exploded")

        blob = _make_fake_vectors(1)[0].tobytes()
        q.put([(999, blob, "test-model")])
        q.put(None)

        t = threading.Thread(target=_db_writer_thread, args=(bad_db, q, error_event))
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        assert error_event.is_set()


# ---------------------------------------------------------------------------
# embed_jobs integration loop
# ---------------------------------------------------------------------------


class TestEmbedJobsLoop:
    def _make_mock_model(self, n_jobs: int):
        model = MagicMock()
        model.encode.return_value = _make_fake_vectors(n_jobs)
        return model

    def test_processes_all_jobs(self, db_manager):
        for i in range(3):
            _insert_extracted_job(db_manager, greenhouse_id=3000 + i)

        with patch("src.embedding.load_model", return_value=self._make_mock_model(3)):
            processed, errors = embed_jobs(
                db_manager, run_id=1, chunk_size=10, batch_size=4,
                model_id="test-model", num_workers=1,
            )

        assert processed == 3
        assert errors == 0
        with db_manager.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM jobs WHERE embedded=1").fetchone()[0]
            emb_count = conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0]
        assert count == 3
        assert emb_count == 3

    def test_idempotent_second_run(self, db_manager):
        _insert_extracted_job(db_manager, greenhouse_id=4000)

        with patch("src.embedding.load_model", return_value=self._make_mock_model(1)):
            embed_jobs(
                db_manager, run_id=1, chunk_size=10, batch_size=4,
                model_id="test-model", num_workers=1,
            )

        with patch("src.embedding.load_model", return_value=self._make_mock_model(0)):
            processed2, errors2 = embed_jobs(
                db_manager, run_id=2, chunk_size=10, batch_size=4,
                model_id="test-model", num_workers=1,
            )

        assert processed2 == 0
        assert errors2 == 0

    def test_embedding_failure_counted_as_error(self, db_manager):
        for i in range(3):
            _insert_extracted_job(db_manager, greenhouse_id=5000 + i)

        bad_model = MagicMock()
        bad_model.encode.side_effect = RuntimeError("OOM")

        with patch("src.embedding.load_model", return_value=bad_model):
            processed, errors = embed_jobs(
                db_manager, run_id=1, chunk_size=10, batch_size=4,
                model_id="test-model", num_workers=1, max_retries=0,
            )

        assert processed == 0
        assert errors == 3
        with db_manager.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM jobs WHERE embedded=1").fetchone()[0]
        assert count == 0
