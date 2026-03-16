import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.database import DatabaseManager
from src.retrieval import (
    build_bm25_index,
    build_user_embedding_string,
    dense_top_k,
    embed_user_profile,
    load_corpus_embeddings,
    load_user_profile,
    reciprocal_rank_fusion,
    retrieve,
    sparse_top_k,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8

VALID_EXTRACTION = {
    "job_title": "Data Scientist",
    "responsibilities": ["Build ML models", "Analyze data"],
    "skills": ["Python", "PyTorch"],
    "tools_and_platforms": ["AWS", "Pandas"],
    "education": "BS",
    "experience": {"min_years": 2, "is_inferred": False},
}


def _make_normalized_vectors(n: int, dim: int = DIM, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _insert_embedded_job(
    db: DatabaseManager,
    greenhouse_id: int,
    vector: np.ndarray,
    model_id: str = "test-model",
    description: str = "Machine learning engineer requiring Python and PyTorch.",
) -> int:
    """Insert a fully-processed job (preprocessed, extracted, embedded) and return its id."""
    base = {
        "greenhouse_id": greenhouse_id,
        "board_token": "test-co",
        "title": "Data Scientist",
        "company": "test-co",
        "location": "San Francisco, CA",
        "raw_description": "<p>Raw</p>",
        "absolute_url": f"https://example.com/jobs/{greenhouse_id}",
        "updated_at_source": "2026-01-01T00:00:00Z",
        "departments": '["Data"]',
        "offices": '["SF"]',
        "collected_at": "2026-01-01T00:00:00Z",
    }
    db.insert_job(base)
    with db.get_connection() as conn:
        conn.execute(
            "UPDATE jobs SET cleaned_description=?, preprocessed=1, extracted=1, embedded=1 WHERE greenhouse_id=?",
            (description, greenhouse_id),
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
        conn.execute(
            "INSERT OR REPLACE INTO job_embeddings (job_id, embedding, model_id) VALUES (?, ?, ?)",
            (job_id, vector.astype(np.float32).tobytes(), model_id),
        )
    return job_id


# ---------------------------------------------------------------------------
# TestGetAllEmbeddings
# ---------------------------------------------------------------------------


class TestGetAllEmbeddings:
    def test_returns_empty_when_no_embeddings(self, db_manager):
        result = db_manager.get_all_embeddings("test-model")
        assert result == []

    def test_returns_blob_for_matching_model_id(self, db_manager):
        vec = _make_normalized_vectors(1)[0]
        _insert_embedded_job(db_manager, 1001, vec, model_id="model-a")
        result = db_manager.get_all_embeddings("model-a")
        assert len(result) == 1

    def test_filters_by_model_id(self, db_manager):
        vec = _make_normalized_vectors(1)[0]
        _insert_embedded_job(db_manager, 1001, vec, model_id="model-a")
        result = db_manager.get_all_embeddings("model-b")
        assert result == []

    def test_returns_plain_tuples_of_int_and_bytes(self, db_manager):
        vec = _make_normalized_vectors(1)[0]
        _insert_embedded_job(db_manager, 1001, vec, model_id="test-model")
        result = db_manager.get_all_embeddings("test-model")
        assert isinstance(result[0], tuple)
        assert isinstance(result[0][0], int)
        assert isinstance(result[0][1], bytes)


# ---------------------------------------------------------------------------
# TestGetAllCleanedDescriptions
# ---------------------------------------------------------------------------


class TestGetAllCleanedDescriptions:
    def test_returns_empty_when_no_jobs(self, db_manager):
        result = db_manager.get_all_cleaned_descriptions()
        assert result == []

    def test_only_returns_embedded_jobs(self, db_manager):
        vecs = _make_normalized_vectors(2)
        _insert_embedded_job(db_manager, 1001, vecs[0])
        # Insert a job that is extracted but NOT embedded
        base = {
            "greenhouse_id": 1002,
            "board_token": "test-co",
            "title": "Analyst",
            "company": "test-co",
            "location": None,
            "raw_description": "<p>raw</p>",
            "absolute_url": "https://example.com",
            "updated_at_source": None,
            "departments": None,
            "offices": None,
            "collected_at": "2026-01-01T00:00:00Z",
        }
        db_manager.insert_job(base)
        with db_manager.get_connection() as conn:
            conn.execute(
                "UPDATE jobs SET cleaned_description='desc', preprocessed=1, extracted=1 WHERE greenhouse_id=1002"
            )
        result = db_manager.get_all_cleaned_descriptions()
        assert len(result) == 1

    def test_returns_correct_job_id_and_description(self, db_manager):
        vec = _make_normalized_vectors(1)[0]
        job_id = _insert_embedded_job(db_manager, 1001, vec, description="Python ML role")
        result = db_manager.get_all_cleaned_descriptions()
        assert result[0][0] == job_id
        assert result[0][1] == "Python ML role"


# ---------------------------------------------------------------------------
# TestInsertJobMatches
# ---------------------------------------------------------------------------


class TestInsertJobMatches:
    def test_writes_rows_to_table(self, db_manager):
        vec = _make_normalized_vectors(1)[0]
        job_id = _insert_embedded_job(db_manager, 1001, vec)
        db_manager.insert_job_matches([(job_id, 0.95, 1, "test-model")])
        with db_manager.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM job_matches").fetchone()[0]
        assert count == 1

    def test_empty_input_is_noop(self, db_manager):
        db_manager.insert_job_matches([])
        with db_manager.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM job_matches").fetchone()[0]
        assert count == 0

    def test_second_run_replaces_first(self, db_manager):
        vecs = _make_normalized_vectors(3)
        ids = [_insert_embedded_job(db_manager, 1000 + i, vecs[i]) for i in range(3)]
        db_manager.insert_job_matches([(ids[0], 0.9, 1, "m"), (ids[1], 0.8, 2, "m"), (ids[2], 0.7, 3, "m")])
        db_manager.insert_job_matches([(ids[0], 0.95, 1, "m")])
        with db_manager.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM job_matches").fetchone()[0]
        assert count == 1

    def test_get_job_matches_returns_rank_order(self, db_manager):
        vecs = _make_normalized_vectors(3)
        ids = [_insert_embedded_job(db_manager, 1000 + i, vecs[i]) for i in range(3)]
        db_manager.insert_job_matches([
            (ids[2], 0.7, 3, "m"),
            (ids[0], 0.9, 1, "m"),
            (ids[1], 0.8, 2, "m"),
        ])
        result = db_manager.get_job_matches()
        ranks = [r[2] for r in result]
        assert ranks == [1, 2, 3]

    def test_get_job_matches_respects_limit(self, db_manager):
        vecs = _make_normalized_vectors(3)
        ids = [_insert_embedded_job(db_manager, 1000 + i, vecs[i]) for i in range(3)]
        db_manager.insert_job_matches([(ids[i], 0.9 - i * 0.1, i + 1, "m") for i in range(3)])
        result = db_manager.get_job_matches(limit=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# TestBuildUserEmbeddingString
# ---------------------------------------------------------------------------


class TestBuildUserEmbeddingString:
    def test_strips_leading_trailing_whitespace(self):
        assert build_user_embedding_string("  hello  ") == "hello"

    def test_returns_full_content(self):
        text = "Python developer with ML experience\nPyTorch, FastAPI"
        assert build_user_embedding_string(text) == text

    def test_empty_string_returns_empty(self):
        assert build_user_embedding_string("   ") == ""


# ---------------------------------------------------------------------------
# TestLoadUserProfile
# ---------------------------------------------------------------------------


class TestLoadUserProfile:
    def test_reads_file_content(self, tmp_path):
        profile = tmp_path / "profile.txt"
        profile.write_text("Python ML developer", encoding="utf-8")
        assert load_user_profile(str(profile)) == "Python ML developer"

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_user_profile("/nonexistent/path/profile.txt")


# ---------------------------------------------------------------------------
# TestLoadCorpusEmbeddings
# ---------------------------------------------------------------------------


class TestLoadCorpusEmbeddings:
    def test_empty_corpus_returns_empty(self, db_manager):
        ids, matrix = load_corpus_embeddings(db_manager, "test-model")
        assert ids == []
        assert matrix.size == 0

    def test_deserializes_blob_correctly(self, db_manager):
        vec = _make_normalized_vectors(1)[0]
        job_id = _insert_embedded_job(db_manager, 1001, vec, model_id="test-model")
        ids, matrix = load_corpus_embeddings(db_manager, "test-model")
        assert ids == [job_id]
        np.testing.assert_allclose(matrix[0], vec, rtol=1e-5)

    def test_job_ids_align_with_matrix_rows(self, db_manager):
        vecs = _make_normalized_vectors(3)
        inserted_ids = [_insert_embedded_job(db_manager, 1000 + i, vecs[i]) for i in range(3)]
        ids, matrix = load_corpus_embeddings(db_manager, "test-model")
        assert ids == sorted(inserted_ids)
        for i, job_id in enumerate(ids):
            idx = inserted_ids.index(job_id)
            np.testing.assert_allclose(matrix[i], vecs[idx], rtol=1e-5)


# ---------------------------------------------------------------------------
# TestDenseTopK
# ---------------------------------------------------------------------------


class TestDenseTopK:
    def test_returns_top_k_results(self):
        vecs = _make_normalized_vectors(10)
        query = vecs[0]
        corpus = vecs
        ids = list(range(10))
        result = dense_top_k(query, corpus, ids, top_k=3)
        assert len(result) == 3

    def test_scores_are_non_increasing(self):
        vecs = _make_normalized_vectors(10)
        query = vecs[0]
        result = dense_top_k(query, vecs, list(range(10)), top_k=5)
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_rank_is_one_based(self):
        vecs = _make_normalized_vectors(5)
        result = dense_top_k(vecs[0], vecs, list(range(5)), top_k=5)
        ranks = [r[2] for r in result]
        assert ranks == list(range(1, 6))

    def test_top_k_clamped_to_corpus_size(self):
        vecs = _make_normalized_vectors(3)
        result = dense_top_k(vecs[0], vecs, list(range(3)), top_k=100)
        assert len(result) == 3

    def test_identical_vector_is_rank_one(self):
        vecs = _make_normalized_vectors(5)
        query = vecs[2].copy()
        result = dense_top_k(query, vecs, list(range(5)), top_k=5)
        assert result[0][0] == 2
        assert abs(result[0][1] - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# TestBuildBm25Index
# ---------------------------------------------------------------------------


class TestBuildBm25Index:
    def test_returns_correct_job_ids(self):
        job_texts = [(10, "Python machine learning role"), (20, "Java backend developer")]
        ids, bm25 = build_bm25_index(job_texts)
        assert ids == [10, 20]

    def test_bm25_object_is_non_none(self):
        job_texts = [(1, "some job description here")]
        _, bm25 = build_bm25_index(job_texts)
        assert bm25 is not None

    def test_handles_empty_description(self):
        job_texts = [(1, ""), (2, "Python developer")]
        ids, bm25 = build_bm25_index(job_texts)
        scores = bm25.get_scores(["python"])
        assert len(scores) == 2


# ---------------------------------------------------------------------------
# TestSparseTopK
# ---------------------------------------------------------------------------


class TestSparseTopK:
    def test_returns_top_k_results(self):
        job_texts = [(i, f"job description number {i} python ml") for i in range(10)]
        ids, bm25 = build_bm25_index(job_texts)
        result = sparse_top_k(bm25, ["python", "ml"], ids, top_k=3)
        assert len(result) == 3

    def test_scores_are_non_increasing(self):
        job_texts = [(i, f"data science machine learning python job {i}") for i in range(10)]
        ids, bm25 = build_bm25_index(job_texts)
        result = sparse_top_k(bm25, ["python", "machine", "learning"], ids, top_k=5)
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_exact_keyword_match_scores_higher(self):
        # Use enough docs so IDF is meaningful; job 1 is the only python/ML one
        job_texts = [
            (1, "python machine learning pytorch data scientist neural networks"),
            (2, "customer success manager sales operations revenue growth"),
            (3, "marketing coordinator social media content creation brand"),
            (4, "sales development representative outbound prospecting crm"),
            (5, "graphic designer adobe illustrator photoshop creative brand"),
        ]
        ids, bm25 = build_bm25_index(job_texts)
        result = sparse_top_k(bm25, ["python", "machine", "learning"], ids, top_k=5)
        assert result[0][0] == 1

    def test_top_k_clamped_to_corpus_size(self):
        job_texts = [(i, f"desc {i}") for i in range(3)]
        ids, bm25 = build_bm25_index(job_texts)
        result = sparse_top_k(bm25, ["desc"], ids, top_k=100)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# TestReciprocalRankFusion
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_job_in_both_lists_scores_higher(self):
        dense = [(1, 0.9, 1), (2, 0.8, 2), (3, 0.7, 3)]
        sparse = [(1, 5.0, 1), (4, 4.0, 2), (5, 3.0, 3)]
        result = reciprocal_rank_fusion(dense, sparse, rrf_k=60, top_k=5)
        scores_by_id = {r[0]: r[1] for r in result}
        # Job 1 appears in both lists, should score higher than job 2 (dense-only) and job 4 (sparse-only)
        assert scores_by_id[1] > scores_by_id.get(2, 0)
        assert scores_by_id[1] > scores_by_id.get(4, 0)

    def test_total_count_equals_top_k(self):
        dense = [(i, 1.0 - i * 0.1, i + 1) for i in range(5)]
        sparse = [(i + 5, 1.0 - i * 0.1, i + 1) for i in range(5)]
        result = reciprocal_rank_fusion(dense, sparse, rrf_k=60, top_k=3)
        assert len(result) == 3

    def test_rank_is_one_based(self):
        dense = [(1, 0.9, 1), (2, 0.8, 2)]
        sparse = [(1, 5.0, 1), (3, 4.0, 2)]
        result = reciprocal_rank_fusion(dense, sparse, rrf_k=60, top_k=3)
        ranks = [r[2] for r in result]
        assert ranks[0] == 1
        assert sorted(ranks) == list(range(1, len(ranks) + 1))

    def test_empty_dense_uses_sparse_only(self):
        sparse = [(10, 5.0, 1), (20, 4.0, 2)]
        result = reciprocal_rank_fusion([], sparse, rrf_k=60, top_k=2)
        ids = [r[0] for r in result]
        assert 10 in ids
        assert 20 in ids

    def test_top_k_clamped_when_fewer_candidates(self):
        dense = [(1, 0.9, 1)]
        sparse = [(2, 5.0, 1)]
        result = reciprocal_rank_fusion(dense, sparse, rrf_k=60, top_k=100)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# TestRetrieveIntegration
# ---------------------------------------------------------------------------


class TestRetrieveIntegration:
    def _make_config(self, profile_path: str, top_k: int = 3, rrf_k: int = 60):
        config = MagicMock()
        config.embedding_model_id = "test-model"
        config.retrieval_user_profile_path = profile_path
        config.retrieval_top_k = top_k
        config.retrieval_rrf_k = rrf_k
        return config

    def _make_mock_model(self, query_vec: np.ndarray):
        model = MagicMock()
        model.encode.return_value = query_vec.reshape(1, -1).astype(np.float32)
        return model

    def test_writes_matches_to_db(self, db_manager, tmp_path):
        profile = tmp_path / "profile.txt"
        profile.write_text("Python machine learning data science PyTorch", encoding="utf-8")

        vecs = _make_normalized_vectors(5)
        for i in range(5):
            _insert_embedded_job(db_manager, 1000 + i, vecs[i])

        config = self._make_config(str(profile), top_k=3)
        mock_model = self._make_mock_model(vecs[0])

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("src.retrieval.load_model", lambda _: mock_model)
            processed, skipped = retrieve(db_manager, run_id=1, config=config)

        assert processed == 3
        assert skipped == 0
        matches = db_manager.get_job_matches()
        assert len(matches) == 3

    def test_empty_corpus_returns_zero(self, db_manager, tmp_path):
        profile = tmp_path / "profile.txt"
        profile.write_text("Python developer", encoding="utf-8")
        config = self._make_config(str(profile))

        mock_model = self._make_mock_model(_make_normalized_vectors(1)[0])
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("src.retrieval.load_model", lambda _: mock_model)
            processed, skipped = retrieve(db_manager, run_id=1, config=config)

        assert processed == 0
        assert skipped == 0

    def test_idempotent_second_run(self, db_manager, tmp_path):
        profile = tmp_path / "profile.txt"
        profile.write_text("Machine learning engineer Python PyTorch", encoding="utf-8")

        vecs = _make_normalized_vectors(5)
        for i in range(5):
            _insert_embedded_job(db_manager, 1000 + i, vecs[i])

        config = self._make_config(str(profile), top_k=3)
        mock_model = self._make_mock_model(vecs[0])

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("src.retrieval.load_model", lambda _: mock_model)
            retrieve(db_manager, run_id=1, config=config)
            retrieve(db_manager, run_id=2, config=config)

        matches = db_manager.get_job_matches()
        assert len(matches) == 3

    def test_respects_top_k(self, db_manager, tmp_path):
        profile = tmp_path / "profile.txt"
        profile.write_text("Python data scientist", encoding="utf-8")

        vecs = _make_normalized_vectors(10)
        for i in range(10):
            _insert_embedded_job(db_manager, 1000 + i, vecs[i])

        config = self._make_config(str(profile), top_k=4)
        mock_model = self._make_mock_model(vecs[0])

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("src.retrieval.load_model", lambda _: mock_model)
            retrieve(db_manager, run_id=1, config=config)

        matches = db_manager.get_job_matches()
        assert len(matches) == 4
