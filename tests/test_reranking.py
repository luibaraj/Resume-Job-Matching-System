import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.database import DatabaseManager
from src.reranking import build_job_text, score_pairs_batched, rerank, _format_input


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_job_with_match(
    db: DatabaseManager,
    greenhouse_id: int,
    score: float,
    rank: int,
    title: str = "Data Scientist",
    description: str = "Python and machine learning experience required.",
    company: str = "Acme Corp",
    model_id: str = "test-model",
) -> int:
    """Insert a job and a corresponding job_matches row. Returns job_id."""
    job = {
        "greenhouse_id": greenhouse_id,
        "board_token": "test-co",
        "title": title,
        "company": company,
        "location": "San Francisco, CA",
        "raw_description": "<p>Raw</p>",
        "absolute_url": f"https://example.com/jobs/{greenhouse_id}",
        "updated_at_source": "2026-01-01T00:00:00Z",
        "departments": '["Data"]',
        "offices": '["SF"]',
        "collected_at": "2026-01-01T00:00:00Z",
    }
    db.insert_job(job)
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
            "INSERT INTO job_matches (job_id, score, rank, model_id) VALUES (?, ?, ?, ?)",
            (job_id, score, rank, model_id),
        )
    return job_id


def _make_mock_model(batch_size: int, vocab_size: int = 1000, yes_logit: float = 5.0, no_logit: float = 0.0):
    """Return a mock causal LM model that outputs fixed yes/no logits."""
    def _forward(input_ids, **kwargs):
        B = input_ids.shape[0]
        logits = torch.zeros(B, 1, vocab_size)
        # yes token id = 1, no token id = 0 (matches _make_mock_tokenizer defaults)
        logits[:, 0, 1] = yes_logit
        logits[:, 0, 0] = no_logit
        output = MagicMock()
        output.logits = logits
        return output

    model = MagicMock(side_effect=_forward)
    model.eval.return_value = model
    return model


def _make_mock_tokenizer(yes_id: int = 1, no_id: int = 0):
    """Return a mock tokenizer."""
    def _call(texts, **kwargs):
        B = len(texts)
        return {
            "input_ids": torch.zeros(B, 10, dtype=torch.long),
            "attention_mask": torch.ones(B, 10, dtype=torch.long),
        }

    tokenizer = MagicMock(side_effect=_call)
    tokenizer.convert_tokens_to_ids.side_effect = lambda t: yes_id if t == "yes" else no_id
    return tokenizer


# ---------------------------------------------------------------------------
# TestBuildJobText
# ---------------------------------------------------------------------------


class TestBuildJobText:
    def test_combines_title_and_description(self):
        result = build_job_text("Data Scientist", "Python and ML required.")
        assert result == "Data Scientist\nPython and ML required."

    def test_strips_whitespace(self):
        result = build_job_text("  Engineer  ", "  Description  ")
        assert result == "Engineer  \n  Description"

    def test_empty_description(self):
        result = build_job_text("Engineer", "")
        assert result == "Engineer"

    def test_empty_title(self):
        result = build_job_text("", "Some description")
        assert result == "Some description"


# ---------------------------------------------------------------------------
# TestFormatInput
# ---------------------------------------------------------------------------


class TestFormatInput:
    def test_contains_query_and_passage(self):
        result = _format_input("my profile", "job description here")
        assert "my profile" in result
        assert "job description here" in result

    def test_contains_instruction(self):
        result = _format_input("q", "d")
        assert "<Instruct>" in result
        assert "<Query>" in result
        assert "<Document>" in result


# ---------------------------------------------------------------------------
# TestScorePairsBatched
# ---------------------------------------------------------------------------


class TestScorePairsBatched:
    def test_returns_one_score_per_passage(self):
        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(batch_size=4)
        passages = ["job A", "job B", "job C"]
        scores = score_pairs_batched(tokenizer, model, "my profile", passages, batch_size=4)
        assert len(scores) == 3

    def test_scores_are_floats_between_0_and_1(self):
        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(batch_size=4)
        passages = ["job A", "job B"]
        scores = score_pairs_batched(tokenizer, model, "query", passages, batch_size=4)
        for s in scores:
            assert isinstance(s, float)
            assert 0.0 <= s <= 1.0

    def test_batching_splits_correctly(self):
        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(batch_size=2)
        passages = ["a", "b", "c", "d", "e"]
        scores = score_pairs_batched(tokenizer, model, "query", passages, batch_size=2)
        assert len(scores) == 5

    def test_higher_yes_logit_gives_higher_score(self):
        # Two separate calls: one with high yes logit, one with low
        tok_high = _make_mock_tokenizer()
        model_high = _make_mock_model(batch_size=4, yes_logit=10.0, no_logit=0.0)
        score_high = score_pairs_batched(tok_high, model_high, "q", ["p"], batch_size=4)[0]

        tok_low = _make_mock_tokenizer()
        model_low = _make_mock_model(batch_size=4, yes_logit=0.0, no_logit=10.0)
        score_low = score_pairs_batched(tok_low, model_low, "q", ["p"], batch_size=4)[0]

        assert score_high > score_low

    def test_empty_passages_returns_empty(self):
        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(batch_size=4)
        scores = score_pairs_batched(tokenizer, model, "query", [], batch_size=4)
        assert scores == []


# ---------------------------------------------------------------------------
# TestRerank (integration with real SQLite, mocked model)
# ---------------------------------------------------------------------------


class TestRerank:
    def _make_config(self, profile_path: str, top_k: int = 3, batch_size: int = 4):
        config = MagicMock()
        config.retrieval_user_profile_path = profile_path
        config.reranking_model_id = "Qwen/Qwen3-Reranker-0.6B"
        config.reranking_top_k = top_k
        config.reranking_batch_size = batch_size
        return config

    def test_rerank_writes_to_db(self, db_manager, tmp_path):
        profile_file = tmp_path / "profile.txt"
        profile_file.write_text("Experienced data scientist with Python skills.")

        # Insert 3 candidates
        _insert_job_with_match(db_manager, 1, score=0.9, rank=1, title="DS Role")
        _insert_job_with_match(db_manager, 2, score=0.8, rank=2, title="ML Engineer")
        _insert_job_with_match(db_manager, 3, score=0.7, rank=3, title="Analyst")

        config = self._make_config(str(profile_file), top_k=3)

        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(batch_size=4)

        with patch("src.reranking.load_reranker", return_value=(tokenizer, model)):
            processed, skipped = rerank(db_manager, run_id=1, config=config)

        assert processed == 3
        assert skipped == 0

        results = db_manager.get_job_matches(limit=10)
        # job_reranked rows should exist
        with db_manager.get_connection() as conn:
            rows = conn.execute("SELECT job_id, rank FROM job_reranked ORDER BY rank").fetchall()
        assert len(rows) == 3
        ranks = [r[1] for r in rows]
        assert ranks == [1, 2, 3]

    def test_rerank_respects_top_k(self, db_manager, tmp_path):
        profile_file = tmp_path / "profile.txt"
        profile_file.write_text("Profile text.")

        for i in range(1, 6):
            _insert_job_with_match(db_manager, i, score=1.0 - i * 0.1, rank=i)

        config = self._make_config(str(profile_file), top_k=2)

        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(batch_size=4)

        with patch("src.reranking.load_reranker", return_value=(tokenizer, model)):
            processed, _ = rerank(db_manager, run_id=1, config=config)

        assert processed == 2

        with db_manager.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM job_reranked").fetchone()[0]
        assert count == 2

    def test_rerank_returns_zero_when_no_candidates(self, db_manager, tmp_path):
        profile_file = tmp_path / "profile.txt"
        profile_file.write_text("Profile text.")

        config = self._make_config(str(profile_file), top_k=5)

        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(batch_size=4)

        with patch("src.reranking.load_reranker", return_value=(tokenizer, model)):
            processed, skipped = rerank(db_manager, run_id=1, config=config)

        assert processed == 0
        assert skipped == 0

    def test_rerank_overwrites_previous_results(self, db_manager, tmp_path):
        profile_file = tmp_path / "profile.txt"
        profile_file.write_text("Profile text.")

        _insert_job_with_match(db_manager, 1, score=0.9, rank=1)
        _insert_job_with_match(db_manager, 2, score=0.8, rank=2)

        config = self._make_config(str(profile_file), top_k=2)

        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(batch_size=4)

        # Run twice — second run should replace first
        with patch("src.reranking.load_reranker", return_value=(tokenizer, model)):
            rerank(db_manager, run_id=1, config=config)
            rerank(db_manager, run_id=2, config=config)

        with db_manager.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM job_reranked").fetchone()[0]
        assert count == 2


# ---------------------------------------------------------------------------
# TestDatabaseMethods
# ---------------------------------------------------------------------------


class TestGetJobMatchesWithText:
    def test_returns_empty_when_no_matches(self, db_manager):
        results = db_manager.get_job_matches_with_text()
        assert results == []

    def test_returns_correct_fields(self, db_manager):
        _insert_job_with_match(
            db_manager, 1001, score=0.9, rank=1,
            title="Engineer", description="Build stuff.", company="Acme"
        )
        results = db_manager.get_job_matches_with_text()
        assert len(results) == 1
        job_id, title, desc, company = results[0]
        assert isinstance(job_id, int)
        assert title == "Engineer"
        assert desc == "Build stuff."
        assert company == "Acme"

    def test_limit_caps_results(self, db_manager):
        for i in range(1, 6):
            _insert_job_with_match(db_manager, i, score=0.9 - i * 0.1, rank=i)
        results = db_manager.get_job_matches_with_text(limit=2)
        assert len(results) == 2

    def test_ordered_by_rank(self, db_manager):
        _insert_job_with_match(db_manager, 1, score=0.5, rank=3)
        _insert_job_with_match(db_manager, 2, score=0.9, rank=1)
        _insert_job_with_match(db_manager, 3, score=0.7, rank=2)
        results = db_manager.get_job_matches_with_text()
        assert len(results) == 3
        # Each result is (job_id, title, desc, company); check job IDs come in rank order
        # We inserted rank=1 for greenhouse_id=2, rank=2 for id=3, rank=3 for id=1
        # So order should be job with rank 1 first
        with db_manager.get_connection() as conn:
            id_rank1 = conn.execute("SELECT id FROM jobs WHERE greenhouse_id=2").fetchone()[0]
        assert results[0][0] == id_rank1


class TestInsertReranked:
    def test_inserts_rows(self, db_manager):
        job_id = _insert_job_with_match(db_manager, 1, score=0.9, rank=1)
        db_manager.insert_reranked([(job_id, 0.95, 1, "reranker-model")])
        with db_manager.get_connection() as conn:
            row = conn.execute("SELECT score, rank, model_id FROM job_reranked WHERE job_id=?", (job_id,)).fetchone()
        assert row[0] == pytest.approx(0.95)
        assert row[1] == 1
        assert row[2] == "reranker-model"

    def test_replaces_existing_rows(self, db_manager):
        job_id1 = _insert_job_with_match(db_manager, 1, score=0.9, rank=1)
        job_id2 = _insert_job_with_match(db_manager, 2, score=0.8, rank=2)

        db_manager.insert_reranked([(job_id1, 0.9, 1, "m"), (job_id2, 0.8, 2, "m")])
        db_manager.insert_reranked([(job_id1, 0.95, 1, "m")])  # only 1 result now

        with db_manager.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM job_reranked").fetchone()[0]
        assert count == 1

    def test_no_op_on_empty(self, db_manager):
        db_manager.insert_reranked([])  # should not raise
        with db_manager.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM job_reranked").fetchone()[0]
        assert count == 0
