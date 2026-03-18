"""Tests for src/evaluation.py NIAH evaluation pipeline."""

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import Config
from src.database import DatabaseManager
from src.evaluation import (
    SyntheticNeedle,
    EvalCase,
    RetrievedItem,
    JudgedItem,
    EvalResult,
    EvalReport,
    GOLDEN_NEEDLE_ID,
    ADVERSARIAL_NEEDLE_ID,
    save_needles_to_db,
    load_needles_from_db,
    save_needles_to_json,
    load_needles_from_json,
    _build_golden_needle_prompt,
    _build_adversarial_needle_prompt,
    generate_needles,
    passes_semantic_check,
    run_retrieval_with_needles,
    run_retrieval_phase,
    _judge_single_item,
    judge_retrieved_items,
    run_judge_phase,
    compute_ndcg_at_k,
    compute_metrics,
    aggregate_results,
    save_results_to_db,
    save_report,
)


class TestTask1EvalTablesCreated:
    """test_task1_eval_tables_created: Schema tables exist."""

    def test_task1_eval_tables_created(self):
        """After initialize_schema, eval_needles and eval_results tables exist."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name
        try:
            db = DatabaseManager(temp_db)
            db.initialize_schema()
            with db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='eval_needles'"
                )
                assert cursor.fetchone() is not None, "eval_needles table not created"

                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='eval_results'"
                )
                assert cursor.fetchone() is not None, "eval_results table not created"
        finally:
            os.unlink(temp_db)


class TestTask1ConfigEvalDefaults:
    """test_task1_config_eval_defaults: Config fields have correct defaults."""

    def test_task1_config_eval_defaults(self, monkeypatch):
        """Config instantiation with no eval env vars has correct defaults."""
        # Clear any existing eval env vars
        for key in [
            "ANTHROPIC_API_KEY",
            "EVAL_JUDGE_MODEL_ID",
            "EVAL_NEEDLE_GEN_MODEL_ID",
            "EVAL_TOP_K",
            "EVAL_NEEDLES_PATH",
            "EVAL_REPORT_PATH",
        ]:
            monkeypatch.delenv(key, raising=False)

        cfg = Config()
        assert cfg.anthropic_api_key == ""
        assert cfg.eval_judge_model_id == "claude-sonnet-4-6"
        assert cfg.eval_needle_gen_model_id == "gemini-2.5-flash"
        assert cfg.eval_top_k == 50
        assert cfg.eval_needles_path == "data/eval_needles.json"
        assert cfg.eval_report_path == "data/eval_report.json"


class TestTask2SyntheticNeedleFields:
    """test_task2_synthetic_needle_fields: SyntheticNeedle has all fields."""

    def test_task2_synthetic_needle_fields(self):
        """Assert SyntheticNeedle has all 7 fields."""
        needle = SyntheticNeedle(
            needle_id=GOLDEN_NEEDLE_ID,
            needle_type="golden",
            title="Senior ML Engineer",
            company="TechCorp",
            description="We are looking for...",
            deal_breaker=None,
            true_relevance=5,
        )
        assert needle.needle_id == GOLDEN_NEEDLE_ID
        assert needle.needle_type == "golden"
        assert needle.title == "Senior ML Engineer"
        assert needle.company == "TechCorp"
        assert needle.description == "We are looking for..."
        assert needle.deal_breaker is None
        assert needle.true_relevance == 5


class TestTask2EvalReportOverallPassTrue:
    """test_task2_eval_report_overall_pass_true: overall_pass returns True."""

    def test_task2_eval_report_overall_pass_true(self):
        """Assert overall_pass() returns True when all thresholds are met."""
        report = EvalReport(
            run_id="2026-03-17T12:00:00Z",
            eval_top_k=50,
            precision_at_k=20,
            n_cases=1,
            mean_recall_at_k=0.96,
            mean_mrr=0.85,
            mean_ndcg_at_k=0.90,
            mean_precision_at_k=0.75,
            thresholds_met={
                "recall_at_k": True,
                "mrr": True,
                "ndcg_at_k": True,
                "precision_at_k": True,
            },
            per_case=[],
            generator_model="gemini-2.5-flash",
            judge_model="claude-sonnet-4-6",
        )
        assert report.overall_pass() is True


class TestTask2EvalReportOverallPassFalse:
    """test_task2_eval_report_overall_pass_false: overall_pass returns False."""

    def test_task2_eval_report_overall_pass_false(self):
        """Assert overall_pass() returns False when one threshold is not met."""
        report = EvalReport(
            run_id="2026-03-17T12:00:00Z",
            eval_top_k=50,
            precision_at_k=20,
            n_cases=1,
            mean_recall_at_k=0.90,
            mean_mrr=0.85,
            mean_ndcg_at_k=0.90,
            mean_precision_at_k=0.75,
            thresholds_met={
                "recall_at_k": False,  # Below target 0.95
                "mrr": True,
                "ndcg_at_k": True,
                "precision_at_k": True,
            },
            per_case=[],
            generator_model="gemini-2.5-flash",
            judge_model="claude-sonnet-4-6",
        )
        assert report.overall_pass() is False


class TestTask2EvalReportAsDict:
    """test_task2_eval_report_as_dict: as_dict() returns plain dict."""

    def test_task2_eval_report_as_dict(self):
        """Assert as_dict() returns plain dict with expected keys."""
        report = EvalReport(
            run_id="2026-03-17T12:00:00Z",
            eval_top_k=50,
            precision_at_k=20,
            n_cases=1,
            mean_recall_at_k=0.96,
            mean_mrr=0.85,
            mean_ndcg_at_k=0.90,
            mean_precision_at_k=0.75,
            thresholds_met={"recall_at_k": True},
            per_case=[],
            generator_model="gemini-2.5-flash",
            judge_model="claude-sonnet-4-6",
        )
        result_dict = report.as_dict()
        assert isinstance(result_dict, dict)
        assert "judge_model" in result_dict
        assert result_dict["judge_model"] == "claude-sonnet-4-6"
        assert result_dict["run_id"] == "2026-03-17T12:00:00Z"


class TestTask3DbRoundtrip:
    """test_task3_db_roundtrip: Save and load EvalCase from DB."""

    def test_task3_db_roundtrip(self):
        """Save 2 EvalCase objects to in-memory DB, load them back, assert equality."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name
        try:
            db = DatabaseManager(temp_db)
            db.initialize_schema()

            # Create two test cases
            golden1 = SyntheticNeedle(
                needle_id=GOLDEN_NEEDLE_ID,
                needle_type="golden",
                title="Senior ML Engineer",
                company="TechCorp",
                description="We are looking for...",
                deal_breaker=None,
                true_relevance=5,
            )
            adversarial1 = SyntheticNeedle(
                needle_id=ADVERSARIAL_NEEDLE_ID,
                needle_type="adversarial",
                title="ML Intern",
                company="SmallCorp",
                description="We are looking for...",
                deal_breaker="Requires 10+ years experience",
                true_relevance=0,
            )
            case1 = EvalCase(resume_id="user_profile_1", resume_text="Resume 1 text", golden=golden1, adversarial=adversarial1)

            golden2 = SyntheticNeedle(
                needle_id=GOLDEN_NEEDLE_ID,
                needle_type="golden",
                title="Data Scientist",
                company="DataCorp",
                description="Looking for DS...",
                deal_breaker=None,
                true_relevance=5,
            )
            adversarial2 = SyntheticNeedle(
                needle_id=ADVERSARIAL_NEEDLE_ID,
                needle_type="adversarial",
                title="Data Analyst",
                company="SmallData",
                description="Looking for DA...",
                deal_breaker="Requires 15 years SQL experience",
                true_relevance=0,
            )
            case2 = EvalCase(resume_id="user_profile_2", resume_text="Resume 2 text", golden=golden2, adversarial=adversarial2)

            cases = [case1, case2]

            # Save to DB
            save_needles_to_db(db, cases, "gemini-2.5-flash")

            # Load from DB
            loaded_cases = load_needles_from_db(db)

            # Assert equality
            assert len(loaded_cases) == 2
            assert loaded_cases[0].resume_id == "user_profile_1"
            assert loaded_cases[0].resume_text == "Resume 1 text"
            assert loaded_cases[0].golden.title == "Senior ML Engineer"
            assert loaded_cases[0].golden.company == "TechCorp"
            assert loaded_cases[0].golden.true_relevance == 5
            assert loaded_cases[0].adversarial.deal_breaker == "Requires 10+ years experience"
            assert loaded_cases[0].adversarial.true_relevance == 0

            assert loaded_cases[1].resume_id == "user_profile_2"
            assert loaded_cases[1].resume_text == "Resume 2 text"
            assert loaded_cases[1].golden.title == "Data Scientist"
            assert loaded_cases[1].adversarial.deal_breaker == "Requires 15 years SQL experience"
        finally:
            os.unlink(temp_db)


class TestTask3JsonRoundtrip:
    """test_task3_json_roundtrip: Save and load EvalCase from JSON."""

    def test_task3_json_roundtrip(self):
        """save_needles_to_json to a tmp file, load_needles_from_json, assert equal."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_json = f.name

        try:
            # Create test cases
            golden = SyntheticNeedle(
                needle_id=GOLDEN_NEEDLE_ID,
                needle_type="golden",
                title="Senior ML Engineer",
                company="TechCorp",
                description="We are looking for...",
                deal_breaker=None,
                true_relevance=5,
            )
            adversarial = SyntheticNeedle(
                needle_id=ADVERSARIAL_NEEDLE_ID,
                needle_type="adversarial",
                title="ML Intern",
                company="SmallCorp",
                description="We are looking for...",
                deal_breaker="Requires UK work authorization",
                true_relevance=0,
            )
            case = EvalCase(resume_id="user_profile", resume_text="Resume text", golden=golden, adversarial=adversarial)

            # Save to JSON
            save_needles_to_json([case], temp_json)

            # Load from JSON
            loaded_cases = load_needles_from_json(temp_json)

            # Assert equality
            assert len(loaded_cases) == 1
            assert loaded_cases[0].resume_id == "user_profile"
            assert loaded_cases[0].resume_text == "Resume text"
            assert loaded_cases[0].golden.title == "Senior ML Engineer"
            assert loaded_cases[0].golden.company == "TechCorp"
            assert loaded_cases[0].golden.deal_breaker is None
            assert loaded_cases[0].adversarial.deal_breaker == "Requires UK work authorization"
            assert loaded_cases[0].adversarial.true_relevance == 0
        finally:
            os.unlink(temp_json)


class TestTask3LoadMissingJsonReturnsEmpty:
    """test_task3_load_missing_json_returns_empty: Missing JSON returns []."""

    def test_task3_load_missing_json_returns_empty(self):
        """call load_needles_from_json on a non-existent path, assert returns []."""
        result = load_needles_from_json("/nonexistent/path/to/eval_needles.json")
        assert result == []


class TestTask4PromptEnforcesLinguisticDistance:
    """test_task4_prompt_enforces_linguistic_distance: Golden prompt includes paraphrase constraint."""

    def test_task4_prompt_enforces_linguistic_distance(self):
        """Call _build_golden_needle_prompt, assert it contains 'verbatim' or 'paraphrase'."""
        from src.evaluation import _build_golden_needle_prompt

        features = {"hard_labels": ["Python"], "soft_labels": ["built scalable systems"], "seniority": "senior", "domain": "ml-engineering"}
        forbidden_words = ["built", "scalable", "systems"]
        prompt = _build_golden_needle_prompt(features, forbidden_words)
        # Verify the constraint is in the prompt
        assert ("verbatim" in prompt.lower() or "paraphrase" in prompt.lower()), (
            f"Expected 'verbatim' or 'paraphrase' in prompt, got: {prompt}"
        )
        assert "DO NOT" in prompt or "do not" in prompt, (
            f"Expected negation constraint in prompt, got: {prompt}"
        )


class TestTask4GenerateNeedlesParseResponse:
    """test_task4_generate_needles_parses_response: Mocked API returns valid JSON."""

    def test_task4_generate_needles_parses_response(self):
        """Mock Gemini to return valid JSON; assert EvalCase has golden and adversarial."""
        from src.evaluation import generate_needles
        import logging
        from unittest.mock import patch

        logger = logging.getLogger("test")

        # Mock Gemini client
        gemini_client = MagicMock()

        # Mock response for golden needle
        golden_response = MagicMock()
        golden_response.text = '{"title": "Senior ML Engineer", "company": "TechCorp", "description": "Leading ML initiatives..."}'

        # Mock response for adversarial needle
        adversarial_response = MagicMock()
        adversarial_response.text = '{"title": "Senior ML Engineer", "company": "TechCorp", "description": "Leading ML initiatives...", "deal_breaker": "Requires 15+ years experience"}'

        # Setup side effects for two calls (golden + adversarial)
        gemini_client.models.generate_content.side_effect = [golden_response, adversarial_response]

        # Patch extract_resume_features and passes_semantic_check to bypass LLM calls
        mock_features = {"hard_labels": ["Python"], "soft_labels": ["led teams"], "seniority": "senior", "domain": "ml-engineering"}
        with patch("src.evaluation.extract_resume_features", return_value=mock_features), \
             patch("src.evaluation.passes_semantic_check", return_value=(True, 0.8)):
            case = generate_needles(
                resume_text="Data scientist with 5 years experience in Python.",
                resume_id="test_user",
                gemini_client=gemini_client,
                model_id="gemini-2.5-flash",
                logger=logger,
                max_retries=3,
            )

            # Assert structure
            assert case.resume_id == "test_user"
            assert case.golden.needle_type == "golden"
            assert case.golden.true_relevance == 5
            assert len(case.golden.description) > 0
            assert case.adversarial.needle_type == "adversarial"
            assert case.adversarial.true_relevance == 0
            assert case.adversarial.deal_breaker == "Requires 15+ years experience"
            assert case.adversarial.description == "Leading ML initiatives..."


class TestTask4GenerateNeedlesRetriesOnBadJson:
    """test_task4_generate_needles_retries_on_bad_json: Retries on invalid JSON."""

    def test_task4_generate_needles_retries_on_bad_json(self):
        """Mock Gemini to return {} first, then valid JSON; assert retried."""
        from src.evaluation import generate_needles
        import logging
        from unittest.mock import patch

        logger = logging.getLogger("test")

        # Mock Gemini client
        gemini_client = MagicMock()

        # First call: bad JSON (missing keys)
        bad_response = MagicMock()
        bad_response.text = "{}"

        # Second call: valid golden JSON
        good_golden = MagicMock()
        good_golden.text = '{"title": "Senior ML Engineer", "company": "TechCorp", "description": "Leading ML initiatives..."}'

        # Third call: valid adversarial JSON
        good_adversarial = MagicMock()
        good_adversarial.text = '{"title": "Senior ML Engineer", "company": "TechCorp", "description": "Leading ML initiatives...", "deal_breaker": "Requires 15+ years"}'

        # Setup side effects: bad response once for golden, then good responses
        gemini_client.models.generate_content.side_effect = [bad_response, good_golden, good_adversarial]

        mock_features = {"hard_labels": ["Python"], "soft_labels": ["led teams"], "seniority": "senior", "domain": "ml-engineering"}
        with patch("src.evaluation.extract_resume_features", return_value=mock_features), \
             patch("src.evaluation.passes_semantic_check", return_value=(True, 0.8)):
            case = generate_needles(
                resume_text="Data scientist with 5 years experience.",
                resume_id="test_user",
                gemini_client=gemini_client,
                model_id="gemini-2.5-flash",
                logger=logger,
                max_retries=3,
            )

        # Assert it succeeded after retry
        assert case.golden.title == "Senior ML Engineer"
        # Verify generate_content was called at least twice (bad + good for golden)
        assert gemini_client.models.generate_content.call_count >= 2


class TestTask4GenerateNeedlesRaisesAfterMaxRetries:
    """test_task4_generate_needles_raises_after_max_retries: Exhausted retries → RuntimeError."""

    def test_task4_generate_needles_raises_after_max_retries(self):
        """Mock Gemini to always return {}; assert RuntimeError raised."""
        from src.evaluation import generate_needles
        import logging
        from unittest.mock import patch

        logger = logging.getLogger("test")

        # Mock Gemini client
        gemini_client = MagicMock()

        # Always return invalid JSON
        bad_response = MagicMock()
        bad_response.text = "{}"
        gemini_client.models.generate_content.return_value = bad_response

        mock_features = {"hard_labels": ["Python"], "soft_labels": ["led teams"], "seniority": "senior", "domain": "ml-engineering"}
        # Should raise RuntimeError after max_retries
        with patch("src.evaluation.extract_resume_features", return_value=mock_features), \
             patch("src.evaluation.passes_semantic_check", return_value=(True, 0.8)):
            with pytest.raises(RuntimeError, match="Failed to generate golden needle"):
                generate_needles(
                    resume_text="Data scientist.",
                    resume_id="test_user",
                    gemini_client=gemini_client,
                    model_id="gemini-2.5-flash",
                    logger=logger,
                    max_retries=2,
                )


class TestTask5GoldenNeedleFlagged:
    """test_task5_golden_needle_flagged: Golden needle is marked is_needle=True."""

    def test_task5_golden_needle_flagged(self, monkeypatch):
        """Mock retrieval to return golden needle; assert is_needle=True, needle_type='golden'."""
        import logging
        import numpy as np
        from unittest.mock import patch

        logger = logging.getLogger("test")

        # Create eval case
        golden = SyntheticNeedle(
            needle_id=GOLDEN_NEEDLE_ID,
            needle_type="golden",
            title="Senior ML Engineer",
            company="TechCorp",
            description="We are looking for...",
            deal_breaker=None,
            true_relevance=5,
        )
        adversarial = SyntheticNeedle(
            needle_id=ADVERSARIAL_NEEDLE_ID,
            needle_type="adversarial",
            title="ML Intern",
            company="SmallCorp",
            description="We are looking for...",
            deal_breaker="Requires 10+ years",
            true_relevance=0,
        )
        case = EvalCase(resume_id="user_profile", resume_text="Resume text", golden=golden, adversarial=adversarial)

        # Mock DB
        db = MagicMock()
        db.get_connection.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
            (1, "Job Title", "Job description")
        ]
        db.get_all_cleaned_descriptions.return_value = [(1, "Job description")]

        # Mock config
        cfg = MagicMock()
        cfg.embedding_model_id = "test-model"
        cfg.eval_top_k = 50
        cfg.retrieval_rrf_k = 60

        # Mock embedding model
        embedding_model = MagicMock()
        embedding_model.encode.return_value = np.ones((1, 384), dtype=np.float32)

        # Mock tokenizer and model
        tokenizer_mock = MagicMock()
        model_mock = MagicMock()

        # Use patches to mock the imported functions
        with patch("src.retrieval.load_corpus_embeddings") as mock_load_corpus, \
             patch("src.retrieval.dense_top_k") as mock_dense, \
             patch("src.retrieval.build_bm25_index") as mock_bm25, \
             patch("src.retrieval.sparse_top_k") as mock_sparse, \
             patch("src.retrieval.reciprocal_rank_fusion") as mock_rrf, \
             patch("src.reranking.score_pairs_batched") as mock_score:

            # Setup mocks
            mock_load_corpus.return_value = ([1], np.ones((1, 384), dtype=np.float32))
            mock_dense.return_value = [(GOLDEN_NEEDLE_ID, 0.9, 1), (1, 0.8, 2)]
            mock_bm25.return_value = ([1, GOLDEN_NEEDLE_ID], MagicMock())
            mock_sparse.return_value = [(GOLDEN_NEEDLE_ID, 0.7, 1)]
            mock_rrf.return_value = [(GOLDEN_NEEDLE_ID, 0.85, 1), (1, 0.75, 2)]
            mock_score.return_value = [0.9, 0.8]

            # Run retrieval
            results = run_retrieval_with_needles(
                case, db, cfg, embedding_model, tokenizer_mock, model_mock, logger
            )

            # Assert golden needle is in results with correct flags
            golden_item = [r for r in results if r.job_id == GOLDEN_NEEDLE_ID]
            assert len(golden_item) == 1
            assert golden_item[0].is_needle is True
            assert golden_item[0].needle_type == "golden"


class TestTask5AdversarialNeedleFlagged:
    """test_task5_adversarial_needle_flagged: Adversarial needle is marked is_needle=True."""

    def test_task5_adversarial_needle_flagged(self):
        """Mock retrieval to return adversarial needle; assert is_needle=True, needle_type='adversarial'."""
        import logging
        import numpy as np
        from unittest.mock import patch

        logger = logging.getLogger("test")

        # Create eval case
        golden = SyntheticNeedle(
            needle_id=GOLDEN_NEEDLE_ID,
            needle_type="golden",
            title="Senior ML Engineer",
            company="TechCorp",
            description="We are looking for...",
            deal_breaker=None,
            true_relevance=5,
        )
        adversarial = SyntheticNeedle(
            needle_id=ADVERSARIAL_NEEDLE_ID,
            needle_type="adversarial",
            title="ML Intern",
            company="SmallCorp",
            description="We are looking for...",
            deal_breaker="Requires 10+ years",
            true_relevance=0,
        )
        case = EvalCase(resume_id="user_profile", resume_text="Resume text", golden=golden, adversarial=adversarial)

        # Mock DB
        db = MagicMock()
        db.get_connection.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
            (1, "Job Title", "Job description")
        ]
        db.get_all_cleaned_descriptions.return_value = [(1, "Job description")]

        # Mock config
        cfg = MagicMock()
        cfg.embedding_model_id = "test-model"
        cfg.eval_top_k = 50
        cfg.retrieval_rrf_k = 60

        # Mock embedding model
        embedding_model = MagicMock()
        embedding_model.encode.return_value = np.ones((1, 384), dtype=np.float32)

        # Mock tokenizer and model
        tokenizer_mock = MagicMock()
        model_mock = MagicMock()

        # Use patches to mock the imported functions
        with patch("src.retrieval.load_corpus_embeddings") as mock_load_corpus, \
             patch("src.retrieval.dense_top_k") as mock_dense, \
             patch("src.retrieval.build_bm25_index") as mock_bm25, \
             patch("src.retrieval.sparse_top_k") as mock_sparse, \
             patch("src.retrieval.reciprocal_rank_fusion") as mock_rrf, \
             patch("src.reranking.score_pairs_batched") as mock_score:

            # Setup mocks
            mock_load_corpus.return_value = ([1], np.ones((1, 384), dtype=np.float32))
            mock_dense.return_value = [(ADVERSARIAL_NEEDLE_ID, 0.9, 1), (1, 0.8, 2)]
            mock_bm25.return_value = ([1, ADVERSARIAL_NEEDLE_ID], MagicMock())
            mock_sparse.return_value = [(ADVERSARIAL_NEEDLE_ID, 0.7, 1)]
            mock_rrf.return_value = [(ADVERSARIAL_NEEDLE_ID, 0.85, 1), (1, 0.75, 2)]
            mock_score.return_value = [0.9, 0.8]

            # Run retrieval
            results = run_retrieval_with_needles(
                case, db, cfg, embedding_model, tokenizer_mock, model_mock, logger
            )

            # Assert adversarial needle is in results with correct flags
            adversarial_item = [r for r in results if r.job_id == ADVERSARIAL_NEEDLE_ID]
            assert len(adversarial_item) == 1
            assert adversarial_item[0].is_needle is True
            assert adversarial_item[0].needle_type == "adversarial"


class TestTask5NoDbWrites:
    """test_task5_no_db_writes: run_retrieval_with_needles never writes to DB."""

    def test_task5_no_db_writes(self):
        """Assert no INSERT or UPDATE SQL is executed during retrieval."""
        import logging
        import numpy as np
        from unittest.mock import patch

        logger = logging.getLogger("test")

        # Create eval case
        golden = SyntheticNeedle(
            needle_id=GOLDEN_NEEDLE_ID,
            needle_type="golden",
            title="Senior ML Engineer",
            company="TechCorp",
            description="We are looking for...",
            deal_breaker=None,
            true_relevance=5,
        )
        adversarial = SyntheticNeedle(
            needle_id=ADVERSARIAL_NEEDLE_ID,
            needle_type="adversarial",
            title="ML Intern",
            company="SmallCorp",
            description="We are looking for...",
            deal_breaker="Requires 10+ years",
            true_relevance=0,
        )
        case = EvalCase(resume_id="user_profile", resume_text="Resume text", golden=golden, adversarial=adversarial)

        # Mock DB with cursor tracking
        db = MagicMock()
        cursor_mock = MagicMock()
        db.get_connection.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
            (1, "Job Title", "Job description")
        ]
        db.get_all_cleaned_descriptions.return_value = [(1, "Job description")]

        # Mock config
        cfg = MagicMock()
        cfg.embedding_model_id = "test-model"
        cfg.eval_top_k = 50
        cfg.retrieval_rrf_k = 60

        # Mock embedding model
        embedding_model = MagicMock()
        embedding_model.encode.return_value = np.ones((1, 384), dtype=np.float32)

        # Mock tokenizer and model
        tokenizer_mock = MagicMock()
        model_mock = MagicMock()

        # Use patches to mock the imported functions
        with patch("src.retrieval.load_corpus_embeddings") as mock_load_corpus, \
             patch("src.retrieval.dense_top_k") as mock_dense, \
             patch("src.retrieval.build_bm25_index") as mock_bm25, \
             patch("src.retrieval.sparse_top_k") as mock_sparse, \
             patch("src.retrieval.reciprocal_rank_fusion") as mock_rrf, \
             patch("src.reranking.score_pairs_batched") as mock_score:

            # Setup mocks
            mock_load_corpus.return_value = ([1], np.ones((1, 384), dtype=np.float32))
            mock_dense.return_value = [(1, 0.8, 1)]
            mock_bm25.return_value = ([1], MagicMock())
            mock_sparse.return_value = [(1, 0.7, 1)]
            mock_rrf.return_value = [(1, 0.75, 1)]
            mock_score.return_value = [0.8]

            # Run retrieval
            results = run_retrieval_with_needles(
                case, db, cfg, embedding_model, tokenizer_mock, model_mock, logger
            )

            # Assert no INSERT/UPDATE SQL was executed
            # Check that cursor was never used for mutations (only SELECT)
            for call in db.get_connection.return_value.__enter__.return_value.execute.call_args_list:
                sql = str(call[0][0]).upper()
                assert "INSERT" not in sql, f"Found INSERT in: {sql}"
                assert "UPDATE" not in sql, f"Found UPDATE in: {sql}"
                assert "DELETE" not in sql, f"Found DELETE in: {sql}"


class TestTask5RetrievalPhaseReturnsAllCases:
    """test_task5_retrieval_phase_returns_all_cases: All cases appear in output."""

    def test_task5_retrieval_phase_returns_all_cases(self):
        """run_retrieval_phase with 2 cases; assert output dict has 2 keys."""
        import logging
        import numpy as np
        from unittest.mock import patch

        logger = logging.getLogger("test")

        # Create two eval cases
        golden1 = SyntheticNeedle(
            needle_id=GOLDEN_NEEDLE_ID,
            needle_type="golden",
            title="Senior ML Engineer",
            company="TechCorp",
            description="We are looking for...",
            deal_breaker=None,
            true_relevance=5,
        )
        adversarial1 = SyntheticNeedle(
            needle_id=ADVERSARIAL_NEEDLE_ID,
            needle_type="adversarial",
            title="ML Intern",
            company="SmallCorp",
            description="We are looking for...",
            deal_breaker="Requires 10+ years",
            true_relevance=0,
        )
        case1 = EvalCase(resume_id="user_1", resume_text="Resume 1", golden=golden1, adversarial=adversarial1)

        golden2 = SyntheticNeedle(
            needle_id=GOLDEN_NEEDLE_ID,
            needle_type="golden",
            title="Data Scientist",
            company="DataCorp",
            description="Looking for DS...",
            deal_breaker=None,
            true_relevance=5,
        )
        adversarial2 = SyntheticNeedle(
            needle_id=ADVERSARIAL_NEEDLE_ID,
            needle_type="adversarial",
            title="Data Analyst",
            company="SmallData",
            description="Looking for DA...",
            deal_breaker="Requires 15 years SQL",
            true_relevance=0,
        )
        case2 = EvalCase(resume_id="user_2", resume_text="Resume 2", golden=golden2, adversarial=adversarial2)

        cases = [case1, case2]

        # Mock DB
        db = MagicMock()
        db.get_connection.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
            (1, "Job Title", "Job description")
        ]
        db.get_all_cleaned_descriptions.return_value = [(1, "Job description")]

        # Mock config
        cfg = MagicMock()
        cfg.embedding_model_id = "test-model"
        cfg.eval_top_k = 50
        cfg.retrieval_rrf_k = 60

        # Mock embedding model
        embedding_model = MagicMock()
        embedding_model.encode.return_value = np.ones((1, 384), dtype=np.float32)

        # Mock tokenizer and model
        tokenizer_mock = MagicMock()
        model_mock = MagicMock()

        # Use patches to mock the imported functions
        with patch("src.retrieval.load_corpus_embeddings") as mock_load_corpus, \
             patch("src.retrieval.dense_top_k") as mock_dense, \
             patch("src.retrieval.build_bm25_index") as mock_bm25, \
             patch("src.retrieval.sparse_top_k") as mock_sparse, \
             patch("src.retrieval.reciprocal_rank_fusion") as mock_rrf, \
             patch("src.reranking.score_pairs_batched") as mock_score:

            # Setup mocks
            mock_load_corpus.return_value = ([1], np.ones((1, 384), dtype=np.float32))
            mock_dense.return_value = [(1, 0.8, 1)]
            mock_bm25.return_value = ([1], MagicMock())
            mock_sparse.return_value = [(1, 0.7, 1)]
            mock_rrf.return_value = [(1, 0.75, 1)]
            mock_score.return_value = [0.8]

            # Run retrieval phase
            results = run_retrieval_phase(
                cases, db, cfg, embedding_model, tokenizer_mock, model_mock, logger
            )

            # Assert output dict has both case resume_ids
            assert len(results) == 2
            assert "user_1" in results
            assert "user_2" in results
            assert isinstance(results["user_1"], list)
            assert isinstance(results["user_2"], list)


class TestTask6GoldenSkipsApi:
    """test_task6_golden_skips_api: Golden needle is deterministic (no API call)."""

    def test_task6_golden_skips_api(self):
        """Judge golden needle without API call; assert score is 5."""
        import logging

        logger = logging.getLogger("test")

        # Create retrieved item for golden needle
        item = RetrievedItem(
            job_id=GOLDEN_NEEDLE_ID,
            rank=1,
            rrf_score=0.9,
            rerank_score=0.9,
            title="Perfect Match",
            description="Perfect match job description",
            is_needle=True,
            needle_type="golden",
        )

        # Mock Anthropic client
        anthropic_client = MagicMock()

        # Judge the item
        judged = judge_retrieved_items(
            resume_text="Resume text",
            items=[item],
            anthropic_client=anthropic_client,
            judge_model_id="claude-sonnet-4-6",
            logger=logger,
        )

        # Assert score is 5 and no API call was made
        assert len(judged) == 1
        assert judged[0].relevance_score == 5
        assert judged[0].judge_reasoning == "golden needle (deterministic)"
        anthropic_client.messages.create.assert_not_called()


class TestTask6AdversarialSkipsApi:
    """test_task6_adversarial_skips_api: Adversarial needle is deterministic (no API call)."""

    def test_task6_adversarial_skips_api(self):
        """Judge adversarial needle without API call; assert score is 0."""
        import logging

        logger = logging.getLogger("test")

        # Create retrieved item for adversarial needle
        item = RetrievedItem(
            job_id=ADVERSARIAL_NEEDLE_ID,
            rank=1,
            rrf_score=0.8,
            rerank_score=0.8,
            title="Deal Breaker Job",
            description="Job with deal breaker",
            is_needle=True,
            needle_type="adversarial",
        )

        # Mock Anthropic client
        anthropic_client = MagicMock()

        # Judge the item
        judged = judge_retrieved_items(
            resume_text="Resume text",
            items=[item],
            anthropic_client=anthropic_client,
            judge_model_id="claude-sonnet-4-6",
            logger=logger,
        )

        # Assert score is 0 and no API call was made
        assert len(judged) == 1
        assert judged[0].relevance_score == 0
        assert judged[0].judge_reasoning == "adversarial needle (deterministic)"
        anthropic_client.messages.create.assert_not_called()


class TestTask6ParsesValidScore:
    """test_task6_parses_valid_score: Valid JSON response is parsed correctly."""

    def test_task6_parses_valid_score(self):
        """Mock Anthropic returning valid JSON; assert score is parsed."""
        import logging
        import json as json_module

        logger = logging.getLogger("test")

        # Create regular retrieved item (not needle)
        item = RetrievedItem(
            job_id=1,
            rank=1,
            rrf_score=0.85,
            rerank_score=0.85,
            title="Good Job",
            description="Good job description",
            is_needle=False,
            needle_type=None,
        )

        # Mock Anthropic client
        anthropic_client = MagicMock()
        response = MagicMock()
        response.content = [MagicMock()]
        response.content[0].text = json_module.dumps({"relevance_score": 3, "reasoning": "good fit"})
        anthropic_client.messages.create.return_value = response

        # Judge the item
        judged = judge_retrieved_items(
            resume_text="Resume text",
            items=[item],
            anthropic_client=anthropic_client,
            judge_model_id="claude-sonnet-4-6",
            logger=logger,
        )

        # Assert score was parsed correctly
        assert len(judged) == 1
        assert judged[0].relevance_score == 3
        assert judged[0].judge_reasoning == "good fit"


class TestTask6RejectsInvalidScore:
    """test_task6_rejects_invalid_score: Invalid scores (e.g., 4) are rejected."""

    def test_task6_rejects_invalid_score(self):
        """Mock Anthropic returning invalid score 4; assert retries and returns error."""
        import logging
        import json as json_module

        logger = logging.getLogger("test")

        # Create regular retrieved item (not needle)
        item = RetrievedItem(
            job_id=1,
            rank=1,
            rrf_score=0.85,
            rerank_score=0.85,
            title="Good Job",
            description="Good job description",
            is_needle=False,
            needle_type=None,
        )

        # Mock Anthropic client
        anthropic_client = MagicMock()
        response = MagicMock()
        response.content = [MagicMock()]
        # Return invalid score 4 every time
        response.content[0].text = json_module.dumps({"relevance_score": 4, "reasoning": "invalid"})
        anthropic_client.messages.create.return_value = response

        # Judge the item
        judged = judge_retrieved_items(
            resume_text="Resume text",
            items=[item],
            anthropic_client=anthropic_client,
            judge_model_id="claude-sonnet-4-6",
            logger=logger,
        )

        # Assert score is 0 (error sentinel) after exhausting retries
        assert len(judged) == 1
        assert judged[0].relevance_score == 0
        assert judged[0].judge_reasoning == "judge_error"


class TestTask7RecallGoldenInTopK:
    """test_task7_recall_golden_in_top_k: Golden at rank 3 → Recall@50 = 1.0."""

    def test_task7_recall_golden_in_top_k(self):
        """Golden needle at rank 3 within eval_top_k=50 → Recall = 1.0."""
        # Create eval case
        golden = SyntheticNeedle(
            needle_id=GOLDEN_NEEDLE_ID,
            needle_type="golden",
            title="Senior ML Engineer",
            company="TechCorp",
            description="We are looking for...",
            deal_breaker=None,
            true_relevance=5,
        )
        adversarial = SyntheticNeedle(
            needle_id=ADVERSARIAL_NEEDLE_ID,
            needle_type="adversarial",
            title="ML Intern",
            company="SmallCorp",
            description="We are looking for...",
            deal_breaker="Requires 10+ years",
            true_relevance=0,
        )
        case = EvalCase(resume_id="user_profile", resume_text="Resume text", golden=golden, adversarial=adversarial)

        # Create retrieved items with golden at rank 3
        retrieved_items = [
            RetrievedItem(job_id=1, rank=1, rrf_score=0.9, rerank_score=0.9, title="Job 1", description="Desc 1", is_needle=False, needle_type=None),
            RetrievedItem(job_id=2, rank=2, rrf_score=0.85, rerank_score=0.85, title="Job 2", description="Desc 2", is_needle=False, needle_type=None),
            RetrievedItem(job_id=GOLDEN_NEEDLE_ID, rank=3, rrf_score=0.8, rerank_score=0.8, title="Golden", description="Golden desc", is_needle=True, needle_type="golden"),
        ]

        # Create judged items
        judged_items = [
            JudgedItem(job_id=1, rank=1, relevance_score=3, judge_reasoning="decent fit", is_needle=False, needle_type=None),
            JudgedItem(job_id=2, rank=2, relevance_score=2, judge_reasoning="okay fit", is_needle=False, needle_type=None),
            JudgedItem(job_id=GOLDEN_NEEDLE_ID, rank=3, relevance_score=5, judge_reasoning="golden needle (deterministic)", is_needle=True, needle_type="golden"),
        ]

        # Compute metrics
        result = compute_metrics(case, retrieved_items, judged_items)

        # Assert recall is 1.0 (golden is within top 50)
        assert result.recall_at_k == 1.0
        assert result.golden_rank == 3


class TestTask7RecallGoldenNotInTopK:
    """test_task7_recall_golden_not_in_top_k: Golden at rank 51 → Recall@50 = 0.0."""

    def test_task7_recall_golden_not_in_top_k(self):
        """Golden needle at rank 51 beyond eval_top_k=50 → Recall = 0.0."""
        # Create eval case
        golden = SyntheticNeedle(
            needle_id=GOLDEN_NEEDLE_ID,
            needle_type="golden",
            title="Senior ML Engineer",
            company="TechCorp",
            description="We are looking for...",
            deal_breaker=None,
            true_relevance=5,
        )
        adversarial = SyntheticNeedle(
            needle_id=ADVERSARIAL_NEEDLE_ID,
            needle_type="adversarial",
            title="ML Intern",
            company="SmallCorp",
            description="We are looking for...",
            deal_breaker="Requires 10+ years",
            true_relevance=0,
        )
        case = EvalCase(resume_id="user_profile", resume_text="Resume text", golden=golden, adversarial=adversarial)

        # Create retrieved items with golden at rank 51 (beyond top 50)
        retrieved_items = [
            RetrievedItem(job_id=i, rank=i, rrf_score=0.9 - i*0.01, rerank_score=0.9 - i*0.01,
                         title=f"Job {i}", description=f"Desc {i}", is_needle=False, needle_type=None)
            for i in range(1, 51)
        ]
        retrieved_items.append(
            RetrievedItem(job_id=GOLDEN_NEEDLE_ID, rank=51, rrf_score=0.3, rerank_score=0.3,
                         title="Golden", description="Golden desc", is_needle=True, needle_type="golden")
        )

        # Create judged items (50 regular + 1 golden)
        judged_items = [
            JudgedItem(job_id=i, rank=i, relevance_score=2, judge_reasoning="okay fit", is_needle=False, needle_type=None)
            for i in range(1, 51)
        ]
        judged_items.append(
            JudgedItem(job_id=GOLDEN_NEEDLE_ID, rank=51, relevance_score=5, judge_reasoning="golden needle (deterministic)", is_needle=True, needle_type="golden")
        )

        # Compute metrics
        result = compute_metrics(case, retrieved_items, judged_items)

        # Assert recall is 0.0 (golden is beyond top 50)
        assert result.recall_at_k == 0.0
        assert result.golden_rank == 51


class TestTask7MrrGoldenAtRank1:
    """test_task7_mrr_golden_at_rank_1: Golden at rank 1 → MRR = 1.0."""

    def test_task7_mrr_golden_at_rank_1(self):
        """Golden needle at rank 1 → MRR = 1.0 / 1 = 1.0."""
        golden = SyntheticNeedle(
            needle_id=GOLDEN_NEEDLE_ID,
            needle_type="golden",
            title="Senior ML Engineer",
            company="TechCorp",
            description="We are looking for...",
            deal_breaker=None,
            true_relevance=5,
        )
        adversarial = SyntheticNeedle(
            needle_id=ADVERSARIAL_NEEDLE_ID,
            needle_type="adversarial",
            title="ML Intern",
            company="SmallCorp",
            description="We are looking for...",
            deal_breaker="Requires 10+ years",
            true_relevance=0,
        )
        case = EvalCase(resume_id="user_profile", resume_text="Resume text", golden=golden, adversarial=adversarial)

        # Create retrieved items with golden at rank 1
        retrieved_items = [
            RetrievedItem(job_id=GOLDEN_NEEDLE_ID, rank=1, rrf_score=0.95, rerank_score=0.95,
                         title="Golden", description="Golden desc", is_needle=True, needle_type="golden"),
        ]

        # Create judged items
        judged_items = [
            JudgedItem(job_id=GOLDEN_NEEDLE_ID, rank=1, relevance_score=5, judge_reasoning="golden needle (deterministic)", is_needle=True, needle_type="golden"),
        ]

        # Compute metrics
        result = compute_metrics(case, retrieved_items, judged_items)

        # Assert MRR is 1.0
        assert result.mrr == 1.0
        assert result.golden_rank == 1


class TestTask7MrrGoldenNotFound:
    """test_task7_mrr_golden_not_found: No golden → MRR = 0.0."""

    def test_task7_mrr_golden_not_found(self):
        """Golden needle not in retrieved list → MRR = 0.0."""
        golden = SyntheticNeedle(
            needle_id=GOLDEN_NEEDLE_ID,
            needle_type="golden",
            title="Senior ML Engineer",
            company="TechCorp",
            description="We are looking for...",
            deal_breaker=None,
            true_relevance=5,
        )
        adversarial = SyntheticNeedle(
            needle_id=ADVERSARIAL_NEEDLE_ID,
            needle_type="adversarial",
            title="ML Intern",
            company="SmallCorp",
            description="We are looking for...",
            deal_breaker="Requires 10+ years",
            true_relevance=0,
        )
        case = EvalCase(resume_id="user_profile", resume_text="Resume text", golden=golden, adversarial=adversarial)

        # Create retrieved items without golden
        retrieved_items = [
            RetrievedItem(job_id=1, rank=1, rrf_score=0.85, rerank_score=0.85,
                         title="Job 1", description="Desc 1", is_needle=False, needle_type=None),
        ]

        # Create judged items
        judged_items = [
            JudgedItem(job_id=1, rank=1, relevance_score=3, judge_reasoning="decent fit", is_needle=False, needle_type=None),
        ]

        # Compute metrics
        result = compute_metrics(case, retrieved_items, judged_items)

        # Assert MRR is 0.0
        assert result.mrr == 0.0
        assert result.golden_rank is None


class TestTask7NdcgPerfectOrder:
    """test_task7_ndcg_perfect_order: Scores [5, 3, 1] → NDCG = 1.0."""

    def test_task7_ndcg_perfect_order(self):
        """Items in perfect order by relevance → NDCG = 1.0."""
        judged_items = [
            JudgedItem(job_id=1, rank=1, relevance_score=5, judge_reasoning="perfect", is_needle=False, needle_type=None),
            JudgedItem(job_id=2, rank=2, relevance_score=3, judge_reasoning="good", is_needle=False, needle_type=None),
            JudgedItem(job_id=3, rank=3, relevance_score=1, judge_reasoning="fair", is_needle=False, needle_type=None),
        ]

        # NDCG when items are in ideal order should be 1.0
        ndcg = compute_ndcg_at_k(judged_items, k=3)
        assert ndcg == 1.0


class TestTask7NdcgAdversarialFirst:
    """test_task7_ndcg_adversarial_ranked_first: Adversarial (0) at rank 1 → NDCG < 1.0."""

    def test_task7_ndcg_adversarial_ranked_first(self):
        """Adversarial (score 0) ranked before golden (score 5) → NDCG < 1.0."""
        judged_items = [
            JudgedItem(job_id=ADVERSARIAL_NEEDLE_ID, rank=1, relevance_score=0, judge_reasoning="adversarial", is_needle=True, needle_type="adversarial"),
            JudgedItem(job_id=GOLDEN_NEEDLE_ID, rank=2, relevance_score=5, judge_reasoning="golden", is_needle=True, needle_type="golden"),
        ]

        # NDCG should be less than 1.0 due to wrong ordering
        ndcg = compute_ndcg_at_k(judged_items, k=2)
        assert ndcg < 1.0
        # Specifically, NDCG should be less than the ideal case
        ideal_ndcg = compute_ndcg_at_k([
            JudgedItem(job_id=GOLDEN_NEEDLE_ID, rank=1, relevance_score=5, judge_reasoning="golden", is_needle=True, needle_type="golden"),
            JudgedItem(job_id=ADVERSARIAL_NEEDLE_ID, rank=2, relevance_score=0, judge_reasoning="adversarial", is_needle=True, needle_type="adversarial"),
        ], k=2)
        assert ndcg < ideal_ndcg


class TestTask7PrecisionAt20:
    """test_task7_precision_at_20: Top 20 items, 14 with score >= 2 → Precision@20 = 0.70."""

    def test_task7_precision_at_20(self):
        """Precision@20: 14 relevant items out of 20 → 14/20 = 0.70."""
        # Create 20 judged items: 14 with score >= 2, 6 with score < 2
        judged_items = []
        for i in range(14):
            judged_items.append(
                JudgedItem(job_id=i+1, rank=i+1, relevance_score=2 if i % 2 == 0 else 3,
                          judge_reasoning="relevant", is_needle=False, needle_type=None)
            )
        for i in range(6):
            judged_items.append(
                JudgedItem(job_id=100+i, rank=14+i+1, relevance_score=1 if i % 2 == 0 else 0,
                          judge_reasoning="not relevant", is_needle=False, needle_type=None)
            )

        # Create dummy eval case and retrieved items
        golden = SyntheticNeedle(
            needle_id=GOLDEN_NEEDLE_ID,
            needle_type="golden",
            title="Senior ML Engineer",
            company="TechCorp",
            description="We are looking for...",
            deal_breaker=None,
            true_relevance=5,
        )
        adversarial = SyntheticNeedle(
            needle_id=ADVERSARIAL_NEEDLE_ID,
            needle_type="adversarial",
            title="ML Intern",
            company="SmallCorp",
            description="We are looking for...",
            deal_breaker="Requires 10+ years",
            true_relevance=0,
        )
        case = EvalCase(resume_id="user_profile", resume_text="Resume text", golden=golden, adversarial=adversarial)

        # Create dummy retrieved items
        retrieved_items = [
            RetrievedItem(job_id=item.job_id, rank=item.rank, rrf_score=0.5, rerank_score=0.5,
                         title=f"Job {item.job_id}", description=f"Desc {item.job_id}", is_needle=False, needle_type=None)
            for item in judged_items
        ]

        # Compute metrics
        result = compute_metrics(case, retrieved_items, judged_items)

        # Assert precision@20 is 0.70
        assert abs(result.precision_at_k - 0.70) < 0.01


# Task 8: Report generation + main entry point
class TestTask8SaveReportCreatesFile:
    """test_task8_save_report_creates_file: save_report writes JSON file with run_id."""

    def test_task8_save_report_creates_file(self, tmp_path):
        """save_report creates a JSON file with correct structure."""
        from src.evaluation import save_report

        report = EvalReport(
            run_id="2026-03-17T12:00:00Z",
            eval_top_k=50,
            precision_at_k=20,
            n_cases=1,
            mean_recall_at_k=0.95,
            mean_mrr=0.85,
            mean_ndcg_at_k=0.88,
            mean_precision_at_k=0.72,
            thresholds_met={"recall_at_k": True, "mrr": True, "ndcg_at_k": True, "precision_at_k": True},
            per_case=[{"resume_id": "user_profile", "recall_at_k": 0.95, "mrr": 0.85, "ndcg_at_k": 0.88, "precision_at_k": 0.72, "golden_rank": 3, "adversarial_rank": 45}],
            generator_model="gemini-2.5-flash",
            judge_model="claude-sonnet-4-6",
        )

        report_path = str(tmp_path / "reports" / "eval_report.json")
        save_report(report, report_path)

        assert Path(report_path).exists()
        with open(report_path) as f:
            data = json.load(f)
        assert data["run_id"] == "2026-03-17T12:00:00Z"
        assert data["n_cases"] == 1


class TestTask8OverallPassTrue:
    """test_task8_overall_pass_true_in_report: All thresholds met → overall_pass: true."""

    def test_task8_overall_pass_true_in_report(self, tmp_path):
        """When all thresholds are met, report has overall_pass: true."""
        from src.evaluation import save_report

        report = EvalReport(
            run_id="2026-03-17T12:00:00Z",
            eval_top_k=50,
            precision_at_k=20,
            n_cases=1,
            mean_recall_at_k=0.96,
            mean_mrr=0.81,
            mean_ndcg_at_k=0.86,
            mean_precision_at_k=0.71,
            thresholds_met={"recall_at_k": True, "mrr": True, "ndcg_at_k": True, "precision_at_k": True},
            per_case=[],
            generator_model="gemini-2.5-flash",
            judge_model="claude-sonnet-4-6",
        )

        report_path = str(tmp_path / "eval_report.json")
        save_report(report, report_path)

        with open(report_path) as f:
            data = json.load(f)
        assert all(data["thresholds_met"].values())


class TestTask8OverallPassFalse:
    """test_task8_overall_pass_false_in_report: One threshold below → overall_pass: false."""

    def test_task8_overall_pass_false_in_report(self, tmp_path):
        """When one threshold is not met, overall_pass is false."""
        from src.evaluation import save_report

        report = EvalReport(
            run_id="2026-03-17T12:00:00Z",
            eval_top_k=50,
            precision_at_k=20,
            n_cases=1,
            mean_recall_at_k=0.90,  # Below 0.95 threshold
            mean_mrr=0.81,
            mean_ndcg_at_k=0.86,
            mean_precision_at_k=0.71,
            thresholds_met={"recall_at_k": False, "mrr": True, "ndcg_at_k": True, "precision_at_k": True},
            per_case=[],
            generator_model="gemini-2.5-flash",
            judge_model="claude-sonnet-4-6",
        )

        report_path = str(tmp_path / "eval_report.json")
        save_report(report, report_path)

        with open(report_path) as f:
            data = json.load(f)
        assert not all(data["thresholds_met"].values())


class TestTask8SaveResultsToDb:
    """test_task8_save_results_to_db_inserts_rows: One case → 4 metric rows."""

    def test_task8_save_results_to_db_inserts_rows(self):
        """save_results_to_db inserts 4 rows (one per metric) for 1 case."""
        from src.evaluation import save_results_to_db

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name
        try:
            db = DatabaseManager(temp_db)
            db.initialize_schema()

            golden = SyntheticNeedle(
                needle_id=GOLDEN_NEEDLE_ID,
                needle_type="golden",
                title="Senior ML Engineer",
                company="TechCorp",
                description="We are looking for...",
                deal_breaker=None,
                true_relevance=5,
            )
            adversarial = SyntheticNeedle(
                needle_id=ADVERSARIAL_NEEDLE_ID,
                needle_type="adversarial",
                title="ML Intern",
                company="SmallCorp",
                description="We are looking for...",
                deal_breaker="Requires 10+ years",
                true_relevance=0,
            )
            case = EvalCase(resume_id="user_profile", resume_text="Resume text", golden=golden, adversarial=adversarial)

            result = EvalResult(
                resume_id="user_profile",
                recall_at_k=0.95,
                mrr=0.85,
                ndcg_at_k=0.88,
                precision_at_k=0.72,
                golden_rank=3,
                adversarial_rank=45,
                judged_items=[],
            )

            run_id = "2026-03-17T12:00:00Z"
            save_results_to_db(db, run_id, [result])

            # Verify 4 rows inserted
            with db.get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM eval_results WHERE run_id = ?", (run_id,))
                count = cursor.fetchone()[0]
            assert count == 4
        finally:
            Path(temp_db).unlink(missing_ok=True)
