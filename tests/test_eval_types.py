"""
Tests for eval.types module.

Verifies that TypedDict definitions have correct fields and types.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.types import PositiveRetrievalStatus, ResumeEvalResult


class TestPositiveRetrievalStatus:
    """Tests for PositiveRetrievalStatus TypedDict."""

    def test_positive_retrieval_status_has_required_fields(self) -> None:
        """Verify all required fields are present in TypedDict."""
        # Create a minimal valid instance
        status: PositiveRetrievalStatus = {
            "positive_id": "test-uuid",
            "resume_id": 1,
            "resume_seniority": "junior",
            "resume_domain": "engineering",
            "positive_title": "Software Engineer",
            "positive_seniority": "junior",
            "positive_domain": "engineering",
            "primary_skills": ["Python", "AWS"],
            "embedding_rank": 5,
            "embedding_hit": True,
            "rerank_rank": 3,
            "reranker_hit": True,
            "miss_type": "hit",
            "seniority_gap": False,
            "domain_gap": False,
        }

        assert status["positive_id"] == "test-uuid"
        assert status["resume_id"] == 1
        assert status["embedding_hit"] is True
        assert status["miss_type"] == "hit"

    def test_positive_retrieval_status_optional_fields_none(self) -> None:
        """Verify optional fields can be None."""
        status: PositiveRetrievalStatus = {
            "positive_id": "test-uuid",
            "resume_id": 1,
            "resume_seniority": "junior",
            "resume_domain": "engineering",
            "positive_title": "Software Engineer",
            "positive_seniority": "junior",
            "positive_domain": "engineering",
            "primary_skills": [],
            "embedding_rank": None,
            "embedding_hit": False,
            "rerank_rank": None,
            "reranker_hit": None,
            "miss_type": "embedding_miss",
            "seniority_gap": True,
            "domain_gap": False,
        }

        assert status["embedding_rank"] is None
        assert status["reranker_hit"] is None
        assert status["miss_type"] == "embedding_miss"


class TestResumeEvalResult:
    """Tests for ResumeEvalResult TypedDict."""

    def test_resume_eval_result_has_required_fields(self) -> None:
        """Verify all required fields are present in TypedDict."""
        result: ResumeEvalResult = {
            "resume_id": 1,
            "seniority": "junior",
            "domain": "engineering",
            "precision_at_5": 0.6,
            "recall_at_10": 0.8,
            "num_positives": 5,
            "positives": [],
        }

        assert result["resume_id"] == 1
        assert result["seniority"] == "junior"
        assert result["precision_at_5"] == 0.6
        assert result["recall_at_10"] == 0.8
        assert result["num_positives"] == 5
        assert isinstance(result["positives"], list)

    def test_resume_eval_result_with_positives(self) -> None:
        """Verify TypedDict can contain PositiveRetrievalStatus items."""
        positive: PositiveRetrievalStatus = {
            "positive_id": "uuid-1",
            "resume_id": 1,
            "resume_seniority": "junior",
            "resume_domain": "engineering",
            "positive_title": "Engineer",
            "positive_seniority": "junior",
            "positive_domain": "engineering",
            "primary_skills": ["Python"],
            "embedding_rank": 1,
            "embedding_hit": True,
            "rerank_rank": 1,
            "reranker_hit": True,
            "miss_type": "hit",
            "seniority_gap": False,
            "domain_gap": False,
        }

        result: ResumeEvalResult = {
            "resume_id": 1,
            "seniority": "junior",
            "domain": "engineering",
            "precision_at_5": 1.0,
            "recall_at_10": 1.0,
            "num_positives": 1,
            "positives": [positive],
        }

        assert len(result["positives"]) == 1
        assert result["positives"][0]["positive_id"] == "uuid-1"
        assert result["positives"][0]["miss_type"] == "hit"
