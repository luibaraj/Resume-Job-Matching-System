"""
Tests for eval.reporting module.

Tests JSON and CSV output generation from evaluation results.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.reporting import write_missed_positives_csv, write_results_json
from eval.types import PositiveRetrievalStatus, ResumeEvalResult


class TestWriteResultsJson:
    """Tests for write_results_json function."""

    def test_write_results_json_creates_valid_json(self, tmp_path) -> None:
        """Should write valid JSON with correct structure."""
        # Create test data
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

        batch_metrics = {
            "mean_precision": {5: 0.8},
            "mean_recall": {10: 0.9},
            "num_queries": 1,
        }

        with patch("eval.reporting.eval_config.TUNE_RESULTS_JSON", str(tmp_path / "results.json")):
            write_results_json([result], batch_metrics, skip_rerank=False)

            # Verify JSON file was created and is valid
            json_file = tmp_path / "results.json"
            assert json_file.exists()

            with open(json_file) as f:
                data = json.load(f)

            # Verify structure
            assert "run_metadata" in data
            assert "aggregate" in data
            assert "per_resume" in data
            assert "miss_analysis" in data

    def test_write_results_json_includes_metadata(self, tmp_path) -> None:
        """Should include run metadata in output."""
        result: ResumeEvalResult = {
            "resume_id": 1,
            "seniority": "junior",
            "domain": "engineering",
            "precision_at_5": 0.8,
            "recall_at_10": 0.9,
            "num_positives": 5,
            "positives": [],
        }

        batch_metrics = {
            "mean_precision": {5: 0.8},
            "mean_recall": {10: 0.9},
            "num_queries": 1,
        }

        with patch("eval.reporting.eval_config.TUNE_RESULTS_JSON", str(tmp_path / "results.json")):
            write_results_json([result], batch_metrics, skip_rerank=True)

            json_file = tmp_path / "results.json"
            with open(json_file) as f:
                data = json.load(f)

            metadata = data["run_metadata"]
            assert "timestamp" in metadata
            assert metadata["skip_rerank"] is True
            assert metadata["num_resumes"] == 1

    def test_write_results_json_computes_miss_analysis(self, tmp_path) -> None:
        """Should compute hit/miss statistics."""
        positive_hit: PositiveRetrievalStatus = {
            "positive_id": "uuid-1",
            "resume_id": 1,
            "resume_seniority": "junior",
            "resume_domain": "engineering",
            "positive_title": "Engineer",
            "positive_seniority": "junior",
            "positive_domain": "engineering",
            "primary_skills": [],
            "embedding_rank": 1,
            "embedding_hit": True,
            "rerank_rank": 1,
            "reranker_hit": True,
            "miss_type": "hit",
            "seniority_gap": False,
            "domain_gap": False,
        }

        positive_miss: PositiveRetrievalStatus = {
            "positive_id": "uuid-2",
            "resume_id": 1,
            "resume_seniority": "junior",
            "resume_domain": "engineering",
            "positive_title": "Engineer2",
            "positive_seniority": "senior",
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

        result: ResumeEvalResult = {
            "resume_id": 1,
            "seniority": "junior",
            "domain": "engineering",
            "precision_at_5": 0.5,
            "recall_at_10": 0.5,
            "num_positives": 2,
            "positives": [positive_hit, positive_miss],
        }

        batch_metrics = {
            "mean_precision": {5: 0.5},
            "mean_recall": {10: 0.5},
            "num_queries": 1,
        }

        with patch("eval.reporting.eval_config.TUNE_RESULTS_JSON", str(tmp_path / "results.json")):
            write_results_json([result], batch_metrics, skip_rerank=False)

            json_file = tmp_path / "results.json"
            with open(json_file) as f:
                data = json.load(f)

            miss_analysis = data["miss_analysis"]
            assert miss_analysis["total_positives"] == 2
            assert miss_analysis["total_hits"] == 1
            assert miss_analysis["embedding_misses"] == 1


class TestWriteMissedPositivesCsv:
    """Tests for write_missed_positives_csv function."""

    def test_write_missed_positives_csv_creates_file(self, tmp_path) -> None:
        """Should create CSV file with missed positives."""
        positive_miss: PositiveRetrievalStatus = {
            "positive_id": "uuid-1",
            "resume_id": 1,
            "resume_seniority": "junior",
            "resume_domain": "engineering",
            "positive_title": "Engineer",
            "positive_seniority": "senior",
            "positive_domain": "engineering",
            "primary_skills": ["Python"],
            "embedding_rank": None,
            "embedding_hit": False,
            "rerank_rank": None,
            "reranker_hit": None,
            "miss_type": "embedding_miss",
            "seniority_gap": True,
            "domain_gap": False,
        }

        result: ResumeEvalResult = {
            "resume_id": 1,
            "seniority": "junior",
            "domain": "engineering",
            "precision_at_5": 0.0,
            "recall_at_10": 0.0,
            "num_positives": 1,
            "positives": [positive_miss],
        }

        positives_df = pd.DataFrame(
            {
                "id": ["uuid-1"],
                "title": ["Engineer"],
                "job_description": ["Build systems"],
                "secondary_skills": ["AWS"],
                "responsibilities": ["Code review"],
            }
        )

        with patch("eval.reporting.eval_config.TUNE_MISSED_CSV", str(tmp_path / "missed.csv")):
            write_missed_positives_csv([result], positives_df)

            csv_file = tmp_path / "missed.csv"
            assert csv_file.exists()

            # Verify CSV contents
            df = pd.read_csv(csv_file)
            assert len(df) == 1
            assert df.iloc[0]["miss_type"] == "embedding_miss"

    def test_write_missed_positives_csv_filters_hits(self, tmp_path) -> None:
        """Should exclude hit positives from CSV."""
        positive_hit: PositiveRetrievalStatus = {
            "positive_id": "uuid-1",
            "resume_id": 1,
            "resume_seniority": "junior",
            "resume_domain": "engineering",
            "positive_title": "Engineer",
            "positive_seniority": "junior",
            "positive_domain": "engineering",
            "primary_skills": [],
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
            "positives": [positive_hit],
        }

        positives_df = pd.DataFrame(
            {
                "id": ["uuid-1"],
                "title": ["Engineer"],
                "job_description": ["Build systems"],
                "secondary_skills": ["AWS"],
                "responsibilities": ["Code review"],
            }
        )

        with patch("eval.reporting.eval_config.TUNE_MISSED_CSV", str(tmp_path / "missed.csv")):
            write_missed_positives_csv([result], positives_df)

            csv_file = tmp_path / "missed.csv"
            # When all are hits, empty DataFrame writes empty CSV (no header/data)
            # Read file directly to check
            content = csv_file.read_text().strip()
            assert content == ""  # Completely empty when no rows

    def test_write_missed_positives_csv_has_expected_columns(self, tmp_path) -> None:
        """Should have all expected columns in output CSV."""
        positive_miss: PositiveRetrievalStatus = {
            "positive_id": "uuid-1",
            "resume_id": 1,
            "resume_seniority": "junior",
            "resume_domain": "engineering",
            "positive_title": "Engineer",
            "positive_seniority": "senior",
            "positive_domain": "data",
            "primary_skills": ["Python", "SQL"],
            "embedding_rank": None,
            "embedding_hit": False,
            "rerank_rank": None,
            "reranker_hit": None,
            "miss_type": "embedding_miss",
            "seniority_gap": True,
            "domain_gap": True,
        }

        result: ResumeEvalResult = {
            "resume_id": 1,
            "seniority": "junior",
            "domain": "engineering",
            "precision_at_5": 0.0,
            "recall_at_10": 0.0,
            "num_positives": 1,
            "positives": [positive_miss],
        }

        positives_df = pd.DataFrame(
            {
                "id": ["uuid-1"],
                "title": ["Engineer"],
                "job_description": ["Build systems"],
                "secondary_skills": ["AWS"],
                "responsibilities": ["Code"],
            }
        )

        with patch("eval.reporting.eval_config.TUNE_MISSED_CSV", str(tmp_path / "missed.csv")):
            write_missed_positives_csv([result], positives_df)

            csv_file = tmp_path / "missed.csv"
            df = pd.read_csv(csv_file)

            expected_columns = [
                "positive_id",
                "resume_id",
                "miss_type",
                "seniority_gap",
                "domain_gap",
                "job_description",
            ]

            for col in expected_columns:
                assert col in df.columns
