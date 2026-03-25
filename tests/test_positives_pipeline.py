"""
Unit tests for the synthetic positives pipeline orchestrator.

Tests the main loop, safety cap, error handling, and JSON output.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ollama

from eval.positive_gen.positives_gen import JobSkeleton
from eval.positive_gen.positives_pipeline import run_pipeline
from eval.positive_gen.positives_validate import ResumeInfo


@pytest.fixture
def resume_text() -> str:
    """Sample resume text."""
    return "Alice Smith, 8 years Python/Go backend engineer. Expert in microservices, PostgreSQL, Docker."


@pytest.fixture
def resume_info() -> ResumeInfo:
    """Sample resume information."""
    return {
        "seniority": "Senior",
        "years_experience": 8,
        "primary_skills": ["Python", "Go", "PostgreSQL"],
        "domain": "backend",
    }


@pytest.fixture
def valid_job() -> JobSkeleton:
    """A sample valid JobSkeleton."""
    return {
        "title": "Senior Backend Engineer",
        "seniority": "Senior",
        "years_required": "5-7",
        "domain": "backend",
        "primary_skills": ["Python", "PostgreSQL"],
        "secondary_skills": ["Docker", "Redis"],
    }


class TestRunPipeline:
    """Tests for run_pipeline."""

    def test_empty_resume_returns_empty_list(
        self, resume_info: ResumeInfo
    ) -> None:
        """Test that empty resume returns empty list without LLM calls."""
        result = run_pipeline("", resume_info, target_count=5)
        assert result == []

    def test_whitespace_only_resume_returns_empty_list(
        self, resume_info: ResumeInfo
    ) -> None:
        """Test that whitespace-only resume returns empty list."""
        result = run_pipeline("   \n  \t  ", resume_info, target_count=5)
        assert result == []

    @patch("eval.positives_pipeline.generate_job_skeleton")
    @patch("eval.positives_pipeline.validate_job_skeleton")
    def test_collects_target_count_on_all_pass(
        self,
        mock_validate: MagicMock,
        mock_generate: MagicMock,
        resume_text: str,
        resume_info: ResumeInfo,
        valid_job: JobSkeleton,
    ) -> None:
        """Test that pipeline collects target_count jobs when all pass."""
        # Generator always returns valid job
        mock_generate.return_value = valid_job
        # Validator always passes
        mock_validate.return_value = {
            "passed": True,
            "failed_check": None,
            "reason": None,
        }

        result = run_pipeline(resume_text, resume_info, target_count=3)

        assert len(result) == 3
        assert all(job == valid_job for job in result)

    @patch("eval.positives_pipeline.generate_job_skeleton")
    @patch("eval.positives_pipeline.validate_job_skeleton")
    @patch("eval.positives_pipeline.repair_job_skeleton")
    def test_collects_repaired_jobs(
        self,
        mock_repair: MagicMock,
        mock_validate: MagicMock,
        mock_generate: MagicMock,
        resume_text: str,
        resume_info: ResumeInfo,
        valid_job: JobSkeleton,
    ) -> None:
        """Test that pipeline collects repaired jobs."""
        mock_generate.return_value = valid_job
        # First call: validate fails; then validate passes after repair
        mock_validate.side_effect = [
            {"passed": False, "failed_check": "structural", "reason": "bad format"},
            {"passed": True, "failed_check": None, "reason": None},
        ]
        # Repair succeeds
        mock_repair.return_value = {
            "success": True,
            "job": valid_job,
            "attempts": 1,
            "discard_reason": None,
        }

        result = run_pipeline(resume_text, resume_info, target_count=1)

        assert len(result) == 1
        assert result[0] == valid_job
        assert mock_repair.call_count == 1

    @patch("eval.positives_pipeline.generate_job_skeleton")
    @patch("eval.positives_pipeline.validate_job_skeleton")
    @patch("eval.positives_pipeline.repair_job_skeleton")
    def test_discards_unrepaired_jobs(
        self,
        mock_repair: MagicMock,
        mock_validate: MagicMock,
        mock_generate: MagicMock,
        resume_text: str,
        resume_info: ResumeInfo,
        valid_job: JobSkeleton,
    ) -> None:
        """Test that pipeline discards jobs that fail repair."""
        mock_generate.return_value = valid_job
        # Validator always fails
        mock_validate.return_value = {
            "passed": False,
            "failed_check": "structural",
            "reason": "bad format",
        }
        # Repair fails
        mock_repair.return_value = {
            "success": False,
            "job": None,
            "attempts": 2,
            "discard_reason": "Cannot fix",
        }

        result = run_pipeline(resume_text, resume_info, target_count=1, model="test-model")

        # No valid jobs collected
        assert len(result) == 0
        # Pipeline should have stopped after safety cap (1 * 10 = 10 attempts)
        assert mock_generate.call_count <= 10

    @patch("eval.positives_pipeline.generate_job_skeleton")
    @patch("eval.positives_pipeline.validate_job_skeleton")
    @patch("eval.positives_pipeline.repair_job_skeleton")
    def test_stops_at_safety_cap(
        self,
        mock_repair: MagicMock,
        mock_validate: MagicMock,
        mock_generate: MagicMock,
        resume_text: str,
        resume_info: ResumeInfo,
        valid_job: JobSkeleton,
    ) -> None:
        """Test that pipeline stops at safety cap (target_count * 10)."""
        mock_generate.return_value = valid_job
        # Validator always fails
        mock_validate.return_value = {
            "passed": False,
            "failed_check": "structural",
            "reason": "bad format",
        }
        # Repair always fails
        mock_repair.return_value = {
            "success": False,
            "job": None,
            "attempts": 2,
            "discard_reason": "Cannot fix",
        }

        result = run_pipeline(resume_text, resume_info, target_count=2)

        # No valid jobs collected
        assert len(result) == 0
        # Safety cap is target_count * 10 = 20
        assert mock_generate.call_count == 20

    @patch("eval.positives_pipeline.generate_job_skeleton")
    def test_skips_generation_value_error(
        self,
        mock_generate: MagicMock,
        resume_text: str,
        resume_info: ResumeInfo,
    ) -> None:
        """Test that ValueError during generation is caught and loop continues."""
        # First call raises ValueError, second call succeeds
        mock_generate.side_effect = [
            ValueError("Parse error"),
            {
                "title": "Engineer",
                "seniority": "Mid",
                "years_required": "3",
                "domain": "frontend",
                "primary_skills": ["JavaScript"],
                "secondary_skills": [],
            },
        ]

        with patch("eval.positives_pipeline.validate_job_skeleton") as mock_validate:
            mock_validate.return_value = {
                "passed": True,
                "failed_check": None,
                "reason": None,
            }

            result = run_pipeline(resume_text, resume_info, target_count=1)

            # Should collect 1 job despite first generation failing
            assert len(result) == 1

    @patch("eval.positives_pipeline.generate_job_skeleton")
    def test_skips_ollama_request_error(
        self,
        mock_generate: MagicMock,
        resume_text: str,
        resume_info: ResumeInfo,
    ) -> None:
        """Test that ollama.RequestError during generation is caught."""
        # First call raises RequestError, second call succeeds
        mock_generate.side_effect = [
            ollama.RequestError("Connection failed"),
            {
                "title": "Engineer",
                "seniority": "Mid",
                "years_required": "3",
                "domain": "frontend",
                "primary_skills": ["JavaScript"],
                "secondary_skills": [],
            },
        ]

        with patch("eval.positives_pipeline.validate_job_skeleton") as mock_validate:
            mock_validate.return_value = {
                "passed": True,
                "failed_check": None,
                "reason": None,
            }

            result = run_pipeline(resume_text, resume_info, target_count=1)

            # Should collect 1 job despite first generation failing
            assert len(result) == 1

    @patch("eval.positives_pipeline.generate_job_skeleton")
    def test_skips_ollama_response_error(
        self,
        mock_generate: MagicMock,
        resume_text: str,
        resume_info: ResumeInfo,
    ) -> None:
        """Test that ollama.ResponseError during generation is caught."""
        # First call raises ResponseError, second call succeeds
        mock_generate.side_effect = [
            ollama.ResponseError("Model error"),
            {
                "title": "Engineer",
                "seniority": "Mid",
                "years_required": "3",
                "domain": "frontend",
                "primary_skills": ["JavaScript"],
                "secondary_skills": [],
            },
        ]

        with patch("eval.positives_pipeline.validate_job_skeleton") as mock_validate:
            mock_validate.return_value = {
                "passed": True,
                "failed_check": None,
                "reason": None,
            }

            result = run_pipeline(resume_text, resume_info, target_count=1)

            # Should collect 1 job despite first generation failing
            assert len(result) == 1

    @patch("eval.positives_pipeline.generate_job_skeleton")
    @patch("eval.positives_pipeline.validate_job_skeleton")
    def test_writes_output_json(
        self,
        mock_validate: MagicMock,
        mock_generate: MagicMock,
        resume_text: str,
        resume_info: ResumeInfo,
        valid_job: JobSkeleton,
        tmp_path: Path,
    ) -> None:
        """Test that pipeline writes collected jobs to JSON file."""
        mock_generate.return_value = valid_job
        mock_validate.return_value = {
            "passed": True,
            "failed_check": None,
            "reason": None,
        }

        output_path = tmp_path / "output.json"
        result = run_pipeline(
            resume_text,
            resume_info,
            target_count=2,
            output_path=str(output_path),
        )

        # Verify JSON file was created
        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["title"] == valid_job["title"]

    @patch("eval.positives_pipeline.generate_job_skeleton")
    @patch("eval.positives_pipeline.validate_job_skeleton")
    def test_no_file_written_when_output_path_none(
        self,
        mock_validate: MagicMock,
        mock_generate: MagicMock,
        resume_text: str,
        resume_info: ResumeInfo,
        valid_job: JobSkeleton,
        tmp_path: Path,
    ) -> None:
        """Test that no file is written when output_path is None."""
        mock_generate.return_value = valid_job
        mock_validate.return_value = {
            "passed": True,
            "failed_check": None,
            "reason": None,
        }

        result = run_pipeline(
            resume_text,
            resume_info,
            target_count=1,
            output_path=None,
        )

        # No file should be written
        assert len(result) == 1

    @patch("eval.positives_pipeline.generate_job_skeleton")
    @patch("eval.positives_pipeline.validate_job_skeleton")
    def test_skips_validation_ollama_error(
        self,
        mock_validate: MagicMock,
        mock_generate: MagicMock,
        resume_text: str,
        resume_info: ResumeInfo,
        valid_job: JobSkeleton,
    ) -> None:
        """Test that ollama.RequestError during validation is caught."""
        mock_generate.return_value = valid_job
        # First call raises RequestError, second iteration succeeds
        mock_validate.side_effect = [
            ollama.RequestError("Connection failed"),
            {
                "passed": True,
                "failed_check": None,
                "reason": None,
            },
        ]

        result = run_pipeline(resume_text, resume_info, target_count=1)

        # Should eventually collect 1 job despite first validation error
        assert len(result) == 1

    @patch("eval.positives_pipeline.generate_job_skeleton")
    @patch("eval.positives_pipeline.validate_job_skeleton")
    @patch("eval.positives_pipeline.repair_job_skeleton")
    def test_repair_ollama_error_increments_discard(
        self,
        mock_repair: MagicMock,
        mock_validate: MagicMock,
        mock_generate: MagicMock,
        resume_text: str,
        resume_info: ResumeInfo,
        valid_job: JobSkeleton,
    ) -> None:
        """Test that ollama.RequestError during repair is caught."""
        mock_generate.return_value = valid_job
        mock_validate.return_value = {
            "passed": False,
            "failed_check": "structural",
            "reason": "bad format",
        }
        # Repair raises error
        mock_repair.side_effect = ollama.RequestError("Connection failed")

        result = run_pipeline(resume_text, resume_info, target_count=1, model="test-model")

        # No valid jobs collected
        assert len(result) == 0
