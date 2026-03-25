"""
Unit tests for the synthetic positives repair module.

Tests the repair loop logic, prompt generation, field merging, and
integration with validation.
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.positives_gen import JobSkeleton
from eval.positives_repair import (
    RepairResult,
    _format_fields_for_prompt,
    _get_fields_for_check,
    _merge_repaired_fields,
    repair_job_skeleton,
)
from eval.positives_validate import ResumeInfo


@pytest.fixture
def sample_job() -> JobSkeleton:
    """A sample failed JobSkeleton."""
    return {
        "title": "Backend Engineer",
        "seniority": "Senior",
        "years_required": "10-12",  # Invalid for Senior (should be 4-8)
        "domain": "backend",
        "primary_skills": ["Python", "Go"],
        "secondary_skills": ["Docker"],
    }


@pytest.fixture
def sample_resume_info() -> ResumeInfo:
    """Sample resume information for validation context."""
    return {
        "seniority": "Senior",
        "years_experience": 8,
        "primary_skills": ["Python", "Go", "Rust"],
        "domain": "backend",
    }


class TestGetFieldsForCheck:
    """Tests for _get_fields_for_check."""

    def test_structural_returns_all_relevant_fields(self) -> None:
        fields = _get_fields_for_check("structural")
        assert set(fields) == {
            "seniority",
            "domain",
            "years_required",
            "primary_skills",
            "title",
        }

    def test_seniority_years_returns_only_seniority_and_years(self) -> None:
        fields = _get_fields_for_check("seniority_years")
        assert set(fields) == {"seniority", "years_required"}

    def test_resume_job_alignment_returns_skills_and_seniority(self) -> None:
        fields = _get_fields_for_check("resume_job_alignment")
        assert set(fields) == {"primary_skills", "seniority"}

    def test_domain_consistency_returns_only_domain(self) -> None:
        fields = _get_fields_for_check("domain_consistency")
        assert fields == ["domain"]

    def test_unknown_check_returns_empty_list(self) -> None:
        fields = _get_fields_for_check("unknown_check")
        assert fields == []


class TestFormatFieldsForPrompt:
    """Tests for _format_fields_for_prompt."""

    def test_formats_selected_fields(self, sample_job: JobSkeleton) -> None:
        fields = ["seniority", "years_required"]
        result = _format_fields_for_prompt(sample_job, fields)
        assert "Seniority: Senior" in result
        assert "YearsRequired: 10-12" in result
        assert "Title:" not in result
        assert "Domain:" not in result

    def test_formats_skills_as_comma_separated(self, sample_job: JobSkeleton) -> None:
        fields = ["primary_skills"]
        result = _format_fields_for_prompt(sample_job, fields)
        assert "PrimarySkills: Python, Go" in result

    def test_empty_fields_list_returns_empty_string(self, sample_job: JobSkeleton) -> None:
        result = _format_fields_for_prompt(sample_job, [])
        assert result == ""


class TestMergeRepairedFields:
    """Tests for _merge_repaired_fields."""

    def test_merges_non_empty_values(self, sample_job: JobSkeleton) -> None:
        repaired: JobSkeleton = {
            "title": "Senior Backend Engineer",
            "seniority": "Senior",
            "years_required": "5",
            "domain": "",  # Empty — should not override original
            "primary_skills": [],  # Empty — should not override original
            "secondary_skills": ["Docker"],
        }
        fields = ["years_required"]
        result = _merge_repaired_fields(sample_job, repaired, fields)
        # years_required was repaired
        assert result["years_required"] == "5"
        # domain and primary_skills were not in fields, so original is kept
        assert result["domain"] == "backend"
        assert result["primary_skills"] == ["Python", "Go"]

    def test_preserves_original_fields_not_in_repair_set(
        self, sample_job: JobSkeleton
    ) -> None:
        repaired: JobSkeleton = {
            "title": sample_job["title"],
            "seniority": "Mid",
            "years_required": sample_job["years_required"],
            "domain": sample_job["domain"],
            "primary_skills": sample_job["primary_skills"],
            "secondary_skills": sample_job["secondary_skills"],
        }
        fields = ["seniority"]  # Only repair seniority
        result = _merge_repaired_fields(sample_job, repaired, fields)
        # Only seniority changed
        assert result["seniority"] == "Mid"
        # Everything else from original
        assert result["title"] == sample_job["title"]
        assert result["domain"] == sample_job["domain"]

    def test_ignores_empty_repaired_values(self, sample_job: JobSkeleton) -> None:
        repaired: JobSkeleton = {
            "title": "",
            "seniority": "Senior",
            "years_required": "",
            "domain": "backend",
            "primary_skills": [],
            "secondary_skills": [],
        }
        fields = ["seniority", "years_required", "title", "primary_skills"]
        result = _merge_repaired_fields(sample_job, repaired, fields)
        # Non-empty values are merged
        assert result["seniority"] == "Senior"
        # Empty values are not merged (original is kept)
        assert result["years_required"] == sample_job["years_required"]
        assert result["title"] == sample_job["title"]
        assert result["primary_skills"] == sample_job["primary_skills"]


class TestRepairJobSkeleton:
    """Tests for the main repair orchestrator."""

    @patch("eval.positives_repair._call_ollama")
    @patch("eval.positives_repair.validate_job_skeleton")
    def test_success_on_first_attempt(
        self,
        mock_validate: MagicMock,
        mock_ollama: MagicMock,
        sample_job: JobSkeleton,
        sample_resume_info: ResumeInfo,
    ) -> None:
        """Test successful repair on attempt 1."""
        # Attempt 1: LLM returns corrected years
        mock_ollama.return_value = "YearsRequired: 5"

        # Validation passes after repair
        mock_validate.return_value = {"passed": True, "failed_check": None, "reason": None}

        result = repair_job_skeleton(
            sample_job,
            "seniority_years",
            "years must match seniority bracket",
            sample_resume_info,
        )

        assert result["success"] is True
        assert result["attempts"] == 1
        assert result["job"] is not None
        assert result["discard_reason"] is None
        assert result["job"]["years_required"] == "5"

    @patch("eval.positives_repair._call_ollama")
    @patch("eval.positives_repair.validate_job_skeleton")
    def test_discard_after_two_failed_attempts(
        self,
        mock_validate: MagicMock,
        mock_ollama: MagicMock,
        sample_job: JobSkeleton,
        sample_resume_info: ResumeInfo,
    ) -> None:
        """Test discard after both attempts fail."""
        # Both attempts: LLM returns valid-looking output
        mock_ollama.return_value = "YearsRequired: 10"

        # Validation always fails
        mock_validate.return_value = {
            "passed": False,
            "failed_check": "seniority_years",
            "reason": "years still out of range",
        }

        result = repair_job_skeleton(
            sample_job,
            "seniority_years",
            "years must match seniority bracket",
            sample_resume_info,
        )

        assert result["success"] is False
        assert result["job"] is None
        assert result["attempts"] == 2
        assert result["discard_reason"] == "years still out of range"

    @patch("eval.positives_repair._call_ollama")
    @patch("eval.positives_repair.validate_job_skeleton")
    def test_uses_lower_temperature_on_attempt_2(
        self,
        mock_validate: MagicMock,
        mock_ollama: MagicMock,
        sample_job: JobSkeleton,
        sample_resume_info: ResumeInfo,
    ) -> None:
        """Test that temperature is lowered on attempt 2."""
        # Attempt 1 fails parsing
        mock_ollama.side_effect = [
            "garbage output",  # Attempt 1: unparseable
            "YearsRequired: 5",  # Attempt 2: valid
        ]

        # Attempt 2 validation passes
        mock_validate.return_value = {"passed": True, "failed_check": None, "reason": None}

        result = repair_job_skeleton(
            sample_job,
            "seniority_years",
            "years must match seniority bracket",
            sample_resume_info,
        )

        assert result["success"] is True
        assert result["attempts"] == 2
        # Verify _call_ollama was called twice with different temperatures
        assert mock_ollama.call_count == 2
        # First call: GENERATION_TEMPERATURE (0.7)
        # Second call: _REPAIR_TEMPERATURE_ATTEMPT2 (0.3)
        first_call_temp = mock_ollama.call_args_list[0][1].get("temperature", 0.7)
        second_call_temp = mock_ollama.call_args_list[1][1].get("temperature", 0.3)
        assert first_call_temp > second_call_temp

    @patch("eval.positives_repair._call_ollama")
    @patch("eval.positives_repair.validate_job_skeleton")
    def test_updates_failed_check_between_attempts(
        self,
        mock_validate: MagicMock,
        mock_ollama: MagicMock,
        sample_job: JobSkeleton,
        sample_resume_info: ResumeInfo,
    ) -> None:
        """Test that if repair shifts the failure, attempt 2 targets the new check."""
        # Both attempts return valid output
        mock_ollama.return_value = "Seniority: Mid\nYearsRequired: 3"

        # Attempt 1: original check still fails
        # Attempt 2: but now a different check fails
        mock_validate.side_effect = [
            {
                "passed": False,
                "failed_check": "seniority_years",
                "reason": "years still out of range",
            },
            {
                "passed": False,
                "failed_check": "resume_job_alignment",
                "reason": "seniority mismatch with resume",
            },
        ]

        result = repair_job_skeleton(
            sample_job,
            "seniority_years",
            "years out of bracket",
            sample_resume_info,
        )

        assert result["success"] is False
        # Discard reason reflects the final failure (resume_job_alignment)
        assert "seniority mismatch" in result["discard_reason"]
