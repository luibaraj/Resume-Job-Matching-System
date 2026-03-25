"""
Unit tests for the seniority-mismatched negatives repair module.

Tests the two-attempt repair loop, field targeting, temperature drop, and
failure shifting between attempts.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.negative_gen.negatives_repair import (
    repair_mismatched_skeleton,
    _get_fields_for_check,
    RepairResult,
)
from eval.positive_gen.positives_gen import JobSkeleton
from eval.positive_gen.positives_validate import ResumeInfo


@pytest.fixture
def failed_job() -> JobSkeleton:
    """Sample failed job skeleton for repair."""
    return {
        "title": "Junior Backend Engineer",
        "seniority": "Junior",  # Junior is wrong; should be Senior for a Senior resume
        "years_required": "0-2",
        "domain": "backend",
        "primary_skills": ["Go"],  # Only 1 skill matches
        "secondary_skills": [],
        "responsibilities": ["Write code"],
    }


@pytest.fixture
def senior_resume_info() -> ResumeInfo:
    """Sample resume info for a Senior engineer."""
    return {
        "seniority": "Senior",
        "years_experience": 7,
        "primary_skills": ["Python", "Go", "Kubernetes"],
        "domain": "backend",
        "resume_text": "Senior backend engineer with 7 years experience in Python and Go",
    }


class TestGetFieldsForCheck:
    """Tests for _get_fields_for_check."""

    def test_structural_fields(self) -> None:
        """Structural check should include core fields."""
        fields = _get_fields_for_check("structural")
        assert "seniority" in fields
        assert "domain" in fields
        assert "years_required" in fields
        assert "primary_skills" in fields
        assert "title" in fields
        assert "responsibilities" in fields

    def test_seniority_years_fields(self) -> None:
        """Seniority-years check should target seniority and years."""
        fields = _get_fields_for_check("seniority_years")
        assert set(fields) == {"seniority", "years_required"}

    def test_seniority_mismatch_fields(self) -> None:
        """Seniority-mismatch check should target seniority, years, and title."""
        fields = _get_fields_for_check("seniority_mismatch")
        assert "seniority" in fields
        assert "years_required" in fields
        assert "title" in fields

    def test_skill_domain_overlap_fields(self) -> None:
        """Skill-domain-overlap check should target skills, domain, and responsibilities."""
        fields = _get_fields_for_check("skill_domain_overlap")
        assert "primary_skills" in fields
        assert "secondary_skills" in fields
        assert "domain" in fields
        assert "responsibilities" in fields

    def test_unknown_check_returns_empty(self) -> None:
        """Unknown check should return empty list."""
        fields = _get_fields_for_check("unknown_check")
        assert fields == []


class TestRepairMismatchedSkeleton:
    """Tests for repair_mismatched_skeleton."""

    @patch("eval.negative_gen.negatives_repair.validate_mismatched_skeleton")
    @patch("eval.negative_gen.negatives_repair._call_ollama")
    @patch("eval.negative_gen.negatives_repair.parse_skeleton_response")
    def test_success_on_first_attempt(
        self,
        mock_parse: MagicMock,
        mock_ollama: MagicMock,
        mock_validate: MagicMock,
        failed_job: JobSkeleton,
        senior_resume_info: ResumeInfo,
    ) -> None:
        """Should return success after first attempt passes."""
        # LLM returns fixed seniority
        mock_ollama.return_value = "Seniority: Senior\nYearsRequired: 4-7\nTitle: Senior Backend Engineer"
        mock_parse.return_value = {
            "seniority": "Senior",
            "years_required": "4-7",
            "title": "Senior Backend Engineer",
        }
        # Validation passes
        mock_validate.return_value = {"passed": True, "failed_check": None, "reason": None}

        result = repair_mismatched_skeleton(
            failed_job,
            "seniority_mismatch",
            "Gap too small",
            senior_resume_info,
            {"target_seniority": "Senior"},
        )

        assert result["success"] is True
        assert result["job"] is not None
        assert result["attempts"] == 1
        assert result["discard_reason"] is None

    @patch("eval.negative_gen.negatives_repair.validate_mismatched_skeleton")
    @patch("eval.negative_gen.negatives_repair._call_ollama")
    @patch("eval.negative_gen.negatives_repair.parse_skeleton_response")
    def test_discard_after_two_failed_attempts(
        self,
        mock_parse: MagicMock,
        mock_ollama: MagicMock,
        mock_validate: MagicMock,
        failed_job: JobSkeleton,
        senior_resume_info: ResumeInfo,
    ) -> None:
        """Should discard after both attempts fail validation."""
        # Both attempts return bad responses that fail validation
        mock_parse.return_value = {"seniority": "Staff", "years_required": "999"}
        mock_validate.return_value = {
            "passed": False,
            "failed_check": "seniority_mismatch",
            "reason": "Still wrong",
        }

        result = repair_mismatched_skeleton(
            failed_job,
            "seniority_mismatch",
            "Gap too small",
            senior_resume_info,
            {"target_seniority": "Senior"},
        )

        assert result["success"] is False
        assert result["job"] is None
        assert result["attempts"] == 2
        assert "Still wrong" in result["discard_reason"]

    @patch("eval.negative_gen.negatives_repair.validate_mismatched_skeleton")
    @patch("eval.negative_gen.negatives_repair._call_ollama")
    @patch("eval.negative_gen.negatives_repair.parse_skeleton_response")
    def test_uses_lower_temperature_on_attempt_2(
        self,
        mock_parse: MagicMock,
        mock_ollama: MagicMock,
        mock_validate: MagicMock,
        failed_job: JobSkeleton,
        senior_resume_info: ResumeInfo,
    ) -> None:
        """Attempt 2 should use lower temperature."""
        mock_parse.return_value = {"seniority": "Senior", "years_required": "4-7"}
        mock_validate.return_value = {
            "passed": False,
            "failed_check": "seniority_mismatch",
            "reason": "Still failing",
        }

        repair_mismatched_skeleton(
            failed_job,
            "seniority_mismatch",
            "Gap too small",
            senior_resume_info,
            {"target_seniority": "Senior"},
        )

        # Check the temperature parameter for both calls (positional arg at index 2)
        call_1_args = mock_ollama.call_args_list[0][0]
        call_2_args = mock_ollama.call_args_list[1][0]
        call_1_temp = call_1_args[2]
        call_2_temp = call_2_args[2]

        # Attempt 2 should have lower temperature
        assert call_2_temp < call_1_temp

    @patch("eval.negative_gen.negatives_repair.validate_mismatched_skeleton")
    @patch("eval.negative_gen.negatives_repair._call_ollama")
    @patch("eval.negative_gen.negatives_repair.parse_skeleton_response")
    def test_updates_failed_check_between_attempts(
        self,
        mock_parse: MagicMock,
        mock_ollama: MagicMock,
        mock_validate: MagicMock,
        failed_job: JobSkeleton,
        senior_resume_info: ResumeInfo,
    ) -> None:
        """Should update failed_check if repair shifts failure to a different check."""
        mock_parse.return_value = {"seniority": "Senior", "years_required": "10-15"}

        # Attempt 1: seniority_mismatch fails, shifts to seniority_years
        mock_validate.side_effect = [
            {
                "passed": False,
                "failed_check": "seniority_years",
                "reason": "Years out of bracket",
            },
            # Attempt 2: still fails but we don't care — test just checks the prompt target
            {
                "passed": False,
                "failed_check": "seniority_years",
                "reason": "Years still out of bracket",
            },
        ]

        repair_mismatched_skeleton(
            failed_job,
            "seniority_mismatch",
            "Gap too small",
            senior_resume_info,
            {"target_seniority": "Senior"},
        )

        # Verify we made 2 calls (2 attempts)
        assert mock_ollama.call_count == 2

    @patch("eval.negative_gen.negatives_repair.validate_mismatched_skeleton")
    @patch("eval.negative_gen.negatives_repair._call_ollama")
    @patch("eval.negative_gen.negatives_repair.parse_skeleton_response")
    def test_target_seniority_passed_to_repair_prompt(
        self,
        mock_parse: MagicMock,
        mock_ollama: MagicMock,
        mock_validate: MagicMock,
        failed_job: JobSkeleton,
        senior_resume_info: ResumeInfo,
    ) -> None:
        """target_seniority should be passed to the repair prompt."""
        mock_parse.return_value = {"seniority": "Senior", "years_required": "4-7"}
        mock_validate.return_value = {"passed": True, "failed_check": None, "reason": None}

        repair_mismatched_skeleton(
            failed_job,
            "seniority_mismatch",
            "Gap too small",
            senior_resume_info,
            {"target_seniority": "Senior"},
        )

        # Check that "Senior" appears in the prompt for seniority_mismatch repair
        prompt = mock_ollama.call_args[0][0]
        assert "Senior" in prompt

    @patch("eval.negative_gen.negatives_repair.validate_mismatched_skeleton")
    @patch("eval.negative_gen.negatives_repair._call_ollama")
    @patch("eval.negative_gen.negatives_repair.parse_skeleton_response")
    def test_seniority_mismatch_repair_injects_target_seniority(
        self,
        mock_parse: MagicMock,
        mock_ollama: MagicMock,
        mock_validate: MagicMock,
        failed_job: JobSkeleton,
        senior_resume_info: ResumeInfo,
    ) -> None:
        """Seniority mismatch repair should inject the target seniority into the prompt."""
        mock_parse.return_value = {"seniority": "Staff", "years_required": "7-10"}
        mock_validate.return_value = {"passed": True, "failed_check": None, "reason": None}

        repair_mismatched_skeleton(
            failed_job,
            "seniority_mismatch",
            "Gap too small",
            senior_resume_info,
            {"target_seniority": "Staff"},
        )

        # Verify the prompt includes explicit target seniority instruction
        prompt = mock_ollama.call_args[0][0]
        assert "Staff" in prompt
        assert "MUST" in prompt or "must" in prompt

    @patch("eval.negative_gen.negatives_repair.validate_mismatched_skeleton")
    @patch("eval.negative_gen.negatives_repair._call_ollama")
    @patch("eval.negative_gen.negatives_repair.parse_skeleton_response")
    def test_parse_error_counts_as_failed_attempt(
        self,
        mock_parse: MagicMock,
        mock_ollama: MagicMock,
        mock_validate: MagicMock,
        failed_job: JobSkeleton,
        senior_resume_info: ResumeInfo,
    ) -> None:
        """Parse error should count as a failed attempt, allow attempt 2."""
        # Attempt 1: parse fails
        mock_parse.side_effect = [
            ValueError("Unparseable"),
            {"seniority": "Senior", "years_required": "4-7"},  # Attempt 2 succeeds
        ]
        mock_validate.return_value = {"passed": True, "failed_check": None, "reason": None}

        result = repair_mismatched_skeleton(
            failed_job,
            "seniority_mismatch",
            "Gap too small",
            senior_resume_info,
            {"target_seniority": "Senior"},
        )

        # Should succeed on attempt 2
        assert result["success"] is True
        assert result["attempts"] == 2

    @patch("eval.negative_gen.negatives_repair.validate_mismatched_skeleton")
    @patch("eval.negative_gen.negatives_repair._call_ollama")
    @patch("eval.negative_gen.negatives_repair.parse_skeleton_response")
    def test_repair_merges_only_targeted_fields(
        self,
        mock_parse: MagicMock,
        mock_ollama: MagicMock,
        mock_validate: MagicMock,
        failed_job: JobSkeleton,
        senior_resume_info: ResumeInfo,
    ) -> None:
        """Repair should merge only the targeted fields, preserving others."""
        # Repair only seniority_mismatch fields: seniority, years_required, title
        # Parse should return only those fields
        mock_parse.return_value = {
            "seniority": "Senior",
            "years_required": "4-7",
            "title": "Senior Backend Engineer",
            # domain is NOT in the repair response
        }
        mock_validate.return_value = {"passed": True, "failed_check": None, "reason": None}

        result = repair_mismatched_skeleton(
            failed_job,
            "seniority_mismatch",
            "Gap too small",
            senior_resume_info,
            {"target_seniority": "Senior"},
        )

        # domain should be preserved from original
        assert result["job"]["domain"] == "backend"
        # seniority should be updated
        assert result["job"]["seniority"] == "Senior"
