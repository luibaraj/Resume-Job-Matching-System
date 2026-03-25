"""
Unit tests for the synthetic positives validation module.

Tests parsing, prompt building, and validation orchestration across all four rule sets.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.positives_gen import JobSkeleton
from eval.positives_validate import (
    ResumeInfo,
    ValidationResult,
    _build_domain_consistency_prompt,
    _build_resume_job_alignment_prompt,
    _build_seniority_years_prompt,
    _build_structural_prompt,
    _parse_validation_response,
    _parse_years_required,
    validate_domain_consistency,
    validate_job_skeleton,
    validate_resume_job_alignment,
    validate_seniority_years,
    validate_structural,
)


@pytest.fixture
def sample_job() -> JobSkeleton:
    """A sample valid JobSkeleton."""
    return {
        "title": "Senior Backend Engineer",
        "seniority": "Senior",
        "years_required": "5-7",
        "domain": "backend",
        "primary_skills": ["Python", "PostgreSQL"],
        "secondary_skills": ["Docker", "Redis"],
    }


@pytest.fixture
def sample_resume_info() -> ResumeInfo:
    """Sample resume information for validation context."""
    return {
        "seniority": "Senior",
        "years_experience": 8,
        "primary_skills": ["Python", "Go", "PostgreSQL"],
        "domain": "backend",
    }


class TestParseYearsRequired:
    """Tests for _parse_years_required."""

    def test_range_returns_max(self) -> None:
        """Test that range string returns the maximum value."""
        assert _parse_years_required("4-6") == 6
        assert _parse_years_required("2-8") == 8
        assert _parse_years_required("1-3") == 3

    def test_plain_int(self) -> None:
        """Test that plain integer is parsed correctly."""
        assert _parse_years_required("3") == 3
        assert _parse_years_required("10") == 10

    def test_empty_string(self) -> None:
        """Test that empty string returns 0."""
        assert _parse_years_required("") == 0

    def test_whitespace_only(self) -> None:
        """Test that whitespace-only string returns 0."""
        assert _parse_years_required("   ") == 0

    def test_unparseable_string(self) -> None:
        """Test that unparseable string returns 0."""
        assert _parse_years_required("many") == 0
        assert _parse_years_required("a lot") == 0

    def test_single_value_range(self) -> None:
        """Test that single-value range (e.g., '5-5') works."""
        assert _parse_years_required("5-5") == 5

    def test_range_with_whitespace(self) -> None:
        """Test that range with whitespace is parsed correctly."""
        assert _parse_years_required("4 - 6") == 6
        assert _parse_years_required(" 3 - 5 ") == 5

    def test_multi_part_range(self) -> None:
        """Test that multi-part ranges return the max."""
        assert _parse_years_required("2-4-6") == 6


class TestParseValidationResponse:
    """Tests for _parse_validation_response."""

    def test_pass_response(self) -> None:
        """Test parsing a PASS response."""
        result = _parse_validation_response("PASS")
        assert result["passed"] is True
        assert result["reason"] is None

    def test_pass_case_insensitive(self) -> None:
        """Test that PASS is case-insensitive."""
        assert _parse_validation_response("pass")["passed"] is True
        assert _parse_validation_response("Pass")["passed"] is True
        assert _parse_validation_response("PASS")["passed"] is True

    def test_fail_with_reason(self) -> None:
        """Test parsing a FAIL response with reason."""
        result = _parse_validation_response("FAIL: seniority mismatch")
        assert result["passed"] is False
        assert result["reason"] == "seniority mismatch"

    def test_fail_case_insensitive(self) -> None:
        """Test that FAIL is case-insensitive."""
        result = _parse_validation_response("fail: bad domain")
        assert result["passed"] is False
        assert result["reason"] == "bad domain"

    def test_fail_with_multiword_reason(self) -> None:
        """Test FAIL with multi-word reason."""
        result = _parse_validation_response("FAIL: This is a multi-word reason")
        assert result["passed"] is False
        assert result["reason"] == "This is a multi-word reason"

    def test_malformed_response(self) -> None:
        """Test parsing a malformed response."""
        result = _parse_validation_response("I dunno")
        assert result["passed"] is False
        assert "Unparseable" in result["reason"]

    def test_strips_whitespace(self) -> None:
        """Test that whitespace is stripped."""
        result = _parse_validation_response("  PASS  ")
        assert result["passed"] is True
        assert result["reason"] is None


class TestBuildStructuralPrompt:
    """Tests for _build_structural_prompt."""

    def test_includes_job_text(self) -> None:
        """Test that prompt includes job text."""
        job_text = "Title: Engineer\nSeniority: Mid"
        prompt = _build_structural_prompt(job_text)
        assert job_text in prompt

    def test_includes_pass_instruction(self) -> None:
        """Test that prompt instructs to output PASS."""
        prompt = _build_structural_prompt("test job")
        assert "PASS" in prompt

    def test_includes_fail_instruction(self) -> None:
        """Test that prompt instructs to output FAIL."""
        prompt = _build_structural_prompt("test job")
        assert "FAIL:" in prompt

    def test_includes_validation_rules(self) -> None:
        """Test that prompt includes validation rules."""
        prompt = _build_structural_prompt("test job")
        assert "Title" in prompt or "title" in prompt
        assert "Seniority" in prompt or "seniority" in prompt


class TestBuildSeniorityYearsPrompt:
    """Tests for _build_seniority_years_prompt."""

    def test_includes_seniority_and_years(self) -> None:
        """Test that prompt includes seniority and years."""
        prompt = _build_seniority_years_prompt("Senior", "5-7")
        assert "Senior" in prompt
        assert "5-7" in prompt

    def test_includes_alignment_rules(self) -> None:
        """Test that prompt includes alignment rules."""
        prompt = _build_seniority_years_prompt("Mid", "3")
        assert "Junior" in prompt or "Mid" in prompt or "Senior" in prompt


class TestBuildResumeJobAlignmentPrompt:
    """Tests for _build_resume_job_alignment_prompt."""

    def test_includes_resume_and_job_info(self) -> None:
        """Test that prompt includes both resume and job info."""
        prompt = _build_resume_job_alignment_prompt(
            "Senior", 8, ["Python", "Go"], "Senior", "5-7", ["Python", "PostgreSQL"]
        )
        assert "Resume" in prompt or "resume" in prompt
        assert "Senior" in prompt
        assert "Python" in prompt

    def test_includes_skill_overlap_rule(self) -> None:
        """Test that prompt mentions skill overlap rule."""
        prompt = _build_resume_job_alignment_prompt(
            "Mid", 5, ["JavaScript"], "Mid", "3", ["JavaScript", "React"]
        )
        assert "2" in prompt or "skill" in prompt.lower()


class TestBuildDomainConsistencyPrompt:
    """Tests for _build_domain_consistency_prompt."""

    def test_includes_domain_and_title(self) -> None:
        """Test that prompt includes domain and title."""
        prompt = _build_domain_consistency_prompt("backend", "backend", "Backend Engineer")
        assert "backend" in prompt
        assert "Backend Engineer" in prompt

    def test_includes_domain_alignment_rule(self) -> None:
        """Test that prompt mentions domain alignment."""
        prompt = _build_domain_consistency_prompt("frontend", "data", "Data Engineer")
        assert "domain" in prompt.lower() or "Domain" in prompt


class TestValidateStructural:
    """Tests for validate_structural."""

    @patch("eval.positives_validate._call_ollama")
    def test_returns_pass_result(self, mock_ollama: MagicMock, sample_job: JobSkeleton) -> None:
        """Test that PASS response is parsed correctly."""
        mock_ollama.return_value = "PASS"
        result = validate_structural(sample_job)

        assert result["passed"] is True
        assert result["reason"] is None

    @patch("eval.positives_validate._call_ollama")
    def test_returns_fail_result(self, mock_ollama: MagicMock, sample_job: JobSkeleton) -> None:
        """Test that FAIL response is parsed correctly."""
        mock_ollama.return_value = "FAIL: invalid seniority"
        result = validate_structural(sample_job)

        assert result["passed"] is False
        assert "invalid seniority" in result["reason"]


class TestValidateSeniorityYears:
    """Tests for validate_seniority_years."""

    @patch("eval.positives_validate._call_ollama")
    def test_returns_pass_result(self, mock_ollama: MagicMock, sample_job: JobSkeleton) -> None:
        """Test that PASS response is parsed correctly."""
        mock_ollama.return_value = "PASS"
        result = validate_seniority_years(sample_job)

        assert result["passed"] is True
        assert result["reason"] is None

    @patch("eval.positives_validate._call_ollama")
    def test_returns_fail_result(self, mock_ollama: MagicMock, sample_job: JobSkeleton) -> None:
        """Test that FAIL response is parsed correctly."""
        mock_ollama.return_value = "FAIL: years out of seniority bracket"
        result = validate_seniority_years(sample_job)

        assert result["passed"] is False
        assert "out of seniority bracket" in result["reason"]


class TestValidateResumeJobAlignment:
    """Tests for validate_resume_job_alignment."""

    @patch("eval.positives_validate._call_ollama")
    def test_returns_pass_result(
        self, mock_ollama: MagicMock, sample_job: JobSkeleton, sample_resume_info: ResumeInfo
    ) -> None:
        """Test that PASS response is parsed correctly."""
        mock_ollama.return_value = "PASS"
        result = validate_resume_job_alignment(sample_job, sample_resume_info)

        assert result["passed"] is True
        assert result["reason"] is None

    @patch("eval.positives_validate._call_ollama")
    def test_returns_fail_result(
        self, mock_ollama: MagicMock, sample_job: JobSkeleton, sample_resume_info: ResumeInfo
    ) -> None:
        """Test that FAIL response is parsed correctly."""
        mock_ollama.return_value = "FAIL: insufficient skill overlap"
        result = validate_resume_job_alignment(sample_job, sample_resume_info)

        assert result["passed"] is False
        assert "skill overlap" in result["reason"]


class TestValidateDomainConsistency:
    """Tests for validate_domain_consistency."""

    @patch("eval.positives_validate._call_ollama")
    def test_returns_pass_result(
        self, mock_ollama: MagicMock, sample_job: JobSkeleton, sample_resume_info: ResumeInfo
    ) -> None:
        """Test that PASS response is parsed correctly."""
        mock_ollama.return_value = "PASS"
        result = validate_domain_consistency(sample_job, sample_resume_info)

        assert result["passed"] is True
        assert result["reason"] is None

    @patch("eval.positives_validate._call_ollama")
    def test_returns_fail_result(
        self, mock_ollama: MagicMock, sample_job: JobSkeleton, sample_resume_info: ResumeInfo
    ) -> None:
        """Test that FAIL response is parsed correctly."""
        mock_ollama.return_value = "FAIL: major domain shift"
        result = validate_domain_consistency(sample_job, sample_resume_info)

        assert result["passed"] is False
        assert "domain shift" in result["reason"]


class TestValidateJobSkeleton:
    """Tests for validate_job_skeleton."""

    @patch("eval.positives_validate.validate_structural")
    @patch("eval.positives_validate.validate_seniority_years")
    @patch("eval.positives_validate.validate_resume_job_alignment")
    @patch("eval.positives_validate.validate_domain_consistency")
    def test_all_pass(
        self,
        mock_domain: MagicMock,
        mock_alignment: MagicMock,
        mock_seniority: MagicMock,
        mock_structural: MagicMock,
        sample_job: JobSkeleton,
        sample_resume_info: ResumeInfo,
    ) -> None:
        """Test that all passing checks return success."""
        # All validators pass
        mock_structural.return_value = {"passed": True, "reason": None}
        mock_seniority.return_value = {"passed": True, "reason": None}
        mock_alignment.return_value = {"passed": True, "reason": None}
        mock_domain.return_value = {"passed": True, "reason": None}

        result = validate_job_skeleton(sample_job, sample_resume_info)

        assert result["passed"] is True
        assert result["failed_check"] is None
        assert result["reason"] is None
        # All four checks should be called
        assert mock_structural.call_count == 1
        assert mock_seniority.call_count == 1
        assert mock_alignment.call_count == 1
        assert mock_domain.call_count == 1

    @patch("eval.positives_validate.validate_structural")
    @patch("eval.positives_validate.validate_seniority_years")
    @patch("eval.positives_validate.validate_resume_job_alignment")
    @patch("eval.positives_validate.validate_domain_consistency")
    def test_fails_at_structural(
        self,
        mock_domain: MagicMock,
        mock_alignment: MagicMock,
        mock_seniority: MagicMock,
        mock_structural: MagicMock,
        sample_job: JobSkeleton,
        sample_resume_info: ResumeInfo,
    ) -> None:
        """Test that failure at structural check short-circuits."""
        mock_structural.return_value = {"passed": False, "reason": "invalid format"}

        result = validate_job_skeleton(sample_job, sample_resume_info)

        assert result["passed"] is False
        assert result["failed_check"] == "structural"
        assert result["reason"] == "invalid format"
        # Only structural check should be called
        assert mock_structural.call_count == 1
        assert mock_seniority.call_count == 0
        assert mock_alignment.call_count == 0
        assert mock_domain.call_count == 0

    @patch("eval.positives_validate.validate_structural")
    @patch("eval.positives_validate.validate_seniority_years")
    @patch("eval.positives_validate.validate_resume_job_alignment")
    @patch("eval.positives_validate.validate_domain_consistency")
    def test_fails_at_seniority_years(
        self,
        mock_domain: MagicMock,
        mock_alignment: MagicMock,
        mock_seniority: MagicMock,
        mock_structural: MagicMock,
        sample_job: JobSkeleton,
        sample_resume_info: ResumeInfo,
    ) -> None:
        """Test that failure at seniority_years check short-circuits."""
        mock_structural.return_value = {"passed": True, "reason": None}
        mock_seniority.return_value = {"passed": False, "reason": "misaligned years"}

        result = validate_job_skeleton(sample_job, sample_resume_info)

        assert result["passed"] is False
        assert result["failed_check"] == "seniority_years"
        # Only first two checks should be called
        assert mock_structural.call_count == 1
        assert mock_seniority.call_count == 1
        assert mock_alignment.call_count == 0
        assert mock_domain.call_count == 0

    @patch("eval.positives_validate.validate_structural")
    @patch("eval.positives_validate.validate_seniority_years")
    @patch("eval.positives_validate.validate_resume_job_alignment")
    @patch("eval.positives_validate.validate_domain_consistency")
    def test_fails_at_resume_job_alignment(
        self,
        mock_domain: MagicMock,
        mock_alignment: MagicMock,
        mock_seniority: MagicMock,
        mock_structural: MagicMock,
        sample_job: JobSkeleton,
        sample_resume_info: ResumeInfo,
    ) -> None:
        """Test that failure at resume_job_alignment check short-circuits."""
        mock_structural.return_value = {"passed": True, "reason": None}
        mock_seniority.return_value = {"passed": True, "reason": None}
        mock_alignment.return_value = {"passed": False, "reason": "insufficient skill overlap"}

        result = validate_job_skeleton(sample_job, sample_resume_info)

        assert result["passed"] is False
        assert result["failed_check"] == "resume_job_alignment"
        # First three checks should be called
        assert mock_structural.call_count == 1
        assert mock_seniority.call_count == 1
        assert mock_alignment.call_count == 1
        assert mock_domain.call_count == 0

    @patch("eval.positives_validate.validate_structural")
    @patch("eval.positives_validate.validate_seniority_years")
    @patch("eval.positives_validate.validate_resume_job_alignment")
    @patch("eval.positives_validate.validate_domain_consistency")
    def test_fails_at_domain_consistency(
        self,
        mock_domain: MagicMock,
        mock_alignment: MagicMock,
        mock_seniority: MagicMock,
        mock_structural: MagicMock,
        sample_job: JobSkeleton,
        sample_resume_info: ResumeInfo,
    ) -> None:
        """Test that failure at domain_consistency check returns failure."""
        mock_structural.return_value = {"passed": True, "reason": None}
        mock_seniority.return_value = {"passed": True, "reason": None}
        mock_alignment.return_value = {"passed": True, "reason": None}
        mock_domain.return_value = {"passed": False, "reason": "major domain shift"}

        result = validate_job_skeleton(sample_job, sample_resume_info)

        assert result["passed"] is False
        assert result["failed_check"] == "domain_consistency"
        # All four checks should be called
        assert mock_structural.call_count == 1
        assert mock_seniority.call_count == 1
        assert mock_alignment.call_count == 1
        assert mock_domain.call_count == 1
