"""
Unit tests for the synthetic positives validation module.

Tests parsing, prompt building, and validation orchestration across all four rule sets.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.positive_gen.positives_gen import JobSkeleton
from eval.positive_gen.positives_validate import (
    ResumeInfo,
    ValidationResult,
    _build_domain_consistency_prompt,
    _build_resume_job_alignment_prompt,
    _build_seniority_years_prompt,
    _build_structural_prompt,
    _normalize_skeleton,
    _parse_validation_response,
    _parse_years_min,
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
        "responsibilities": ["Design scalable APIs", "Lead database optimization", "Mentor junior engineers"],
    }


@pytest.fixture
def sample_resume_info() -> ResumeInfo:
    """Sample resume information for validation context."""
    return {
        "seniority": "Senior",
        "years_experience": 8,
        "primary_skills": ["Python", "Go", "PostgreSQL"],
        "domain": "backend",
        "resume_text": "Senior Backend Engineer at TechCorp with 8 years experience. Designed microservices, optimized databases, led teams. Skills: Python, Go, PostgreSQL, Docker, AWS.",
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


class TestParseYearsMin:
    """Tests for _parse_years_min."""

    def test_range_returns_min(self) -> None:
        """Test that range string returns the minimum value."""
        assert _parse_years_min("4-6") == 4
        assert _parse_years_min("2-8") == 2
        assert _parse_years_min("1-3") == 1

    def test_plain_int(self) -> None:
        """Test that plain integer is parsed correctly."""
        assert _parse_years_min("3") == 3
        assert _parse_years_min("10") == 10

    def test_empty_string(self) -> None:
        """Test that empty string returns 0."""
        assert _parse_years_min("") == 0

    def test_whitespace_only(self) -> None:
        """Test that whitespace-only string returns 0."""
        assert _parse_years_min("   ") == 0

    def test_unparseable_string(self) -> None:
        """Test that unparseable string returns 0."""
        assert _parse_years_min("many") == 0
        assert _parse_years_min("a lot") == 0

    def test_single_value_range(self) -> None:
        """Test that single-value range (e.g., '5-5') works."""
        assert _parse_years_min("5-5") == 5

    def test_range_with_whitespace(self) -> None:
        """Test that range with whitespace is parsed correctly."""
        assert _parse_years_min("4 - 6") == 4
        assert _parse_years_min(" 3 - 5 ") == 3

    def test_multi_part_range(self) -> None:
        """Test that multi-part ranges return the min."""
        assert _parse_years_min("2-4-6") == 2


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

    def test_pass_with_trailing_explanation(self) -> None:
        """LLM returns PASS followed by explanation text — should still pass."""
        result = _parse_validation_response("PASS\n\nAll checks succeeded.")
        assert result["passed"] is True
        assert result["reason"] is None

    def test_pass_with_bullet_list(self) -> None:
        """LLM returns PASS followed by a bullet list — should still pass."""
        result = _parse_validation_response("PASS\n- skill overlap ok\n- seniority ok")
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
            "Senior", 8, ["Python", "Go"], "Senior backend engineer with 8 years experience",
            "Senior", "5-7", ["Python", "PostgreSQL"], ["Design APIs", "Optimize DBs", "Lead teams"]
        )
        assert "Resume" in prompt or "resume" in prompt
        assert "Senior" in prompt
        assert "Python" in prompt

    def test_includes_skill_overlap_rule(self) -> None:
        """Test that prompt mentions skill overlap rule."""
        prompt = _build_resume_job_alignment_prompt(
            "Mid", 5, ["JavaScript"], "Mid-level frontend engineer with 5 years experience",
            "Mid", "3", ["JavaScript", "React"], ["Build UIs", "Write tests", "Collaborate"]
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
    """Tests for validate_seniority_years (deterministic check)."""

    # Junior bracket: max ≤ 2
    def test_junior_2_years_passes(self, sample_job: JobSkeleton) -> None:
        """Test that Junior with 2 years passes."""
        job = {**sample_job, "seniority": "Junior", "years_required": "1-2"}
        assert validate_seniority_years(job)["passed"] is True

    def test_junior_over_2_years_fails(self, sample_job: JobSkeleton) -> None:
        """Test that Junior with 3 years fails."""
        job = {**sample_job, "seniority": "Junior", "years_required": "3"}
        result = validate_seniority_years(job)
        assert result["passed"] is False
        assert "Junior" in result["reason"]

    # Mid bracket: max ≤ 5
    def test_mid_3_years_passes(self, sample_job: JobSkeleton) -> None:
        """Test that Mid with 3 years passes."""
        job = {**sample_job, "seniority": "Mid", "years_required": "3"}
        assert validate_seniority_years(job)["passed"] is True

    def test_mid_5_years_passes(self, sample_job: JobSkeleton) -> None:
        """Test that Mid with 3-5 years passes."""
        job = {**sample_job, "seniority": "Mid", "years_required": "3-5"}
        assert validate_seniority_years(job)["passed"] is True

    def test_mid_6_years_fails(self, sample_job: JobSkeleton) -> None:
        """Test that Mid with 6 years fails."""
        job = {**sample_job, "seniority": "Mid", "years_required": "6"}
        result = validate_seniority_years(job)
        assert result["passed"] is False

    # Senior bracket: 2 ≤ max ≤ 8
    def test_senior_5_7_passes(self, sample_job: JobSkeleton) -> None:
        """Test that Senior with 5-7 years passes."""
        job = {**sample_job, "seniority": "Senior", "years_required": "5-7"}
        assert validate_seniority_years(job)["passed"] is True

    def test_senior_4_8_passes(self, sample_job: JobSkeleton) -> None:
        """Test that Senior with 4-8 years passes."""
        job = {**sample_job, "seniority": "Senior", "years_required": "4-8"}
        assert validate_seniority_years(job)["passed"] is True

    def test_senior_9_years_fails(self, sample_job: JobSkeleton) -> None:
        """Test that Senior with 9 years fails."""
        job = {**sample_job, "seniority": "Senior", "years_required": "9"}
        result = validate_seniority_years(job)
        assert result["passed"] is False

    # Staff bracket: max ≥ 6
    def test_staff_8_years_passes(self, sample_job: JobSkeleton) -> None:
        """Test that Staff with 8 years passes."""
        job = {**sample_job, "seniority": "Staff", "years_required": "8"}
        assert validate_seniority_years(job)["passed"] is True

    def test_staff_5_years_fails(self, sample_job: JobSkeleton) -> None:
        """Test that Staff with 5 years fails."""
        job = {**sample_job, "seniority": "Staff", "years_required": "5"}
        result = validate_seniority_years(job)
        assert result["passed"] is False

    # Unknown seniority
    def test_unknown_seniority_fails(self, sample_job: JobSkeleton) -> None:
        """Test that unknown seniority fails."""
        job = {**sample_job, "seniority": "Mid-level", "years_required": "3"}
        result = validate_seniority_years(job)
        assert result["passed"] is False
        assert "Unknown seniority" in result["reason"]


class TestNormalizeSkeleton:
    """Tests for _normalize_skeleton."""

    def test_mid_level_normalized_to_mid(self, sample_job: JobSkeleton) -> None:
        """Test that 'Mid-level' normalizes to 'Mid'."""
        job = {**sample_job, "seniority": "Mid-level"}
        result = _normalize_skeleton(job)
        assert result["seniority"] == "Mid"

    def test_fullstack_case_normalized(self, sample_job: JobSkeleton) -> None:
        """Test that 'Fullstack' normalizes to 'fullstack'."""
        job = {**sample_job, "domain": "Fullstack"}
        result = _normalize_skeleton(job)
        assert result["domain"] == "fullstack"

    def test_full_dash_stack_normalized(self, sample_job: JobSkeleton) -> None:
        """Test that 'Full-stack' normalizes to 'fullstack'."""
        job = {**sample_job, "domain": "Full-stack"}
        result = _normalize_skeleton(job)
        assert result["domain"] == "fullstack"

    def test_known_seniority_unchanged(self, sample_job: JobSkeleton) -> None:
        """Test that known seniority values are preserved."""
        job = {**sample_job, "seniority": "Senior"}
        assert _normalize_skeleton(job)["seniority"] == "Senior"

    def test_known_domain_unchanged(self, sample_job: JobSkeleton) -> None:
        """Test that known domain values are preserved."""
        job = {**sample_job, "domain": "backend"}
        assert _normalize_skeleton(job)["domain"] == "backend"

    def test_other_fields_unchanged(self, sample_job: JobSkeleton) -> None:
        """Test that non-seniority/domain fields are unchanged."""
        result = _normalize_skeleton(sample_job)
        assert result["title"] == sample_job["title"]
        assert result["primary_skills"] == sample_job["primary_skills"]
        assert result["secondary_skills"] == sample_job["secondary_skills"]
        assert result["years_required"] == sample_job["years_required"]


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
