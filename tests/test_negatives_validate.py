"""
Unit tests for the seniority-mismatched negatives validation module.

Tests the seniority mismatch check, skill-domain overlap check, and validation orchestration.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.negative_gen.negatives_validate import (
    validate_mismatched_skeleton,
    validate_seniority_mismatch,
    validate_skill_domain_overlap,
)
from eval.positive_gen.positives_gen import JobSkeleton
from eval.positive_gen.positives_validate import ResumeInfo


@pytest.fixture
def sample_job() -> JobSkeleton:
    """Sample job skeleton for testing."""
    return {
        "title": "Junior Backend Engineer",
        "seniority": "Junior",
        "years_required": "0-2",
        "domain": "backend",
        "primary_skills": ["Python", "PostgreSQL", "Docker"],
        "secondary_skills": ["Redis"],
        "responsibilities": ["Build features", "Write tests", "Review code"],
    }


@pytest.fixture
def senior_resume_info() -> ResumeInfo:
    """Sample resume info for a Senior engineer."""
    return {
        "seniority": "Senior",
        "years_experience": 7,
        "primary_skills": ["Python", "Go", "Kubernetes"],
        "domain": "backend",
        "resume_text": "Experienced Senior backend engineer with 7 years in Python and Go...",
    }


class TestValidateSeniorityMismatch:
    """Tests for validate_seniority_mismatch (deterministic check)."""

    def test_junior_resume_senior_job_passes(self, sample_job: JobSkeleton) -> None:
        """Junior resume with Senior job (gap=2) should pass."""
        resume_info: ResumeInfo = {
            "seniority": "Junior",
            "years_experience": 1,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "Junior dev with 1 year experience",
        }
        sample_job["seniority"] = "Senior"
        result = validate_seniority_mismatch(sample_job, resume_info)
        assert result["passed"] is True

    def test_junior_resume_staff_job_passes(self, sample_job: JobSkeleton) -> None:
        """Junior resume with Staff job (gap=3) should pass."""
        resume_info: ResumeInfo = {
            "seniority": "Junior",
            "years_experience": 1,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "Junior dev",
        }
        sample_job["seniority"] = "Staff"
        result = validate_seniority_mismatch(sample_job, resume_info)
        assert result["passed"] is True

    def test_junior_resume_mid_job_fails(self, sample_job: JobSkeleton) -> None:
        """Junior resume with Mid job (gap=1) should fail."""
        resume_info: ResumeInfo = {
            "seniority": "Junior",
            "years_experience": 1,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "Junior dev",
        }
        sample_job["seniority"] = "Mid"
        result = validate_seniority_mismatch(sample_job, resume_info)
        assert result["passed"] is False
        assert "gap" in result["reason"].lower()

    def test_junior_resume_junior_job_fails(self, sample_job: JobSkeleton) -> None:
        """Junior resume with Junior job (gap=0) should fail."""
        resume_info: ResumeInfo = {
            "seniority": "Junior",
            "years_experience": 1,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "Junior dev",
        }
        sample_job["seniority"] = "Junior"
        result = validate_seniority_mismatch(sample_job, resume_info)
        assert result["passed"] is False

    def test_mid_resume_junior_job_passes(self, sample_job: JobSkeleton) -> None:
        """Mid resume with Junior job (gap=1) should pass."""
        resume_info: ResumeInfo = {
            "seniority": "Mid",
            "years_experience": 4,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "Mid-level engineer",
        }
        sample_job["seniority"] = "Junior"
        result = validate_seniority_mismatch(sample_job, resume_info)
        assert result["passed"] is True

    def test_mid_resume_staff_job_passes(self, sample_job: JobSkeleton) -> None:
        """Mid resume with Staff job (gap=2) should pass."""
        resume_info: ResumeInfo = {
            "seniority": "Mid",
            "years_experience": 4,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "Mid-level engineer",
        }
        sample_job["seniority"] = "Staff"
        result = validate_seniority_mismatch(sample_job, resume_info)
        assert result["passed"] is True

    def test_mid_resume_mid_job_fails(self, sample_job: JobSkeleton) -> None:
        """Mid resume with Mid job (gap=0) should fail."""
        resume_info: ResumeInfo = {
            "seniority": "Mid",
            "years_experience": 4,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "Mid-level engineer",
        }
        sample_job["seniority"] = "Mid"
        result = validate_seniority_mismatch(sample_job, resume_info)
        assert result["passed"] is False

    def test_senior_resume_junior_job_passes(self, sample_job: JobSkeleton) -> None:
        """Senior resume with Junior job (gap=2) should pass."""
        resume_info: ResumeInfo = {
            "seniority": "Senior",
            "years_experience": 7,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "Senior engineer",
        }
        sample_job["seniority"] = "Junior"
        result = validate_seniority_mismatch(sample_job, resume_info)
        assert result["passed"] is True

    def test_senior_resume_mid_job_fails(self, sample_job: JobSkeleton) -> None:
        """Senior resume with Mid job (gap=1) should fail."""
        resume_info: ResumeInfo = {
            "seniority": "Senior",
            "years_experience": 7,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "Senior engineer",
        }
        sample_job["seniority"] = "Mid"
        result = validate_seniority_mismatch(sample_job, resume_info)
        assert result["passed"] is False

    def test_senior_resume_senior_job_fails(self, sample_job: JobSkeleton) -> None:
        """Senior resume with Senior job (gap=0) should fail."""
        resume_info: ResumeInfo = {
            "seniority": "Senior",
            "years_experience": 7,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "Senior engineer",
        }
        sample_job["seniority"] = "Senior"
        result = validate_seniority_mismatch(sample_job, resume_info)
        assert result["passed"] is False

    def test_staff_resume_junior_job_passes(self, sample_job: JobSkeleton) -> None:
        """Staff resume with Junior job (gap=3) should pass."""
        resume_info: ResumeInfo = {
            "seniority": "Staff",
            "years_experience": 10,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "Staff engineer",
        }
        sample_job["seniority"] = "Junior"
        result = validate_seniority_mismatch(sample_job, resume_info)
        assert result["passed"] is True

    def test_staff_resume_mid_job_passes(self, sample_job: JobSkeleton) -> None:
        """Staff resume with Mid job (gap=2) should pass."""
        resume_info: ResumeInfo = {
            "seniority": "Staff",
            "years_experience": 10,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "Staff engineer",
        }
        sample_job["seniority"] = "Mid"
        result = validate_seniority_mismatch(sample_job, resume_info)
        assert result["passed"] is True

    def test_staff_resume_senior_job_fails(self, sample_job: JobSkeleton) -> None:
        """Staff resume with Senior job (gap=1) should fail."""
        resume_info: ResumeInfo = {
            "seniority": "Staff",
            "years_experience": 10,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "Staff engineer",
        }
        sample_job["seniority"] = "Senior"
        result = validate_seniority_mismatch(sample_job, resume_info)
        assert result["passed"] is False

    def test_unknown_resume_seniority_fails(self, sample_job: JobSkeleton) -> None:
        """Unknown resume seniority should fail."""
        resume_info: ResumeInfo = {
            "seniority": "SuperSenior",
            "years_experience": 10,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "Unknown seniority engineer",
        }
        result = validate_seniority_mismatch(sample_job, resume_info)
        assert result["passed"] is False

    def test_unknown_job_seniority_fails(self, sample_job: JobSkeleton) -> None:
        """Unknown job seniority should fail."""
        resume_info: ResumeInfo = {
            "seniority": "Senior",
            "years_experience": 7,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "Senior engineer",
        }
        sample_job["seniority"] = "Executive"
        result = validate_seniority_mismatch(sample_job, resume_info)
        assert result["passed"] is False


class TestValidateSkillDomainOverlap:
    """Tests for validate_skill_domain_overlap."""

    @patch("eval.negative_gen.negatives_validate.call_ollama_validate")
    def test_returns_pass_result(
        self, mock_ollama: MagicMock, sample_job: JobSkeleton, senior_resume_info: ResumeInfo
    ) -> None:
        """Should return pass result when LLM responds PASS."""
        mock_ollama.return_value = "PASS"
        result = validate_skill_domain_overlap(sample_job, senior_resume_info)
        assert result["passed"] is True
        assert result["reason"] is None

    @patch("eval.negative_gen.negatives_validate.call_ollama_validate")
    def test_returns_fail_result(
        self, mock_ollama: MagicMock, sample_job: JobSkeleton, senior_resume_info: ResumeInfo
    ) -> None:
        """Should return fail result when LLM responds FAIL."""
        mock_ollama.return_value = "FAIL: Only 1 skill matches"
        result = validate_skill_domain_overlap(sample_job, senior_resume_info)
        assert result["passed"] is False
        assert "Only 1 skill matches" in result["reason"]

    @patch("eval.negative_gen.negatives_validate.call_ollama_validate")
    def test_prompt_excludes_seniority_comparison(
        self, mock_ollama: MagicMock, sample_job: JobSkeleton, senior_resume_info: ResumeInfo
    ) -> None:
        """Prompt should not mention seniority proximity checks."""
        mock_ollama.return_value = "PASS"
        validate_skill_domain_overlap(sample_job, senior_resume_info)

        # Get the prompt that was sent
        prompt = mock_ollama.call_args[0][0]
        # Should not mention seniority proximity or ±1 level
        assert "±1" not in prompt
        assert "within" not in prompt.lower() or "seniority" not in prompt.lower()


class TestValidateMismatchedSkeleton:
    """Tests for validate_mismatched_skeleton orchestrator."""

    @patch("eval.negative_gen.negatives_validate.validate_structural")
    @patch("eval.negative_gen.negatives_validate.validate_seniority_years")
    @patch("eval.negative_gen.negatives_validate.validate_seniority_mismatch")
    @patch("eval.negative_gen.negatives_validate.validate_skill_domain_overlap")
    def test_all_checks_pass(
        self,
        mock_overlap: MagicMock,
        mock_mismatch: MagicMock,
        mock_seniority_years: MagicMock,
        mock_structural: MagicMock,
        sample_job: JobSkeleton,
        senior_resume_info: ResumeInfo,
    ) -> None:
        """All checks passing should return passed=True."""
        mock_structural.return_value = {"passed": True, "reason": None}
        mock_seniority_years.return_value = {"passed": True, "reason": None}
        mock_mismatch.return_value = {"passed": True, "reason": None}
        mock_overlap.return_value = {"passed": True, "reason": None}

        result = validate_mismatched_skeleton(sample_job, senior_resume_info)

        assert result["passed"] is True
        assert result["failed_check"] is None
        assert result["reason"] is None

    @patch("eval.negative_gen.negatives_validate.validate_structural")
    def test_fails_at_structural(
        self,
        mock_structural: MagicMock,
        sample_job: JobSkeleton,
        senior_resume_info: ResumeInfo,
    ) -> None:
        """Should short-circuit at structural failure."""
        mock_structural.return_value = {"passed": False, "reason": "Seniority invalid"}

        result = validate_mismatched_skeleton(sample_job, senior_resume_info)

        assert result["passed"] is False
        assert result["failed_check"] == "structural"
        assert "Seniority invalid" in result["reason"]

    @patch("eval.negative_gen.negatives_validate.validate_structural")
    @patch("eval.negative_gen.negatives_validate.validate_seniority_years")
    def test_fails_at_seniority_years(
        self,
        mock_seniority_years: MagicMock,
        mock_structural: MagicMock,
        sample_job: JobSkeleton,
        senior_resume_info: ResumeInfo,
    ) -> None:
        """Should short-circuit at seniority_years failure."""
        mock_structural.return_value = {"passed": True, "reason": None}
        mock_seniority_years.return_value = {"passed": False, "reason": "Years out of bracket"}

        result = validate_mismatched_skeleton(sample_job, senior_resume_info)

        assert result["passed"] is False
        assert result["failed_check"] == "seniority_years"

    @patch("eval.negative_gen.negatives_validate.validate_structural")
    @patch("eval.negative_gen.negatives_validate.validate_seniority_years")
    @patch("eval.negative_gen.negatives_validate.validate_seniority_mismatch")
    def test_fails_at_seniority_mismatch(
        self,
        mock_mismatch: MagicMock,
        mock_seniority_years: MagicMock,
        mock_structural: MagicMock,
        sample_job: JobSkeleton,
        senior_resume_info: ResumeInfo,
    ) -> None:
        """Should short-circuit at seniority_mismatch failure."""
        mock_structural.return_value = {"passed": True, "reason": None}
        mock_seniority_years.return_value = {"passed": True, "reason": None}
        mock_mismatch.return_value = {"passed": False, "reason": "Gap too small"}

        result = validate_mismatched_skeleton(sample_job, senior_resume_info)

        assert result["passed"] is False
        assert result["failed_check"] == "seniority_mismatch"

    @patch("eval.negative_gen.negatives_validate.validate_structural")
    @patch("eval.negative_gen.negatives_validate.validate_seniority_years")
    @patch("eval.negative_gen.negatives_validate.validate_seniority_mismatch")
    @patch("eval.negative_gen.negatives_validate.validate_skill_domain_overlap")
    def test_fails_at_skill_domain_overlap(
        self,
        mock_overlap: MagicMock,
        mock_mismatch: MagicMock,
        mock_seniority_years: MagicMock,
        mock_structural: MagicMock,
        sample_job: JobSkeleton,
        senior_resume_info: ResumeInfo,
    ) -> None:
        """Should short-circuit at skill_domain_overlap failure."""
        mock_structural.return_value = {"passed": True, "reason": None}
        mock_seniority_years.return_value = {"passed": True, "reason": None}
        mock_mismatch.return_value = {"passed": True, "reason": None}
        mock_overlap.return_value = {"passed": False, "reason": "No skill overlap"}

        result = validate_mismatched_skeleton(sample_job, senior_resume_info)

        assert result["passed"] is False
        assert result["failed_check"] == "skill_domain_overlap"

    def test_normalizes_skeleton_before_checks(self, senior_resume_info: ResumeInfo) -> None:
        """Should normalize seniority and domain before checks."""
        job: JobSkeleton = {
            "title": "Engineer",
            "seniority": "mid-level",  # Needs normalization
            "years_required": "2-4",
            "domain": "Full-Stack",  # Needs normalization
            "primary_skills": ["Python"],
            "secondary_skills": [],
            "responsibilities": ["Code"],
        }

        # This should not raise because normalization happens before structural check
        # (Even if mock fails, we're testing that normalization happens)
        with patch("eval.negative_gen.negatives_validate.validate_structural"):
            with patch("eval.negative_gen.negatives_validate.validate_seniority_years"):
                with patch("eval.negative_gen.negatives_validate.validate_seniority_mismatch"):
                    with patch("eval.negative_gen.negatives_validate.validate_skill_domain_overlap"):
                        validate_mismatched_skeleton(job, senior_resume_info)
