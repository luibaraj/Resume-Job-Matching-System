"""
Unit tests for the negatives generation module.

Tests the target seniority selection and skeleton generation with deterministic fields.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.negative_gen.negatives_gen import (
    SENIORITY_ORDER,
    generate_mismatched_skeleton,
    get_target_seniority,
)


class TestGetTargetSeniority:
    """Tests for get_target_seniority."""

    def test_junior_maps_to_senior_or_staff(self) -> None:
        """Junior candidate should map to Senior or Staff."""
        for _ in range(20):  # Try multiple times due to randomness
            target = get_target_seniority("Junior")
            assert target in ["Senior", "Staff"]

    def test_mid_maps_to_junior_or_staff(self) -> None:
        """Mid candidate should map to Junior or Staff."""
        for _ in range(20):
            target = get_target_seniority("Mid")
            assert target in ["Junior", "Staff"]

    def test_senior_maps_to_junior(self) -> None:
        """Senior candidate should map to Junior."""
        target = get_target_seniority("Senior")
        assert target == "Junior"

    def test_staff_maps_to_junior_or_mid(self) -> None:
        """Staff candidate should map to Junior or Mid."""
        for _ in range(20):
            target = get_target_seniority("Staff")
            assert target in ["Junior", "Mid"]

    def test_invalid_seniority_raises_value_error(self) -> None:
        """Invalid seniority should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid resume_seniority"):
            get_target_seniority("SuperSenior")

    def test_never_maps_to_same_seniority(self) -> None:
        """Target should never be the same as resume seniority."""
        for seniority in SENIORITY_ORDER:
            for _ in range(10):
                target = get_target_seniority(seniority)
                assert target != seniority

    @patch("eval.negative_gen.negatives_gen.random.choice")
    def test_uses_random_choice(self, mock_choice: MagicMock) -> None:
        """Should use random.choice for multi-option levels."""
        mock_choice.return_value = "Staff"
        target = get_target_seniority("Junior")
        assert target == "Staff"
        mock_choice.assert_called_once()


class TestGenerateMismatchedSkeleton:
    """Tests for generate_mismatched_skeleton."""

    @patch("eval.negative_gen.negatives_gen._generate_single_responsibility")
    @patch("eval.negative_gen.negatives_gen._generate_skills")
    @patch("eval.negative_gen.negatives_gen._generate_deterministic_fields")
    @patch("eval.negative_gen.negatives_gen._extract_years_experience")
    @patch("eval.negative_gen.negatives_gen.get_target_seniority")
    def test_seniority_mismatch_returns_tuple(
        self,
        mock_get_target: MagicMock,
        mock_extract_years: MagicMock,
        mock_det_fields: MagicMock,
        mock_skills: MagicMock,
        mock_resp: MagicMock,
    ) -> None:
        """Should return tuple of (JobSkeleton dict, mismatch_context dict) for seniority mismatch."""
        mock_get_target.return_value = "Staff"
        mock_extract_years.return_value = 5
        mock_det_fields.return_value = {
            "title": "Staff Backend Engineer",
            "seniority": "Staff",
            "domain": "backend",
            "years_required": "6-10",
        }
        mock_skills.return_value = (["Python", "Go"], ["Kubernetes"])
        # Need to provide enough return values for the loop to generate 5 responsibilities
        mock_resp.side_effect = [
            "Design distributed systems",
            "Lead architecture decisions",
            "Mentor junior engineers",
            "Review designs",
            "Build scalable systems",
        ]

        resume_info = {
            "resume_text": "5 years backend experience",
            "seniority": "Junior",
            "domain": "backend",
            "primary_skills": ["Python"],
        }

        skeleton, context = generate_mismatched_skeleton(
            resume_info=resume_info,
            mismatch_type="seniority"
        )

        assert isinstance(skeleton, dict)
        assert skeleton["title"] == "Staff Backend Engineer"
        assert skeleton["seniority"] == "Staff"
        assert isinstance(context, dict)
        assert context["target_seniority"] == "Staff"
        assert len(skeleton["responsibilities"]) > 0

    @patch("eval.negative_gen.negatives_gen.ollama.chat")
    @patch("eval.negative_gen.negatives_gen._generate_skills")
    @patch("eval.negative_gen.negatives_gen._generate_deterministic_fields")
    @patch("eval.negative_gen.negatives_gen._extract_years_experience")
    def test_responsibility_mismatch_uses_deterministic_fields(
        self,
        mock_extract_years: MagicMock,
        mock_det_fields: MagicMock,
        mock_skills: MagicMock,
        mock_chat: MagicMock,
    ) -> None:
        """Should use deterministic fields matching resume for responsibility mismatch."""
        mock_extract_years.return_value = 5
        mock_det_fields.return_value = {
            "title": "Mid Backend Engineer",
            "seniority": "Mid",
            "domain": "backend",
            "years_required": "2-5",
        }
        mock_skills.return_value = (["Python", "Docker"], ["Kubernetes"])
        # Mock ollama.chat to always return a valid responsibility (up to 20 calls to be safe)
        mock_chat.return_value = {"message": {"content": "Deploy and monitor production services"}}

        resume_info = {
            "resume_text": "5 years backend experience",
            "seniority": "Mid",
            "domain": "backend",
            "primary_skills": ["Python", "Docker"],
        }

        skeleton, context = generate_mismatched_skeleton(
            resume_info=resume_info,
            mismatch_type="responsibility"
        )

        # Fields should match resume
        assert skeleton["seniority"] == "Mid"
        assert skeleton["domain"] == "backend"
        assert context.get("mismatch_dimension") == "responsibility"
        # Should have generated 5 responsibilities (TARGET_RESPONSIBILITY_COUNT)
        assert len(skeleton["responsibilities"]) > 0
