"""
Unit tests for the seniority-mismatched negatives generation module.

Tests the target seniority selection, years range mapping, prompt building,
and skeleton generation.
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
    _years_range_for_seniority,
    _build_mismatched_skeleton_prompt,
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


class TestYearsRangeForSeniority:
    """Tests for _years_range_for_seniority."""

    def test_junior_returns_correct_range(self) -> None:
        """Junior should return 0-2."""
        assert _years_range_for_seniority("Junior") == "0-2"

    def test_mid_returns_correct_range(self) -> None:
        """Mid should return 2-4."""
        assert _years_range_for_seniority("Mid") == "2-4"

    def test_senior_returns_correct_range(self) -> None:
        """Senior should return 4-7."""
        assert _years_range_for_seniority("Senior") == "4-7"

    def test_staff_returns_correct_range(self) -> None:
        """Staff should return 7-10."""
        assert _years_range_for_seniority("Staff") == "7-10"

    def test_invalid_seniority_raises_value_error(self) -> None:
        """Invalid seniority should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid seniority"):
            _years_range_for_seniority("Consultant")


class TestBuildMismatchedSkeletonPrompt:
    """Tests for _build_mismatched_skeleton_prompt."""

    def test_includes_target_seniority(self) -> None:
        """Prompt should include target seniority."""
        prompt = _build_mismatched_skeleton_prompt(
            "I have 8 years of backend experience", "Junior", "0-2"
        )
        assert "Junior" in prompt

    def test_includes_years_range(self) -> None:
        """Prompt should include pre-computed years range."""
        prompt = _build_mismatched_skeleton_prompt(
            "I have 8 years of backend experience", "Senior", "4-7"
        )
        assert "4-7" in prompt

    def test_includes_resume_text(self) -> None:
        """Prompt should include resume text for context."""
        resume_text = "Python, Django, PostgreSQL expertise"
        prompt = _build_mismatched_skeleton_prompt(resume_text, "Staff", "7-10")
        assert "Python" in prompt
        assert "Django" in prompt

    def test_emphasizes_seniority_mismatch(self) -> None:
        """Prompt should emphasize that seniority must be mismatched."""
        prompt = _build_mismatched_skeleton_prompt(
            "Resume text", "Senior", "4-7"
        )
        assert "MUST be" in prompt or "must be" in prompt
        assert "Senior" in prompt


class TestGenerateMismatchedSkeleton:
    """Tests for generate_mismatched_skeleton."""

    @patch("eval.negative_gen.negatives_gen._call_ollama")
    @patch("eval.negative_gen.negatives_gen.get_target_seniority")
    def test_returns_tuple_of_skeleton_and_target_seniority(
        self, mock_get_target: MagicMock, mock_call_ollama: MagicMock
    ) -> None:
        """Should return tuple of (JobSkeleton dict, target_seniority str)."""
        mock_get_target.return_value = "Staff"
        mock_call_ollama.return_value = """Title: Staff Data Engineer
Seniority: Staff
YearsRequired: 7-9
Domain: data
PrimarySkills: Python, Spark, SQL
SecondarySkills: Kubernetes
Responsibilities: Design data pipelines; Lead architecture; Mentor team"""

        skeleton, target = generate_mismatched_skeleton(
            "Resume text", "Junior"
        )

        assert isinstance(skeleton, dict)
        assert skeleton["title"] == "Staff Data Engineer"
        assert skeleton["seniority"] == "Staff"
        assert target == "Staff"

    @patch("eval.negative_gen.negatives_gen._call_ollama")
    @patch("eval.negative_gen.negatives_gen.get_target_seniority")
    def test_calls_get_target_seniority(
        self, mock_get_target: MagicMock, mock_call_ollama: MagicMock
    ) -> None:
        """Should call get_target_seniority with resume_seniority."""
        mock_get_target.return_value = "Senior"
        mock_call_ollama.return_value = """Title: Senior Engineer
Seniority: Senior
YearsRequired: 4-6
Domain: backend
PrimarySkills: Go, Rust
SecondarySkills: Kubernetes
Responsibilities: Build systems; Review code; Unblock team"""

        generate_mismatched_skeleton("Resume text", "Mid")

        mock_get_target.assert_called_once_with("Mid")

    @patch("eval.negative_gen.negatives_gen._call_ollama")
    @patch("eval.negative_gen.negatives_gen.get_target_seniority")
    def test_invalid_resume_seniority_raises_value_error(
        self, mock_get_target: MagicMock, mock_call_ollama: MagicMock
    ) -> None:
        """Invalid resume_seniority should raise ValueError."""
        mock_get_target.side_effect = ValueError("Invalid")

        with pytest.raises(ValueError):
            generate_mismatched_skeleton("Resume text", "BadSeniority")

    @patch("eval.negative_gen.negatives_gen._call_ollama")
    @patch("eval.negative_gen.negatives_gen.get_target_seniority")
    def test_bad_llm_response_raises_value_error(
        self, mock_get_target: MagicMock, mock_call_ollama: MagicMock
    ) -> None:
        """Unparseable LLM response should raise ValueError."""
        mock_get_target.return_value = "Junior"
        mock_call_ollama.return_value = "Completely unparseable gibberish"

        with pytest.raises(ValueError, match="No recognizable fields"):
            generate_mismatched_skeleton("Resume text", "Senior")

    @patch("eval.negative_gen.negatives_gen._call_ollama")
    @patch("eval.negative_gen.negatives_gen.get_target_seniority")
    def test_accepts_custom_model(
        self, mock_get_target: MagicMock, mock_call_ollama: MagicMock
    ) -> None:
        """Should accept custom model parameter."""
        mock_get_target.return_value = "Junior"
        mock_call_ollama.return_value = """Title: Junior Engineer
Seniority: Junior
YearsRequired: 0-2
Domain: backend
PrimarySkills: Python, JavaScript
SecondarySkills: Git
Responsibilities: Write features; Read docs; Learn from team"""

        generate_mismatched_skeleton("Resume text", "Senior", model="custom_model")

        # Verify _call_ollama was called with the custom model (positional arg at index 1)
        call_args = mock_call_ollama.call_args
        assert call_args[0][1] == "custom_model"
