"""
Unit tests for the synthetic positives generation module.

Tests the job skeleton parsing, prompt building, and generation orchestration.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.positive_gen.positives_gen import (
    JobSkeleton,
    generate_job_skeleton,
    parse_skeleton_response,
    _build_skeleton_prompt,
)


class TestParseSkeletonResponse:
    """Tests for parse_skeleton_response."""

    def test_parses_valid_response(self) -> None:
        """Test parsing a complete valid response."""
        response = """Title: Senior Backend Engineer
Seniority: Senior
YearsRequired: 4-6
Domain: backend
PrimarySkills: Python, PostgreSQL, Docker
SecondarySkills: Redis, Kubernetes
Responsibilities: Design scalable microservices; Lead database optimization; Mentor junior engineers"""
        result = parse_skeleton_response(response)

        assert result["title"] == "Senior Backend Engineer"
        assert result["seniority"] == "Senior"
        assert result["years_required"] == "4-6"
        assert result["domain"] == "backend"
        assert result["primary_skills"] == ["Python", "PostgreSQL", "Docker"]
        assert result["secondary_skills"] == ["Redis", "Kubernetes"]
        assert result["responsibilities"] == ["Design scalable microservices", "Lead database optimization", "Mentor junior engineers"]

    def test_parses_skills_as_list(self) -> None:
        """Test that skills are parsed as lists."""
        response = """Title: Engineer
Seniority: Mid
YearsRequired: 3
Domain: frontend
PrimarySkills: JavaScript, React, TypeScript
SecondarySkills: CSS, Webpack
Responsibilities: Build responsive interfaces; Collaborate with designers; Write unit tests"""
        result = parse_skeleton_response(response)

        assert isinstance(result["primary_skills"], list)
        assert isinstance(result["secondary_skills"], list)
        assert isinstance(result["responsibilities"], list)
        assert result["primary_skills"] == ["JavaScript", "React", "TypeScript"]
        assert result["secondary_skills"] == ["CSS", "Webpack"]
        assert result["responsibilities"] == ["Build responsive interfaces", "Collaborate with designers", "Write unit tests"]

    def test_missing_optional_field_defaults_empty(self) -> None:
        """Test that missing fields default to empty string or list."""
        response = """Title: Engineer
Seniority: Junior
YearsRequired: 2
Domain: data
PrimarySkills: Python, SQL
Responsibilities: Analyze datasets; Generate reports; Support senior analysts"""
        result = parse_skeleton_response(response)

        assert result["secondary_skills"] == []
        assert result["responsibilities"] == ["Analyze datasets", "Generate reports", "Support senior analysts"]

    def test_raises_on_empty_response(self) -> None:
        """Test that empty response raises ValueError."""
        with pytest.raises(ValueError, match="Empty LLM response"):
            parse_skeleton_response("")

    def test_raises_on_whitespace_only_response(self) -> None:
        """Test that whitespace-only response raises ValueError."""
        with pytest.raises(ValueError, match="Empty LLM response"):
            parse_skeleton_response("   \n  \t  ")

    def test_raises_on_no_recognizable_fields(self) -> None:
        """Test that response with no recognizable fields raises ValueError."""
        with pytest.raises(ValueError, match="No recognizable fields"):
            parse_skeleton_response("some random text without colons")

    def test_key_normalization_case_insensitive(self) -> None:
        """Test that key normalization handles case insensitivity."""
        response = """TITLE: Senior Engineer
seniority: Mid
YearsRequired: 5
domain: backend
primaryskills: Python, Go
secondaryskills: Docker"""
        result = parse_skeleton_response(response)

        assert result["title"] == "Senior Engineer"
        assert result["seniority"] == "Mid"
        assert result["domain"] == "backend"

    def test_strips_whitespace_from_values(self) -> None:
        """Test that whitespace is stripped from field values."""
        response = """Title:   Senior Backend Engineer
Seniority:  Senior
YearsRequired:   4-6
Domain:  backend
PrimarySkills:  Python , Docker , Go
SecondarySkills:  Redis , Kubernetes  """
        result = parse_skeleton_response(response)

        assert result["title"] == "Senior Backend Engineer"
        assert result["seniority"] == "Senior"
        assert result["years_required"] == "4-6"
        assert result["primary_skills"] == ["Python", "Docker", "Go"]
        assert result["secondary_skills"] == ["Redis", "Kubernetes"]

    def test_ignores_lines_without_colon(self) -> None:
        """Test that lines without colons are skipped."""
        response = """Title: Engineer
This is junk
Seniority: Mid
More garbage here
YearsRequired: 3
Domain: frontend
PrimarySkills: JavaScript, React"""
        result = parse_skeleton_response(response)

        assert result["title"] == "Engineer"
        assert result["seniority"] == "Mid"


class TestBuildSkeletonPrompt:
    """Tests for _build_skeleton_prompt."""

    def test_includes_resume_text(self) -> None:
        """Test that prompt includes the resume text."""
        resume = "Alice Smith, 8 years Python backend engineer"
        prompt = _build_skeleton_prompt(resume)

        assert resume in prompt

    def test_includes_expected_field_names(self) -> None:
        """Test that prompt includes all expected field names."""
        resume = "test resume"
        prompt = _build_skeleton_prompt(resume)

        assert "Title:" in prompt
        assert "Seniority:" in prompt
        assert "YearsRequired:" in prompt
        assert "Domain:" in prompt
        assert "PrimarySkills:" in prompt
        assert "SecondarySkills:" in prompt
        assert "Responsibilities:" in prompt

    def test_includes_instruction_to_output_only_fields(self) -> None:
        """Test that prompt instructs to output only the fields."""
        resume = "test resume"
        prompt = _build_skeleton_prompt(resume)

        assert "ONLY" in prompt or "only" in prompt


class TestGenerateJobSkeleton:
    """Tests for generate_job_skeleton."""

    @patch("eval.positives_gen._call_ollama")
    def test_calls_ollama_and_parses_result(self, mock_ollama: MagicMock) -> None:
        """Test that generate_job_skeleton calls Ollama and parses the result."""
        mock_ollama.return_value = """Title: Senior Backend Engineer
Seniority: Senior
YearsRequired: 5-7
Domain: backend
PrimarySkills: Python, PostgreSQL
SecondarySkills: Docker, Kubernetes
Responsibilities: Design scalable APIs; Review code; Optimize databases"""

        result = generate_job_skeleton("test resume")

        assert isinstance(result, dict)
        assert result["title"] == "Senior Backend Engineer"
        assert result["seniority"] == "Senior"
        assert result["domain"] == "backend"
        assert result["responsibilities"] == ["Design scalable APIs", "Review code", "Optimize databases"]
        # Verify Ollama was called
        assert mock_ollama.call_count == 1

    @patch("eval.positives_gen._call_ollama")
    def test_raises_value_error_on_bad_response(self, mock_ollama: MagicMock) -> None:
        """Test that ValueError is propagated when parsing fails."""
        mock_ollama.return_value = "garbage response with no colons"

        with pytest.raises(ValueError, match="No recognizable fields"):
            generate_job_skeleton("test resume")

    @patch("eval.positives_gen._call_ollama")
    def test_accepts_custom_model(self, mock_ollama: MagicMock) -> None:
        """Test that custom model parameter is passed to Ollama."""
        mock_ollama.return_value = """Title: Engineer
Seniority: Mid
YearsRequired: 3
Domain: frontend
PrimarySkills: JavaScript, React
SecondarySkills: CSS
Responsibilities: Build UI components; Write tests; Collaborate with backend team"""

        generate_job_skeleton("test resume", model="custom-model")

        # Verify model was passed to _call_ollama
        call_args = mock_ollama.call_args
        assert call_args[0][1] == "custom-model"
