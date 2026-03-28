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
)
from eval.positive_gen.positives_validate import ResumeInfo


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


class TestBuildYearsExtractionPrompt:
    """Tests for _build_years_extraction_prompt."""

    def test_includes_resume_text(self) -> None:
        """Test that prompt includes the resume text."""
        from eval.positive_gen.positives_gen import _build_years_extraction_prompt

        resume = "Alice Smith, 8 years Python backend engineer"
        prompt = _build_years_extraction_prompt(resume)

        assert resume in prompt

    def test_requests_single_line_format(self) -> None:
        """Test that prompt requests YearsExperience: format."""
        from eval.positive_gen.positives_gen import _build_years_extraction_prompt

        prompt = _build_years_extraction_prompt("test resume")

        assert "YearsExperience:" in prompt

    def test_instructs_to_output_integer(self) -> None:
        """Test that prompt instructs to return a single integer."""
        from eval.positive_gen.positives_gen import _build_years_extraction_prompt

        prompt = _build_years_extraction_prompt("test resume")

        assert "integer" in prompt.lower()


class TestExtractYearsExperience:
    """Tests for _extract_years_experience."""

    @patch("eval.positive_gen.positives_gen._call_ollama")
    def test_parses_valid_response(self, mock_ollama: MagicMock) -> None:
        """Test parsing valid YearsExperience response."""
        from eval.positive_gen.positives_gen import _extract_years_experience

        mock_ollama.return_value = "YearsExperience: 7"

        result = _extract_years_experience("test resume", "model")

        assert result == 7
        assert isinstance(result, int)

    @patch("eval.positive_gen.positives_gen._call_ollama")
    def test_raises_on_missing_field(self, mock_ollama: MagicMock) -> None:
        """Test that ValueError raised when YearsExperience field missing."""
        from eval.positive_gen.positives_gen import _extract_years_experience

        mock_ollama.return_value = "SomeOtherField: 5"

        with pytest.raises(ValueError):
            _extract_years_experience("test resume", "model")

    @patch("eval.positive_gen.positives_gen._call_ollama")
    def test_raises_on_non_integer_value(self, mock_ollama: MagicMock) -> None:
        """Test that ValueError raised when value is not an integer."""
        from eval.positive_gen.positives_gen import _extract_years_experience

        mock_ollama.return_value = "YearsExperience: five"

        with pytest.raises(ValueError):
            _extract_years_experience("test resume", "model")

    @patch("eval.positive_gen.positives_gen._call_ollama")
    def test_strips_whitespace_before_parsing(self, mock_ollama: MagicMock) -> None:
        """Test that whitespace is stripped before parsing."""
        from eval.positive_gen.positives_gen import _extract_years_experience

        mock_ollama.return_value = "  YearsExperience:  3  "

        result = _extract_years_experience("test resume", "model")

        assert result == 3

    @patch("eval.positive_gen.positives_gen._call_ollama")
    def test_uses_extract_max_tokens(self, mock_ollama: MagicMock) -> None:
        """Test that _call_ollama is called with RESUME_EXTRACT_MAX_TOKENS."""
        from eval.positive_gen.positives_gen import _extract_years_experience
        from config import RESUME_EXTRACT_MAX_TOKENS

        mock_ollama.return_value = "YearsExperience: 5"

        _extract_years_experience("test resume", "model")

        # Verify max_tokens parameter
        call_kwargs = mock_ollama.call_args[1]
        assert call_kwargs.get("max_tokens") == RESUME_EXTRACT_MAX_TOKENS


class TestConstrainYearsRequired:
    """Tests for _constrain_years_required."""

    def test_returns_bracket_unchanged_when_experience_covers_max(self) -> None:
        """Test bracket unchanged when experience >= max."""
        from eval.positive_gen.positives_gen import _constrain_years_required

        result = _constrain_years_required("4-8", 10)

        assert result == "4-8"

    def test_returns_bracket_unchanged_when_experience_equals_max(self) -> None:
        """Test bracket unchanged when experience == max."""
        from eval.positive_gen.positives_gen import _constrain_years_required

        result = _constrain_years_required("4-8", 8)

        assert result == "4-8"

    def test_lowers_range_when_experience_below_max(self) -> None:
        """Test range lowered when experience < max."""
        from eval.positive_gen.positives_gen import (
            _constrain_years_required,
            _parse_years_required,
        )

        result = _constrain_years_required("4-8", 5)

        # Max of returned range should be <= 5
        max_years = _parse_years_required(result)
        assert max_years <= 5

    def test_handles_single_value_bracket(self) -> None:
        """Test handling of single value bracket."""
        from eval.positive_gen.positives_gen import (
            _constrain_years_required,
            _parse_years_required,
        )

        result = _constrain_years_required("5", 3)

        max_years = _parse_years_required(result)
        assert max_years <= 3

    def test_handles_single_value_bracket_at_or_above(self) -> None:
        """Test single value bracket when experience >= value."""
        from eval.positive_gen.positives_gen import _constrain_years_required

        result = _constrain_years_required("5", 6)

        assert result == "5"

    def test_result_is_valid_years_required_string(self) -> None:
        """Test that returned string is valid and parseable."""
        from eval.positive_gen.positives_gen import (
            _constrain_years_required,
            _parse_years_required,
        )

        result = _constrain_years_required("4-8", 5)

        # Should not raise
        max_years = _parse_years_required(result)
        assert isinstance(max_years, int)


class TestGenerateDeterministicFields:
    """Tests for _generate_deterministic_fields."""

    def test_title_uses_seniority_and_domain(self) -> None:
        """Test that title is constructed from seniority and domain."""
        from eval.positive_gen.positives_gen import _generate_deterministic_fields

        result = _generate_deterministic_fields("Senior", "backend", 8)

        assert result["title"] == "Senior Backend Engineer"

    @pytest.mark.parametrize(
        "domain,expected_role",
        [
            ("backend", "Backend Engineer"),
            ("frontend", "Frontend Engineer"),
            ("fullstack", "Full Stack Engineer"),
            ("data", "Data Engineer"),
        ],
    )
    def test_title_all_domain_mappings(
        self, domain: str, expected_role: str
    ) -> None:
        """Test all domain-to-role mappings."""
        from eval.positive_gen.positives_gen import _generate_deterministic_fields

        result = _generate_deterministic_fields("Mid", domain, 5)

        assert result["title"] == f"Mid {expected_role}"

    def test_seniority_passes_through(self) -> None:
        """Test that seniority is returned unchanged."""
        from eval.positive_gen.positives_gen import _generate_deterministic_fields

        result = _generate_deterministic_fields("Senior", "backend", 8)

        assert result["seniority"] == "Senior"

    def test_domain_passes_through(self) -> None:
        """Test that domain is returned unchanged."""
        from eval.positive_gen.positives_gen import _generate_deterministic_fields

        result = _generate_deterministic_fields("Mid", "frontend", 4)

        assert result["domain"] == "frontend"

    def test_years_required_constrained_by_experience(self) -> None:
        """Test that years_required is constrained by experience."""
        from eval.positive_gen.positives_gen import (
            _generate_deterministic_fields,
            _parse_years_required,
        )

        # Senior bracket is "4-8", but experience is only 3
        result = _generate_deterministic_fields("Senior", "backend", 3)

        max_years = _parse_years_required(result["years_required"])
        assert max_years <= 3

    def test_years_required_unconstrained_when_experience_is_sufficient(self) -> None:
        """Test that years_required is unchanged when experience is sufficient."""
        from eval.positive_gen.positives_gen import _generate_deterministic_fields

        result = _generate_deterministic_fields("Senior", "backend", 10)

        # Senior bracket is "4-8", should be unchanged
        assert result["years_required"] == "4-8"


class TestBuildSkillsPrompt:
    """Tests for _build_skills_prompt."""

    def test_includes_resume_text(self) -> None:
        """Test that prompt includes resume text."""
        from eval.positive_gen.positives_gen import _build_skills_prompt

        prompt = _build_skills_prompt("test resume with Python", "Senior", "backend")

        assert "test resume with Python" in prompt

    def test_includes_seniority_and_domain(self) -> None:
        """Test that prompt includes seniority and domain."""
        from eval.positive_gen.positives_gen import _build_skills_prompt

        prompt = _build_skills_prompt("test resume", "Senior", "backend")

        assert "Senior" in prompt
        assert "backend" in prompt

    def test_specifies_primary_skills_label(self) -> None:
        """Test that prompt specifies PrimarySkills label."""
        from eval.positive_gen.positives_gen import _build_skills_prompt

        prompt = _build_skills_prompt("test resume", "Mid", "frontend")

        assert "PrimarySkills:" in prompt

    def test_specifies_secondary_skills_label(self) -> None:
        """Test that prompt specifies SecondarySkills label."""
        from eval.positive_gen.positives_gen import _build_skills_prompt

        prompt = _build_skills_prompt("test resume", "Mid", "frontend")

        assert "SecondarySkills:" in prompt

    def test_specifies_primary_count_constraint(self) -> None:
        """Test that prompt specifies primary skills count constraints."""
        from eval.positive_gen.positives_gen import _build_skills_prompt

        prompt = _build_skills_prompt("test resume", "Mid", "backend")

        # Should mention 2-4 range
        assert "2" in prompt and "4" in prompt

    def test_specifies_secondary_count_constraint(self) -> None:
        """Test that prompt specifies secondary skills count constraints."""
        from eval.positive_gen.positives_gen import _build_skills_prompt

        prompt = _build_skills_prompt("test resume", "Mid", "backend")

        # Should mention 1-3 range
        assert "1" in prompt and "3" in prompt


class TestGenerateSkills:
    """Tests for _generate_skills."""

    @patch("eval.positive_gen.positives_gen._call_ollama")
    def test_returns_parsed_primary_and_secondary_skills(
        self, mock_ollama: MagicMock
    ) -> None:
        """Test that skills are parsed and returned as tuple."""
        from eval.positive_gen.positives_gen import _generate_skills

        mock_ollama.return_value = """PrimarySkills: Python, Go, PostgreSQL
SecondarySkills: Redis, Docker"""

        resume_info: ResumeInfo = {
            "seniority": "Senior",
            "years_experience": 8,
            "primary_skills": [],
            "domain": "backend",
            "resume_text": "test",
        }

        primary, secondary = _generate_skills(resume_info, "model")

        assert primary == ["Python", "Go", "PostgreSQL"]
        assert secondary == ["Redis", "Docker"]

    @patch("eval.positive_gen.positives_gen._call_ollama")
    def test_uses_skills_max_tokens(self, mock_ollama: MagicMock) -> None:
        """Test that _call_ollama is called with SKILLS_MAX_TOKENS."""
        from eval.positive_gen.positives_gen import _generate_skills
        from config import SKILLS_MAX_TOKENS

        mock_ollama.return_value = """PrimarySkills: Python, Go
SecondarySkills: Redis"""

        resume_info: ResumeInfo = {
            "seniority": "Mid",
            "years_experience": 5,
            "primary_skills": [],
            "domain": "backend",
            "resume_text": "test",
        }

        _generate_skills(resume_info, "model")

        call_kwargs = mock_ollama.call_args[1]
        assert call_kwargs.get("max_tokens") == SKILLS_MAX_TOKENS

    @patch("eval.positive_gen.positives_gen._call_ollama")
    def test_raises_on_missing_primary_skills(self, mock_ollama: MagicMock) -> None:
        """Test that ValueError raised when PrimarySkills missing."""
        from eval.positive_gen.positives_gen import _generate_skills

        mock_ollama.return_value = "SecondarySkills: Redis"

        resume_info: ResumeInfo = {
            "seniority": "Mid",
            "years_experience": 5,
            "primary_skills": [],
            "domain": "backend",
            "resume_text": "test",
        }

        with pytest.raises(ValueError):
            _generate_skills(resume_info, "model")


class TestBuildResponsibilityPrompt:
    """Tests for _build_responsibility_prompt."""

    def test_includes_resume_text(self) -> None:
        """Test that prompt includes resume text."""
        from eval.positive_gen.positives_gen import _build_responsibility_prompt

        prompt = _build_responsibility_prompt(
            "test resume with Python", "Senior", "backend", ["Python", "Go"], []
        )

        assert "test resume with Python" in prompt

    def test_includes_primary_skills(self) -> None:
        """Test that prompt includes at least one primary skill."""
        from eval.positive_gen.positives_gen import _build_responsibility_prompt

        prompt = _build_responsibility_prompt(
            "test resume", "Mid", "frontend", ["JavaScript", "React"], []
        )

        assert "JavaScript" in prompt or "React" in prompt

    def test_includes_already_generated_as_do_not_repeat(self) -> None:
        """Test that already_generated items appear as do-not-repeat list."""
        from eval.positive_gen.positives_gen import _build_responsibility_prompt

        already_generated = [
            "Designed distributed systems",
            "Mentored junior engineers",
        ]
        prompt = _build_responsibility_prompt(
            "test resume", "Senior", "backend", ["Python"], already_generated
        )

        for responsibility in already_generated:
            assert responsibility in prompt

    def test_empty_already_generated_omits_do_not_repeat_section(self) -> None:
        """Test that empty already_generated doesn't include do-not-repeat section."""
        from eval.positive_gen.positives_gen import _build_responsibility_prompt

        prompt = _build_responsibility_prompt(
            "test resume", "Mid", "backend", ["Python"], []
        )

        # Should not have "do not repeat" boilerplate for empty list
        assert "do not repeat" not in prompt.lower() or "already written" not in prompt.lower()

    def test_requires_distinct_sentence(self) -> None:
        """Test that prompt mentions distinct or non-repetitive."""
        from eval.positive_gen.positives_gen import _build_responsibility_prompt

        prompt = _build_responsibility_prompt(
            "test resume", "Senior", "backend", ["Python"], ["Already done"]
        )

        assert "distinct" in prompt.lower() or "not repeat" in prompt.lower()


class TestGenerateSingleResponsibility:
    """Tests for _generate_single_responsibility."""

    @patch("eval.positive_gen.positives_gen._call_ollama")
    def test_returns_stripped_sentence(self, mock_ollama: MagicMock) -> None:
        """Test that returned sentence is stripped of whitespace."""
        from eval.positive_gen.positives_gen import _generate_single_responsibility

        mock_ollama.return_value = "  Built distributed caching layer using Redis and Python.  "

        resume_info: ResumeInfo = {
            "seniority": "Senior",
            "years_experience": 8,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "test",
        }

        result = _generate_single_responsibility(
            resume_info, ["Python"], [], "model"
        )

        assert result == "Built distributed caching layer using Redis and Python."

    @patch("eval.positive_gen.positives_gen._call_ollama")
    def test_raises_on_empty_response(self, mock_ollama: MagicMock) -> None:
        """Test that ValueError raised on empty response."""
        from eval.positive_gen.positives_gen import _generate_single_responsibility

        mock_ollama.return_value = ""

        resume_info: ResumeInfo = {
            "seniority": "Mid",
            "years_experience": 5,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "test",
        }

        with pytest.raises(ValueError):
            _generate_single_responsibility(resume_info, ["Python"], [], "model")

    @patch("eval.positive_gen.positives_gen._call_ollama")
    def test_raises_on_fewer_than_10_words(self, mock_ollama: MagicMock) -> None:
        """Test that ValueError raised when response has < 10 words."""
        from eval.positive_gen.positives_gen import _generate_single_responsibility

        mock_ollama.return_value = "Short sentence"

        resume_info: ResumeInfo = {
            "seniority": "Mid",
            "years_experience": 5,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "test",
        }

        with pytest.raises(ValueError):
            _generate_single_responsibility(resume_info, ["Python"], [], "model")

    @patch("eval.positive_gen.positives_gen._call_ollama")
    def test_uses_responsibility_max_tokens(self, mock_ollama: MagicMock) -> None:
        """Test that _call_ollama is called with RESPONSIBILITY_MAX_TOKENS."""
        from eval.positive_gen.positives_gen import _generate_single_responsibility
        from config import RESPONSIBILITY_MAX_TOKENS

        mock_ollama.return_value = (
            "Built and maintained scalable backend systems using Python and PostgreSQL."
        )

        resume_info: ResumeInfo = {
            "seniority": "Senior",
            "years_experience": 8,
            "primary_skills": ["Python"],
            "domain": "backend",
            "resume_text": "test",
        }

        _generate_single_responsibility(resume_info, ["Python"], [], "model")

        call_kwargs = mock_ollama.call_args[1]
        assert call_kwargs.get("max_tokens") == RESPONSIBILITY_MAX_TOKENS


class TestGenerateJobSkeleton:
    """Tests for generate_job_skeleton with new signature."""

    @pytest.fixture
    def sample_resume_info(self) -> ResumeInfo:
        """Provide sample ResumeInfo for tests."""
        return {
            "seniority": "Senior",
            "years_experience": 8,
            "primary_skills": ["Python", "Go"],
            "domain": "backend",
            "resume_text": "Alice Smith, 8 years Python/Go backend engineer...",
        }

    @patch("eval.positive_gen.positives_gen._generate_single_responsibility")
    @patch("eval.positive_gen.positives_gen._generate_skills")
    @patch("eval.positive_gen.positives_gen._extract_years_experience")
    def test_accepts_resume_info_dict(
        self,
        mock_extract_years: MagicMock,
        mock_gen_skills: MagicMock,
        mock_gen_resp: MagicMock,
        sample_resume_info: ResumeInfo,
    ) -> None:
        """Test that new signature accepts ResumeInfo dict."""
        mock_extract_years.return_value = 8
        mock_gen_skills.return_value = (["Python", "Go"], ["Docker"])
        mock_gen_resp.side_effect = [
            "Built distributed systems with Python.",
            "Optimized database queries.",
            "Led team of five engineers.",
            "Mentored junior developers.",
        ]

        result = generate_job_skeleton(sample_resume_info, "model")

        assert isinstance(result, dict)
        assert result["seniority"] == "Senior"

    @patch("eval.positive_gen.positives_gen._generate_single_responsibility")
    @patch("eval.positive_gen.positives_gen._generate_skills")
    @patch("eval.positive_gen.positives_gen._extract_years_experience")
    def test_extracts_years_from_resume_text(
        self,
        mock_extract_years: MagicMock,
        mock_gen_skills: MagicMock,
        mock_gen_resp: MagicMock,
        sample_resume_info: ResumeInfo,
    ) -> None:
        """Test that years_experience is extracted from resume text."""
        mock_extract_years.return_value = 8
        mock_gen_skills.return_value = (["Python", "Go"], ["Docker"])
        mock_gen_resp.side_effect = [
            "Built distributed systems with Python.",
            "Optimized database queries.",
            "Led team of five engineers.",
            "Mentored junior developers.",
        ]

        generate_job_skeleton(sample_resume_info, "model")

        mock_extract_years.assert_called_once()
        # Verify resume_text is passed
        call_args = mock_extract_years.call_args[0]
        assert call_args[0] == sample_resume_info["resume_text"]

    @patch("eval.positive_gen.positives_gen._generate_single_responsibility")
    @patch("eval.positive_gen.positives_gen._generate_skills")
    @patch("eval.positive_gen.positives_gen._extract_years_experience")
    def test_generates_skills_once(
        self,
        mock_extract_years: MagicMock,
        mock_gen_skills: MagicMock,
        mock_gen_resp: MagicMock,
        sample_resume_info: ResumeInfo,
    ) -> None:
        """Test that skills are generated once."""
        mock_extract_years.return_value = 8
        mock_gen_skills.return_value = (["Python", "Go"], ["Docker"])
        mock_gen_resp.side_effect = [
            "Built distributed systems with Python.",
            "Optimized database queries.",
            "Led team of five engineers.",
            "Mentored junior developers.",
        ]

        generate_job_skeleton(sample_resume_info, "model")

        mock_gen_skills.assert_called_once()

    @patch("eval.positive_gen.positives_gen._generate_single_responsibility")
    @patch("eval.positive_gen.positives_gen._generate_skills")
    @patch("eval.positive_gen.positives_gen._extract_years_experience")
    def test_collects_target_responsibility_count(
        self,
        mock_extract_years: MagicMock,
        mock_gen_skills: MagicMock,
        mock_gen_resp: MagicMock,
        sample_resume_info: ResumeInfo,
    ) -> None:
        """Test that responsibilities are collected until TARGET_RESPONSIBILITY_COUNT."""
        from config import TARGET_RESPONSIBILITY_COUNT

        mock_extract_years.return_value = 8
        mock_gen_skills.return_value = (["Python", "Go"], ["Docker"])
        mock_gen_resp.side_effect = [
            "Built distributed systems with Python.",
            "Optimized database queries.",
            "Led team of five engineers.",
            "Mentored junior developers.",
            "Wrote comprehensive documentation.",
        ]

        result = generate_job_skeleton(sample_resume_info, "model")

        # Should collect TARGET_RESPONSIBILITY_COUNT responsibilities
        assert len(result["responsibilities"]) == TARGET_RESPONSIBILITY_COUNT

    @patch("eval.positive_gen.positives_gen._generate_single_responsibility")
    @patch("eval.positive_gen.positives_gen._generate_skills")
    @patch("eval.positive_gen.positives_gen._extract_years_experience")
    def test_raises_when_fewer_than_3_responsibilities(
        self,
        mock_extract_years: MagicMock,
        mock_gen_skills: MagicMock,
        mock_gen_resp: MagicMock,
        sample_resume_info: ResumeInfo,
    ) -> None:
        """Test that ValueError raised when fewer than 3 responsibilities generated."""
        mock_extract_years.return_value = 8
        mock_gen_skills.return_value = (["Python", "Go"], ["Docker"])
        mock_gen_resp.side_effect = ValueError("Bad response")

        with pytest.raises(ValueError, match="Only generated"):
            generate_job_skeleton(sample_resume_info, "model")

    @patch("eval.positive_gen.positives_gen._generate_single_responsibility")
    @patch("eval.positive_gen.positives_gen._generate_skills")
    @patch("eval.positive_gen.positives_gen._extract_years_experience")
    def test_returns_valid_job_skeleton(
        self,
        mock_extract_years: MagicMock,
        mock_gen_skills: MagicMock,
        mock_gen_resp: MagicMock,
        sample_resume_info: ResumeInfo,
    ) -> None:
        """Test that returned object is a valid JobSkeleton with all fields."""
        mock_extract_years.return_value = 8
        mock_gen_skills.return_value = (["Python", "Go"], ["Docker"])
        mock_gen_resp.side_effect = [
            "Built distributed systems with Python.",
            "Optimized database queries.",
            "Led team of five engineers.",
            "Mentored junior developers.",
        ]

        result = generate_job_skeleton(sample_resume_info, "model")

        # Verify all 7 JobSkeleton fields present
        assert "title" in result
        assert "seniority" in result
        assert "years_required" in result
        assert "domain" in result
        assert "primary_skills" in result
        assert "secondary_skills" in result
        assert "responsibilities" in result

    @patch("eval.positive_gen.positives_gen._generate_single_responsibility")
    @patch("eval.positive_gen.positives_gen._generate_skills")
    @patch("eval.positive_gen.positives_gen._extract_years_experience")
    def test_years_required_le_years_experience(
        self,
        mock_extract_years: MagicMock,
        mock_gen_skills: MagicMock,
        mock_gen_resp: MagicMock,
        sample_resume_info: ResumeInfo,
    ) -> None:
        """Test that years_required max is <= years_experience."""
        from eval.positive_gen.positives_validate import _parse_years_required

        mock_extract_years.return_value = 5  # Only 5 years experience
        mock_gen_skills.return_value = (["Python", "Go"], ["Docker"])
        mock_gen_resp.side_effect = [
            "Built distributed systems with Python.",
            "Optimized database queries.",
            "Led team of five engineers.",
            "Mentored junior developers.",
        ]

        result = generate_job_skeleton(sample_resume_info, "model")

        max_required = _parse_years_required(result["years_required"])
        assert max_required <= 5

    @patch("eval.positive_gen.positives_gen._generate_single_responsibility")
    @patch("eval.positive_gen.positives_gen._generate_skills")
    @patch("eval.positive_gen.positives_gen._extract_years_experience")
    def test_skips_failed_responsibility_attempts(
        self,
        mock_extract_years: MagicMock,
        mock_gen_skills: MagicMock,
        mock_gen_resp: MagicMock,
        sample_resume_info: ResumeInfo,
    ) -> None:
        """Test that failed responsibility attempts are skipped."""
        from config import TARGET_RESPONSIBILITY_COUNT

        mock_extract_years.return_value = 8
        mock_gen_skills.return_value = (["Python", "Go"], ["Docker"])
        # Mix of failures and successes
        mock_gen_resp.side_effect = [
            ValueError("Bad"),
            "Built distributed systems with Python.",
            ValueError("Bad"),
            "Optimized database queries.",
            "Led team of five engineers.",
            ValueError("Bad"),
            "Mentored junior developers.",
        ]

        result = generate_job_skeleton(sample_resume_info, "model")

        # Should have exactly TARGET_RESPONSIBILITY_COUNT successful ones
        assert len(result["responsibilities"]) == TARGET_RESPONSIBILITY_COUNT
        # All should be strings (no ValueErrors)
        assert all(isinstance(r, str) for r in result["responsibilities"])

    @patch("eval.positive_gen.positives_gen._generate_single_responsibility")
    @patch("eval.positive_gen.positives_gen._generate_skills")
    @patch("eval.positive_gen.positives_gen._extract_years_experience")
    def test_accepts_custom_model(
        self,
        mock_extract_years: MagicMock,
        mock_gen_skills: MagicMock,
        mock_gen_resp: MagicMock,
        sample_resume_info: ResumeInfo,
    ) -> None:
        """Test that custom model is passed through to all internal calls."""
        mock_extract_years.return_value = 8
        mock_gen_skills.return_value = (["Python", "Go"], ["Docker"])
        mock_gen_resp.side_effect = [
            "Built distributed systems with Python.",
            "Optimized database queries.",
            "Led team of five engineers.",
            "Mentored junior developers.",
        ]

        generate_job_skeleton(sample_resume_info, "custom-model")

        # Verify model passed to extract_years
        assert mock_extract_years.call_args[0][1] == "custom-model"
        # Verify model passed to generate_skills
        assert mock_gen_skills.call_args[0][1] == "custom-model"
        # Verify model passed to generate_single_responsibility
        assert mock_gen_resp.call_args[0][2] == "custom-model"
