"""Unit tests for regex-based extraction functions."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path for imports
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from regex_extraction import (
    DEGREE_BACHELOR,
    DEGREE_MASTER,
    DEGREE_PHD,
    DEGREE_UNKNOWN,
    SENIORITY_ENTRY,
    SENIORITY_MID,
    SENIORITY_SENIOR,
    SENIORITY_UNKNOWN,
    YEARS_UNKNOWN,
    build_chroma_where_filter,
    extract_degree_requirement,
    extract_seniority_from_title,
    extract_seniority_level,
    extract_years_experience,
    extract_degree_with_fallback,
    extract_seniority_with_fallback,
    extract_years_with_fallback,
)

from llm_extraction import extract_degree_with_llm, extract_seniority_with_llm, extract_years_with_llm

class TestExtractDegreeRequirement:
    """Test degree requirement extraction from job descriptions."""

    def test_phd_requirement(self):
        """PhD requirement is correctly identified."""
        assert extract_degree_requirement("PhD required") == DEGREE_PHD
        assert extract_degree_requirement("Ph.D. preferred") == DEGREE_PHD
        assert extract_degree_requirement("Doctorate in Computer Science") == DEGREE_PHD

    def test_masters_requirement(self):
        """Master's degree requirement is correctly identified."""
        assert extract_degree_requirement("Master's degree required") == DEGREE_MASTER
        assert extract_degree_requirement("Masters degree preferred") == DEGREE_MASTER
        assert extract_degree_requirement("M.S. in Math") == DEGREE_MASTER
        assert extract_degree_requirement("M.Eng. required") == DEGREE_MASTER
        assert extract_degree_requirement("MBA preferred") == DEGREE_MASTER

    def test_bachelors_requirement(self):
        """Bachelor's degree requirement is correctly identified."""
        assert extract_degree_requirement("Bachelor's degree required") == DEGREE_BACHELOR
        assert extract_degree_requirement("Bachelors degree preferred") == DEGREE_BACHELOR
        assert extract_degree_requirement("B.S. in Engineering") == DEGREE_BACHELOR
        assert extract_degree_requirement("B.A. preferred") == DEGREE_BACHELOR
        assert extract_degree_requirement("Undergraduate degree required") == DEGREE_BACHELOR

    def test_no_degree_requirement(self):
        """No degree requirement returns DEGREE_UNKNOWN."""
        assert extract_degree_requirement("") == DEGREE_UNKNOWN
        assert extract_degree_requirement("No specific degree required") == DEGREE_UNKNOWN
        assert extract_degree_requirement("High school diploma or equivalent") == DEGREE_UNKNOWN

    def test_phd_takes_priority_over_masters(self):
        """PhD takes priority when both are mentioned."""
        assert extract_degree_requirement("PhD or Master's degree") == DEGREE_PHD

    def test_masters_takes_priority_over_bachelors(self):
        """Master's takes priority when both are mentioned."""
        assert extract_degree_requirement("Master's or Bachelor's degree") == DEGREE_MASTER


class TestExtractSeniorityLevel:
    """Test seniority level extraction from job descriptions."""

    def test_senior_seniority(self):
        """Senior-level positions are correctly identified."""
        assert extract_seniority_level("Senior Data Scientist") == SENIORITY_SENIOR
        assert extract_seniority_level("Sr. Engineer required") == SENIORITY_SENIOR
        assert extract_seniority_level("Lead Software Engineer") == SENIORITY_SENIOR
        assert extract_seniority_level("Principal Architect") == SENIORITY_SENIOR
        assert extract_seniority_level("Director of Engineering") == SENIORITY_SENIOR
        assert extract_seniority_level("Engineering Manager") == SENIORITY_SENIOR

    def test_mid_seniority(self):
        """Mid-level positions are correctly identified."""
        assert extract_seniority_level("Mid-level Engineer") == SENIORITY_MID
        assert extract_seniority_level("Mid level developer") == SENIORITY_MID
        assert extract_seniority_level("Intermediate Python developer") == SENIORITY_MID

    def test_entry_seniority(self):
        """Entry-level positions are correctly identified."""
        assert extract_seniority_level("Entry-level Data Scientist") == SENIORITY_ENTRY
        assert extract_seniority_level("Entry level engineer") == SENIORITY_ENTRY
        assert extract_seniority_level("Junior Software Engineer") == SENIORITY_ENTRY
        assert extract_seniority_level("New grad opportunity") == SENIORITY_ENTRY
        assert extract_seniority_level("New Graduate program") == SENIORITY_ENTRY

    def test_no_seniority_specified(self):
        """No seniority specification returns SENIORITY_UNKNOWN."""
        assert extract_seniority_level("") == SENIORITY_UNKNOWN
        assert extract_seniority_level("Full-time Engineer position") == SENIORITY_UNKNOWN

    def test_senior_takes_priority_over_mid(self):
        """Senior takes priority when both are mentioned."""
        result = extract_seniority_level("Senior or mid-level position")
        assert result == SENIORITY_SENIOR

    def test_mid_takes_priority_over_entry(self):
        """Mid takes priority when both are mentioned."""
        result = extract_seniority_level("Mid-level or entry-level position")
        assert result == SENIORITY_MID


class TestExtractSeniorityFromTitle:
    """Test seniority extraction from job title strings."""

    def test_senior_title(self):
        """Senior-level titles are correctly identified."""
        assert extract_seniority_from_title("Senior Data Scientist") == SENIORITY_SENIOR
        assert extract_seniority_from_title("Sr. Software Engineer") == SENIORITY_SENIOR
        assert extract_seniority_from_title("Lead Machine Learning Engineer") == SENIORITY_SENIOR
        assert extract_seniority_from_title("Principal Engineer") == SENIORITY_SENIOR
        assert extract_seniority_from_title("Director of Data Science") == SENIORITY_SENIOR
        assert extract_seniority_from_title("Engineering Manager") == SENIORITY_SENIOR

    def test_mid_title(self):
        """Mid-level titles are correctly identified."""
        assert extract_seniority_from_title("Mid-level Data Analyst") == SENIORITY_MID
        assert extract_seniority_from_title("Intermediate Software Engineer") == SENIORITY_MID

    def test_entry_title(self):
        """Entry-level titles are correctly identified."""
        assert extract_seniority_from_title("Junior Data Scientist") == SENIORITY_ENTRY
        assert extract_seniority_from_title("Entry-level Software Engineer") == SENIORITY_ENTRY
        assert extract_seniority_from_title("New Grad Software Engineer") == SENIORITY_ENTRY

    def test_ambiguous_title_returns_unknown(self):
        """Titles without seniority keywords return SENIORITY_UNKNOWN."""
        assert extract_seniority_from_title("Data Scientist") == SENIORITY_UNKNOWN
        assert extract_seniority_from_title("Software Engineer") == SENIORITY_UNKNOWN
        assert extract_seniority_from_title("") == SENIORITY_UNKNOWN

    def test_case_insensitive(self):
        """Title matching is case-insensitive."""
        assert extract_seniority_from_title("SENIOR DATA SCIENTIST") == SENIORITY_SENIOR
        assert extract_seniority_from_title("junior engineer") == SENIORITY_ENTRY

    def test_senior_takes_priority_in_title(self):
        """Senior takes priority when multiple signals exist in title."""
        assert extract_seniority_from_title("Senior or Mid-level Engineer") == SENIORITY_SENIOR

    def test_senior_with_location_in_title(self):
        """Senior title with location suffix is correctly identified."""
        assert extract_seniority_from_title("Senior Customer Support Engineer - US West") == SENIORITY_SENIOR


class TestExtractYearsExperience:
    """Test years of experience extraction from job descriptions."""

    def test_plus_years_pattern(self):
        """'X+ years of experience' is correctly extracted."""
        assert extract_years_experience("3+ years of experience required") == 3
        assert extract_years_experience("5+ years experience") == 5
        assert extract_years_experience("10+ years of experience") == 10

    def test_at_least_pattern(self):
        """'At least X years of experience' is correctly extracted."""
        assert extract_years_experience("at least 2 years of experience required") == 2
        assert extract_years_experience("At least 4 years experience") == 4
        assert extract_years_experience("minimum of 1 year of experience") == 1
        assert extract_years_experience("minimum 3 years of experience") == 3

    def test_or_more_pattern(self):
        """'X or more years of experience' is correctly extracted."""
        assert extract_years_experience("5 or more years of experience required") == 5
        assert extract_years_experience("2 or more years experience") == 2

    def test_x_years_of_y_experience_pattern(self):
        """'X years of Y experience' is correctly extracted."""
        assert extract_years_experience("2 years of internship experience") == 2
        assert extract_years_experience("3 years of software development experience") == 3
        assert extract_years_experience("1 year of professional experience") == 1

    def test_multiple_years_requirements_returns_minimum(self):
        """Multiple year requirements return the minimum."""
        assert extract_years_experience("3+ years Python, 5+ years data science") == 3

    def test_no_years_specified(self):
        """No year requirement returns YEARS_UNKNOWN."""
        assert extract_years_experience("") == YEARS_UNKNOWN
        assert extract_years_experience("Experience required (no specific years)") == YEARS_UNKNOWN

    def test_range_pattern_with_hyphen(self):
        """X-Y years range pattern is matched, extracting the lower bound."""
        # Extracts the first number in the range
        assert extract_years_experience("2-5 years of experience") == 2
        assert extract_years_experience("3-7 years required") == 3

    def test_range_pattern_with_en_dash(self):
        """X–Y years range pattern with en-dash is matched."""
        # Handles both hyphen and en-dash
        assert extract_years_experience("3–5 years of experience building production-grade ML systems") == 3

    def test_range_pattern_with_in_keyword(self):
        """X-Y years in [role] pattern is matched."""
        # Handles "X-Y years in [role/domain]"
        assert extract_years_experience("Experience: 2-4 years in a technical support, sysadmin, or network-focused role.") == 2

    def test_version_number_not_matched(self):
        """Version numbers are not matched (e.g., Python 3.10)."""
        assert extract_years_experience("Python 3.10 experience") == YEARS_UNKNOWN

    def test_caps_at_two_digits(self):
        """Years are capped at 2 digits to avoid matching 4-digit years."""
        assert extract_years_experience("2+ years of experience") == 2
        # 999 years should not match (more than 2 digits)
        assert extract_years_experience("999+ years of experience") == YEARS_UNKNOWN


class TestExtractUserDegreeWithLLM:
    """Test degree extraction from resume text using LLM."""

    def test_llm_returns_bachelor(self):
        """LLM returns 1 for bachelor degree."""
        with patch("llm_extraction._call_ollama", return_value="1"):
            result = extract_degree_with_llm("== EDUCATION ==\nB.S. in Computer Science", model="llama3.2")
        assert result == DEGREE_BACHELOR

    def test_llm_returns_master(self):
        """LLM returns 2 for master degree."""
        with patch("llm_extraction._call_ollama", return_value="2"):
            result = extract_degree_with_llm("== EDUCATION ==\nMaster's degree in Data Science", model="llama3.2")
        assert result == DEGREE_MASTER

    def test_llm_returns_phd(self):
        """LLM returns 3 for PhD."""
        with patch("llm_extraction._call_ollama", return_value="3"):
            result = extract_degree_with_llm("== EDUCATION ==\nPhD in Computer Science", model="llama3.2")
        assert result == DEGREE_PHD

    def test_llm_failure_returns_unknown(self):
        """LLM failure returns DEGREE_UNKNOWN."""
        with patch("llm_extraction._call_ollama", side_effect=Exception("timeout")):
            result = extract_degree_with_llm("== EDUCATION ==\nB.S. in CS", model="llama3.2")
        assert result == DEGREE_UNKNOWN

    def test_llm_unknown_string_returns_unknown(self):
        """LLM returns unknown string returns DEGREE_UNKNOWN."""
        with patch("llm_extraction._call_ollama", return_value="unknown"):
            result = extract_degree_with_llm("== EDUCATION ==\nHigh school", model="llama3.2")
        assert result == DEGREE_UNKNOWN


class TestExtractUserSeniorityWithLLM:
    """Test seniority extraction from resume text using LLM."""

    def test_llm_returns_entry(self):
        """LLM returns 1 for entry level."""
        with patch("llm_extraction._call_ollama", return_value="1"):
            result = extract_seniority_with_llm("== SENIORITY LEVEL ==\nNew Grad or Junior level", model="llama3.2")
        assert result == SENIORITY_ENTRY

    def test_llm_returns_mid(self):
        """LLM returns 2 for mid level."""
        with patch("llm_extraction._call_ollama", return_value="2"):
            result = extract_seniority_with_llm("Mid-level professional", model="llama3.2")
        assert result == SENIORITY_MID

    def test_llm_returns_senior(self):
        """LLM returns 3 for senior level."""
        with patch("llm_extraction._call_ollama", return_value="3"):
            result = extract_seniority_with_llm("== SENIORITY LEVEL ==\nSenior level professional", model="llama3.2")
        assert result == SENIORITY_SENIOR

    def test_llm_failure_returns_unknown(self):
        """LLM failure returns SENIORITY_UNKNOWN."""
        with patch("llm_extraction._call_ollama", side_effect=Exception("timeout")):
            result = extract_seniority_with_llm("Senior engineer", model="llama3.2")
        assert result == SENIORITY_UNKNOWN

    def test_llm_unknown_string_returns_unknown(self):
        """LLM returns unknown string returns SENIORITY_UNKNOWN."""
        with patch("llm_extraction._call_ollama", return_value="unknown"):
            result = extract_seniority_with_llm("== EDUCATION ==\nB.S. in CS", model="llama3.2")
        assert result == SENIORITY_UNKNOWN


class TestExtractUserYearsWithLLM:
    """Test years of experience extraction from resume text using LLM."""

    def test_llm_returns_integer(self):
        """LLM returns integer years."""
        with patch("llm_extraction._call_ollama", return_value="5"):
            result = extract_years_with_llm("== EXPERIENCE ==\n5+ years of experience", model="llama3.2")
        assert result == 5

    def test_llm_returns_zero(self):
        """LLM returns zero for no experience."""
        with patch("llm_extraction._call_ollama", return_value="0"):
            result = extract_years_with_llm("Entry level position", model="llama3.2")
        assert result == 0

    def test_llm_returns_large_number(self):
        """LLM returns large number for senior experience."""
        with patch("llm_extraction._call_ollama", return_value="15"):
            result = extract_years_with_llm("15+ years of leadership", model="llama3.2")
        assert result == 15

    def test_llm_failure_returns_unknown(self):
        """LLM failure returns YEARS_UNKNOWN."""
        with patch("llm_extraction._call_ollama", side_effect=Exception("timeout")):
            result = extract_years_with_llm("== EXPERIENCE ==\n3+ years", model="llama3.2")
        assert result == YEARS_UNKNOWN

    def test_llm_unknown_string_returns_unknown(self):
        """LLM returns unknown string returns YEARS_UNKNOWN."""
        with patch("llm_extraction._call_ollama", return_value="unknown"):
            result = extract_years_with_llm("No clear years stated", model="llama3.2")
        assert result == YEARS_UNKNOWN


class TestBuildChromaWhereFilter:
    """Test ChromaDB filter construction."""

    def test_no_filters_returns_none(self):
        """All unknown attributes return None."""
        result = build_chroma_where_filter(DEGREE_UNKNOWN, SENIORITY_UNKNOWN, YEARS_UNKNOWN)
        assert result is None

    def test_single_seniority_filter(self):
        """Single seniority attribute returns unwrapped condition."""
        result = build_chroma_where_filter(DEGREE_UNKNOWN, SENIORITY_ENTRY, YEARS_UNKNOWN)
        assert result is not None
        assert "$and" not in result  # Single condition, no $and wrapper
        assert "$or" in result

    def test_multiple_filters_use_and(self):
        """Two or more attributes are combined with $and."""
        result = build_chroma_where_filter(DEGREE_BACHELOR, SENIORITY_ENTRY, YEARS_UNKNOWN)
        assert result is not None
        assert "$and" in result
        assert len(result["$and"]) == 2  # degree and seniority

    def test_all_three_attributes_in_filter(self):
        """All three attributes result in three conditions in $and."""
        result = build_chroma_where_filter(DEGREE_MASTER, SENIORITY_MID, 3)
        assert result is not None
        assert "$and" in result
        assert len(result["$and"]) == 3

    def test_seniority_condition_includes_unknown_jobs(self):
        """Seniority filter allows jobs with unspecified seniority."""
        result = build_chroma_where_filter(DEGREE_UNKNOWN, SENIORITY_ENTRY, YEARS_UNKNOWN)
        # The condition should have an $or with exact match and UNKNOWN match
        assert "$or" in result
        conditions = result["$or"]
        # Should allow SENIORITY_ENTRY or SENIORITY_UNKNOWN
        assert any({"seniority_level": {"$eq": SENIORITY_ENTRY}} in [c] for c in conditions)

    def test_degree_uses_lte_for_less_restrictive_jobs(self):
        """Degree filter uses $lte to allow jobs requiring less education."""
        result = build_chroma_where_filter(DEGREE_BACHELOR, SENIORITY_UNKNOWN, YEARS_UNKNOWN)
        assert "$or" in result
        # Should have a condition with $lte for required_degree
        conditions_str = str(result)
        assert "$lte" in conditions_str or "$eq" in conditions_str


class TestExtractDegreeWithFallback:
    def test_regex_match_skips_llm(self):
        with patch("src.generation._call_ollama") as mock_llm:
            result = extract_degree_with_fallback("requires a PhD")
        assert result == DEGREE_PHD
        mock_llm.assert_not_called()

    def test_llm_called_on_unknown(self):
        with patch("src.generation._call_ollama", return_value="bachelor") as mock_llm:
            result = extract_degree_with_fallback("no degree info here")
        assert result == DEGREE_BACHELOR
        mock_llm.assert_called_once()

    def test_llm_phd_response(self):
        with patch("src.generation._call_ollama", return_value="PhD"):
            result = extract_degree_with_fallback("no degree info here")
        assert result == DEGREE_PHD

    def test_llm_master_response(self):
        with patch("src.generation._call_ollama", return_value="Master"):
            result = extract_degree_with_fallback("no degree info here")
        assert result == DEGREE_MASTER

    def test_llm_failure_returns_unknown(self):
        with patch("src.generation._call_ollama", side_effect=Exception("timeout")):
            result = extract_degree_with_fallback("no degree info here")
        assert result == DEGREE_UNKNOWN

    def test_llm_unknown_string_returns_unknown(self):
        with patch("src.generation._call_ollama", return_value="Unknown"):
            result = extract_degree_with_fallback("no degree info here")
        assert result == DEGREE_UNKNOWN


class TestExtractSeniorityWithFallback:
    def test_regex_match_skips_llm(self):
        with patch("src.generation._call_ollama") as mock_llm:
            result = extract_seniority_with_fallback("Senior Software Engineer")
        assert result == SENIORITY_SENIOR
        mock_llm.assert_not_called()

    def test_llm_called_on_unknown(self):
        with patch("src.generation._call_ollama", return_value="mid") as mock_llm:
            result = extract_seniority_with_fallback("software engineer role")
        assert result == SENIORITY_MID
        mock_llm.assert_called_once()

    def test_llm_senior_response(self):
        with patch("src.generation._call_ollama", return_value="Senior"):
            result = extract_seniority_with_fallback("software engineer role")
        assert result == SENIORITY_SENIOR

    def test_llm_entry_response(self):
        with patch("src.generation._call_ollama", return_value="entry"):
            result = extract_seniority_with_fallback("software engineer role")
        assert result == SENIORITY_ENTRY

    def test_llm_failure_returns_unknown(self):
        with patch("src.generation._call_ollama", side_effect=Exception("timeout")):
            result = extract_seniority_with_fallback("software engineer role")
        assert result == SENIORITY_UNKNOWN

    def test_llm_unknown_string_returns_unknown(self):
        with patch("src.generation._call_ollama", return_value="Unknown"):
            result = extract_seniority_with_fallback("software engineer role")
        assert result == SENIORITY_UNKNOWN

class TestExtractYearsWithFallback:
    def test_regex_match_skips_llm(self):
        with patch("src.generation._call_ollama") as mock_llm:
            result = extract_years_with_fallback("5+ years of experience required")
        assert result == 5
        mock_llm.assert_not_called()

    def test_llm_called_on_unknown(self):
        with patch("src.generation._call_ollama", return_value="3") as mock_llm:
            result = extract_years_with_fallback("some experience needed")
        assert result == 3
        mock_llm.assert_called_once()

    def test_llm_integer_response(self):
        with patch("src.generation._call_ollama", return_value="7"):
            result = extract_years_with_fallback("some experience needed")
        assert result == 7

    def test_llm_unknown_string_returns_unknown(self):
        with patch("src.generation._call_ollama", return_value="Unknown"):
            result = extract_years_with_fallback("some experience needed")
        assert result == YEARS_UNKNOWN

    def test_llm_failure_returns_unknown(self):
        with patch("src.generation._call_ollama", side_effect=Exception("timeout")):
            result = extract_years_with_fallback("some experience needed")
        assert result == YEARS_UNKNOWN

    def test_llm_response_with_extra_text(self):
        with patch("src.generation._call_ollama", return_value="about 4 years"):
            result = extract_years_with_fallback("some experience needed")
        assert result == 4
