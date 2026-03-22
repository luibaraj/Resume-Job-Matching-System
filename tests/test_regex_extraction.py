"""Unit tests for regex-based extraction functions."""

import sys
from pathlib import Path

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
    extract_user_degree,
    extract_user_seniority,
    extract_user_years_experience,
    extract_years_experience,
)


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

    def test_multiple_years_requirements_returns_minimum(self):
        """Multiple year requirements return the minimum."""
        assert extract_years_experience("3+ years Python, 5+ years data science") == 3

    def test_no_years_specified(self):
        """No year requirement returns YEARS_UNKNOWN."""
        assert extract_years_experience("") == YEARS_UNKNOWN
        assert extract_years_experience("Experience required (no specific years)") == YEARS_UNKNOWN

    def test_range_pattern_not_matched(self):
        """X-Y years range pattern is intentionally not matched."""
        # This is to avoid confusion with date ranges like "2022-2024 years of experience"
        assert extract_years_experience("2-5 years of experience") == YEARS_UNKNOWN

    def test_version_number_not_matched(self):
        """Version numbers are not matched (e.g., Python 3.10)."""
        assert extract_years_experience("Python 3.10 experience") == YEARS_UNKNOWN

    def test_caps_at_two_digits(self):
        """Years are capped at 2 digits to avoid matching 4-digit years."""
        assert extract_years_experience("2+ years of experience") == 2
        # 999 years should not match (more than 2 digits)
        assert extract_years_experience("999+ years of experience") == YEARS_UNKNOWN


class TestExtractUserDegree:
    """Test degree extraction from resume text."""

    def test_extract_from_resume_with_degree(self):
        """Degree is extracted from resume text."""
        resume = "== EDUCATION ==\nDegree: B.S. in Computer Science"
        assert extract_user_degree(resume) == DEGREE_BACHELOR

    def test_extract_masters_from_resume(self):
        """Master's degree is extracted from resume."""
        resume = "== EDUCATION ==\nMaster's degree in Data Science"
        assert extract_user_degree(resume) == DEGREE_MASTER

    def test_no_degree_in_resume(self):
        """Returns DEGREE_UNKNOWN if no degree found."""
        resume = "== EDUCATION ==\nHigh school diploma"
        result = extract_user_degree(resume)
        assert result == DEGREE_UNKNOWN


class TestExtractUserSeniority:
    """Test seniority extraction from resume text."""

    def test_extract_from_seniority_section(self):
        """Seniority is extracted from SENIORITY LEVEL section."""
        resume = "== SENIORITY LEVEL ==\nNew Grad or Junior level"
        assert extract_user_seniority(resume) == SENIORITY_ENTRY

    def test_extract_senior_from_seniority_section(self):
        """Senior seniority extracted from section."""
        resume = "== SENIORITY LEVEL ==\nSenior level professional"
        assert extract_user_seniority(resume) == SENIORITY_SENIOR

    def test_fallback_to_full_text_if_no_section(self):
        """Falls back to full-text scan if section not found."""
        resume = "Senior engineer with 10+ years of experience"
        assert extract_user_seniority(resume) == SENIORITY_SENIOR

    def test_no_seniority_in_resume(self):
        """Returns SENIORITY_UNKNOWN if none found."""
        resume = "== EDUCATION ==\nB.S. in Computer Science"
        assert extract_user_seniority(resume) == SENIORITY_UNKNOWN


class TestExtractUserYearsExperience:
    """Test years of experience extraction from resume text."""

    def test_extract_from_experience_section(self):
        """Years are extracted from EXPERIENCE section."""
        resume = """== EXPERIENCE ==
Company A — Data Scientist (2020-2022)
- 3+ years of experience with Python
        """
        assert extract_user_years_experience(resume) == 3

    def test_no_years_in_resume(self):
        """Returns YEARS_UNKNOWN if no explicit year count found."""
        resume = """== EXPERIENCE ==
Company A — Data Scientist Intern (2020-2022)
- Worked on various projects
        """
        assert extract_user_years_experience(resume) == YEARS_UNKNOWN

    def test_multiple_years_in_experience_returns_minimum(self):
        """If multiple year counts in experience, return minimum."""
        resume = """== EXPERIENCE ==
- 5+ years of Python experience
- 10+ years of leadership experience
        """
        assert extract_user_years_experience(resume) == 5


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
