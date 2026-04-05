"""
Regex-based extraction of structured job requirements and resume attributes.

All patterns are designed for precision over recall: ambiguous matches return UNKNOWN
rather than risk filtering out valid candidates. A false negative (missing a value)
simply skips a filter condition; a false positive could incorrectly exclude good matches.
"""

import re
import warnings
from typing import Optional

from src.config import (
    DEGREE_BACHELOR,
    DEGREE_MASTER,
    DEGREE_PHD,
    DEGREE_UNKNOWN,
    SENIORITY_ENTRY,
    SENIORITY_MID,
    SENIORITY_SENIOR,
    SENIORITY_UNKNOWN,
    YEARS_UNKNOWN,
)

# ===== Degree requirement patterns =====
# Require full phrases or dotted abbreviations only (no bare "bs"/"ms").

_DEGREE_PHD_RE = re.compile(
    r"\b(ph\.d\.?|phd|doctorate|doctoral\s+degree)\b",
    re.IGNORECASE,
)

_DEGREE_MASTER_RE = re.compile(
    r"(?:master(?:'?s)?(?:\s+degree)?|m\.s\.(?:\s|$)|m\.eng\.(?:\s|$)|mba)(?:\s|$|\b)",
    re.IGNORECASE,
)

_DEGREE_BACHELOR_RE = re.compile(
    r"(?:bachelor(?:'?s)?(?:\s+degree)?|b\.s\.(?:\s|$)|b\.a\.(?:\s|$)|undergraduate(?:\s+degree)?)(?:\s|$|\b)",
    re.IGNORECASE,
)

# ===== Seniority level patterns =====
# Dropped ambiguous terms: "staff", "associate", "head of", standalone "entry"/"mid".

_SENIORITY_SENIOR_RE = re.compile(
    r"\b(senior|sr\.\s|lead|principal|director|manager)\b",
    re.IGNORECASE,
)

_SENIORITY_MID_RE = re.compile(
    r"\b(mid-level|mid\s+level|intermediate)\b",
    re.IGNORECASE,
)

_SENIORITY_ENTRY_RE = re.compile(
    r"\b(entry-level|entry\s+level|junior|new\s+grad(?:uate)?)\b",
    re.IGNORECASE,
)

# ===== Years of experience patterns =====
# Require "experience" or "years" to avoid matching dates/versions.
# Pattern: (1-2 digits)+ optional_whitespace years(?) optional("of experience" or just "experience")
# Supports: "3+ years", "at least 5 years", "5 or more years", "3-5 years", "3–5 years", "2 years of X experience"

_YEARS_PLUS_RE = re.compile(
    r"(?<!\d)(\d{1,2})\+\s*years?\b(?:\s+(?:of\s+)?experience)?",
    re.IGNORECASE,
)

_YEARS_AT_LEAST_RE = re.compile(
    r"(?:at\s+least|minimum\s+of|minimum)\s+(?<!\d)(\d{1,2})\s+years?\b(?:\s+(?:of\s+)?experience)?",
    re.IGNORECASE,
)

_YEARS_OR_MORE_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s+or\s+more\s+years?\b(?:\s+(?:of\s+)?experience)?",
    re.IGNORECASE,
)

_YEARS_RANGE_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s*[-–]\s*\d{1,2}\s+years?\b(?:\s+(?:of\s+)?(?:in\s+)?(?:[\w\s]+?\s+)?experience)?",
    re.IGNORECASE,
)

_YEARS_OF_EXPERIENCE_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s+years?\s+of\s+[\w\s]+?experience\b",
    re.IGNORECASE,
)

# ===== Resume section extraction patterns =====

_RESUME_SENIORITY_SECTION_RE = re.compile(
    r"==\s*SENIORITY LEVEL\s*==(.+?)(?:==|\Z)",
    re.IGNORECASE | re.DOTALL,
)

_RESUME_EXPERIENCE_SECTION_RE = re.compile(
    r"==\s*EXPERIENCE\s*==(.+?)(?:==|\Z)",
    re.IGNORECASE | re.DOTALL,
)


# ===== Job description extraction =====


def extract_degree_requirement(text: str) -> int:
    """
    Extract the highest degree requirement from a job description.

    Checks PhD → Master's → Bachelor's in priority order.

    Args:
        text: Cleaned plain-text job description.

    Returns:
        DEGREE_PHD (3), DEGREE_MASTER (2), DEGREE_BACHELOR (1), or DEGREE_UNKNOWN (0).
    """
    if not text:
        return DEGREE_UNKNOWN
    if _DEGREE_PHD_RE.search(text):
        return DEGREE_PHD
    if _DEGREE_MASTER_RE.search(text):
        return DEGREE_MASTER
    if _DEGREE_BACHELOR_RE.search(text):
        return DEGREE_BACHELOR
    return DEGREE_UNKNOWN


def extract_seniority_level(text: str) -> int:
    """
    Extract seniority level from a job description.

    Checks senior → mid → entry in priority order.

    Args:
        text: Cleaned plain-text job description.

    Returns:
        SENIORITY_SENIOR (3), SENIORITY_MID (2), SENIORITY_ENTRY (1),
        or SENIORITY_UNKNOWN (0).
    """
    if not text:
        return SENIORITY_UNKNOWN
    if _SENIORITY_SENIOR_RE.search(text):
        return SENIORITY_SENIOR
    if _SENIORITY_MID_RE.search(text):
        return SENIORITY_MID
    if _SENIORITY_ENTRY_RE.search(text):
        return SENIORITY_ENTRY
    return SENIORITY_UNKNOWN


def extract_seniority_from_title(title: str) -> int:
    """
    Extract seniority level from a job title string.

    Delegates to extract_seniority_level, reusing the same regex patterns.
    Intended as a fallback when the job description yields SENIORITY_UNKNOWN.

    Args:
        title: Raw job title string (e.g., "Senior Data Scientist").

    Returns:
        SENIORITY_SENIOR (3), SENIORITY_MID (2), SENIORITY_ENTRY (1),
        or SENIORITY_UNKNOWN (0).
    """
    return extract_seniority_level(title)


def extract_years_experience(text: str) -> int:
    """
    Extract minimum years of experience required from a job description.

    Collects all matches from all patterns; returns the minimum found value.
    Handles: "3+ years of experience", "at least 2 years of experience",
    "5 or more years of experience", "3-5 years of experience", "2 years of internship experience".

    Args:
        text: Cleaned plain-text job description.

    Returns:
        Minimum years found as int, or YEARS_UNKNOWN (-1) if none found.
    """
    if not text:
        return YEARS_UNKNOWN

    found: list[int] = []
    for pattern in (_YEARS_PLUS_RE, _YEARS_AT_LEAST_RE, _YEARS_OR_MORE_RE, _YEARS_RANGE_RE, _YEARS_OF_EXPERIENCE_RE):
        for match in pattern.finditer(text):
            val = int(match.group(1))
            found.append(val)

    return min(found) if found else YEARS_UNKNOWN


# ===== Resume text extraction =====


def extract_user_degree(resume_text: str) -> int:
    """
    Extract degree level from resume text.

    Reuses the same patterns as job descriptions.

    Args:
        resume_text: Full resume text.

    Returns:
        DEGREE_* constant.
    """
    warnings.warn(
        "extract_user_degree is deprecated. Use llm_extraction.extract_degree_with_llm instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return extract_degree_requirement(resume_text)


def extract_user_seniority(resume_text: str) -> int:
    """
    Extract seniority level from resume's SENIORITY LEVEL section.

    Extracts and searches the `== SENIORITY LEVEL ==` section if present;
    falls back to full-text scan if section not found.

    Args:
        resume_text: Full resume text.

    Returns:
        SENIORITY_* constant.
    """
    warnings.warn(
        "extract_user_seniority is deprecated. Use llm_extraction.extract_seniority_with_llm instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not resume_text:
        return SENIORITY_UNKNOWN

    section_match = _RESUME_SENIORITY_SECTION_RE.search(resume_text)
    search_text = section_match.group(1) if section_match else resume_text

    if _SENIORITY_SENIOR_RE.search(search_text):
        return SENIORITY_SENIOR
    if _SENIORITY_MID_RE.search(search_text):
        return SENIORITY_MID
    if _SENIORITY_ENTRY_RE.search(search_text):
        return SENIORITY_ENTRY
    return SENIORITY_UNKNOWN


def extract_user_years_experience(resume_text: str) -> int:
    """
    Extract years of experience from resume text.

    Scans the entire resume text for year patterns.

    Args:
        resume_text: Full resume text.

    Returns:
        Minimum years found as int, or YEARS_UNKNOWN (-1).
    """
    warnings.warn(
        "extract_user_years_experience is deprecated. Use llm_extraction.extract_years_with_llm instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not resume_text:
        return YEARS_UNKNOWN

    found: list[int] = []
    for pattern in (_YEARS_PLUS_RE, _YEARS_AT_LEAST_RE, _YEARS_OR_MORE_RE, _YEARS_RANGE_RE, _YEARS_OF_EXPERIENCE_RE):
        for match in pattern.finditer(resume_text):
            found.append(int(match.group(1)))

    return min(found) if found else YEARS_UNKNOWN


# ===== Filter construction =====


def build_chroma_where_filter(
    user_degree: int,
    user_seniority: int,
    user_years: int,
) -> Optional[dict]:
    """
    Construct a ChromaDB `where` filter dict from user profile attributes.

    Only includes a condition if the user attribute was successfully extracted
    (not a sentinel). Returns None if no attributes were extracted (no filtering).

    Filtering logic per attribute:
    - seniority: jobs must match user seniority level OR have unspecified seniority
    - degree: jobs must require at most user's degree level OR have no stated requirement
    - years: jobs must require at most user's years OR have no stated requirement

    Args:
        user_degree: DEGREE_* constant from extract_user_degree().
        user_seniority: SENIORITY_* constant from extract_user_seniority().
        user_years: int from extract_user_years_experience(), or YEARS_UNKNOWN.

    Returns:
        A ChromaDB-compatible where dict, or None if nothing to filter on.
    """
    conditions: list[dict] = []

    if user_seniority != SENIORITY_UNKNOWN:
        conditions.append(
            {
                "$or": [
                    {"seniority_level": {"$eq": user_seniority}},
                    {"seniority_level": {"$eq": SENIORITY_UNKNOWN}},
                ]
            }
        )

    if user_degree != DEGREE_UNKNOWN:
        conditions.append(
            {
                "$or": [
                    {"required_degree": {"$lte": user_degree}},
                    {"required_degree": {"$eq": DEGREE_UNKNOWN}},
                ]
            }
        )

    if user_years != YEARS_UNKNOWN:
        conditions.append(
            {
                "$or": [
                    {"min_years_experience": {"$lt": user_years}},
                    {"min_years_experience": {"$eq": YEARS_UNKNOWN}},
                ]
            }
        )

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def describe_chroma_filter(where_filter: Optional[dict]) -> str:
    """
    Convert a ChromaDB where filter dict to a human-readable description.

    Translates numeric constants and query operators into plain language.

    Args:
        where_filter: ChromaDB where dict (nested $and/$or operators), or None.

    Returns:
        A readable string describing the active filters.
    """
    if not where_filter:
        return "No filters applied"

    # Map numeric constants to labels
    SENIORITY_MAP = {
        SENIORITY_UNKNOWN: "Unknown",
        SENIORITY_ENTRY: "Entry-level",
        SENIORITY_MID: "Mid-level",
        SENIORITY_SENIOR: "Senior",
    }
    DEGREE_MAP = {
        DEGREE_UNKNOWN: "Unknown",
        DEGREE_BACHELOR: "Bachelor's or lower",
        DEGREE_MASTER: "Master's or lower",
        DEGREE_PHD: "PhD or lower",
    }

    descriptions = []

    # Parse $and conditions at root level
    conditions_to_check = where_filter.get("$and", [where_filter])

    for condition in conditions_to_check:
        if "$or" in condition:
            # This is an OR clause — describe it as a single condition
            parts = condition["$or"]
            main_val = None
            field_name = None

            # Extract the main requirement value (the non-UNKNOWN one)
            for part in parts:
                for field, op_dict in part.items():
                    field_name = field
                    if "$eq" in op_dict:
                        val = op_dict["$eq"]
                        if val != SENIORITY_UNKNOWN and val != DEGREE_UNKNOWN and val != YEARS_UNKNOWN:
                            main_val = val
                    elif "$lte" in op_dict:
                        main_val = op_dict["$lte"]
                    elif "$lt" in op_dict:
                        main_val = op_dict["$lt"]

            # Format based on field type
            if field_name == "seniority_level" and main_val is not None:
                descriptions.append(f"Seniority: {SENIORITY_MAP.get(main_val, main_val)}")
            elif field_name == "required_degree" and main_val is not None:
                descriptions.append(f"Degree: ≤{DEGREE_MAP.get(main_val, main_val)}")
            elif field_name == "min_years_experience" and main_val is not None:
                descriptions.append(f"Years of experience: ≥{main_val}")

    return ", ".join(descriptions) if descriptions else "No filters applied"
