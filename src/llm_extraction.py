"""
LLM-based extraction of resume metadata using Ollama.

Functions extract degree level, seniority, and years of experience
from resume text by calling an LLM directly, returning standardized
integer codes or UNKNOWN sentinels on failure.
"""

import logging
from src.config import (
    DEGREE_UNKNOWN,
    DEGREE_BACHELOR,
    DEGREE_MASTER,
    DEGREE_PHD,
    SENIORITY_UNKNOWN,
    SENIORITY_ENTRY,
    SENIORITY_MID,
    SENIORITY_SENIOR,
    YEARS_UNKNOWN,
)
from src.generation import _call_ollama

logger = logging.getLogger(__name__)


def _parse_int_response(response: str, default: int) -> int:
    """
    Parse integer from LLM response.

    Attempts to extract the first integer from response text.
    Returns default if parsing fails or response is empty.

    Args:
        response: The LLM response string.
        default: Value to return on parse failure.

    Returns:
        Parsed integer or default.
    """
    if not response or not response.strip():
        return default

    # Try to find an integer in the response (first sequence of digits)
    import re
    match = re.search(r'-?\d+', response.strip())
    if match:
        try:
            return int(match.group())
        except (ValueError, AttributeError):
            return default
    return default


def extract_degree_with_llm(resume_text: str, model: str) -> int:
    """
    Extract highest degree level from resume using LLM.

    Args:
        resume_text: Full resume text.
        model: Ollama model name (e.g., "llama3.2:3b-instruct-q4_K_M").

    Returns:
        DEGREE_PHD (3), DEGREE_MASTER (2), DEGREE_BACHELOR (1),
        or DEGREE_UNKNOWN (0) on failure.
    """
    prompt = f"""Extract the highest degree level from this resume.
Return ONLY a single number:
- 3 if PhD or Doctoral degree
- 2 if Master's degree
- 1 if Bachelor's degree
- 0 if no degree found or uncertain

Resume:
{resume_text[:2000]}

Answer:"""

    try:
        response = _call_ollama(prompt, model)
        degree = _parse_int_response(response, DEGREE_UNKNOWN)
        logger.debug("degree raw=%r parsed=%d", response, degree)
        # Clamp to valid range
        if degree not in (DEGREE_UNKNOWN, DEGREE_BACHELOR, DEGREE_MASTER, DEGREE_PHD):
            logger.warning("Invalid degree value %d, returning UNKNOWN", degree)
            return DEGREE_UNKNOWN
        return degree
    except Exception as e:
        logger.error("Failed to extract degree: %s", e)
        return DEGREE_UNKNOWN


def extract_seniority_with_llm(resume_text: str, model: str) -> int:
    """
    Extract seniority level from resume using LLM.

    Args:
        resume_text: Full resume text.
        model: Ollama model name.

    Returns:
        SENIORITY_SENIOR (3), SENIORITY_MID (2), SENIORITY_ENTRY (1),
        or SENIORITY_UNKNOWN (0) on failure.
    """
    prompt = f"""Extract the career seniority level from this resume.
Return ONLY a single number:
- 3 if Senior/Staff/Principal/Lead role
- 2 if Mid-level (e.g., Software Engineer II, Senior Developer)
- 1 if Entry-level (e.g., Junior, Associate, Graduate)
- 0 if unclear

Resume:
{resume_text[:2000]}

Answer:"""

    try:
        response = _call_ollama(prompt, model)
        seniority = _parse_int_response(response, SENIORITY_UNKNOWN)
        logger.debug("seniority raw=%r parsed=%d", response, seniority)
        # Clamp to valid range
        if seniority not in (
            SENIORITY_UNKNOWN,
            SENIORITY_ENTRY,
            SENIORITY_MID,
            SENIORITY_SENIOR,
        ):
            logger.warning("Invalid seniority value %d, returning UNKNOWN", seniority)
            return SENIORITY_UNKNOWN
        return seniority
    except Exception as e:
        logger.error("Failed to extract seniority: %s", e)
        return SENIORITY_UNKNOWN


def extract_years_with_llm(resume_text: str, model: str) -> int:
    """
    Extract years of professional experience from resume using LLM.

    Args:
        resume_text: Full resume text.
        model: Ollama model name.

    Returns:
        Total years as integer, or YEARS_UNKNOWN (-1) on failure.
    """
    prompt = f"""Extract the total years of professional experience from this resume.
Return ONLY a single number representing years:
- Return the total years (e.g., 5 for "5 years", 8 for "8+ years")
- Return -1 if no experience info found or unclear

Resume:
{resume_text[:2000]}

Answer:"""

    try:
        response = _call_ollama(prompt, model)
        years = _parse_int_response(response, YEARS_UNKNOWN)
        logger.debug("years raw=%r parsed=%d", response, years)
        # Validate: years should be >= 0 or exactly YEARS_UNKNOWN
        if years < 0 and years != YEARS_UNKNOWN:
            logger.warning("Invalid years value %d, returning UNKNOWN", years)
            return YEARS_UNKNOWN
        return years
    except Exception as e:
        logger.error("Failed to extract years: %s", e)
        return YEARS_UNKNOWN
