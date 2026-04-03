"""
Utility functions for filtering jobs based on user profile.
"""
import re
from typing import Optional, Dict, Any


def extract_degree_from_text(text: str) -> Optional[int]:
    """
    Extract highest degree from resume text.

    Returns:
        0: No degree
        1: Bachelor's
        2: Master's
        3: PhD
    """
    text_lower = text.lower()
    if "phd" in text_lower or "doctorate" in text_lower:
        return 3
    if "master" in text_lower or "ms" in text_lower or "m.s." in text_lower:
        return 2
    if "bachelor" in text_lower or "bs" in text_lower or "b.s." in text_lower:
        return 1
    return 0


def extract_seniority_from_text(text: str) -> Optional[int]:
    """
    Extract seniority level from resume text.

    Returns:
        0: Intern
        1: Entry / New Grad
        2: Mid-level
        3: Senior
        4: Lead / Principal
    """
    text_lower = text.lower()
    if "lead" in text_lower or "principal" in text_lower or "director" in text_lower:
        return 4
    if "senior" in text_lower:
        return 3
    if "mid" in text_lower or "mid-level" in text_lower:
        return 2
    if "entry" in text_lower or "new grad" in text_lower or "junior" in text_lower:
        return 1
    if "intern" in text_lower:
        return 0
    return 1  # default to entry


def extract_years_from_text(text: str) -> Optional[int]:
    """
    Extract years of experience from resume text.

    Returns:
        Integer years (rounded down).
    """
    # Look for patterns like "X years" or "X+ years"
    patterns = [
        r"(\d+)\+?\s*years?",
        r"(\d+)\+?\s*yr",
        r"(\d+)\s*years? of experience",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                return int(matches[0])
            except ValueError:
                continue
    return 0
