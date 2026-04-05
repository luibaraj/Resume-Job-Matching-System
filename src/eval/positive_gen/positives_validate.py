"""
Synthetic positives validation — Step 2: LLM-based validation.

Validates job skeletons generated in Step 1 across four rule sets using LLM calls:
1. Structural validation (format compliance)
2. Seniority-years alignment (consistency)
3. Resume-job alignment (skill/seniority match)
4. Domain consistency (domain alignment with title)

Validated skeletons proceed to Step 3 (expansion). Failed skeletons route to Stage 3 (fix/discard).

All LLM calls use a locally hosted LLaMA 3.2 3B model via Ollama.
"""

import logging
import sys
from pathlib import Path
from typing import TypedDict

import ollama

# Allow running as a script from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    GENERATION_TEMPERATURE,
    GENERATION_TOP_P,
    OLLAMA_MODEL,
    VALIDATION_MAX_TOKENS,
)
from eval.eval_utils import call_ollama_validate
from .positives_gen import JobSkeleton

logger = logging.getLogger(__name__)


class ValidationResult(TypedDict):
    """Result of a single validation rule check."""

    passed: bool
    reason: str | None  # None when passed; populated with failure description on FAIL


class ResumeInfo(TypedDict):
    """Resume-derived fields needed by the validation orchestrator."""

    seniority: str  # Junior / Mid / Senior / Staff
    years_experience: int  # Total years of professional experience
    primary_skills: list[str]
    domain: str  # backend / frontend / fullstack / data
    resume_text: str  # Full resume text, used for responsibilities alignment




def _parse_years_required(years_str: str) -> int:
    """
    Parse a raw years_required string to an integer.

    For range strings like "4-6", returns the maximum (6).
    For plain integers like "3", returns 3.
    Returns 0 if the string cannot be parsed.

    Args:
        years_str: Raw years string from JobSkeleton (e.g., "4-6" or "3").

    Returns:
        Integer years value (max of range, or plain value). 0 on parse failure.
    """
    if not years_str.strip():
        return 0

    # Try parsing as a range (e.g., "4-6")
    if "-" in years_str:
        parts = years_str.split("-")
        try:
            return max(int(p.strip()) for p in parts if p.strip())
        except ValueError:
            return 0

    # Try parsing as a plain integer
    try:
        return int(years_str.strip())
    except ValueError:
        return 0


def _parse_years_min(years_str: str) -> int:
    """
    Parse a raw years_required string to an integer (minimum of range).

    For range strings like "4-6", returns the minimum (4).
    For plain integers like "3", returns 3.
    Returns 0 if the string cannot be parsed.

    Args:
        years_str: Raw years string from JobSkeleton (e.g., "4-6" or "3").

    Returns:
        Integer years value (min of range, or plain value). 0 on parse failure.
    """
    if not years_str.strip():
        return 0

    # Try parsing as a range (e.g., "4-6")
    if "-" in years_str:
        parts = years_str.split("-")
        try:
            return min(int(p.strip()) for p in parts if p.strip())
        except ValueError:
            return 0

    # Try parsing as a plain integer
    try:
        return int(years_str.strip())
    except ValueError:
        return 0


def validate_responsibilities(
    responsibilities: list[str],
) -> ValidationResult:
    """
    Run deterministic responsibilities validation.

    Checks that responsibilities list has 3–5 items, each with ≥10 words,
    and no empty items or literal periods.

    Args:
        responsibilities: List of responsibility strings.

    Returns:
        ValidationResult with passed=True or passed=False and a reason string.
    """
    if not responsibilities:
        msg = "Responsibilities list is empty"
        logger.warning("responsibilities: FAIL — %s", msg)
        return {"passed": False, "reason": msg}

    if len(responsibilities) < 3 or len(responsibilities) > 5:
        msg = f"Must have 3–5 responsibilities, got {len(responsibilities)}"
        logger.warning("responsibilities: FAIL — %s", msg)
        return {"passed": False, "reason": msg}

    for i, resp in enumerate(responsibilities):
        resp_stripped = resp.strip()
        if not resp_stripped or resp_stripped == ".":
            msg = f"Responsibility {i} is empty or a literal period"
            logger.warning("responsibilities: FAIL — %s", msg)
            return {"passed": False, "reason": msg}

        word_count = len(resp_stripped.split())
        if word_count < 10:
            msg = f"Responsibility {i} has {word_count} words, need ≥10: {resp_stripped!r}"
            logger.warning("responsibilities: FAIL — %s", msg)
            return {"passed": False, "reason": msg}

    logger.info("responsibilities: PASS — %d items, all ≥10 words", len(responsibilities))
    return {"passed": True, "reason": None}


def _parse_validation_response(raw_response: str) -> ValidationResult:
    """
    Parse an LLM validation response into a ValidationResult.

    Accepts:
        "PASS" → {"passed": True, "reason": None}
        "FAIL: <explanation>" → {"passed": False, "reason": "<explanation>"}

    Matching is case-insensitive and strips surrounding whitespace.
    If the response does not match either pattern (malformed LLM output),
    treats the result as a failure with reason "Unparseable LLM response: <raw>".

    Args:
        raw_response: Raw string returned by the LLM.

    Returns:
        ValidationResult dict.
    """
    stripped = raw_response.strip()

    if stripped.upper().startswith("PASS"):
        return {"passed": True, "reason": None}

    if stripped.upper().startswith("FAIL:"):
        reason = stripped[5:].strip()  # Everything after "FAIL:"
        return {"passed": False, "reason": reason}

    # Unparseable response
    return {"passed": False, "reason": f"Unparseable LLM response: {raw_response}"}


def _build_structural_prompt(job_text: str) -> str:
    """
    Build the structural validation prompt.

    Args:
        job_text: Formatted string representation of the job skeleton.

    Returns:
        Prompt string ready for LLM call.
    """
    return f"""Generated job:
{job_text}

Check format compliance. Verify:
1. Title field exists and is a real job title (not gibberish)
2. Seniority is one of: Junior, Mid, Senior, Staff
3. YearsRequired is a number between 0–20
4. Domain is one of: backend, frontend, fullstack, data
5. PrimarySkills has 2–4 items (comma-separated)
6. No fields are missing

Respond ONLY with:
- PASS if all checks succeed
- FAIL: [specific issue] if any check fails (e.g., "FAIL: Seniority is 'Very Senior', not in enum")

Be strict. If anything is malformed, FAIL."""


def _build_seniority_years_prompt(seniority: str, years: str) -> str:
    """
    Build the seniority-years alignment prompt.

    Args:
        seniority: Seniority level string (e.g., "Senior").
        years: Raw years_required string (e.g., "4-6").

    Returns:
        Prompt string ready for LLM call.
    """
    return f"""Generated job:
Seniority: {seniority}
YearsRequired: {years}

Check seniority-years alignment. Verify:
- Junior should require ≤ 2 years
- Mid should require 2–5 years
- Senior should require 4–8 years
- Staff should require ≥ 6 years

Is the seniority level aligned with years required?

Respond ONLY with:
- PASS if aligned
- FAIL: [explanation] (e.g., "FAIL: Senior should require 4–8 years, but specifies 12")

Be strict."""


def _build_resume_job_alignment_prompt(
    resume_seniority: str,
    resume_years: int,
    resume_skills: list[str],
    resume_text: str,
    job_seniority: str,
    job_years: str,
    job_skills: list[str],
    job_responsibilities: list[str],
) -> str:
    """
    Build the resume-job alignment prompt.

    Args:
        resume_seniority: Resume seniority level (e.g., "Mid").
        resume_years: Resume years of experience (integer).
        resume_skills: Resume primary skills list.
        resume_text: Full resume text for responsibilities alignment.
        job_seniority: Job skeleton seniority.
        job_years: Job skeleton years_required raw string.
        job_skills: Job skeleton primary_skills list.
        job_responsibilities: Job skeleton responsibilities list.

    Returns:
        Prompt string ready for LLM call.
    """
    return f"""Resume:
- Seniority: {resume_seniority}
- Experience: {resume_years} years
- Primary Skills: {", ".join(resume_skills)}
- Resume text:
{resume_text}

Generated Job:
- Seniority: {job_seniority}
- YearsRequired: {job_years}
- PrimarySkills: {", ".join(job_skills)}
- Responsibilities: {"; ".join(job_responsibilities)}

Check resume-job alignment. Verify:
1. At least 2 of the resume's skills appear in the job's skills
2. Job seniority must exactly match resume seniority
3. Job years required ≤ resume experience (no stretch allowed)
4. Each job responsibility aligns with at least one area of work described in the resume
5. Overall, is this a plausible match for this candidate?

Respond ONLY with:
- PASS if all checks succeed
- FAIL: [specific issue] (e.g., "FAIL: Only 1 skill matches; need 2. Domain also shifted from backend to data.")

Be strict. Seniority must match exactly and years must not exceed resume experience."""


def _build_domain_consistency_prompt(
    resume_domain: str,
    job_domain: str,
    job_title: str,
) -> str:
    """
    Build the domain consistency prompt.

    Args:
        resume_domain: Resume domain (e.g., "backend").
        job_domain: Job skeleton domain.
        job_title: Job skeleton title.

    Returns:
        Prompt string ready for LLM call.
    """
    return f"""Resume: {resume_domain} engineer
Generated Job: {job_domain} domain, title: {job_title}

Check domain consistency. Verify:
1. Job domain aligns with resume domain (exact match or adjacent, e.g., fullstack→backend OK)
2. Domain matches the job title (e.g., "Data Engineer" should be domain: data)
3. No major domain shifts (backend → data, frontend → infrastructure)

Respond ONLY with:
- PASS if domain is consistent
- FAIL: [explanation] (e.g., "FAIL: Resume is backend, job is data; major domain shift")

Be strict."""


def validate_structural(
    job: JobSkeleton,
    model: str = OLLAMA_MODEL,
) -> ValidationResult:
    """
    Run Rule Set 1: structural format compliance check.

    Verifies that all required fields are present, title is non-gibberish,
    seniority is one of the four valid levels, years_required is 0–20,
    domain is one of the four valid domains, primary_skills has 2–4 items,
    and responsibilities has 3–5 items.

    Args:
        job: JobSkeleton dict from Step 1.
        model: Ollama model name (default: OLLAMA_MODEL from config).

    Returns:
        ValidationResult with passed=True or passed=False and a reason string.

    Raises:
        ollama.RequestError: If Ollama is not reachable.
        ollama.ResponseError: If the model returns an error response.
    """
    # Format job skeleton as text for the prompt
    job_text = (
        f"Title: {job['title']}\n"
        f"Seniority: {job['seniority']}\n"
        f"YearsRequired: {job['years_required']}\n"
        f"Domain: {job['domain']}\n"
        f"PrimarySkills: {', '.join(job['primary_skills'])}\n"
        f"SecondarySkills: {', '.join(job['secondary_skills'])}\n"
        f"Responsibilities: {'; '.join(job['responsibilities'])}"
    )

    prompt = _build_structural_prompt(job_text)
    raw_response = call_ollama_validate(prompt, model)
    logger.debug("Structural validation response: %s", raw_response)

    result = _parse_validation_response(raw_response)

    if result["passed"]:
        logger.info("structural: PASS — title: %s", job["title"])
    else:
        logger.warning("structural: FAIL — %s", result["reason"])

    return result


def validate_seniority_years(
    job: JobSkeleton,
    model: str = OLLAMA_MODEL,
) -> ValidationResult:
    """
    Run Rule Set 2: seniority-to-years alignment check (deterministic).

    Checks that years_required falls within the accepted bracket for the
    given seniority level using thresholds:
      Junior: max ≤ 2
      Mid:    max ≤ 5
      Senior: 2 ≤ max ≤ 8
      Staff:  max ≥ 6

    No LLM call is made — this is a purely numeric check.

    Args:
        job: JobSkeleton dict from Step 1.
        model: Ollama model name (kept for signature compatibility, not used).

    Returns:
        ValidationResult with passed=True or passed=False and a reason string.
    """
    seniority = job["seniority"]
    years_str = job["years_required"]

    max_years = _parse_years_required(years_str)

    brackets: dict[str, tuple[int, int]] = {
        "Junior": (0, 2),
        "Mid":    (0, 5),
        "Senior": (2, 8),
        "Staff":  (6, 99),
    }

    if seniority not in brackets:
        msg = f"Unknown seniority: {seniority!r}"
        logger.warning("seniority_years: FAIL — %s", msg)
        return {"passed": False, "reason": msg}

    lo, hi = brackets[seniority]
    if max_years < lo or max_years > hi:
        msg = f"{seniority} requires {lo}–{hi} years (max), got {years_str!r}"
        logger.warning("seniority_years: FAIL — %s", msg)
        return {"passed": False, "reason": msg}

    logger.info("seniority_years: PASS — %s / %s years", seniority, years_str)
    return {"passed": True, "reason": None}


def validate_resume_job_alignment(
    job: JobSkeleton,
    resume_info: ResumeInfo,
    model: str = OLLAMA_MODEL,
) -> ValidationResult:
    """
    Run Rule Set 3: resume-to-job alignment check.

    Verifies skill overlap (≥ 2 shared skills), seniority match (exact),
    and experience ceiling (job years ≤ resume years, no stretch).

    Args:
        job: JobSkeleton dict from Step 1.
        resume_info: ResumeInfo dict with seniority, years_experience,
                     primary_skills, and domain.
        model: Ollama model name.

    Returns:
        ValidationResult with passed=True or passed=False and a reason string.

    Raises:
        ollama.RequestError: If Ollama is not reachable.
        ollama.ResponseError: If the model returns an error response.
    """
    # Guard 1: Exact seniority match (deterministic, no LLM)
    if job["seniority"] != resume_info["seniority"]:
        reason = f"Seniority mismatch: job is {job['seniority']!r}, resume is {resume_info['seniority']!r}"
        logger.warning("resume_job_alignment: FAIL — %s", reason)
        return {"passed": False, "reason": reason}

    # Guard 2: Years ceiling (no stretch) — deterministic, no LLM
    min_years = _parse_years_min(job["years_required"])
    if min_years > resume_info["years_experience"]:
        reason = f"Years required ({job['years_required']}) exceeds resume experience ({resume_info['years_experience']})"
        logger.warning("resume_job_alignment: FAIL — %s", reason)
        return {"passed": False, "reason": reason}

    prompt = _build_resume_job_alignment_prompt(
        resume_info["seniority"],
        resume_info["years_experience"],
        resume_info["primary_skills"],
        resume_info["resume_text"],
        job["seniority"],
        job["years_required"],
        job["primary_skills"],
        job["responsibilities"],
    )
    raw_response = call_ollama_validate(prompt, model)
    logger.debug("Resume-job alignment validation response: %s", raw_response)

    result = _parse_validation_response(raw_response)

    if result["passed"]:
        logger.info("resume_job_alignment: PASS — skills and seniority match")
    else:
        logger.warning("resume_job_alignment: FAIL — %s", result["reason"])

    return result


def validate_domain_consistency(
    job: JobSkeleton,
    resume_info: ResumeInfo,
    model: str = OLLAMA_MODEL,
) -> ValidationResult:
    """
    Run Rule Set 4: domain consistency check.

    Verifies the job domain aligns with the resume domain (exact or adjacent)
    and that the domain is consistent with the job title.

    Args:
        job: JobSkeleton dict from Step 1.
        resume_info: ResumeInfo dict (only `domain` field is used here).
        model: Ollama model name.

    Returns:
        ValidationResult with passed=True or passed=False and a reason string.

    Raises:
        ollama.RequestError: If Ollama is not reachable.
        ollama.ResponseError: If the model returns an error response.
    """
    prompt = _build_domain_consistency_prompt(
        resume_info["domain"],
        job["domain"],
        job["title"],
    )
    raw_response = call_ollama_validate(prompt, model)
    logger.debug("Domain consistency validation response: %s", raw_response)

    result = _parse_validation_response(raw_response)

    if result["passed"]:
        logger.info(
            "domain_consistency: PASS — %s matches resume domain %s",
            job["domain"],
            resume_info["domain"],
        )
    else:
        logger.warning("domain_consistency: FAIL — %s", result["reason"])

    return result


_SENIORITY_ALIASES: dict[str, str] = {
    "junior":    "Junior",
    "mid":       "Mid",
    "mid-level": "Mid",
    "midlevel":  "Mid",
    "mid level": "Mid",
    "senior":    "Senior",
    "staff":     "Staff",
}

_DOMAIN_ALIASES: dict[str, str] = {
    "backend":           "backend",
    "frontend":          "frontend",
    "front-end":         "frontend",
    "front end":         "frontend",
    "fullstack":         "fullstack",
    "full-stack":        "fullstack",
    "full stack":        "fullstack",
    "data":              "data",
    "data science":      "data",
    "data engineering":  "data",
}


def _normalize_skeleton(job: JobSkeleton) -> JobSkeleton:
    """
    Return a copy of job with seniority and domain normalized to canonical enum values.

    Maps common variants (e.g., "Mid-level" → "Mid", "Full-stack" → "fullstack")
    to their canonical forms. Unknown values are left as-is after title-casing.

    Args:
        job: JobSkeleton dict.

    Returns:
        New JobSkeleton dict with normalized fields.
    """
    normalized = dict(job)
    seniority_lower = job["seniority"].strip().lower()
    normalized["seniority"] = _SENIORITY_ALIASES.get(
        seniority_lower, job["seniority"].strip().title()
    )
    domain_lower = job["domain"].strip().lower()
    normalized["domain"] = _DOMAIN_ALIASES.get(domain_lower, domain_lower)
    return normalized


def validate_job_skeleton(
    job: JobSkeleton,
    resume_info: ResumeInfo,
    model: str = OLLAMA_MODEL,
) -> dict[str, bool | str | None]:
    """
    Run all validation rule sets in sequence.

    Executes responsibilities (deterministic) → structural → seniority_years →
    resume_job_alignment → domain_consistency. Stops at the first failure and
    returns which check failed. All checks must pass for the skeleton to be accepted.

    Before validation, normalizes seniority and domain fields to canonical
    enum values (e.g., "Mid-level" → "Mid", "Full-stack" → "fullstack").

    Args:
        job: JobSkeleton dict from Step 1.
        resume_info: ResumeInfo dict with seniority, years_experience,
                     primary_skills, and domain.
        model: Ollama model name (default: OLLAMA_MODEL from config).

    Returns:
        Dict with keys:
            "passed" (bool): True only if all checks passed.
            "failed_check" (str | None): Name of the first failed check
                ("responsibilities", "structural", "seniority_years",
                "resume_job_alignment", "domain_consistency"), or None if all passed.
            "reason" (str | None): Failure reason from the failing check,
                or None if all passed.

    Raises:
        ollama.RequestError: If Ollama is not reachable.
        ollama.ResponseError: If the model returns an error response.
    """
    job = _normalize_skeleton(job)
    checks = [
        ("responsibilities", lambda: validate_responsibilities(job["responsibilities"])),
        ("structural", lambda: validate_structural(job, model)),
        ("seniority_years", lambda: validate_seniority_years(job, model)),
        ("resume_job_alignment", lambda: validate_resume_job_alignment(job, resume_info, model)),
        ("domain_consistency", lambda: validate_domain_consistency(job, resume_info, model)),
    ]

    for check_name, check_fn in checks:
        result = check_fn()
        if not result["passed"]:
            logger.warning(
                "validate_job_skeleton: FAIL at %s — %s",
                check_name,
                result["reason"],
            )
            return {
                "passed": False,
                "failed_check": check_name,
                "reason": result["reason"],
            }

    logger.info("validate_job_skeleton: PASS — all checks passed")
    return {
        "passed": True,
        "failed_check": None,
        "reason": None,
    }
