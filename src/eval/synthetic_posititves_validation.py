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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    GENERATION_TEMPERATURE,
    GENERATION_TOP_P,
    OLLAMA_MODEL,
    VALIDATION_MAX_TOKENS,
)
from eval.synthetic_positives_generation import JobSkeleton

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


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Call Ollama chat endpoint and return response content."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": GENERATION_TEMPERATURE,
            "top_p": GENERATION_TOP_P,
            "num_predict": VALIDATION_MAX_TOKENS,
        },
    )
    return response["message"]["content"]


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

    if stripped.upper() == "PASS":
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
    job_seniority: str,
    job_years: str,
    job_skills: list[str],
) -> str:
    """
    Build the resume-job alignment prompt.

    Args:
        resume_seniority: Resume seniority level (e.g., "Mid").
        resume_years: Resume years of experience (integer).
        resume_skills: Resume primary skills list.
        job_seniority: Job skeleton seniority.
        job_years: Job skeleton years_required raw string.
        job_skills: Job skeleton primary_skills list.

    Returns:
        Prompt string ready for LLM call.
    """
    return f"""Resume:
- Seniority: {resume_seniority}
- Experience: {resume_years} years
- Primary Skills: {", ".join(resume_skills)}

Generated Job:
- Seniority: {job_seniority}
- YearsRequired: {job_years}
- PrimarySkills: {", ".join(job_skills)}

Check resume-job alignment. Verify:
1. At least 2 of the resume's skills appear in the job's skills
2. Job seniority is within ±1 level of resume seniority
3. Job years required ≤ resume experience + 2 (allow 2-year stretch)
4. Overall, is this a plausible match for this candidate?

Respond ONLY with:
- PASS if all checks succeed
- FAIL: [specific issue] (e.g., "FAIL: Only 1 skill matches; need 2. Domain also shifted from backend to data.")

Be strict but fair. Minor skill gaps or 1-level seniority difference are acceptable if the job is otherwise well-matched."""


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
    domain is one of the four valid domains, and primary_skills has 2–4 items.

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
        f"SecondarySkills: {', '.join(job['secondary_skills'])}"
    )

    prompt = _build_structural_prompt(job_text)
    raw_response = _call_ollama(prompt, model)
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
    Run Rule Set 2: seniority-to-years alignment check.

    Checks that the years_required in the skeleton is consistent with
    the seniority level using the thresholds: Junior ≤ 2, Mid 2–5,
    Senior 4–8, Staff ≥ 6.

    Args:
        job: JobSkeleton dict from Step 1.
        model: Ollama model name.

    Returns:
        ValidationResult with passed=True or passed=False and a reason string.

    Raises:
        ollama.RequestError: If Ollama is not reachable.
        ollama.ResponseError: If the model returns an error response.
    """
    prompt = _build_seniority_years_prompt(job["seniority"], job["years_required"])
    raw_response = _call_ollama(prompt, model)
    logger.debug("Seniority-years validation response: %s", raw_response)

    result = _parse_validation_response(raw_response)

    if result["passed"]:
        logger.info(
            "seniority_years: PASS — %s / %s years",
            job["seniority"],
            job["years_required"],
        )
    else:
        logger.warning("seniority_years: FAIL — %s", result["reason"])

    return result


def validate_resume_job_alignment(
    job: JobSkeleton,
    resume_info: ResumeInfo,
    model: str = OLLAMA_MODEL,
) -> ValidationResult:
    """
    Run Rule Set 3: resume-to-job alignment check.

    Verifies skill overlap (≥ 2 shared skills), seniority compatibility
    (within ±1 level), and experience ceiling (job years ≤ resume years + 2).

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
    prompt = _build_resume_job_alignment_prompt(
        resume_info["seniority"],
        resume_info["years_experience"],
        resume_info["primary_skills"],
        job["seniority"],
        job["years_required"],
        job["primary_skills"],
    )
    raw_response = _call_ollama(prompt, model)
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
    raw_response = _call_ollama(prompt, model)
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


def validate_job_skeleton(
    job: JobSkeleton,
    resume_info: ResumeInfo,
    model: str = OLLAMA_MODEL,
) -> dict[str, bool | str | None]:
    """
    Run all four validation rule sets in sequence.

    Executes structural → seniority_years → resume_job_alignment →
    domain_consistency. Stops at the first failure and returns which
    check failed. All four must pass for the skeleton to be accepted.

    Args:
        job: JobSkeleton dict from Step 1.
        resume_info: ResumeInfo dict with seniority, years_experience,
                     primary_skills, and domain.
        model: Ollama model name (default: OLLAMA_MODEL from config).

    Returns:
        Dict with keys:
            "passed" (bool): True only if all four checks passed.
            "failed_check" (str | None): Name of the first failed check
                ("structural", "seniority_years", "resume_job_alignment",
                "domain_consistency"), or None if all passed.
            "reason" (str | None): Failure reason from the failing check,
                or None if all passed.

    Raises:
        ollama.RequestError: If Ollama is not reachable.
        ollama.ResponseError: If the model returns an error response.
    """
    checks = [
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
