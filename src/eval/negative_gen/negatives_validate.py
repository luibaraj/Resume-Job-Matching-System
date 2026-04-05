"""
Module for validating seniority-mismatched job skeletons.

Validates job skeletons generated in the negative pipeline across four rule sets:
1. Structural validation (format compliance) — reused from positives
2. Seniority-years alignment (consistency) — reused from positives
3. Seniority mismatch (deterministic check, NEW)
4. Skill-domain overlap (LLM check, NEW)
"""

import logging
import sys
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    GENERATION_TEMPERATURE,
    GENERATION_TOP_P,
    OLLAMA_MODEL,
    VALIDATION_MAX_TOKENS,
)
from src.eval.eval_utils import call_ollama_validate
from src.eval.positive_gen.positives_gen import JobSkeleton
from src.eval.positive_gen.positives_validate import (
    ResumeInfo,
    ValidationResult,
    _normalize_skeleton,
    _parse_validation_response,
    _parse_years_required,
    validate_structural,
    validate_seniority_years,
)
from .negatives_gen import SENIORITY_ORDER, MismatchType

logger = logging.getLogger(__name__)

# Minimum gap (in SENIORITY_ORDER positions) for seniority mismatch to be valid
_MIN_MISMATCH_GAP = {
    "Junior": 2,
    "Mid": 1,
    "Senior": 2,
    "Staff": 2,
}





def validate_seniority_mismatch(
    job: JobSkeleton,
    resume_info: ResumeInfo,
) -> ValidationResult:
    """
    Run Rule Set 3: seniority mismatch check (deterministic).

    Verifies that the job seniority is sufficiently different from the resume
    seniority. Gap requirements:
      - Junior resume: job ≥2 away (Senior or Staff only)
      - Mid resume: job ≥1 away (Junior or Staff)
      - Senior resume: job ≥2 away (Junior only)
      - Staff resume: job ≥2 away (Junior or Mid)

    No LLM call is made — this is a purely logic check.

    Args:
        job: JobSkeleton dict from the generation step.
        resume_info: ResumeInfo dict with resume seniority.

    Returns:
        ValidationResult with passed=True or passed=False and a reason string.
    """
    job_seniority = job["seniority"]
    resume_seniority = resume_info["seniority"]

    # Handle unknown seniorities
    if resume_seniority not in SENIORITY_ORDER:
        msg = f"Unknown resume seniority: {resume_seniority!r}"
        logger.warning("seniority_mismatch: FAIL — %s", msg)
        return {"passed": False, "reason": msg}

    if job_seniority not in SENIORITY_ORDER:
        msg = f"Unknown job seniority: {job_seniority!r}"
        logger.warning("seniority_mismatch: FAIL — %s", msg)
        return {"passed": False, "reason": msg}

    # Calculate gap
    resume_idx = SENIORITY_ORDER.index(resume_seniority)
    job_idx = SENIORITY_ORDER.index(job_seniority)
    gap = abs(job_idx - resume_idx)

    min_gap = _MIN_MISMATCH_GAP[resume_seniority]

    if gap < min_gap:
        msg = f"Seniority mismatch gap {gap} is less than minimum {min_gap} for {resume_seniority} resume (job is {job_seniority})"
        logger.warning("seniority_mismatch: FAIL — %s", msg)
        return {"passed": False, "reason": msg}

    logger.info("seniority_mismatch: PASS — gap=%d, resume=%s, job=%s", gap, resume_seniority, job_seniority)
    return {"passed": True, "reason": None}


def _build_skill_domain_overlap_prompt(
    resume_skills: list[str],
    resume_domain: str,
    job_domain: str,
    job_skills: list[str],
    job_responsibilities: list[str],
) -> str:
    """
    Build the skill-domain overlap validation prompt.

    Checks that the job shares domain and skills with the resume (making it
    realistic for the candidate) WITHOUT considering seniority (mismatch is intentional).

    Args:
        resume_skills: Resume primary skills.
        resume_domain: Resume domain.
        job_domain: Job skeleton domain.
        job_skills: Job skeleton primary_skills.
        job_responsibilities: Job skeleton responsibilities.

    Returns:
        Prompt string ready for LLM call.
    """
    return f"""Resume:
- Domain: {resume_domain}
- Primary Skills: {", ".join(resume_skills)}

Generated Job:
- Domain: {job_domain}
- Primary Skills: {", ".join(job_skills)}
- Responsibilities: {"; ".join(job_responsibilities)}

Check skill and domain overlap. Verify:
1. At least 2 of the resume's skills appear in the job's skills
2. Job domain matches resume domain (exact or adjacent, e.g., fullstack→backend OK)
3. Job responsibilities are plausible for this domain

NOTE: Do NOT check seniority — the seniority mismatch is intentional.

Respond ONLY with:
- PASS if all checks succeed
- FAIL: [specific issue] (e.g., "FAIL: Only 1 skill matches; need 2. Also job domain is data, resume is backend; too large a shift.")

Be strict about skill overlap and domain alignment."""


def validate_skill_domain_overlap(
    job: JobSkeleton,
    resume_info: ResumeInfo,
    model: str = OLLAMA_MODEL,
) -> ValidationResult:
    """
    Run Rule Set 4: skill-domain overlap check (NEW, LLM-based).

    Verifies that the job shares enough skills and domain with the resume
    to be a realistic job (just at a different seniority level).

    This replaces validate_domain_consistency from the positive pipeline,
    which it doesn't directly use; instead, this check includes both domain
    and skill overlap in one call.

    Args:
        job: JobSkeleton dict from the generation step.
        resume_info: ResumeInfo dict with skills and domain.
        model: Ollama model name.

    Returns:
        ValidationResult with passed=True or passed=False and a reason string.

    Raises:
        ollama.RequestError: If Ollama is not reachable.
        ollama.ResponseError: If the model returns an error response.
    """
    prompt = _build_skill_domain_overlap_prompt(
        resume_info["primary_skills"],
        resume_info["domain"],
        job["domain"],
        job["primary_skills"],
        job["responsibilities"],
    )
    raw_response = call_ollama_validate(prompt, model)
    logger.debug("Skill-domain overlap validation response: %s", raw_response)

    result = _parse_validation_response(raw_response)

    if result["passed"]:
        logger.info("skill_domain_overlap: PASS — skills and domain realistic")
    else:
        logger.warning("skill_domain_overlap: FAIL — %s", result["reason"])

    return result




def _build_responsibility_mismatch_prompt(
    resume_text: str,
    job_responsibilities: list[str],
    job_domain: str,
    job_seniority: str,
) -> str:
    """
    Build the responsibility mismatch validation prompt.

    Checks that job responsibilities describe work genuinely different from
    what the resume candidate does, even though seniority and domain match.

    Args:
        resume_text: Full resume text.
        job_responsibilities: Job skeleton responsibilities.
        job_domain: Job skeleton domain.
        job_seniority: Job skeleton seniority.

    Returns:
        Prompt string ready for LLM call.
    """
    return f"""Resume:
{resume_text}

Generated Job:
- Domain: {job_domain}
- Seniority: {job_seniority}
- Responsibilities: {"; ".join(job_responsibilities)}

Check responsibility mismatch. Verify:
1. The job responsibilities describe a DIFFERENT type of engineering work than what the resume candidate does
2. The candidate would NOT be qualified or experienced in these responsibilities based on their resume
3. The responsibilities are internally consistent for a {job_domain} {job_seniority} role (just the wrong type)

NOTE: Seniority and domain matching the resume is EXPECTED and CORRECT here.
We are only checking that the day-to-day responsibilities are misaligned.

Respond ONLY with:
- PASS if responsibilities are genuinely misaligned with the resume
- FAIL: [specific issue] (e.g., "FAIL: Responsibilities closely match what the resume describes")

Be strict: if responsibilities describe anything the candidate demonstrably does, FAIL."""


def validate_responsibility_mismatch(
    job: JobSkeleton,
    resume_info: ResumeInfo,
    model: str = OLLAMA_MODEL,
) -> ValidationResult:
    """
    Run Rule Set 3 (responsibility mismatch): LLM check that responsibilities
    are for a genuinely different role type than the resume candidate's work.

    Args:
        job: JobSkeleton dict from the generation step.
        resume_info: ResumeInfo dict with resume text and domain.
        model: Ollama model name.

    Returns:
        ValidationResult with passed=True or passed=False and a reason string.

    Raises:
        ollama.RequestError: If Ollama is not reachable.
        ollama.ResponseError: If the model returns an error response.
    """
    prompt = _build_responsibility_mismatch_prompt(
        resume_info["resume_text"],
        job["responsibilities"],
        job["domain"],
        job["seniority"],
    )
    raw_response = call_ollama_validate(prompt, model)
    logger.debug("Responsibility mismatch validation response: %s", raw_response)

    result = _parse_validation_response(raw_response)

    if result["passed"]:
        logger.info("responsibility_mismatch: PASS")
    else:
        logger.warning("responsibility_mismatch: FAIL — %s", result["reason"])

    return result


def validate_mismatched_skeleton(
    job: JobSkeleton,
    resume_info: ResumeInfo,
    model: str = OLLAMA_MODEL,
    mismatch_type: MismatchType = "seniority",
) -> dict[str, bool | str | None]:
    """
    Run all validation checks for a mismatched skeleton.

    The checks run depend on mismatch_type:
    - "seniority": structural → seniority_years → seniority_mismatch → skill_domain_overlap
    - "responsibility": structural → seniority_years → responsibility_mismatch

    Stops at the first failure. Before validation, normalizes seniority and domain
    fields to canonical enum values (e.g., "Mid-level" → "Mid", "Full-stack" → "fullstack").

    Args:
        job: JobSkeleton dict from the generation step.
        resume_info: ResumeInfo dict with seniority, years_experience,
                     primary_skills, and domain.
        model: Ollama model name (default: OLLAMA_MODEL from config).
        mismatch_type: Type of mismatch ("seniority", "responsibility").

    Returns:
        Dict with keys:
            "passed" (bool): True only if all checks passed.
            "failed_check" (str | None): Name of the first failed check, or None if all passed.
            "reason" (str | None): Failure reason from the failing check, or None if all passed.

    Raises:
        ValueError: If mismatch_type is not recognized.
        ollama.RequestError: If Ollama is not reachable.
        ollama.ResponseError: If the model returns an error response.
    """
    job = _normalize_skeleton(job)

    if mismatch_type == "seniority":
        checks = [
            ("structural", lambda: validate_structural(job, model)),
            ("seniority_years", lambda: validate_seniority_years(job, model)),
            ("seniority_mismatch", lambda: validate_seniority_mismatch(job, resume_info)),
            ("skill_domain_overlap", lambda: validate_skill_domain_overlap(job, resume_info, model)),
        ]
    elif mismatch_type == "responsibility":
        checks = [
            ("structural", lambda: validate_structural(job, model)),
            ("seniority_years", lambda: validate_seniority_years(job, model)),
            ("responsibility_mismatch", lambda: validate_responsibility_mismatch(job, resume_info, model)),
        ]
    else:
        raise ValueError(
            f"Unknown mismatch_type: {mismatch_type!r}. "
            f"Must be one of 'seniority', 'responsibility'."
        )

    for check_name, check_fn in checks:
        result = check_fn()
        if not result["passed"]:
            logger.warning(
                "validate_mismatched_skeleton: FAIL at %s — %s",
                check_name,
                result["reason"],
            )
            return {
                "passed": False,
                "failed_check": check_name,
                "reason": result["reason"],
            }

    logger.info("validate_mismatched_skeleton: PASS — all checks passed (mismatch_type=%s)", mismatch_type)
    return {
        "passed": True,
        "failed_check": None,
        "reason": None,
    }
