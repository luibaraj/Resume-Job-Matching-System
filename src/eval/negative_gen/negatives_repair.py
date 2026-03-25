"""
Module for repairing seniority-mismatched job skeletons that fail validation.

Attempts to fix a failed JobSkeleton through a two-attempt repair loop.
If both attempts fail, the skeleton is discarded. Each repair attempt sends
a targeted fix prompt to the LLM based on which validation rule set failed.
"""

import logging
import sys
from pathlib import Path
from typing import TypedDict

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    GENERATION_TEMPERATURE,
    GENERATION_TOP_P,
    OLLAMA_MODEL,
    REPAIR_MAX_TOKENS,
)
from eval.positive_gen.positives_gen import JobSkeleton, parse_skeleton_response
from eval.positive_gen.positives_validate import ResumeInfo
from .negatives_validate import validate_mismatched_skeleton

logger = logging.getLogger(__name__)

# Lowered temperature for attempt 2 to get more deterministic output
_REPAIR_TEMPERATURE_ATTEMPT2: float = 0.3


class RepairResult(TypedDict):
    """Result of the repair loop for a failed JobSkeleton."""

    success: bool  # True if a repaired skeleton passed validation
    job: JobSkeleton | None  # Repaired skeleton, or None if discarded
    attempts: int  # Number of repair attempts made (1 or 2)
    discard_reason: str | None  # Reason for discard; None on success


def _get_fields_for_check(failed_check: str) -> list[str]:
    """
    Maps each failed_check to the minimal set of JobSkeleton field keys
    that need to be shown to the LLM for repair.

    Extends the positive_repair mapping with two new checks specific to negatives.
    """
    field_map = {
        "structural": [
            "seniority",
            "domain",
            "years_required",
            "primary_skills",
            "title",
            "responsibilities",
        ],
        "seniority_years": ["seniority", "years_required"],
        "seniority_mismatch": ["seniority", "years_required", "title"],
        "skill_domain_overlap": [
            "primary_skills",
            "secondary_skills",
            "domain",
            "responsibilities",
        ],
    }
    return field_map.get(failed_check, [])


def _format_fields_for_prompt(job: JobSkeleton, fields: list[str]) -> str:
    """
    Formats only the specified fields from a JobSkeleton into canonical
    'Field: value' lines for prompt injection.
    """
    lines = []
    field_map = {
        "title": f"Title: {job['title']}",
        "seniority": f"Seniority: {job['seniority']}",
        "years_required": f"YearsRequired: {job['years_required']}",
        "domain": f"Domain: {job['domain']}",
        "primary_skills": f"PrimarySkills: {', '.join(job['primary_skills'])}",
        "secondary_skills": f"SecondarySkills: {', '.join(job['secondary_skills'])}",
        "responsibilities": f"Responsibilities: {'; '.join(job['responsibilities'])}",
    }
    for field in fields:
        if field in field_map:
            lines.append(field_map[field])
    return "\n".join(lines)


def _merge_repaired_fields(
    original: JobSkeleton, repaired: JobSkeleton, fields: list[str]
) -> JobSkeleton:
    """
    Merges only the repaired field values back into the original skeleton.
    Fields not in the repair response are kept from original unchanged.
    Only non-empty repaired values are applied — guards against the parser
    returning empty defaults for fields the model didn't output.
    """
    merged = dict(original)
    for field in fields:
        value = repaired.get(field)
        # Only apply non-empty values — empty means the model didn't output this field
        if value:
            merged[field] = value
    return merged


def _build_repair_prompt(
    job: JobSkeleton,
    failed_check: str,
    reason: str | None,
    attempt: int,
    resume_info: ResumeInfo,
    target_seniority: str,
) -> str:
    """
    Builds a targeted fix prompt for the failing check.
    Only the relevant fields are shown to the LLM.

    Args:
        job: The current JobSkeleton (may be partially repaired from a previous attempt).
        failed_check: The validation check that failed (e.g., "seniority_mismatch").
        reason: The failure reason from validation.
        attempt: Attempt number (1 or 2).
        resume_info: Resume information used for context in some fix instructions.
        target_seniority: The target mismatched seniority that generation determined.
                         Used for seniority_mismatch and skill_domain_overlap repairs.

    Returns:
        The prompt string to send to the LLM.
    """
    fields = _get_fields_for_check(failed_check)
    fields_text = _format_fields_for_prompt(job, fields)
    failure_msg = reason or failed_check

    # Map seniority to its bracket
    seniority_brackets = {
        "Junior": "0–2",
        "Mid": "2–4",
        "Senior": "4–7",
        "Staff": "7–10",
    }

    # Build fix instructions per failed_check
    if failed_check == "structural":
        fix_instruction = (
            "Fix any malformed fields:\n"
            "- Seniority must be one of: Junior, Mid, Senior, Staff\n"
            "- Domain must be one of: backend, frontend, fullstack, data\n"
            "- YearsRequired must be a number between 1 and 20 (e.g., '4-6' or '5')\n"
            "- PrimarySkills must have 2 to 4 items\n"
            "- Title must be a valid job title (non-empty)\n"
            "- Responsibilities must have 3 to 5 non-empty items (semicolon-separated)"
        )
    elif failed_check == "seniority_years":
        fix_instruction = (
            "Fix YearsRequired to match the seniority bracket:\n"
            f"- Junior: 0–2 years\n"
            f"- Mid: 2–4 years\n"
            f"- Senior: 4–7 years\n"
            f"- Staff: 7–10 years\n"
            f"Current seniority is '{job['seniority']}'. Adjust YearsRequired accordingly."
        )
    elif failed_check == "seniority_mismatch":
        target_bracket = seniority_brackets.get(target_seniority, "unknown")
        fix_instruction = (
            f"Fix the seniority and years to match the target: {target_seniority}\n"
            f"- Seniority MUST be: {target_seniority}\n"
            f"- YearsRequired MUST be in range: {target_bracket} years\n"
            f"Example Title format: '{target_seniority} [Role] Engineer'\n"
            f"This is intentionally different from the resume seniority — that is correct."
        )
    elif failed_check == "skill_domain_overlap":
        fix_instruction = (
            "Fix skills and domain to align with the resume:\n"
            "- At least 2 of the resume's skills must appear in PrimarySkills\n"
            "- Domain must match or be adjacent to the resume domain\n"
            "- Responsibilities must be plausible for this domain\n"
            f"Resume domain: {resume_info['domain']}\n"
            f"Resume primary skills: {', '.join(resume_info['primary_skills'])}\n"
            f"NOTE: Do NOT change seniority — {target_seniority} mismatch is intentional."
        )
    else:
        fix_instruction = "Fix the above fields to pass validation."

    if attempt == 1:
        # Attempt 1: surgical, targeted prompt
        prompt = (
            f"The following job fields failed validation.\n"
            f"\n"
            f"{fields_text}\n"
            f"\n"
            f"Validation failure: {failure_msg}\n"
            f"\n"
            f"{fix_instruction}\n"
            f"\n"
            f"Output ONLY the corrected fields above, one per line, in the same format.\n"
            f"Do not output any other fields. Do not add explanation or extra text."
        )
    else:
        # Attempt 2: stricter with enum hints and imperative close
        format_hints = []
        for field in fields:
            if field == "seniority":
                format_hints.append("Seniority: [Junior/Mid/Senior/Staff]")
            elif field == "domain":
                format_hints.append("Domain: [backend/frontend/fullstack/data]")
            elif field == "years_required":
                format_hints.append("YearsRequired: [e.g., 4-6]")
            elif field == "title":
                format_hints.append("Title: [Job Title]")
            elif field == "primary_skills":
                format_hints.append("PrimarySkills: skill1, skill2, skill3")
            elif field == "secondary_skills":
                format_hints.append("SecondarySkills: skill4, skill5")
            elif field == "responsibilities":
                format_hints.append("Responsibilities: [resp1; resp2; resp3]")
        format_template = "\n".join(format_hints)

        prompt = (
            f"The following job fields failed validation.\n"
            f"\n"
            f"{fields_text}\n"
            f"\n"
            f"Validation failure: {failure_msg}\n"
            f"\n"
            f"{fix_instruction}\n"
            f"\n"
            f"You MUST correct the above failure.\n"
            f"\n"
            f"Output ONLY the corrected fields, one per line:\n"
            f"{format_template}\n"
            f"\n"
            f"Do not add explanation, formatting, or extra text. You must correct this."
        )

    return prompt


def _call_ollama(
    prompt: str,
    model: str = OLLAMA_MODEL,
    temperature: float = GENERATION_TEMPERATURE,
) -> str:
    """
    Call Ollama chat endpoint and return response content.

    Args:
        prompt: The prompt to send.
        model: Ollama model name.
        temperature: Sampling temperature (0.0 to 1.0+).

    Returns:
        The LLM response text.

    Raises:
        ollama.RequestError: If the request fails.
        ollama.ResponseError: If the model returns an error.
    """
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": temperature,
            "top_p": GENERATION_TOP_P,
            "num_predict": REPAIR_MAX_TOKENS,
        },
    )
    return response["message"]["content"]


def repair_mismatched_skeleton(
    job: JobSkeleton,
    failed_check: str,
    reason: str | None,
    resume_info: ResumeInfo,
    target_seniority: str,
    model: str = OLLAMA_MODEL,
) -> RepairResult:
    """
    Attempt to repair a failed mismatched JobSkeleton through up to 2 targeted fix attempts.

    Each attempt sends a prompt to the LLM with only the relevant fields for the
    failing check. If the repaired skeleton passes all validations, it is returned
    immediately. If both attempts fail, the skeleton is discarded.

    CRITICAL: target_seniority is passed as a parameter (not re-computed via
    get_target_seniority) to ensure the repair targets the same seniority that
    generation determined.

    Args:
        job: The JobSkeleton that failed validation.
        failed_check: The validation check that failed (e.g., "seniority_mismatch").
        reason: The failure reason from validation.
        resume_info: Resume information (used in fix instructions for some checks).
        target_seniority: The target mismatched seniority (from generation, not re-randomized).
        model: Ollama model name (defaults to OLLAMA_MODEL).

    Returns:
        RepairResult with success, job, attempts, and discard_reason fields.
    """
    current_job = job  # track the merged skeleton across attempts

    for attempt in (1, 2):
        fields = _get_fields_for_check(failed_check)
        temperature = (
            GENERATION_TEMPERATURE
            if attempt == 1
            else _REPAIR_TEMPERATURE_ATTEMPT2
        )
        prompt = _build_repair_prompt(
            current_job, failed_check, reason, attempt, resume_info, target_seniority
        )
        raw = _call_ollama(prompt, model, temperature)

        try:
            repaired_partial = parse_skeleton_response(raw)
        except ValueError:
            logger.warning(
                "repair attempt %d: parse failed — raw: %r", attempt, raw
            )
            continue  # count as failed attempt

        # Merge only the repaired fields back into the current skeleton
        candidate = _merge_repaired_fields(current_job, repaired_partial, fields)

        validation = validate_mismatched_skeleton(candidate, resume_info, model)

        if validation["passed"]:
            logger.info("repair attempt %d: PASS — %s", attempt, candidate["title"])
            return RepairResult(
                success=True,
                job=candidate,
                attempts=attempt,
                discard_reason=None,
            )
        else:
            logger.warning(
                "repair attempt %d: FAIL — check=%s reason=%s",
                attempt,
                validation["failed_check"],
                validation["reason"],
            )
            # Update state for next iteration — failure may have shifted
            current_job = candidate
            failed_check = validation["failed_check"] or failed_check
            reason = validation["reason"]

    # Both attempts exhausted
    logger.warning("repair: DISCARD after 2 attempts — reason=%s", reason)
    return RepairResult(success=False, job=None, attempts=2, discard_reason=reason)
