"""
Synthetic positives repair — Step 3: Failure recovery.

Attempts to fix a failed JobSkeleton through a two-attempt repair loop.
If both attempts fail, the skeleton is discarded (returns RepairResult with
success=False and job=None). Each repair attempt sends a targeted fix prompt
to the LLM based on which validation rule set failed. Only the relevant fields
for the failing check are shown to the model and expected in its output, keeping
noise minimal and preventing inadvertent mutations of valid fields. Attempt 2
lowers temperature and adds stricter constraints.

All LLM calls use a locally hosted LLaMA 3.2 3B model via Ollama.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    GENERATION_TEMPERATURE,
    OLLAMA_MODEL,
)
from eval.eval_utils import (
    _REPAIR_TEMPERATURE_ATTEMPT2,
    RepairResult,
    call_ollama_repair,
    format_fields_for_prompt,
    merge_repaired_fields,
    build_attempt1_prompt,
    build_attempt2_prompt,
)
from .positives_gen import JobSkeleton, parse_skeleton_response
from .positives_validate import ResumeInfo, validate_job_skeleton

logger = logging.getLogger(__name__)

# Backwards compatibility: tests may import private names from this module
_format_fields_for_prompt = format_fields_for_prompt
_merge_repaired_fields = merge_repaired_fields


def _get_fields_for_check(failed_check: str) -> list[str]:
    """
    Maps each failed_check to the minimal set of JobSkeleton field keys
    that need to be shown to the LLM for repair.
    """
    field_map = {
        "responsibilities": ["responsibilities"],
        "structural": [
            "seniority",
            "domain",
            "years_required",
            "primary_skills",
            "title",
            "responsibilities",
        ],
        "seniority_years": ["seniority", "years_required"],
        "resume_job_alignment": ["primary_skills", "seniority", "years_required", "responsibilities"],
        "domain_consistency": ["domain"],
    }
    return field_map.get(failed_check, [])




def _build_repair_prompt(
    job: JobSkeleton,
    failed_check: str,
    reason: str | None,
    attempt: int,
    resume_info: ResumeInfo,
) -> str:
    """
    Builds a targeted fix prompt for the failing check.
    Only the relevant fields are shown to the LLM.

    Args:
        job: The current JobSkeleton (may be partially repaired from a previous attempt)
        failed_check: The validation check that failed (e.g., "seniority_years")
        reason: The failure reason from validation
        attempt: Attempt number (1 or 2)
        resume_info: Resume information used for context in some fix instructions

    Returns:
        The prompt string to send to the LLM
    """
    fields = _get_fields_for_check(failed_check)
    fields_text = format_fields_for_prompt(job, fields)
    failure_msg = reason or failed_check

    # Build fix instructions per failed_check
    if failed_check == "responsibilities":
        fix_instruction = (
            "Rewrite the Responsibilities field. Requirements:\n"
            "- Must have 3 to 5 items (semicolon-separated)\n"
            "- Each item must be a distinct, non-repetitive complete sentence of at least 10 words\n"
            "- Each item must describe a real engineering task matching the resume skills"
        )
    elif failed_check == "structural":
        fix_instruction = (
            "Fix any malformed fields:\n"
            "- Seniority must be one of: Junior, Mid, Senior, Staff\n"
            "- Domain must be one of: backend, frontend, fullstack, data\n"
            "- YearsRequired must be a number between 1 and 20 (e.g., '4-6' or '5')\n"
            "- PrimarySkills must have 2 to 4 items\n"
            "- Title must be a valid job title (non-empty)"
        )
    elif failed_check == "seniority_years":
        fix_instruction = (
            "Fix YearsRequired to match the seniority bracket:\n"
            f"- Junior: 0–2 years\n"
            f"- Mid: 2–5 years\n"
            f"- Senior: 4–8 years\n"
            f"- Staff: 6+ years\n"
            f"Current seniority is '{job['seniority']}'. Adjust YearsRequired accordingly."
        )
    elif failed_check == "resume_job_alignment":
        fix_instruction = (
            "Fix skills, seniority, years, and responsibilities so the job aligns with the resume:\n"
            "- At least 2 of the resume's primary skills must appear in PrimarySkills\n"
            "- Job seniority must exactly match resume seniority\n"
            f"- Job years required must be ≤ {resume_info['years_experience']} (resume experience)\n"
            "- Each responsibility must align with at least one area of work in the resume\n"
            f"Resume seniority: {resume_info['seniority']}\n"
            f"Resume years experience: {resume_info['years_experience']}\n"
            f"Resume primary skills: {', '.join(resume_info['primary_skills'])}"
        )
    elif failed_check == "domain_consistency":
        fix_instruction = (
            "Fix the Domain field to match the resume and the job title:\n"
            f"- Resume domain: {resume_info['domain']}\n"
            "- Domain must be one of: backend, frontend, fullstack, data\n"
            "- Domain must be consistent with the Title field\n"
            "- Adjacent domains are allowed, but not major shifts"
        )
    else:
        fix_instruction = "Fix the above fields to pass validation."

    if attempt == 1:
        prompt = build_attempt1_prompt(fields_text, failure_msg, fix_instruction)
    else:
        prompt = build_attempt2_prompt(fields_text, failure_msg, fix_instruction, fields)

    return prompt




def repair_job_skeleton(
    job: JobSkeleton,
    failed_check: str,
    reason: str | None,
    resume_info: ResumeInfo,
    model: str = OLLAMA_MODEL,
) -> RepairResult:
    """
    Attempt to repair a failed JobSkeleton through up to 2 targeted fix attempts.

    Each attempt sends a prompt to the LLM with only the relevant fields for the
    failing check. If the repaired skeleton passes all validations, it is returned
    immediately. If both attempts fail, the skeleton is discarded.

    Args:
        job: The JobSkeleton that failed validation
        failed_check: The validation check that failed (e.g., "seniority_years")
        reason: The failure reason from validation
        resume_info: Resume information (used in fix instructions for some checks)
        model: Ollama model name (defaults to OLLAMA_MODEL)

    Returns:
        RepairResult with success, job, attempts, and discard_reason fields
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
            current_job, failed_check, reason, attempt, resume_info
        )
        raw = call_ollama_repair(prompt, model, temperature)

        try:
            repaired_partial = parse_skeleton_response(raw)
        except ValueError:
            logger.warning(
                "repair attempt %d: parse failed — raw: %r", attempt, raw
            )
            continue  # count as failed attempt

        # Merge only the repaired fields back into the current skeleton
        candidate = merge_repaired_fields(current_job, repaired_partial, fields)

        validation = validate_job_skeleton(candidate, resume_info, model)

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
