"""
Module for repairing seniority-mismatched job skeletons that fail validation.

Attempts to fix a failed JobSkeleton through a two-attempt repair loop.
If both attempts fail, the skeleton is discarded. Each repair attempt sends
a targeted fix prompt to the LLM based on which validation rule set failed.
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
from eval.positive_gen.positives_gen import JobSkeleton, parse_skeleton_response
from eval.positive_gen.positives_validate import ResumeInfo
from .negatives_validate import validate_mismatched_skeleton
from .negatives_gen import MismatchType

logger = logging.getLogger(__name__)

# Backwards compatibility: tests may import private names from this module
_format_fields_for_prompt = format_fields_for_prompt
_merge_repaired_fields = merge_repaired_fields


def _get_fields_for_check(failed_check: str) -> list[str]:
    """
    Maps each failed_check to the minimal set of JobSkeleton field keys
    that need to be shown to the LLM for repair.

    Extends the positive_repair mapping with new checks specific to negatives.
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
        "responsibility_mismatch": ["responsibilities", "primary_skills", "secondary_skills"],
    }
    return field_map.get(failed_check, [])




def _build_repair_prompt(
    job: JobSkeleton,
    failed_check: str,
    reason: str | None,
    attempt: int,
    resume_info: ResumeInfo,
    mismatch_context: dict,
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
        mismatch_context: Dict with mismatch metadata, e.g.:
                         {"target_seniority": "Senior"} for seniority mismatch
                         {"target_seniority": "Senior", "target_domain": "data", "resume_domain": "backend"} for domain mismatch
                         {"target_seniority": "Senior", "resume_domain": "backend"} for responsibility mismatch

    Returns:
        The prompt string to send to the LLM.
    """
    fields = _get_fields_for_check(failed_check)
    fields_text = format_fields_for_prompt(job, fields)
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
        target_seniority = mismatch_context.get("target_seniority", "")
        target_bracket = seniority_brackets.get(target_seniority, "unknown")
        fix_instruction = (
            f"Fix the seniority and years to match the target: {target_seniority}\n"
            f"- Seniority MUST be: {target_seniority}\n"
            f"- YearsRequired MUST be in range: {target_bracket} years\n"
            f"Example Title format: '{target_seniority} [Role] Engineer'\n"
            f"This is intentionally different from the resume seniority — that is correct."
        )
    elif failed_check == "skill_domain_overlap":
        target_seniority = mismatch_context.get("target_seniority", "")
        fix_instruction = (
            "Fix skills and domain to align with the resume:\n"
            "- At least 2 of the resume's skills must appear in PrimarySkills\n"
            "- Domain must match or be adjacent to the resume domain\n"
            "- Responsibilities must be plausible for this domain\n"
            f"Resume domain: {resume_info['domain']}\n"
            f"Resume primary skills: {', '.join(resume_info['primary_skills'])}\n"
            f"NOTE: Do NOT change seniority — {target_seniority} mismatch is intentional."
        )
    elif failed_check == "responsibility_mismatch":
        resume_domain = mismatch_context.get("resume_domain", resume_info["domain"])
        fix_instruction = (
            "Rewrite responsibilities to describe a DIFFERENT type of engineering work.\n"
            "- Choose a different sub-role within the same domain\n"
            "- Responsibilities must NOT describe work the resume candidate demonstrably does\n"
            f"Resume domain: {resume_domain}\n"
            f"Resume primary skills: {', '.join(resume_info['primary_skills'])}\n"
            "Focus on a clearly different function: e.g., if candidate builds APIs, write "
            "responsibilities for infrastructure automation, data pipelines, or ML serving."
        )
    else:
        fix_instruction = "Fix the above fields to pass validation."

    if attempt == 1:
        prompt = build_attempt1_prompt(fields_text, failure_msg, fix_instruction)
    else:
        prompt = build_attempt2_prompt(fields_text, failure_msg, fix_instruction, fields)

    return prompt




def repair_mismatched_skeleton(
    job: JobSkeleton,
    failed_check: str,
    reason: str | None,
    resume_info: ResumeInfo,
    mismatch_context: dict,
    model: str = OLLAMA_MODEL,
    mismatch_type: MismatchType = "seniority",
) -> RepairResult:
    """
    Attempt to repair a failed mismatched JobSkeleton through up to 2 targeted fix attempts.

    Each attempt sends a prompt to the LLM with only the relevant fields for the
    failing check. If the repaired skeleton passes all validations, it is returned
    immediately. If both attempts fail, the skeleton is discarded.

    CRITICAL: mismatch_context is passed as a parameter (not re-computed during repair)
    to ensure the repair maintains the same mismatch target that generation determined.

    Args:
        job: The JobSkeleton that failed validation.
        failed_check: The validation check that failed (e.g., "seniority_mismatch").
        reason: The failure reason from validation.
        resume_info: Resume information (used in fix instructions for some checks).
        mismatch_context: Dict with mismatch metadata from generation (e.g., target_seniority, target_domain).
        model: Ollama model name (defaults to OLLAMA_MODEL).
        mismatch_type: Type of mismatch ("seniority", "domain", "responsibility").

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
            current_job, failed_check, reason, attempt, resume_info, mismatch_context
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

        validation = validate_mismatched_skeleton(candidate, resume_info, model, mismatch_type)

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
