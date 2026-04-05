"""
Shared utility functions for the eval pipeline.

Provides common LLM interfaces, prompt building, and data structures used by
the positive and negative generation, validation, and repair modules.
"""

import logging
import sys
import time
from pathlib import Path
from typing import TypedDict

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    GENERATION_TEMPERATURE,
    GENERATION_TOP_P,
    OLLAMA_MODEL,
    REPAIR_MAX_TOKENS,
    VALIDATION_MAX_TOKENS,
)
from src.eval.types import JobSkeleton

logger = logging.getLogger(__name__)

__all__ = [
    "_REPAIR_TEMPERATURE_ATTEMPT2",
    "RepairResult",
    "call_ollama_validate",
    "call_ollama_repair",
    "format_fields_for_prompt",
    "merge_repaired_fields",
    "build_attempt1_prompt",
    "build_attempt2_prompt",
]

# Lowered temperature for attempt 2 to get more deterministic output
_REPAIR_TEMPERATURE_ATTEMPT2: float = 0.3


class RepairResult(TypedDict):
    """Result of the repair loop for a failed JobSkeleton."""

    success: bool  # True if a repaired skeleton passed validation
    job: JobSkeleton | None  # Repaired skeleton, or None if discarded
    attempts: int  # Number of repair attempts made (1 or 2)
    discard_reason: str | None  # Reason for discard; None on success


def call_ollama_validate(prompt: str, model: str = OLLAMA_MODEL, max_retries: int = 1) -> str:
    """
    Call Ollama chat endpoint for validation and return response content.

    Retries once on transient RequestError with short exponential backoff.

    Args:
        prompt: The prompt to send.
        model: Ollama model name.
        max_retries: Number of retry attempts on transient errors.

    Returns:
        The LLM response text.

    Raises:
        ollama.RequestError: If the request fails after all retries.
        ollama.ResponseError: If the model returns an error.
    """
    attempt = 0
    while True:
        try:
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
        except ollama.RequestError as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "call_ollama_validate failed after %d attempts: %s",
                    max_retries,
                    e,
                )
                raise
            # Short exponential backoff: 0.3s, then 0.6s
            delay = 0.3 * (2 ** (attempt - 1))
            logger.warning(
                "call_ollama_validate attempt %d/%d failed: %s. Retrying in %.1fs...",
                attempt,
                max_retries,
                e,
                delay,
            )
            time.sleep(delay)


def call_ollama_repair(
    prompt: str,
    model: str = OLLAMA_MODEL,
    temperature: float = GENERATION_TEMPERATURE,
    max_retries: int = 2,
) -> str:
    """
    Call Ollama chat endpoint for repair and return response content.

    Retries up to twice on transient RequestError with short exponential backoff.

    Args:
        prompt: The prompt to send.
        model: Ollama model name.
        temperature: Sampling temperature (0.0 to 1.0+).
        max_retries: Number of retry attempts on transient errors.

    Returns:
        The LLM response text.

    Raises:
        ollama.RequestError: If the request fails after all retries.
        ollama.ResponseError: If the model returns an error.
    """
    attempt = 0
    while True:
        try:
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
        except ollama.RequestError as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "call_ollama_repair failed after %d attempts: %s",
                    max_retries,
                    e,
                )
                raise
            # Short exponential backoff: 0.3s, then 0.6s
            delay = 0.3 * (2 ** (attempt - 1))
            logger.warning(
                "call_ollama_repair attempt %d/%d failed: %s. Retrying in %.1fs...",
                attempt,
                max_retries,
                e,
                delay,
            )
            time.sleep(delay)


def format_fields_for_prompt(job: JobSkeleton, fields: list[str]) -> str:
    """
    Format only the specified fields from a JobSkeleton into canonical
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


def merge_repaired_fields(
    original: JobSkeleton, repaired: JobSkeleton, fields: list[str]
) -> JobSkeleton:
    """
    Merge only the repaired field values back into the original skeleton.

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
    return merged  # type: ignore[return-value]


def build_attempt1_prompt(
    fields_text: str,
    failure_msg: str,
    fix_instruction: str,
) -> str:
    """
    Build attempt-1 repair prompt.

    Args:
        fields_text: Formatted fields from format_fields_for_prompt().
        failure_msg: Validation failure reason.
        fix_instruction: Module-specific fix instructions.

    Returns:
        Prompt string ready for LLM call.
    """
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
    return prompt


def build_attempt2_prompt(
    fields_text: str,
    failure_msg: str,
    fix_instruction: str,
    fields: list[str],
) -> str:
    """
    Build attempt-2 repair prompt with stricter constraints and format hints.

    Args:
        fields_text: Formatted fields from format_fields_for_prompt().
        failure_msg: Validation failure reason.
        fix_instruction: Module-specific fix instructions.
        fields: List of field names to rebuild format hints from.

    Returns:
        Prompt string ready for LLM call.
    """
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
