"""
Module for generating seniority-mismatched job skeletons from resumes.

This module handles the generation phase of the negative pipeline, creating
job descriptions with intentionally mismatched seniority levels to the resume.
"""

import random
import sys
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    GENERATION_TOP_P,
    GENERATION_TEMPERATURE,
    OLLAMA_MODEL,
    SKELETON_MAX_TOKENS,
)
from eval.positive_gen.positives_gen import JobSkeleton, parse_skeleton_response

SENIORITY_ORDER = ["Junior", "Mid", "Senior", "Staff"]

# Mismatched targets — minimum gap preserved where possible
_MISMATCH_TARGETS = {
    "Junior": ["Senior", "Staff"],  # both ≥2 away
    "Mid": ["Junior", "Staff"],  # Junior=1 away (extreme under), Staff=2 away (extreme over)
    "Senior": ["Junior"],  # 2 away going down
    "Staff": ["Junior", "Mid"],  # 3 and 2 away
}

# Canonical years ranges for each seniority level
_YEARS_FOR_SENIORITY = {
    "Junior": "0-2",
    "Mid": "2-4",
    "Senior": "4-7",
    "Staff": "7-10",
}


def get_target_seniority(resume_seniority: str) -> str:
    """
    Return a mismatched target seniority level for the given resume seniority.

    Uses random.choice for levels with multiple valid targets.

    Args:
        resume_seniority: Canonical seniority string from the resume
                         ("Junior", "Mid", "Senior", or "Staff").

    Returns:
        A seniority string that is mismatched with the resume level.

    Raises:
        ValueError: If resume_seniority is not in SENIORITY_ORDER.
    """
    if resume_seniority not in _MISMATCH_TARGETS:
        raise ValueError(
            f"Invalid resume_seniority: {resume_seniority}. "
            f"Must be one of {SENIORITY_ORDER}."
        )
    targets = _MISMATCH_TARGETS[resume_seniority]
    return random.choice(targets)


def _years_range_for_seniority(seniority: str) -> str:
    """
    Return the canonical years range string for a given seniority level.

    Args:
        seniority: Canonical seniority string ("Junior", "Mid", "Senior", "Staff").

    Returns:
        Years range as a string (e.g., "0-2").

    Raises:
        ValueError: If seniority is not recognized.
    """
    if seniority not in _YEARS_FOR_SENIORITY:
        raise ValueError(
            f"Invalid seniority: {seniority}. Must be one of {SENIORITY_ORDER}."
        )
    return _YEARS_FOR_SENIORITY[seniority]


def _build_mismatched_skeleton_prompt(
    resume_text: str, target_seniority: str, years_range: str
) -> str:
    """
    Build a prompt for generating a seniority-mismatched job skeleton.

    Embeds target_seniority and years_range as hard constraints to prevent LLM drift.
    Resume text provides domain and skills context only.

    Args:
        resume_text: Full resume text.
        target_seniority: The forced mismatched seniority (e.g., "Senior").
        years_range: Pre-computed years range (e.g., "4-7").

    Returns:
        Prompt string ready for LLM.
    """
    return f"""Resume:
{resume_text}

Generate 1 job description skeleton for a {target_seniority} role.

CRITICAL: The seniority MUST be {target_seniority} — do NOT match the candidate's level.
Use the resume's domain and skills as context for the job content only.

Output ONLY these 7 fields, one per line. Do not add explanation or extra text.

Title: {target_seniority} [Role] Engineer
Seniority: {target_seniority}
YearsRequired: {years_range}
Domain: [backend/frontend/fullstack/data]
PrimarySkills: [skill1, skill2, skill3]
SecondarySkills: [skill4, skill5]
Responsibilities: [responsibility1; responsibility2; responsibility3]
"""


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Call Ollama chat endpoint and return response content.

    Args:
        prompt: Prompt string to send to the model.
        model: Model name (default from config).

    Returns:
        Response content string.

    Raises:
        ollama.RequestError: On connection or request failure.
        ollama.ResponseError: On invalid response from model.
    """
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        options={
            "temperature": GENERATION_TEMPERATURE,
            "top_p": GENERATION_TOP_P,
            "num_predict": SKELETON_MAX_TOKENS,
        },
    )
    return response["message"]["content"]


def generate_mismatched_skeleton(
    resume_text: str, resume_seniority: str, model: str = OLLAMA_MODEL
) -> tuple[JobSkeleton, str]:
    """
    Generate a JobSkeleton with seniority intentionally mismatched to the resume.

    Determines target seniority, injects it as a hard constraint in the prompt
    (along with pre-computed years_range), calls LLM, and parses the response.

    Args:
        resume_text: Full resume text.
        resume_seniority: Candidate's seniority level (e.g., "Senior").
        model: Ollama model name.

    Returns:
        Tuple of (JobSkeleton dict, target_seniority str).
        The target_seniority string is needed so the pipeline can pass it consistently to repair.

    Raises:
        ValueError: If resume_seniority is invalid or LLM response cannot be parsed.
        ollama.RequestError: On Ollama connection failure.
        ollama.ResponseError: On invalid response from model.
    """
    target_seniority = get_target_seniority(resume_seniority)
    years_range = _years_range_for_seniority(target_seniority)
    prompt = _build_mismatched_skeleton_prompt(resume_text, target_seniority, years_range)
    response = _call_ollama(prompt, model)
    skeleton = parse_skeleton_response(response)
    return skeleton, target_seniority
