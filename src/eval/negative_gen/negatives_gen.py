"""
Module for generating seniority-mismatched job skeletons from resumes.

This module handles the generation phase of the negative pipeline, creating
job descriptions with intentionally mismatched seniority levels to the resume.
"""

import random
import sys
from pathlib import Path
from typing import Literal

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    GENERATION_TOP_P,
    GENERATION_TEMPERATURE,
    OLLAMA_MODEL,
    SKELETON_MAX_TOKENS,
)
from eval.positive_gen.positives_gen import JobSkeleton, parse_skeleton_response

MismatchType = Literal["seniority", "domain", "responsibility"]

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

DOMAIN_ORDER = ["backend", "frontend", "fullstack", "data"]

# Domain mismatch targets — prefer domains with minimal skill overlap
_DOMAIN_MISMATCH_TARGETS: dict[str, list[str]] = {
    "backend": ["frontend", "data"],
    "frontend": ["backend", "data"],
    "fullstack": ["data"],
    "data": ["frontend", "backend"],
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


def get_target_domain(resume_domain: str) -> str:
    """
    Return a mismatched target domain for the given resume domain.

    Uses random.choice for domains with multiple valid targets.

    Args:
        resume_domain: Canonical domain string from the resume
                      ("backend", "frontend", "fullstack", or "data").

    Returns:
        A domain string that is mismatched with the resume domain.

    Raises:
        ValueError: If resume_domain is not in DOMAIN_ORDER.
    """
    if resume_domain not in _DOMAIN_MISMATCH_TARGETS:
        raise ValueError(
            f"Invalid resume_domain: {resume_domain}. "
            f"Must be one of {DOMAIN_ORDER}."
        )
    targets = _DOMAIN_MISMATCH_TARGETS[resume_domain]
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


def _build_domain_mismatch_prompt(
    resume_text: str, resume_seniority: str, target_domain: str
) -> str:
    """
    Build a prompt for generating a domain-mismatched job skeleton.

    Seniority matches the resume; domain is forced to target_domain.
    Skills and responsibilities reflect the target domain.

    Args:
        resume_text: Full resume text.
        resume_seniority: Resume seniority level (e.g., "Senior").
        target_domain: The forced mismatched domain (e.g., "data").

    Returns:
        Prompt string ready for LLM.
    """
    return f"""Resume:
{resume_text}

Generate 1 job description skeleton for a {resume_seniority} role.

CRITICAL: The domain MUST be {target_domain} — do NOT match the candidate's domain.
The seniority MUST match the candidate's level: {resume_seniority}.
Use skills and responsibilities typical for {target_domain} engineering at this level.

Output ONLY these 7 fields, one per line. Do not add explanation or extra text.

Title: {resume_seniority} [Role] Engineer
Seniority: {resume_seniority}
YearsRequired: [appropriate range for {resume_seniority}]
Domain: {target_domain}
PrimarySkills: [skill1, skill2, skill3]
SecondarySkills: [skill4, skill5]
Responsibilities: [responsibility1; responsibility2; responsibility3]
"""


def _build_responsibility_mismatch_prompt(
    resume_text: str, resume_seniority: str, resume_domain: str
) -> str:
    """
    Build a prompt for generating a responsibility-mismatched job skeleton.

    Seniority and domain match the resume, but responsibilities describe
    a different engineering sub-role within the same domain.

    Args:
        resume_text: Full resume text.
        resume_seniority: Resume seniority level (e.g., "Senior").
        resume_domain: Resume domain (e.g., "backend").

    Returns:
        Prompt string ready for LLM.
    """
    return f"""Resume:
{resume_text}

Generate 1 job description skeleton for a {resume_seniority} {resume_domain} role.

CRITICAL: Seniority MUST be {resume_seniority}. Domain MUST be {resume_domain}.
HOWEVER: The responsibilities MUST be for a completely different engineering sub-role
within {resume_domain} — for example, if the candidate builds APIs, write responsibilities
for someone doing infrastructure, data pipelines, or ML model deployment instead.
The skills can reflect this different sub-role.

Output ONLY these 7 fields, one per line. Do not add explanation or extra text.

Title: {resume_seniority} [Different Sub-Role] Engineer
Seniority: {resume_seniority}
YearsRequired: [appropriate range for {resume_seniority}]
Domain: {resume_domain}
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
    resume_text: str,
    resume_seniority: str,
    model: str = OLLAMA_MODEL,
    mismatch_type: MismatchType = "seniority",
    resume_domain: str = "",
) -> tuple[JobSkeleton, dict]:
    """
    Generate a JobSkeleton with an intentional mismatch to the resume.

    The mismatch type determines what aspect of the job differs from the resume:
    - "seniority": job seniority is deliberately far from resume seniority; domain/skills match
    - "domain": job domain is deliberately different; seniority matches
    - "responsibility": job responsibilities describe a different sub-role; seniority/domain match

    Args:
        resume_text: Full resume text.
        resume_seniority: Candidate's seniority level (e.g., "Senior").
        model: Ollama model name.
        mismatch_type: Type of mismatch to generate ("seniority", "domain", "responsibility").
        resume_domain: Resume domain (required for "domain" and "responsibility" types).

    Returns:
        Tuple of (JobSkeleton dict, mismatch_context dict).
        The mismatch_context dict contains metadata needed by repair, with keys like:
        - "target_seniority": the target seniority level
        - "target_domain": the target domain (if domain mismatch)
        - "resume_domain": the original resume domain (if domain/responsibility mismatch)

    Raises:
        ValueError: If resume_seniority/resume_domain is invalid or LLM response cannot be parsed.
        ollama.RequestError: On Ollama connection failure.
        ollama.ResponseError: On invalid response from model.
    """
    if mismatch_type == "seniority":
        target_seniority = get_target_seniority(resume_seniority)
        years_range = _years_range_for_seniority(target_seniority)
        prompt = _build_mismatched_skeleton_prompt(resume_text, target_seniority, years_range)
        mismatch_context = {"target_seniority": target_seniority}

    elif mismatch_type == "domain":
        if not resume_domain:
            raise ValueError(
                "resume_domain is required for mismatch_type='domain'"
            )
        target_domain = get_target_domain(resume_domain)
        years_range = _years_range_for_seniority(resume_seniority)
        prompt = _build_domain_mismatch_prompt(resume_text, resume_seniority, target_domain)
        mismatch_context = {
            "target_seniority": resume_seniority,
            "target_domain": target_domain,
            "resume_domain": resume_domain,
        }

    elif mismatch_type == "responsibility":
        if not resume_domain:
            raise ValueError(
                "resume_domain is required for mismatch_type='responsibility'"
            )
        prompt = _build_responsibility_mismatch_prompt(
            resume_text, resume_seniority, resume_domain
        )
        mismatch_context = {
            "target_seniority": resume_seniority,
            "resume_domain": resume_domain,
        }

    else:
        raise ValueError(
            f"Unknown mismatch_type: {mismatch_type!r}. "
            f"Must be one of 'seniority', 'domain', 'responsibility'."
        )

    response = _call_ollama(prompt, model)
    skeleton = parse_skeleton_response(response)
    return skeleton, mismatch_context
