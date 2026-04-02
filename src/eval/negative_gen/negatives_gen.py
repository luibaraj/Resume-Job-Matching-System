"""
Module for generating mismatched job skeletons from resumes.

This module handles the generation phase of the negative pipeline, creating
job descriptions with intentionally mismatched seniority or responsibilities
to the resume. Deterministic fields (title, seniority, domain, years_required)
are generated using the same approach as the positives pipeline to ensure
structural consistency and reduce validation failures.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Literal

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    GENERATION_TEMPERATURE,
    GENERATION_TOP_P,
    OLLAMA_MODEL,
    RESPONSIBILITY_MAX_TOKENS,
    TARGET_RESPONSIBILITY_COUNT,
)
from eval.positive_gen.positives_gen import (
    JobSkeleton,
    _constrain_years_required,
    _extract_years_experience,
    _generate_deterministic_fields,
    _generate_single_responsibility,
    _generate_skills,
    parse_skeleton_response,
)

MismatchType = Literal["seniority", "responsibility"]

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

def get_target_seniority(resume_seniority: str, rng: random.Random | None = None) -> str:
    """
    Return a mismatched target seniority level for the given resume seniority.

    Uses random.choice (or seeded RNG) for levels with multiple valid targets.

    Args:
        resume_seniority: Canonical seniority string from the resume
                         ("Junior", "Mid", "Senior", or "Staff").
        rng: Optional seeded random.Random instance for reproducible selection.
             If None, uses the global random module (default unseeded behavior).

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
    # Use provided RNG if available; otherwise use global random module
    chooser = rng.choice if rng is not None else random.choice
    return chooser(targets)


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

Generate ONE responsibility for a {resume_seniority} {resume_domain} role.

CRITICAL: The responsibility MUST be for a completely different engineering sub-role
within {resume_domain} — for example, if the candidate builds APIs, write for someone
doing infrastructure, data pipelines, or ML model deployment instead.
The responsibility must be a single sentence with at least 10 words.

Output ONLY the responsibility sentence (no label, no extra text)."""


def generate_mismatched_skeleton(
    resume_info: dict,
    model: str = OLLAMA_MODEL,
    mismatch_type: MismatchType = "seniority",
) -> tuple[JobSkeleton, dict]:
    """
    Generate a JobSkeleton with an intentional mismatch to the resume.

    Uses deterministic field generation (title, seniority, domain, years_required)
    matching the positives pipeline approach to ensure structural consistency.

    The mismatch type determines what aspect of the job differs from the resume:
    - "seniority": job seniority is deliberately far from resume seniority; domain/skills match
    - "responsibility": job responsibilities describe a different sub-role; seniority/domain match

    Args:
        resume_info: Resume information dict with keys: resume_text, seniority, domain.
        model: Ollama model name.
        mismatch_type: Type of mismatch to generate ("seniority", "responsibility").

    Returns:
        Tuple of (JobSkeleton dict, mismatch_context dict).
        The mismatch_context dict contains metadata needed by repair, with keys:
        - "target_seniority": the target seniority level (for seniority mismatch)
        - "mismatch_dimension": "responsibility" (for responsibility mismatch)

    Raises:
        ValueError: If resume_seniority is invalid or generation fails.
        ollama.RequestError: On Ollama connection failure.
        ollama.ResponseError: On invalid response from model.
    """
    resume_text = resume_info["resume_text"]
    resume_seniority = resume_info["seniority"]
    resume_domain = resume_info["domain"]

    if mismatch_type == "seniority":
        # Seniority mismatch: deterministic fields from target seniority
        # Use seeded RNG for reproducible seniority selection
        rng = random.Random(42)
        target_seniority = get_target_seniority(resume_seniority, rng=rng)

        # Extract years experience and generate deterministic fields
        years_experience = _extract_years_experience(resume_text, model)
        det = _generate_deterministic_fields(target_seniority, resume_domain, years_experience)

        # For seniority mismatch, override years_required to NOT clamp to candidate experience
        # Use the raw bracket for target seniority
        det["years_required"] = _YEARS_FOR_SENIORITY[target_seniority]

        # Generate skills from the deterministic target seniority
        target_resume_info = resume_info.copy()  # type: ignore[union-attr]
        target_resume_info["seniority"] = target_seniority
        primary_skills, secondary_skills = _generate_skills(target_resume_info, model)  # type: ignore[arg-type]

        # Generate responsibilities using standard responsibility generator
        responsibilities: list[str] = []
        max_resp_attempts = TARGET_RESPONSIBILITY_COUNT * 4
        attempts = 0

        while len(responsibilities) < TARGET_RESPONSIBILITY_COUNT and attempts < max_resp_attempts:
            attempts += 1
            try:
                resp = _generate_single_responsibility(
                    target_resume_info, primary_skills, model, responsibilities  # type: ignore[arg-type]
                )
                responsibilities.append(resp)
            except ValueError:
                continue

        if len(responsibilities) < 3:
            raise ValueError(f"Failed to generate ≥3 responsibilities for seniority mismatch")

        skeleton: JobSkeleton = {
            "title": det["title"],
            "seniority": det["seniority"],
            "years_required": det["years_required"],
            "domain": det["domain"],
            "primary_skills": primary_skills,
            "secondary_skills": secondary_skills,
            "responsibilities": responsibilities[:TARGET_RESPONSIBILITY_COUNT],
        }
        mismatch_context = {"target_seniority": target_seniority}

    elif mismatch_type == "responsibility":
        # Responsibility mismatch: deterministic fields from resume, but responsibilities differ
        years_experience = _extract_years_experience(resume_text, model)
        det = _generate_deterministic_fields(resume_seniority, resume_domain, years_experience)

        # Generate skills normally
        primary_skills, secondary_skills = _generate_skills(resume_info, model)  # type: ignore[arg-type]

        # Generate responsibilities that describe a different sub-role
        responsibilities: list[str] = []
        max_resp_attempts = TARGET_RESPONSIBILITY_COUNT * 4
        attempts = 0

        while len(responsibilities) < TARGET_RESPONSIBILITY_COUNT and attempts < max_resp_attempts:
            attempts += 1
            try:
                prompt = _build_responsibility_mismatch_prompt(
                    resume_text, resume_seniority, resume_domain
                )
                raw = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    options={
                        "temperature": GENERATION_TEMPERATURE,
                        "top_p": GENERATION_TOP_P,
                        "num_predict": RESPONSIBILITY_MAX_TOKENS,
                    },
                )
                resp = raw["message"]["content"].strip()

                if not resp:
                    continue

                word_count = len(resp.split())
                if word_count < 5:
                    continue

                responsibilities.append(resp)
            except (ValueError, KeyError):
                continue

        if len(responsibilities) < 3:
            raise ValueError(f"Failed to generate ≥3 responsibilities for responsibility mismatch")

        skeleton: JobSkeleton = {
            "title": det["title"],
            "seniority": det["seniority"],
            "years_required": det["years_required"],
            "domain": det["domain"],
            "primary_skills": primary_skills,
            "secondary_skills": secondary_skills,
            "responsibilities": responsibilities[:TARGET_RESPONSIBILITY_COUNT],
        }
        mismatch_context = {"mismatch_dimension": "responsibility"}

    else:
        raise ValueError(
            f"Unknown mismatch_type: {mismatch_type!r}. "
            f"Must be one of 'seniority', 'responsibility'."
        )

    return skeleton, mismatch_context
