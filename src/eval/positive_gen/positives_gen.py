"""
Synthetic positives generation — Step 1: Job skeleton generation.

Generates a structured job description skeleton from a resume using an LLM.
The skeleton captures: title, seniority, years required, domain, and primary/secondary skills.

This is Stage 1 of the synthetic positives pipeline. The skeleton is intentionally
minimal and well-structured so Stage 2 can expand it into a full job description
without repeating the LLM's tendency to hallucinate free-form content.

All LLM calls use a locally hosted LLaMA 3.2 3B model via Ollama.
"""

import logging
import sys
from pathlib import Path
from typing import TypedDict

import ollama

# Allow running as a script from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    GENERATION_TEMPERATURE,
    GENERATION_TOP_P,
    OLLAMA_MODEL,
    SKELETON_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


class JobSkeleton(TypedDict):
    """Parsed output from the job skeleton LLM call."""

    title: str
    seniority: str
    years_required: str  # Raw string (e.g., "4-6"); not converted to int
    domain: str  # backend/frontend/fullstack/data
    primary_skills: list[str]  # Parsed from comma-separated string
    secondary_skills: list[str]
    responsibilities: list[str]  # 3–5 bullet-point responsibilities


def _build_skeleton_prompt(resume_text: str, resume_seniority: str) -> str:
    """Build prompt for generating a job skeleton from a resume.

    Args:
        resume_text: Full resume text.
        resume_seniority: The candidate's seniority level (e.g., "Senior", "Mid").
                         This is fixed and must not be changed by the LLM.
    """
    return f"""Resume: {resume_text}

Generate 1 job description skeleton matching this resume. The job MUST:
- Use seniority: {resume_seniority}  ← FIXED, do not change
- Require no more years of experience than the candidate has
- Each responsibility must be a complete sentence of at least 10 words

Output ONLY these fields, one per line:

Title: [{resume_seniority}] [Role] Engineer
Seniority: {resume_seniority}
YearsRequired: [4-6]
Domain: [backend/frontend/fullstack/data]
PrimarySkills: [skill1, skill2, skill3]
SecondarySkills: [skill4, skill5]
Responsibilities: [10+ word responsibility sentence; 10+ word responsibility sentence; 10+ word responsibility sentence]

Do not add explanation, formatting, or extra text."""


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Call Ollama chat endpoint and return response content."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": GENERATION_TEMPERATURE,
            "top_p": GENERATION_TOP_P,
            "num_predict": SKELETON_MAX_TOKENS,
        },
    )
    return response["message"]["content"]


def parse_skeleton_response(response: str) -> dict:
    """
    Parse a structured LLM skeleton response into a Python dict.

    Expected format (one field per line):
        Title: Senior Backend Engineer
        Seniority: Senior
        YearsRequired: 4-6
        Domain: backend
        PrimarySkills: Python, PostgreSQL, Docker
        SecondarySkills: Redis, Kubernetes
        Responsibilities: resp1; resp2; resp3

    Args:
        response: Raw LLM response string.

    Returns:
        Dict with keys: title, seniority, years_required, domain,
        primary_skills, secondary_skills, responsibilities. Missing fields default to
        empty string or empty list.

    Raises:
        ValueError: If response is empty or contains no parseable key:value lines.
    """
    FIELD_MAP = {
        "title": "title",
        "seniority": "seniority",
        "yearsrequired": "years_required",
        "domain": "domain",
        "primaryskills": "primary_skills",
        "secondaryskills": "secondary_skills",
        "responsibilities": "responsibilities",
    }

    stripped = response.strip()
    if not stripped:
        raise ValueError("Empty LLM response — cannot parse skeleton")

    raw: dict[str, str] = {}
    for line in stripped.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        normalized_key = key.strip().lower().replace(" ", "")
        raw[normalized_key] = value.strip()

    if not any(k in raw for k in FIELD_MAP):
        raise ValueError(f"No recognizable fields in LLM response: {response!r}")

    def split_skills(s: str) -> list[str]:
        return [skill.strip() for skill in s.split(",") if skill.strip()]

    def split_responsibilities(s: str) -> list[str]:
        import re as _re
        items = _re.split(r"[;,]", s)
        return [item.strip() for item in items if item.strip()]

    return {
        "title": raw.get("title", ""),
        "seniority": raw.get("seniority", ""),
        "years_required": raw.get("yearsrequired", ""),
        "domain": raw.get("domain", ""),
        "primary_skills": split_skills(raw.get("primaryskills", "")),
        "secondary_skills": split_skills(raw.get("secondaryskills", "")),
        "responsibilities": split_responsibilities(raw.get("responsibilities", "")),
    }


def generate_job_skeleton(
    resume_text: str,
    model: str = OLLAMA_MODEL,
    resume_seniority: str = "",
) -> dict:
    """
    Generate a job skeleton matching the given resume text.

    Builds a prompt, calls the LLM, and parses the structured response
    into a dict. This is Step 1 of the synthetic positives pipeline;
    the skeleton is designed to be expanded into a full job description
    in a subsequent step.

    Args:
        resume_text: Full text of the candidate's resume.
        model: Ollama model name (default: OLLAMA_MODEL from config).
        resume_seniority: The candidate's seniority level (e.g., "Senior", "Mid").
                         If provided, seniority is deterministically set to this value,
                         ignoring any LLM output. Default: "" (no override).

    Returns:
        JobSkeleton dict with keys: title, seniority, years_required,
        domain, primary_skills, secondary_skills, responsibilities.

    Raises:
        ValueError: If the LLM response cannot be parsed.
        ollama.RequestError: If Ollama is not reachable.
        ollama.ResponseError: If the model returns an error response.
    """
    prompt = _build_skeleton_prompt(resume_text, resume_seniority)
    raw_response = _call_ollama(prompt, model)
    logger.debug("Raw skeleton response: %s", raw_response)
    skeleton = parse_skeleton_response(raw_response)

    # Deterministically override seniority if provided
    if resume_seniority:
        skeleton["seniority"] = resume_seniority

    logger.info(
        "Generated skeleton — title: %s, seniority: %s, domain: %s",
        skeleton["title"],
        skeleton["seniority"],
        skeleton["domain"],
    )
    return skeleton


