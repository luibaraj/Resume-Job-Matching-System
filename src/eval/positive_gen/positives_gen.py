"""
Synthetic positives generation — Step 1: Job skeleton generation.

Generates a structured job description skeleton from a resume using an LLM.
The skeleton captures: title, seniority, years required, domain, and primary/secondary skills.

This is Stage 1 of the synthetic positives pipeline. The skeleton is intentionally
minimal and well-structured so Stage 2 can expand it into a full job description
without repeating the LLM's tendency to hallucinate free-form content.

Per-field strategy: deterministic fields (title, seniority, domain, years_required)
are assembled directly from resume_info. LLM is used only for skills and responsibilities,
with per-responsibility generation and anti-repetition context injection.

All LLM calls use a locally hosted LLaMA 3.2 3B model via Ollama.
"""

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import ollama

# Allow running as a script from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    GENERATION_TEMPERATURE,
    GENERATION_TOP_P,
    OLLAMA_MODEL,
    RESPONSIBILITY_MAX_TOKENS,
    RESUME_EXTRACT_MAX_TOKENS,
    SKILLS_MAX_TOKENS,
    TARGET_RESPONSIBILITY_COUNT,
)

if TYPE_CHECKING:
    from eval.positive_gen.positives_validate import ResumeInfo

logger = logging.getLogger(__name__)


# Lazy import for _parse_years_required to avoid circular imports
def _parse_years_required(years_str: str) -> int:
    """Parse a raw years_required string to an integer (max of range).

    This is a re-export from positives_validate to avoid circular imports
    while keeping it available for tests.
    """
    from eval.positive_gen.positives_validate import _parse_years_required as _parse
    return _parse(years_str)


class JobSkeleton(TypedDict):
    """Parsed output from the job skeleton LLM call."""

    title: str
    seniority: str
    years_required: str  # Raw string (e.g., "4-6"); not converted to int
    domain: str  # backend/frontend/fullstack/data
    primary_skills: list[str]  # Parsed from comma-separated string
    secondary_skills: list[str]
    responsibilities: list[str]  # 3–5 bullet-point responsibilities


def _call_ollama(
    prompt: str,
    model: str = OLLAMA_MODEL,
    temperature: float = GENERATION_TEMPERATURE,
    max_tokens: int = SKILLS_MAX_TOKENS,
) -> str:
    """Call Ollama chat endpoint and return response content.

    Args:
        prompt: The prompt to send to the LLM.
        model: Ollama model name (default: OLLAMA_MODEL from config).
        temperature: Sampling temperature (default: GENERATION_TEMPERATURE).
        max_tokens: Maximum tokens in response (default: SKILLS_MAX_TOKENS).

    Returns:
        The response content string.
    """
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": temperature,
            "top_p": GENERATION_TOP_P,
            "num_predict": max_tokens,
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


def _build_years_extraction_prompt(resume_text: str) -> str:
    """Build prompt to extract years of experience from resume text.

    Args:
        resume_text: Full resume text.

    Returns:
        Prompt string requesting single-line output with YearsExperience field.
    """
    return f"""Resume: {resume_text}

Extract the total years of professional work experience from this resume.
Sum all positions and roles to get a total integer.

Output ONLY:
YearsExperience: [integer]

For example:
YearsExperience: 7

Do not add explanation or extra text."""


def _extract_years_experience(resume_text: str, model: str = OLLAMA_MODEL) -> int:
    """Extract years of experience from resume text via LLM.

    Args:
        resume_text: Full resume text.
        model: Ollama model name (default: OLLAMA_MODEL from config).

    Returns:
        Integer years of experience.

    Raises:
        ValueError: If response cannot be parsed or field is missing.
    """
    prompt = _build_years_extraction_prompt(resume_text)
    raw = _call_ollama(prompt, model, temperature=GENERATION_TEMPERATURE, max_tokens=RESUME_EXTRACT_MAX_TOKENS)
    logger.debug("Years extraction response: %s", raw)

    # Parse response: expecting "YearsExperience: <int>"
    for line in raw.strip().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        if key.strip().lower().replace(" ", "") == "yearsexperience":
            try:
                return int(value.strip())
            except ValueError:
                raise ValueError(f"Cannot parse YearsExperience value: {value!r}")

    raise ValueError(f"YearsExperience field not found in response: {raw!r}")


def _constrain_years_required(bracket: str, years_experience: int) -> str:
    """Constrain a years_required bracket to not exceed user's experience.

    If the maximum of the bracket is <= years_experience, returns bracket unchanged.
    If the maximum exceeds years_experience, returns a new range ending at or below years_experience.

    Args:
        bracket: Years bracket string (e.g., "4-8" or "5").
        years_experience: User's total years of experience (integer).

    Returns:
        Constrained years_required string.
    """
    # Parse the bracket to extract max required
    if "-" in bracket:
        parts = bracket.split("-")
        try:
            max_required = max(int(p.strip()) for p in parts if p.strip())
        except ValueError:
            return bracket  # Unparseable, return as-is
    else:
        try:
            max_required = int(bracket.strip())
        except ValueError:
            return bracket

    # If max required <= experience, no constraint needed
    if max_required <= years_experience:
        return bracket

    # Constrain: create range ending at or below years_experience
    # Use a reasonable minimum (e.g., years_experience - 2, but at least 0)
    min_new = max(0, years_experience - 2)
    return f"{min_new}-{years_experience}"


def _generate_deterministic_fields(
    seniority: str,
    domain: str,
    years_experience: int,
) -> dict:
    """Generate deterministic fields (title, seniority, domain, years_required).

    Args:
        seniority: Seniority level (e.g., "Senior").
        domain: Domain (e.g., "backend").
        years_experience: User's years of experience (integer).

    Returns:
        Dict with keys: title, seniority, domain, years_required.
    """
    domain_role = {
        "backend": "Backend Engineer",
        "frontend": "Frontend Engineer",
        "fullstack": "Full Stack Engineer",
        "data": "Data Engineer",
    }
    role = domain_role.get(domain, "Engineer")
    _title_seniority_prefix = {
        "Junior": "Junior",
        "Mid": "Mid",
        "Senior": "Senior",
        "Staff": "Staff",
    }
    title = f"{_title_seniority_prefix.get(seniority, seniority)} {role}"

    years_bracket = {
        "Junior": "0-2",
        "Mid": "2-5",
        "Senior": "4-8",
        "Staff": "6-10",
    }
    bracket = years_bracket.get(seniority, "0-5")
    years_required = _constrain_years_required(bracket, years_experience)

    return {
        "title": title,
        "seniority": seniority,
        "domain": domain,
        "years_required": years_required,
    }


def _build_skills_prompt(resume_text: str, seniority: str, domain: str) -> str:
    """Build prompt to generate primary and secondary skills.

    Args:
        resume_text: Full resume text.
        seniority: Job seniority level.
        domain: Job domain.

    Returns:
        Prompt string requesting skills output.
    """
    return f"""Resume: {resume_text}

Generate skills for a {seniority} {domain} engineer job that matches this resume.

Output ONLY these fields, one per line:
PrimarySkills: [skill1, skill2, skill3] (2-4 items, comma-separated)
SecondarySkills: [skill4, skill5] (1-3 items, comma-separated)

Example:
PrimarySkills: Python, PostgreSQL, Docker
SecondarySkills: Redis, Kubernetes

Do not add explanation or extra text."""


def _generate_skills(
    resume_info: dict, model: str = OLLAMA_MODEL
) -> tuple[list[str], list[str]]:
    """Generate primary and secondary skills via LLM.

    Args:
        resume_info: Resume information dict (ResumeInfo TypedDict).
        model: Ollama model name (default: OLLAMA_MODEL from config).

    Returns:
        Tuple of (primary_skills, secondary_skills).

    Raises:
        ValueError: If response cannot be parsed.
    """
    prompt = _build_skills_prompt(
        resume_info["resume_text"],
        resume_info["seniority"],
        resume_info["domain"],
    )
    raw = _call_ollama(
        prompt, model, temperature=GENERATION_TEMPERATURE, max_tokens=SKILLS_MAX_TOKENS
    )
    logger.debug("Skills generation response: %s", raw)

    # Parse using existing parse_skeleton_response
    parsed = parse_skeleton_response(raw)
    primary = parsed.get("primary_skills", [])
    secondary = parsed.get("secondary_skills", [])

    if not primary:
        raise ValueError("PrimarySkills field missing or empty in response")

    return primary, secondary


def _build_responsibility_prompt(
    resume_text: str,
    seniority: str,
    domain: str,
    primary_skills: list[str],
    already_generated: list[str],
) -> str:
    """Build prompt to generate a single responsibility.

    Args:
        resume_text: Full resume text.
        seniority: Job seniority level.
        domain: Job domain.
        primary_skills: Primary skills list for context.
        already_generated: List of responsibilities already generated (do not repeat).

    Returns:
        Prompt string requesting a single responsibility.
    """
    prompt = f"""Resume: {resume_text}

Generate ONE responsibility for a {seniority} {domain} engineer job.
The responsibility must be a single sentence with at least 10 words.
It must align with the resume and use skills from: {", ".join(primary_skills)}.
Write each responsibility as a distinct, non-repetitive sentence."""

    if already_generated:
        prompt += "\n\nAlready written (do not repeat or rephrase):\n"
        for resp in already_generated:
            prompt += f"- {resp}\n"

    prompt += "\nOutput ONLY the responsibility sentence (no label, no extra text)."

    return prompt


def _generate_single_responsibility(
    resume_info: dict,
    primary_skills: list[str],
    model: str = OLLAMA_MODEL,
    already_generated: list[str] = None,
) -> str:
    """Generate a single responsibility via LLM.

    Args:
        resume_info: Resume information dict (ResumeInfo TypedDict).
        primary_skills: Primary skills list for context.
        model: Ollama model name (default: OLLAMA_MODEL from config).
        already_generated: List of responsibilities already generated (default: []).

    Returns:
        Single responsibility sentence (stripped).

    Raises:
        ValueError: If response is empty or has fewer than 5 words.
    """
    if already_generated is None:
        already_generated = []

    prompt = _build_responsibility_prompt(
        resume_info["resume_text"],
        resume_info["seniority"],
        resume_info["domain"],
        primary_skills,
        already_generated,
    )
    raw = _call_ollama(
        prompt,
        model,
        temperature=GENERATION_TEMPERATURE,
        max_tokens=RESPONSIBILITY_MAX_TOKENS,
    )
    logger.debug("Responsibility generation response: %s", raw)

    resp = raw.strip()
    if not resp:
        raise ValueError("Empty responsibility response")

    word_count = len(resp.split())
    if word_count < 5:
        raise ValueError(f"Responsibility has {word_count} words, need ≥5")

    return resp


def generate_job_skeleton(
    resume_info: dict, model: str = OLLAMA_MODEL
) -> JobSkeleton:
    """
    Generate a job skeleton matching the given resume info.

    Uses a per-field strategy:
    1. Extract years_experience from resume text via LLM
    2. Assemble deterministic fields (title, seniority, domain, years_required)
       directly from resume_info with years_required constrained by experience
    3. Generate skills via LLM
    4. Generate responsibilities one at a time with anti-repetition context

    This is Step 1 of the synthetic positives pipeline.

    Args:
        resume_info: ResumeInfo dict with seniority, years_experience,
                     primary_skills, domain, and resume_text.
        model: Ollama model name (default: OLLAMA_MODEL from config).

    Returns:
        JobSkeleton dict with keys: title, seniority, years_required,
        domain, primary_skills, secondary_skills, responsibilities.

    Raises:
        ValueError: If generation fails (e.g., <3 responsibilities).
        ollama.RequestError: If Ollama is not reachable.
        ollama.ResponseError: If the model returns an error response.
    """
    # Step 1: Extract years from resume text
    years_experience = _extract_years_experience(resume_info["resume_text"], model)
    logger.info("Extracted years of experience: %d", years_experience)

    # Step 2: Assemble deterministic fields
    det = _generate_deterministic_fields(
        resume_info["seniority"], resume_info["domain"], years_experience
    )
    logger.info("Generated deterministic fields: title=%s", det["title"])

    # Step 3: Generate skills
    primary_skills, secondary_skills = _generate_skills(resume_info, model)
    logger.info("Generated skills: primary=%s, secondary=%s", primary_skills, secondary_skills)

    # Step 4: Generate responsibilities one at a time
    responsibilities: list[str] = []
    max_resp_attempts = TARGET_RESPONSIBILITY_COUNT * 4
    attempts = 0

    while len(responsibilities) < TARGET_RESPONSIBILITY_COUNT and attempts < max_resp_attempts:
        attempts += 1
        try:
            resp = _generate_single_responsibility(
                resume_info, primary_skills, model, responsibilities
            )
            responsibilities.append(resp)
            logger.info(
                "Generated responsibility %d/%d: %s",
                len(responsibilities),
                TARGET_RESPONSIBILITY_COUNT,
                resp[:50] + "...",
            )
        except (ValueError, ollama.RequestError, ollama.ResponseError) as e:
            logger.debug("Responsibility generation attempt %d failed: %s", attempts, e)
            continue

    if len(responsibilities) < 3:
        raise ValueError(
            f"Only generated {len(responsibilities)} responsibilities (need ≥3)"
        )

    logger.info(
        "Generated skeleton — title: %s, seniority: %s, domain: %s",
        det["title"],
        det["seniority"],
        det["domain"],
    )

    return JobSkeleton(
        title=det["title"],
        seniority=det["seniority"],
        years_required=det["years_required"],
        domain=det["domain"],
        primary_skills=primary_skills,
        secondary_skills=secondary_skills,
        responsibilities=responsibilities,
    )


