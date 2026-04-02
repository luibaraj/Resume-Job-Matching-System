"""
Generation pipeline for grounding resume-job matches and explaining fit.

Processes batches of (resume, job_posting) pairs by:
1. Extracting required skills from job postings via LLM
2. Searching resumes for matching text spans
3. Filtering pairs with zero validated matches
4. Generating brief fit explanations using only grounded pairs
5. Logging results and flagging hallucinations

All LLM calls use a locally hosted LLaMA 3.2 3B model via Ollama.
All output spans are exact text excerpts, validated via substring search.
"""

import logging
import re
from typing import TypedDict

import ollama

from config import (
    CORPUS_LIMITATION_MESSAGE,
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    GENERATION_TOP_P,
    MAX_BATCH_SIZE,
    OLLAMA_MODEL,
    PROMPT_MAX_CHARS,
)

logger = logging.getLogger(__name__)

# Compiled regex for whitespace normalization
_WHITESPACE_RE = re.compile(r'\s+')


class RequirementMatch(TypedDict):
    """One grounded pair: a requirement and the resume text that matches it."""
    requirement: str   # exact span from job posting
    resume_match: str  # exact span from resume


class PairResult(TypedDict):
    """Output for one (resume, job_posting) pair."""
    explanation: str
    validated_pairs: list[RequirementMatch]
    num_validated_pairs: int
    hallucination_count: int
    flagged_for_review: bool


# ============================================================================
# Private helpers: no LLM dependency
# ============================================================================


def _span_exists_in_text(span: str, text: str) -> bool:
    """Check if span exists as a substring in text (case-sensitive).

    Uses regex to handle spans containing metacharacters like C++, .NET, etc.
    """
    if not span:
        return True  # Empty span matches (documents intended behavior)
    return re.search(re.escape(span), text) is not None


def _normalize_whitespace(text: str) -> str:
    """Collapse all whitespace to single spaces and strip.

    Converts tabs, newlines, and multiple spaces to single spaces.
    """
    return _WHITESPACE_RE.sub(' ', text).strip()


def _parse_requirements(raw_response: str) -> list[str]:
    """Parse LLM response into a list of requirement spans.

    Strips numbering (1., 2., -, etc.) and blank lines.
    """
    if not raw_response.strip():
        return []

    spans = []
    for line in raw_response.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Strip common numbering: "1.", "1)", "-", "*"
        line = re.sub(r'^[\d]+[.)]\s*', '', line)
        line = re.sub(r'^[-*]\s*', '', line)
        line = line.strip()
        if line:
            spans.append(line)
    return spans


def _parse_resume_match(raw_response: str) -> str | None:
    """Parse LLM response for a single resume span.

    Returns None if response is "NOT FOUND" (exact, case-insensitive).
    Otherwise returns trimmed span.
    """
    trimmed = raw_response.strip()
    if not trimmed or trimmed.upper() == "NOT FOUND":
        return None
    return trimmed


# ============================================================================
# Prompt builders
# ============================================================================


def _build_requirements_prompt(job_posting: str) -> str:
    """Build prompt for extracting required skills from job posting.

    Validates job_posting length against PROMPT_MAX_CHARS to prevent token overflow.
    """
    if len(job_posting) > PROMPT_MAX_CHARS:
        logger.warning(
            "Job posting length %d exceeds PROMPT_MAX_CHARS %d; may cause truncation",
            len(job_posting),
            PROMPT_MAX_CHARS,
        )

    return f"""You are an expert at identifying key job requirements from posting text.
Your task: Extract the top 3-5 required skills or qualifications from this job posting.

CRITICAL: Copy the text EXACTLY as it appears in the posting. Do not paraphrase or add details.
Output as a numbered list with one item per line.

Job posting:
{job_posting}

Required skills (copy exact text, one per line):"""


def _build_resume_match_prompt(resume: str, requirement: str) -> str:
    """Build prompt for finding a resume span that matches a requirement.

    Validates resume length against PROMPT_MAX_CHARS to prevent token overflow.
    """
    if len(resume) > PROMPT_MAX_CHARS:
        logger.warning(
            "Resume length %d exceeds PROMPT_MAX_CHARS %d; may cause truncation",
            len(resume),
            PROMPT_MAX_CHARS,
        )

    return f"""You are an expert at finding evidence in resumes.
Your task: Find the shortest exact phrase in this resume that demonstrates the requirement "{requirement}".

CRITICAL: Copy the exact text from the resume. If you cannot find a clear match, reply with "NOT FOUND" and nothing else.

Resume:
{resume}

Evidence from resume (copy exact text or "NOT FOUND"):"""


def _build_explanation_prompt(validated_pairs: list[RequirementMatch]) -> str:
    """Build prompt for generating a fit explanation."""
    pairs_text = '\n'.join(
        f"Requirement: {pair['requirement']}\nResume evidence: {pair['resume_match']}"
        for pair in validated_pairs
    )
    return f"""You are a recruiter writing a brief, evidence-based fit explanation.
Your task: Write 1-2 sentences explaining how this candidate matches the role.

Use ONLY the provided requirement-evidence pairs. Do not speculate or add information beyond what is listed.

Matched pairs:
{pairs_text}

Fit explanation (1-2 sentences):"""


# ============================================================================
# LLM invocation helper
# ============================================================================


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Call Ollama chat endpoint and return response content."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": GENERATION_TEMPERATURE,
            "top_p": GENERATION_TOP_P,
            "num_predict": GENERATION_MAX_TOKENS,
        },
    )
    return response["message"]["content"]


# ============================================================================
# Public pipeline functions
# ============================================================================


def extract_requirements(
    job_posting: str,
    model: str = OLLAMA_MODEL,
) -> list[str]:
    """Extract required skills from job posting via LLM.

    Validates that each extracted span exists in the job posting text.
    Discards hallucinated spans.

    Args:
        job_posting: Job posting text to extract from.
        model: Ollama model name.

    Returns:
        List of validated exact-text spans from job posting.
    """
    prompt = _build_requirements_prompt(job_posting)
    response = _call_ollama(prompt, model)
    candidate_spans = _parse_requirements(response)

    # Validate each span
    validated = [span for span in candidate_spans if _span_exists_in_text(span, job_posting)]

    logger.debug(f"Extracted {len(candidate_spans)} requirements, validated {len(validated)}")
    return validated


def find_resume_matches(
    resume: str,
    requirements: list[str],
    model: str = OLLAMA_MODEL,
) -> tuple[list[RequirementMatch], int]:
    """Find resume spans matching each requirement.

    Normalizes whitespace and validates each span exists in resume.
    Counts hallucinations (LLM returned non-None but span not found).

    Args:
        resume: Resume text to search.
        requirements: List of requirement spans to match.
        model: Ollama model name.

    Returns:
        Tuple of (validated_pairs, hallucination_count).
    """
    validated_pairs = []
    hallucination_count = 0

    for requirement in requirements:
        prompt = _build_resume_match_prompt(resume, requirement)
        response = _call_ollama(prompt, model)
        resume_match = _parse_resume_match(response)

        if resume_match is None:
            # "NOT FOUND" — not counted as hallucination
            continue

        # Normalize and validate
        normalized_match = _normalize_whitespace(resume_match)
        if _span_exists_in_text(normalized_match, resume):
            # Valid match
            validated_pairs.append({
                "requirement": requirement,
                "resume_match": normalized_match,
            })
        else:
            # Hallucination: LLM returned a span that isn't in the resume
            hallucination_count += 1
            logger.debug(f"Hallucination detected: '{normalized_match}' not in resume")

    logger.debug(f"Found {len(validated_pairs)} matches, {hallucination_count} hallucinations")
    return validated_pairs, hallucination_count


def filter_pairs(
    pairs: list[tuple[str, str]],
    model: str = OLLAMA_MODEL,
) -> tuple[list[tuple[str, str, list[RequirementMatch], int]], str | None]:
    """Process batch and filter out pairs with zero validated matches.

    Args:
        pairs: List of (resume, job_posting) tuples.
        model: Ollama model name.

    Returns:
        Tuple of (retained_pairs, corpus_message).
        If all pairs scrapped: ([], CORPUS_LIMITATION_MESSAGE).
        Otherwise: (retained, None).
        Each retained element is (resume, job_posting, validated_pairs, hallucination_count).
    """
    retained = []

    for resume, job_posting in pairs:
        # Extract requirements from job posting
        requirements = extract_requirements(job_posting, model)

        # Find resume matches
        validated_pairs, hallucination_count = find_resume_matches(resume, requirements, model)

        # Keep only if we have at least one validated match
        if len(validated_pairs) > 0:
            retained.append((resume, job_posting, validated_pairs, hallucination_count))

    # Check if all pairs were scrapped
    if len(retained) == 0:
        logger.warning("All pairs scrapped — corpus limitation detected")
        return [], CORPUS_LIMITATION_MESSAGE

    logger.debug(f"Retained {len(retained)} out of {len(pairs)} pairs")
    return retained, None


def generate_explanation(
    validated_pairs: list[RequirementMatch],
    model: str = OLLAMA_MODEL,
) -> str:
    """Generate brief fit explanation from validated pairs.

    Args:
        validated_pairs: List of (requirement, resume_match) pairs.
        model: Ollama model name.

    Returns:
        Explanation string (1-2 sentences).
    """
    prompt = _build_explanation_prompt(validated_pairs)
    response = _call_ollama(prompt, model)
    return response.strip()


def log_result(result: PairResult) -> None:
    """Log result with hallucination flag if needed.

    Args:
        result: PairResult to log.
    """
    logger.info(
        f"Explanation: {result['explanation']} | "
        f"Validated pairs: {result['num_validated_pairs']} | "
        f"Hallucinations: {result['hallucination_count']}"
    )
    if result["flagged_for_review"]:
        logger.warning(
            f"Pair flagged for manual review due to {result['hallucination_count']} hallucination(s)"
        )


def run_generation_pipeline(
    pairs: list[tuple[str, str]],
    model: str = OLLAMA_MODEL,
) -> list[PairResult] | str:
    """Run full generation pipeline on a batch of (resume, job_posting) pairs.

    Processes the batch, filters out pairs with no validated matches,
    generates explanations, and logs results.

    Args:
        pairs: List of (resume, job_posting) tuples (max 10).
        model: Ollama model name.

    Returns:
        List of PairResult dicts, or CORPUS_LIMITATION_MESSAGE string if all filtered.

    Raises:
        ValueError: If len(pairs) > MAX_BATCH_SIZE.
    """
    if len(pairs) > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size {len(pairs)} exceeds max {MAX_BATCH_SIZE}")

    # Filter pairs
    retained, corpus_message = filter_pairs(pairs, model)

    # If all filtered, return corpus message
    if corpus_message is not None:
        return corpus_message

    # Generate explanations and build results
    results = []
    for resume, job_posting, validated_pairs, hallucination_count in retained:
        explanation = generate_explanation(validated_pairs, model)
        result: PairResult = {
            "explanation": explanation,
            "validated_pairs": validated_pairs,
            "num_validated_pairs": len(validated_pairs),
            "hallucination_count": hallucination_count,
            "flagged_for_review": hallucination_count > 0,
        }
        log_result(result)
        results.append(result)

    return results
