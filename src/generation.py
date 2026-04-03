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
import time
from typing import TypedDict

import ollama

from src.config import (
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


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL, max_retries: int = 1) -> str:
    """Call Ollama chat endpoint and return response content.

    Retries once on transient RequestError with exponential backoff.
    Sets a timeout to prevent indefinite hangs.

    Args:
        prompt: The prompt to send to Ollama.
        model: Ollama model name.
        max_retries: Number of retry attempts on transient errors.

    Returns:
        Response content string.

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
                    "num_predict": GENERATION_MAX_TOKENS,
                },
            )
            return response["message"]["content"]
        except ollama.RequestError as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "_call_ollama failed after %d attempts: %s",
                    max_retries,
                    e,
                )
                raise
            # Exponential backoff: 0.5s, then 1.0s
            delay = 0.5 * (2 ** (attempt - 1))
            logger.warning(
                "_call_ollama attempt %d/%d failed: %s. Retrying in %.1fs...",
                attempt,
                max_retries,
                e,
                delay,
            )
            time.sleep(delay)


# ============================================================================
# Public pipeline functions
# ============================================================================


def extract_requirements(
    job_posting: str,
    model: str = OLLAMA_MODEL,
    run_id: str | None = None,
) -> list[str]:
    """Extract required skills from job posting via LLM.

    Validates that each extracted span exists in the job posting text.
    Discards hallucinated spans.

    Args:
        job_posting: Job posting text to extract from.
        model: Ollama model name.
        run_id: Optional trace ID for request tracing.

    Returns:
        List of validated exact-text spans from job posting.
    """
    start_time = time.monotonic()
    prompt = _build_requirements_prompt(job_posting)
    response = _call_ollama(prompt, model)
    candidate_spans = _parse_requirements(response)

    # Validate each span
    validated = [span for span in candidate_spans if _span_exists_in_text(span, job_posting)]

    elapsed_time = time.monotonic() - start_time
    if run_id:
        logger.info(
            "extract_requirements (run_id=%s) completed in %.3fs",
            run_id,
            elapsed_time,
        )
    else:
        logger.info("extract_requirements completed in %.3fs", elapsed_time)
    logger.debug("Extracted %d requirements, validated %d", len(candidate_spans), len(validated))
    return validated


def find_resume_matches(
    resume: str,
    requirements: list[str],
    model: str = OLLAMA_MODEL,
    run_id: str | None = None,
) -> tuple[list[RequirementMatch], int]:
    """Find resume spans matching each requirement.

    Normalizes whitespace and validates each span exists in resume.
    Counts hallucinations (LLM returned non-None but span not found).

    Args:
        resume: Resume text to search.
        requirements: List of requirement spans to match.
        model: Ollama model name.
        run_id: Optional trace ID for request tracing.

    Returns:
        Tuple of (validated_pairs, hallucination_count).
    """
    start_time = time.monotonic()
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
            logger.debug("Hallucination detected: '%s' not in resume", normalized_match)

    elapsed_time = time.monotonic() - start_time
    if run_id:
        logger.info(
            "find_resume_matches (run_id=%s) completed in %.3fs",
            run_id,
            elapsed_time,
        )
    else:
        logger.info("find_resume_matches completed in %.3fs", elapsed_time)
    logger.debug("Found %d matches, %d hallucinations", len(validated_pairs), hallucination_count)
    return validated_pairs, hallucination_count


def filter_pairs(
    pairs: list[tuple[str, str]],
    model: str = OLLAMA_MODEL,
    run_id: str | None = None,
) -> tuple[list[tuple[str, str, list[RequirementMatch], int]], str | None]:
    """Process batch and retain all pairs (even with zero validated matches).

    Args:
        pairs: List of (resume, job_posting) tuples.
        model: Ollama model name.
        run_id: Optional trace ID for request tracing.

    Returns:
        Tuple of (all_pairs, corpus_message).
        If any pair has zero validated matches: (all_pairs, CORPUS_LIMITATION_MESSAGE).
        Otherwise: (all_pairs, None).
        Each element is (resume, job_posting, validated_pairs, hallucination_count).
    """
    all_pairs = []
    has_zero_matches = False

    for resume, job_posting in pairs:
        # Extract requirements from job posting
        requirements = extract_requirements(job_posting, model, run_id=run_id)

        # Find resume matches
        validated_pairs, hallucination_count = find_resume_matches(resume, requirements, model, run_id=run_id)

        # Keep all pairs, track if any have zero validated matches
        all_pairs.append((resume, job_posting, validated_pairs, hallucination_count))
        if not validated_pairs:
            has_zero_matches = True

    corpus_message = CORPUS_LIMITATION_MESSAGE if has_zero_matches else None
    if has_zero_matches:
        logger.warning("Some pairs have zero validated matches — corpus limitation warning")
    else:
        logger.debug("All %d pairs have validated matches", len(all_pairs))
    return all_pairs, corpus_message


def generate_explanation(
    validated_pairs: list[RequirementMatch],
    model: str = OLLAMA_MODEL,
    run_id: str | None = None,
) -> str:
    """Generate brief fit explanation from validated pairs.

    Args:
        validated_pairs: List of (requirement, resume_match) pairs.
        model: Ollama model name.
        run_id: Optional trace ID for request tracing.

    Returns:
        Explanation string (1-2 sentences).
    """
    start_time = time.monotonic()
    prompt = _build_explanation_prompt(validated_pairs)
    response = _call_ollama(prompt, model)
    explanation = response.strip()

    elapsed_time = time.monotonic() - start_time
    if run_id:
        logger.info(
            "generate_explanation (run_id=%s) completed in %.3fs",
            run_id,
            elapsed_time,
        )
    else:
        logger.info("generate_explanation completed in %.3fs", elapsed_time)

    return explanation


def log_result(result: PairResult) -> None:
    """Log result with hallucination flag if needed.

    Args:
        result: PairResult to log.
    """
    logger.info(
        "Explanation: %s | Validated pairs: %d | Hallucinations: %d",
        result['explanation'],
        result['num_validated_pairs'],
        result['hallucination_count'],
    )
    if result["flagged_for_review"]:
        logger.warning(
            "Pair flagged for manual review due to %d hallucination(s)",
            result['hallucination_count'],
        )


def run_generation_pipeline(
    pairs: list[tuple[str, str]],
    model: str = OLLAMA_MODEL,
    run_id: str | None = None,
) -> tuple[list[PairResult], str | None]:
    """Run full generation pipeline on a batch of (resume, job_posting) pairs.

    Processes the batch, generates explanations for all pairs (or None if no matches),
    and logs results.

    Args:
        pairs: List of (resume, job_posting) tuples (max 10).
        model: Ollama model name.
        run_id: Optional trace ID for request tracing.

    Returns:
        Tuple of (list of PairResult dicts, corpus_message or None).
        corpus_message is set if any pair has zero validated matches.

    Raises:
        ValueError: If len(pairs) > MAX_BATCH_SIZE.
    """
    if len(pairs) > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size {len(pairs)} exceeds max {MAX_BATCH_SIZE}")

    # Filter pairs (now keeps all pairs)
    all_pairs, corpus_message = filter_pairs(pairs, model, run_id=run_id)

    # Generate explanations and build results
    results = []
    for _, _, validated_pairs, hallucination_count in all_pairs:
        # Skip explanation generation if no validated matches
        if validated_pairs:
            explanation = generate_explanation(validated_pairs, model, run_id=run_id)
        else:
            explanation = None

        result: PairResult = {
            "explanation": explanation,
            "validated_pairs": validated_pairs,
            "num_validated_pairs": len(validated_pairs),
            "hallucination_count": hallucination_count,
            "flagged_for_review": hallucination_count > 0,
        }
        log_result(result)
        results.append(result)

    return results, corpus_message
