"""
Generation layer: synthesize structured explanations of job-candidate matches.

Pipeline:
1. Chain-of-Note (CoN) relevance filter: quick check job is relevant
2. Generation with Chain-of-Thought: synthesize top-3 similarities with citations
3. Reference-free LLM judge: evaluate quality
4. Parse citations and persist to DB

Follows src/extraction.py patterns: Gemini client init, temperature, response_mime_type, retries.
"""

import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import Optional

import jsonschema
from google import genai

from src.utils import setup_logging


@dataclass
class JobContext:
    """Complete job context from DB, used as input to generation pipeline."""

    job_id: int
    rank: int
    score: float
    title: str
    company: str
    location: Optional[str]
    absolute_url: Optional[str]
    cleaned_description: str
    responsibilities: list[str]
    skills: list[str]
    tools_and_platforms: list[str]
    experience_min_years: Optional[int]


@dataclass
class GenerationResult:
    """Output of generation pipeline for one job."""

    job_id: int
    rank: int
    summary: str
    citations: list[dict]  # [{\"source\": \"resume|job\", \"label\": \"slug\"}, ...]
    evaluation: dict
    passed_eval: bool
    model_id: str


def _build_job_context(row: tuple, deserialize_list_fields: list[str]) -> JobContext:
    """
    Convert DB tuple row to JobContext, deserializing JSON list fields.

    Args:
        row: tuple of (job_id, rank, score, title, company, location, absolute_url,
                       cleaned_description, responsibilities, skills, tools_and_platforms,
                       experience_min_years)
        deserialize_list_fields: list of field names (in order as they appear in row)
                                that should be deserialized from JSON

    Returns:
        JobContext dataclass instance
    """
    (
        job_id,
        rank,
        score,
        title,
        company,
        location,
        absolute_url,
        cleaned_description,
        responsibilities,
        skills,
        tools_and_platforms,
        experience_min_years,
    ) = row

    # Deserialize JSON list fields
    for field_name in deserialize_list_fields:
        if field_name == "responsibilities" and isinstance(responsibilities, str):
            responsibilities = json.loads(responsibilities) if responsibilities else []
        elif field_name == "skills" and isinstance(skills, str):
            skills = json.loads(skills) if skills else []
        elif field_name == "tools_and_platforms" and isinstance(tools_and_platforms, str):
            tools_and_platforms = (
                json.loads(tools_and_platforms) if tools_and_platforms else []
            )

    return JobContext(
        job_id=job_id,
        rank=rank,
        score=score,
        title=title,
        company=company,
        location=location,
        absolute_url=absolute_url,
        cleaned_description=cleaned_description,
        responsibilities=responsibilities,
        skills=skills,
        tools_and_platforms=tools_and_platforms,
        experience_min_years=experience_min_years,
    )


def _validate_summary_structure(text: str) -> list[str]:
    """
    Regex validation of summary structure.

    Checks:
    - Presence of <thinking>...</thinking> block
    - Exactly 3 **Similarity N** sections
    - At least one [R:...] and one [J:...] citation

    Args:
        text: raw summary text from generation

    Returns:
        List of validation issues (empty list = pass)
    """
    issues = []

    # Check for <thinking> block
    if not re.search(r"<thinking>.*?</thinking>", text, re.DOTALL):
        issues.append("Missing <thinking> block")

    # Check for exactly 3 **Similarity N** sections
    similarity_matches = re.findall(r"\*\*Similarity\s+\d+", text, re.IGNORECASE)
    if len(similarity_matches) != 3:
        issues.append(
            f"Expected exactly 3 **Similarity** sections, found {len(similarity_matches)}"
        )

    # Check for at least one [R:...] citation
    if not re.search(r"\[R:[a-zA-Z0-9_-]+\]", text):
        issues.append("Missing [R:...] citation (resume)")

    # Check for at least one [J:...] citation
    if not re.search(r"\[J:[a-zA-Z0-9_-]+\]", text):
        issues.append("Missing [J:...] citation (job)")

    return issues


def _parse_citations(text: str) -> list[dict]:
    """
    Extract all [R:label] and [J:label] citations from summary text.

    Uses regex: \\[(R|J):([a-zA-Z0-9_-]+)\\]
    Deduplicates by (source, label) pairs.

    Args:
        text: summary text with citations

    Returns:
        List of dicts: [{\"source\": \"resume|job\", \"label\": \"slug\"}, ...]
    """
    # Extract all citations: [R:label] or [J:label]
    citation_pattern = r"\[(R|J):([a-zA-Z0-9_-]+)\]"
    matches = re.findall(citation_pattern, text)

    # Deduplicate: track (source, label) pairs
    seen = set()
    citations = []

    for source_letter, label in matches:
        source = "resume" if source_letter == "R" else "job"
        key = (source, label)

        if key not in seen:
            seen.add(key)
            citations.append({"source": source, "label": label})

    return citations


# JSON Schemas for Gemini response validation

CON_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance_verdict": {
            "type": "string",
            "enum": ["relevant", "irrelevant", "contradictory"],
            "description": "Relevance classification"
        },
        "relevance_reasoning": {
            "type": "string",
            "description": "One-sentence explanation of the verdict"
        },
        "contradictions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Hard conflicts (e.g., experience level mismatch)"
        },
        "strong_alignments": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key overlaps between resume and job"
        }
    },
    "required": ["relevance_verdict", "relevance_reasoning", "contradictions", "strong_alignments"]
}

EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "justification": {"type": "string"},
                "flags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["score", "justification", "flags"]
        },
        "completeness": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "justification": {"type": "string"}
            },
            "required": ["score", "justification"]
        },
        "structural_adherence": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "justification": {"type": "string"},
                "issues": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["score", "justification", "issues"]
        },
        "overall_pass": {"type": "boolean"}
    },
    "required": ["faithfulness", "completeness", "structural_adherence", "overall_pass"]
}


def _run_con_filter(
    job: JobContext,
    resume_text: str,
    client,
    model_id: str,
    logger: logging.Logger
) -> dict | None:
    """
    Chain-of-Note relevance filter.

    Returns notes dict on \"relevant\", None on \"irrelevant\"/\"contradictory\"/error.

    Args:
        job: JobContext with job details
        resume_text: candidate resume text
        client: Gemini genai.Client instance
        model_id: Gemini model ID
        logger: logger instance

    Returns:
        dict with relevance_verdict, reasoning, contradictions, strong_alignments; or None
    """
    system_prompt = """You are a career matching assistant. Read the candidate resume and job posting, then produce structured reading notes."""

    extracted_block = f"""Extracted Requirements:
  - Responsibilities: {', '.join(job.responsibilities) if job.responsibilities else 'None'}
  - Skills: {', '.join(job.skills) if job.skills else 'None'}
  - Tools: {', '.join(job.tools_and_platforms) if job.tools_and_platforms else 'None'}
  - Experience: {job.experience_min_years} years minimum{' (inferred)' if job.experience_min_years is None else ''}"""

    user_prompt = f"""CANDIDATE RESUME:
{resume_text}

JOB POSTING: {job.title} | {job.company} | {job.location or 'Remote'}
{job.cleaned_description}

{extracted_block}

TASK: Return ONLY valid JSON with this structure:
{{
  "relevance_verdict": "relevant" | "irrelevant" | "contradictory",
  "relevance_reasoning": "<one sentence>",
  "contradictions": [...],
  "strong_alignments": [...]
}}

Rules:
- "irrelevant": fewer than 2 meaningful overlaps
- "contradictory": hard conflict (e.g., requires 5+ years, candidate has <2)
- "relevant": ≥2 overlaps, no hard contradictions

Return ONLY JSON. relevance_verdict must be exactly: relevant | irrelevant | contradictory."""

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=[
                {"role": "user", "parts": [{"text": system_prompt + "\n\n" + user_prompt}]}
            ],
            config=genai.types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )
        parsed = json.loads(response.text)
        jsonschema.validate(parsed, CON_SCHEMA)
        return parsed
    except json.JSONDecodeError as e:
        logger.warning("Job %d (CoN filter): invalid JSON: %s", job.job_id, e)
        return None
    except jsonschema.ValidationError as e:
        logger.warning("Job %d (CoN filter): schema validation failed: %s", job.job_id, e.message)
        return None
    except Exception as e:
        logger.warning("Job %d (CoN filter): API error: %s", job.job_id, e)
        return None


def _run_generation(
    job: JobContext,
    resume_text: str,
    con_notes: dict,
    client,
    model_id: str,
    logger: logging.Logger,
    max_retries: int = 2
) -> str | None:
    """
    Main Chain-of-Thought + citation generation.

    Returns raw summary text or None on failure after retries.

    Args:
        job: JobContext with job details
        resume_text: candidate resume text
        con_notes: Chain-of-Note output (contains strong_alignments)
        client: Gemini genai.Client instance
        model_id: Gemini model ID
        logger: logger instance
        max_retries: max retries on structural failure

    Returns:
        str (summary text) or None
    """
    system_prompt = """You are an expert career coach writing a personalized job match analysis. Cite sources inline."""

    extracted_block = f"""Extracted Requirements:
  - Responsibilities: {', '.join(job.responsibilities) if job.responsibilities else 'None'}
  - Skills: {', '.join(job.skills) if job.skills else 'None'}
  - Tools: {', '.join(job.tools_and_platforms) if job.tools_and_platforms else 'None'}
  - Experience: {job.experience_min_years} years minimum"""

    guidance = (
        "\n".join(con_notes.get("strong_alignments", []))
        if con_notes.get("strong_alignments")
        else "No specific guidance from relevance filter"
    )

    few_shot_ml = """FEW-SHOT EXAMPLE 1: ML Engineering Domain
RESUME: Candidate built a PyTorch autoencoder [R:autoencoder] for anomaly detection and led a 3-person team [R:team-lead] on iterative improvements.
JOB: Acme seeks a Senior ML Engineer to design scalable ML infrastructure [J:ml-infra] and mentor junior engineers [J:mentorship].

<thinking>PyTorch and autoencoder directly align with scalable ML infrastructure needs. Leading a team aligns with mentorship responsibilities. Both roles involve iterative ML development cycles.</thinking>

**Match Analysis: Senior ML Engineer at Acme**

**Similarity 1 — Deep Learning Engineering**
Candidate built a PyTorch autoencoder [R:autoencoder] for production use, directly matching the role's need for scalable ML infrastructure [J:ml-infra]. Both involve designing efficient neural network systems.

**Similarity 2 — Technical Leadership**
The candidate's experience leading a 3-person team [R:team-lead] maps directly to the role's mentorship needs [J:mentorship]. Both require guiding junior engineers through complex ML problems.

**Similarity 3 — Iterative ML Development**
Both the candidate's background and the role require rapid experimentation cycles, model evaluation, and continuous improvement of ML systems."""

    few_shot_ds = """FEW-SHOT EXAMPLE 2: Data Science / NLP Domain
RESUME: Candidate conducted SQL-based data analysis [R:sql-analysis] and built a sentiment analysis model [R:nlp-model] using NLTK and scikit-learn.
JOB: TechCorp seeks a Data Scientist to perform statistical analysis [J:stat-analysis] and develop NLP solutions [J:nlp-solutions].

<thinking>SQL experience aligns with statistical analysis work. NLP background directly matches the role's need for NLP solutions. Both use Python-based ML tools.</thinking>

**Match Analysis: Data Scientist at TechCorp**

**Similarity 1 — Data Manipulation & Analysis**
The candidate's SQL-based data analysis background [R:sql-analysis] directly supports the role's statistical analysis requirements [J:stat-analysis]. Both involve extracting insights from structured data.

**Similarity 2 — NLP Development**
Building a sentiment analysis model [R:nlp-model] with NLTK and scikit-learn is directly applicable to the role's NLP solution development [J:nlp-solutions]. Both require understanding of text processing pipelines.

**Similarity 3 — Python ML Stack**
The candidate's experience with scikit-learn and Python aligns with the role's tech stack [J:nlp-solutions]. Both emphasize open-source Python tools for rapid prototyping."""

    user_prompt = f"""CITATION FORMAT:
- [R:slug] for claims from resume
- [J:slug] for claims from job description

{few_shot_ml}

{few_shot_ds}

NOW ANALYZE THIS JOB:

CANDIDATE RESUME:
{resume_text}

JOB: {job.title} | {job.company}
{job.cleaned_description}

{extracted_block}

Reading Notes (guidance):
{guidance}

IMPORTANT: Your response MUST include:
1. A <thinking> block with analysis
2. Exactly 3 **Similarity** sections (numbered 1-3)
3. At least one [R:...] citation and at least one [J:...] citation per section

Return ONLY the summary text. Do not include JSON wrapper."""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=[
                    {"role": "user", "parts": [{"text": system_prompt + "\n\n" + user_prompt}]}
                ],
                config=genai.types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1024,
                ),
            )
            summary_text = response.text

            # Validate structure
            issues = _validate_summary_structure(summary_text)
            if not issues:
                return summary_text

            # Log failure and retry if not last attempt
            if attempt < max_retries - 1:
                logger.warning(
                    "Job %d: generation structural validation failed (attempt %d/%d): %s",
                    job.job_id, attempt + 1, max_retries, "; ".join(issues)
                )
            else:
                logger.error(
                    "Job %d: generation structural validation failed after %d attempts: %s",
                    job.job_id, max_retries, "; ".join(issues)
                )
                return None
        except Exception as e:
            logger.warning("Job %d: generation API error (attempt %d/%d): %s",
                          job.job_id, attempt + 1, max_retries, e)
            if attempt == max_retries - 1:
                return None

    return None


def _run_evaluator(
    job: JobContext,
    resume_text: str,
    summary: str,
    client,
    model_id: str,
    logger: logging.Logger
) -> dict | None:
    """
    Reference-free LLM judge for summary quality.

    Returns evaluation dict or None on failure.

    Args:
        job: JobContext with job details
        resume_text: candidate resume text
        summary: generated summary text
        client: Gemini genai.Client instance
        model_id: Gemini model ID
        logger: logger instance

    Returns:
        dict with evaluation scores and overall_pass, or None
    """
    system_prompt = """You are a quality evaluator for AI-generated job match summaries."""

    extracted_block = f"""Extracted Requirements:
  - Responsibilities: {', '.join(job.responsibilities) if job.responsibilities else 'None'}
  - Skills: {', '.join(job.skills) if job.skills else 'None'}
  - Tools: {', '.join(job.tools_and_platforms) if job.tools_and_platforms else 'None'}"""

    user_prompt = f"""CANDIDATE RESUME:
{resume_text}

JOB: {job.title} | {job.company}
{job.cleaned_description}

{extracted_block}

GENERATED SUMMARY:
{summary}

Return ONLY valid JSON with this structure:
{{
  "faithfulness": {{"score": <0-10>, "justification": "...", "flags": [...]}},
  "completeness": {{"score": <0-10>, "justification": "..."}},
  "structural_adherence": {{"score": <0-10>, "justification": "...", "issues": [...]}},
  "overall_pass": <true|false>
}}

Scoring:
- faithfulness: deduct 2 per fabricated claim, max 10
- completeness: check for 3 distinct similarities, ≥1 R + J citation each
- structural_adherence: check for <thinking> block, 3 sections, correct citation format
- overall_pass: true if ALL scores ≥ 7

Return ONLY JSON. overall_pass must be a boolean."""

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=[
                {"role": "user", "parts": [{"text": system_prompt + "\n\n" + user_prompt}]}
            ],
            config=genai.types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )
        parsed = json.loads(response.text)
        jsonschema.validate(parsed, EVAL_SCHEMA)
        return parsed
    except json.JSONDecodeError as e:
        logger.warning("Job %d (evaluator): invalid JSON: %s", job.job_id, e)
        return None
    except jsonschema.ValidationError as e:
        logger.warning("Job %d (evaluator): schema validation failed: %s", job.job_id, e.message)
        return None
    except Exception as e:
        logger.warning("Job %d (evaluator): API error: %s", job.job_id, e)
        return None


def _process_single_job(
    job: JobContext,
    resume_text: str,
    client,
    model_id: str,
    max_retries: int,
    logger: logging.Logger
) -> GenerationResult | None:
    """
    Orchestrate generation pipeline for a single job.

    Runs: CoN filter → (if passes) generation with retries → evaluation → parse citations.
    Returns GenerationResult if successful, None if CoN drops job or generation fails.

    Error handling:
    - CoN filter: API/JSON/schema error or irrelevant/contradictory → None (drop job)
    - Generation: API error or structural fail → Retry up to max_retries, then None
    - Evaluator: Any failure → passed_eval=False, empty {} — still write summary

    Args:
        job: JobContext with job details
        resume_text: candidate resume text
        client: Gemini genai.Client instance
        model_id: Gemini model ID
        max_retries: max retries for generation
        logger: logger instance

    Returns:
        GenerationResult on success, None if job dropped or generation failed
    """
    # Stage 1: CoN filter
    con_notes = _run_con_filter(job, resume_text, client, model_id, logger)
    if con_notes is None:
        logger.info("Job %d: dropped by CoN filter (API/schema error)", job.job_id)
        return None

    verdict = con_notes.get("relevance_verdict")
    if verdict != "relevant":
        logger.info("Job %d: dropped by CoN filter (verdict: %s)", job.job_id, verdict)
        return None

    logger.info("Job %d: passed CoN filter", job.job_id)

    # Stage 2: Generation with retries
    summary_text = _run_generation(job, resume_text, con_notes, client, model_id, logger, max_retries)
    if summary_text is None:
        logger.warning("Job %d: generation failed after %d retries", job.job_id, max_retries)
        return None

    logger.info("Job %d: generation succeeded", job.job_id)

    # Stage 3: Evaluation
    evaluation = _run_evaluator(job, resume_text, summary_text, client, model_id, logger)
    if evaluation is None:
        logger.warning("Job %d: evaluator API/schema error (still writing summary)", job.job_id)
        evaluation = {}
        passed_eval = False
    else:
        passed_eval = evaluation.get("overall_pass", False)
        logger.info("Job %d: evaluation complete (overall_pass=%s)", job.job_id, passed_eval)

    # Stage 4: Parse citations
    citations = _parse_citations(summary_text)

    return GenerationResult(
        job_id=job.job_id,
        rank=job.rank,
        summary=summary_text,
        citations=citations,
        evaluation=evaluation,
        passed_eval=passed_eval,
        model_id=model_id,
    )


def generate_summaries(db, config) -> tuple[int, int]:
    """
    Main generation pipeline entry point.

    Loads resume, initializes Gemini client, fetches reranked jobs, processes each,
    and writes summaries to DB.

    Args:
        db: DatabaseManager instance
        config: Config instance with generation settings

    Returns:
        Tuple of (summaries_written, jobs_dropped_or_failed)
    """
    logger = logging.getLogger("generation")

    # Load resume
    from src.retrieval import load_user_profile
    try:
        resume_text = load_user_profile(config.retrieval_user_profile_path)
        logger.info("Loaded resume from %s", config.retrieval_user_profile_path)
    except FileNotFoundError as e:
        logger.error("Resume file not found: %s", e)
        raise

    # Initialize Gemini client
    client = genai.Client(api_key=config.google_api_key)
    logger.info("Initialized Gemini client with model %s", config.generation_model_id)

    # Fetch reranked jobs with full text
    rows = db.get_reranked_with_full_text(config.generation_top_k)
    logger.info("Fetched %d reranked jobs for generation", len(rows))

    if not rows:
        logger.warning("No reranked jobs found in database")
        return 0, 0

    # Process each job
    results = []
    dropped_count = 0

    for row in rows:
        job_context = _build_job_context(
            row,
            deserialize_list_fields=[
                "responsibilities",
                "skills",
                "tools_and_platforms",
            ],
        )

        result = _process_single_job(
            job_context,
            resume_text,
            client,
            config.generation_model_id,
            config.generation_max_retries,
            logger,
        )

        if result is None:
            dropped_count += 1
        else:
            results.append(result)

    logger.info("Processed %d jobs: %d successful, %d dropped", len(rows), len(results), dropped_count)

    # Serialize results and insert into DB
    if results:
        summaries_data = [
            {
                "job_id": r.job_id,
                "rank": r.rank,
                "summary": r.summary,
                "citations_json": json.dumps(r.citations),
                "evaluation_json": json.dumps(r.evaluation),
                "passed_eval": 1 if r.passed_eval else 0,
                "model_id": r.model_id,
            }
            for r in results
        ]
        db.insert_summaries(summaries_data)
        logger.info("Wrote %d summaries to job_summaries table", len(results))

    return len(results), dropped_count


def main() -> None:
    """CLI entrypoint called by Docker container.

    Reads config, initializes DB, runs generation, updates pipeline_runs.
    Exits with code 0 on success, 1 on failure.
    """
    from datetime import datetime

    from src.config import load_config
    from src.database import DatabaseManager

    logger = setup_logging(name="generation_main")
    run_id = None
    try:
        config = load_config()
        logger.setLevel(config.log_level)

        db = DatabaseManager(config.db_path)
        db.initialize_schema()

        run_date = datetime.utcnow().strftime("%Y-%m-%d")
        run_id = db.create_pipeline_run(run_date, "generation")

        processed, skipped = generate_summaries(db, config)
        db.finish_pipeline_run(run_id, "success", jobs_processed=processed, jobs_skipped=skipped)
        logger.info("Generation step completed: %d summaries written", processed)

    except Exception as e:
        logger.exception("Generation step failed")
        if run_id is not None:
            try:
                from src.config import load_config
                from src.database import DatabaseManager

                config = load_config()
                db = DatabaseManager(config.db_path)
                db.finish_pipeline_run(run_id, "failed", jobs_processed=0, jobs_skipped=0)
            except Exception as cleanup_error:
                logger.exception("Failed to update pipeline_runs on error: %s", cleanup_error)
        exit(1)
