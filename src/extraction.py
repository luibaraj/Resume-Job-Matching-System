import asyncio
import json
import logging
import os

import jsonschema
from google import genai

from src.utils import setup_logging


"""
System prompt for Gemini-based Information Extraction
Aligned with job matching pipeline (data collection → extraction → embeddings → retrieval → reranking)
"""

EXTRACTION_SYSTEM_PROMPT = """\
Extract structured data from the job description below. Fill in the JSON skeleton provided — replace placeholder values with the correct extracted values.

Field rules:
- "job_title": restate the title exactly as given
- "responsibilities": verb+object phrases (e.g., "Design backend APIs")
- "skills": technical and professional competencies; include explicit soft skills
- "tools_and_platforms": languages, frameworks, databases, cloud/SaaS tools
- "education": minimum degree if stated; else "unknown"
- "experience":
  - If stated: min_years=N, is_inferred=false
  - If not stated: min_years=-1, is_inferred=true

Use empty lists [] when a field has no content. Return ONLY valid JSON — no markdown, no preamble."""

SKELETON_TEMPLATE = """\
Job Description:
{text}

Fill this skeleton:
{{"job_title":"","responsibilities":[],"skills":[],"tools_and_platforms":[],"education":"","experience":{{"min_years":-1,"is_inferred":true}}}}"""

# JSON Schema for Gemini output validation
EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "job_title": {
            "type": "string",
            "description": "Job title as stated in the job posting"
        },
        "responsibilities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Action-oriented job responsibilities and duties"
        },
        "skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Technical and professional competencies required"
        },
        "tools_and_platforms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific technologies, frameworks, languages, and platforms"
        },
        "education": {
            "type": "string",
            "description": "Minimum education requirement or 'unknown'"
        },
        "experience": {
            "type": "object",
            "properties": {
                "min_years": {
                    "type": "integer",
                    "description": "Minimum years of experience (-1 if inferred/not stated)"
                },
                "is_inferred": {
                    "type": "boolean",
                    "description": "Whether experience was inferred vs explicitly stated"
                }
            },
            "required": ["min_years", "is_inferred"]
        }
    },
    "required": ["job_title", "responsibilities", "skills", "tools_and_platforms", "education", "experience"]
}


async def _extract_job_async(
    record: tuple[int, str | None, str],
    semaphore: asyncio.Semaphore,
    client,
    model_id: str,
) -> tuple[int, dict] | None:
    """Run extraction on a single job record asynchronously.

    Args:
        record: (job_id, cleaned_description, title)
        semaphore: Semaphore to limit concurrent in-flight requests
        client: Gemini genai.Client instance
        model_id: Gemini model ID (e.g. "gemini-2.5-flash")

    Returns:
        (job_id, extracted_dict) on success, None on any failure
    """
    logger = logging.getLogger("extract_job")
    job_id, cleaned_description, title = record

    if not cleaned_description:
        logger.warning("Job %d has no cleaned_description, skipping", job_id)
        return None

    prompt = EXTRACTION_SYSTEM_PROMPT + "\n\n" + SKELETON_TEMPLATE.replace("{text}", cleaned_description)

    # Initialize content for safe debug logging in except blocks
    content = "<not yet received>"

    logger.debug(
        "Job %d (%s): sending prompt (%d chars) to Gemini",
        job_id, title or "no title", len(prompt)
    )

    try:
        async with semaphore:
            response = await client.aio.models.generate_content(
                model=model_id,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
        content = response.text
        parsed = json.loads(content)
        jsonschema.validate(parsed, EXTRACTION_JSON_SCHEMA)
        return (job_id, parsed)
    except json.JSONDecodeError as e:
        logger.warning("Job %d: invalid JSON from Gemini: %s", job_id, e)
        logger.debug("Job %d: raw Gemini response:\n%s", job_id, content)
        return None
    except jsonschema.ValidationError as e:
        logger.warning(
            "Job %d: schema validation failed — field: %s, message: %s",
            job_id,
            " -> ".join(str(p) for p in e.absolute_path) or "(root)",
            e.message,
        )
        logger.debug("Job %d: raw Gemini response:\n%s", job_id, content)
        return None
    except Exception as e:
        logger.warning("Job %d: extraction failed (%s): %s", job_id, type(e).__name__, e)
        logger.debug("Job %d: raw Gemini response (if any):\n%s", job_id, content)
        return None


async def _extract_chunk_async(records, semaphore, client, model_id):
    """Run extraction on a chunk of records concurrently.

    Args:
        records: List of (job_id, cleaned_description, title) tuples
        semaphore: Semaphore to limit concurrent in-flight requests
        client: Gemini genai.Client instance
        model_id: Gemini model ID

    Returns:
        List of (job_id, extracted_dict) results or None for each
    """
    tasks = [_extract_job_async(r, semaphore, client, model_id) for r in records]
    return await asyncio.gather(*tasks)


def extract_jobs(
    db,
    run_id: int,
    chunk_size: int,
    api_key: str,
    model_id: str,
    max_retries: int = 2,
    concurrency: int = 10,
) -> tuple[int, int]:
    """Run extraction over all preprocessed, unextracted jobs in chunked batches.

    Mirrors the offset-0 pattern from preprocessing: committed rows drop out of
    WHERE extracted=0, so re-querying from offset 0 correctly returns the next batch.
    Uses async/await with asyncio.gather to run API calls concurrently per chunk.

    Args:
        db: DatabaseManager instance
        run_id: Pipeline run ID for audit logging
        chunk_size: Number of records per batch
        api_key: Google API key for Gemini
        model_id: Gemini model ID (e.g. "gemini-2.5-flash")
        max_retries: Max retries for failed records per chunk
        concurrency: Max number of concurrent in-flight Gemini requests (default: 10)

    Returns:
        Tuple of (processed_count, error_count)
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger = setup_logging(log_level=log_level, name="extract_jobs")

    client = genai.Client(api_key=api_key)
    logger.info("Gemini client initialized with model %s (concurrency: %d)", model_id, concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    total_processed = 0
    total_errors = 0

    while True:
        records = db.get_unextracted_jobs_chunked(chunk_size, 0)
        if not records:
            break

        logger.info("Processing chunk: %d records", len(records))

        results = list(asyncio.run(_extract_chunk_async(records, semaphore, client, model_id)))
        successes = [r for r in results if r is not None]
        errors_in_chunk = len(records) - len(successes)

        # Retry failed records
        input_ids = {r[0] for r in records}
        output_ids = {r[0] for r in successes}
        missing_ids = input_ids - output_ids

        for attempt in range(1, max_retries + 1):
            if not missing_ids:
                break

            logger.warning(
                "%d record(s) missing (attempt %d/%d), retrying IDs: %s",
                len(missing_ids), attempt, max_retries, sorted(missing_ids),
            )
            records_to_retry = [r for r in records if r[0] in missing_ids]
            retry_results = list(asyncio.run(_extract_chunk_async(records_to_retry, semaphore, client, model_id)))
            retry_successes = [r for r in retry_results if r is not None]
            successes.extend(retry_successes)
            output_ids = {r[0] for r in successes}
            missing_ids = input_ids - output_ids

        if missing_ids:
            logger.error(
                "%d record(s) permanently failed after %d retries: %s",
                len(missing_ids), max_retries, sorted(missing_ids),
            )

        if successes:
            try:
                db.update_extraction_batch(successes)
            except Exception as e:
                logger.error("DB write failed; stopping extraction: %s", e, exc_info=True)
                total_errors += errors_in_chunk
                break
        else:
            # No progress — all records failed. Log and stop to avoid infinite loop.
            logger.error(
                "All %d records in chunk failed extraction. Stopping to avoid infinite loop.",
                len(records),
            )
            total_errors += errors_in_chunk
            break

        total_processed += len(successes)
        total_errors += errors_in_chunk
        logger.info("Chunk complete: %d extracted, %d errors", len(successes), errors_in_chunk)

    logger.info("Extraction complete: %d extracted, %d errors", total_processed, total_errors)
    return total_processed, total_errors


def main() -> None:
    """CLI entrypoint called by Docker container.

    Reads config, initializes DB, runs extraction, updates pipeline_runs.
    Exits with code 0 on success, 1 on failure.
    """
    from datetime import datetime

    from src.config import load_config
    from src.database import DatabaseManager

    logger = setup_logging(name="extraction_main")
    try:
        config = load_config()
        logger.setLevel(config.log_level)

        db = DatabaseManager(config.db_path)
        db.initialize_schema()

        run_date = datetime.utcnow().strftime("%Y-%m-%d")
        run_id = db.create_pipeline_run(run_date, "extraction")

        processed, errors = extract_jobs(
            db,
            run_id,
            chunk_size=config.extraction_chunk_size,
            api_key=config.google_api_key,
            model_id=config.extraction_model_id,
            max_retries=config.extraction_max_retries,
            concurrency=config.extraction_concurrency,
        )
        db.finish_pipeline_run(run_id, "success", jobs_processed=processed, jobs_skipped=errors)
        logger.info("Extraction step completed successfully")

    except Exception as e:
        logger.exception("Extraction step failed")
        try:
            from src.config import load_config
            from src.database import DatabaseManager

            config = load_config()
            db = DatabaseManager(config.db_path)
            run_date = datetime.utcnow().strftime("%Y-%m-%d")
            run_id = db.create_pipeline_run(run_date, "extraction")
            db.finish_pipeline_run(run_id, "failed", 0, 0, str(e))
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
