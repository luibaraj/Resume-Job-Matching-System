import asyncio
import json
import logging
import os
import re
from pathlib import Path

import jsonschema
from dotenv import load_dotenv
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


def _repair_json(raw: str) -> dict | None:
    """Attempt to parse and repair a potentially malformed JSON string.

    Strategies applied in order:
    1. Strip markdown code fences if present (```json ... ```)
    2. Locate and extract the first complete JSON object by bracket matching
    3. If bracket extraction fails to parse, attempt missing-comma repair and retry

    Args:
        raw: Raw response text from Gemini

    Returns:
        Parsed dict on success, None if all repair strategies fail
    """
    logger = logging.getLogger("repair_json")

    # Strategy 1: Strip markdown fences
    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:].lstrip()
    if text.startswith("```"):
        text = text[3:].lstrip()
    if text.endswith("```"):
        text = text[:-3].rstrip()

    # Strategy 2: Extract first complete JSON object by bracket scan
    brace_start = text.find("{")
    if brace_start != -1:
        depth = 0
        in_string = False
        escape_next = False

        for i in range(brace_start, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        # Found complete object
                        candidate = text[brace_start : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            logger.debug("Bracket extraction produced unparseable JSON: %s", candidate[:100])
                            break

    # Strategy 3: Comma repair — fix missing commas between fields
    # Pattern: closing delimiter (}, ], or ") followed by an opening quote (for next field)
    # This catches: }{"key" -> },"key" or ]"key" -> ],"key"
    text_for_repair = raw.strip()
    if text_for_repair.startswith("```"):
        text_for_repair = text_for_repair.split("```", 2)[1] if "```" in text_for_repair else text_for_repair

    repaired = re.sub(r'([}\]"])\s*\n?\s*(")', r"\1,\2", text_for_repair)
    if repaired != text_for_repair:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            logger.debug("Comma repair failed to produce valid JSON")

    return None


async def _extract_job_async(
    record: tuple[int, str | None, str],
    semaphore: asyncio.Semaphore,
    client,
    model_id: str,
    max_output_tokens: int = 1024,
) -> tuple[int, dict] | tuple[int, None, str, str]:
    """Run extraction on a single job record asynchronously.

    Args:
        record: (job_id, cleaned_description, title)
        semaphore: Semaphore to limit concurrent in-flight requests
        client: Gemini genai.Client instance
        model_id: Gemini model ID (e.g. "gemini-2.5-flash")

    Returns:
        (job_id, extracted_dict) on success
        (job_id, None, error_type, error_message) on failure
    """
    logger = logging.getLogger("extract_job")
    job_id, cleaned_description, title = record

    if not cleaned_description:
        logger.warning("Job %d has no cleaned_description, skipping", job_id)
        return (job_id, None, "missing_description", "No cleaned_description available")

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
                    max_output_tokens=max_output_tokens,
                ),
            )
        content = response.text

        # Check if model was cut off mid-output
        if response.candidates and response.candidates[0].finish_reason == "MAX_TOKENS":
            logger.warning("Job %d: response truncated (MAX_TOKENS), attempting repair anyway", job_id)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.debug("Job %d: direct parse failed, attempting repair", job_id)
            parsed = _repair_json(content)
            if parsed is None:
                logger.warning("Job %d: JSON repair failed", job_id)
                logger.debug("Job %d: raw response:\n%s", job_id, content)
                return (job_id, None, "json_parse_error", "Failed to parse and repair JSON from Gemini")
            logger.info("Job %d: JSON repaired successfully", job_id)

        jsonschema.validate(parsed, EXTRACTION_JSON_SCHEMA)
        return (job_id, parsed)
    except json.JSONDecodeError as e:
        logger.warning("Job %d: invalid JSON from Gemini: %s", job_id, e)
        logger.debug("Job %d: raw Gemini response:\n%s", job_id, content)
        return (job_id, None, "json_parse_error", str(e))
    except jsonschema.ValidationError as e:
        error_msg = f"Field: {' -> '.join(str(p) for p in e.absolute_path) or '(root)'}, Message: {e.message}"
        logger.warning(
            "Job %d: schema validation failed — %s",
            job_id,
            error_msg,
        )
        logger.debug("Job %d: raw Gemini response:\n%s", job_id, content)
        return (job_id, None, "schema_validation_error", error_msg)
    except Exception as e:
        logger.warning("Job %d: extraction failed (%s): %s", job_id, type(e).__name__, e)
        logger.debug("Job %d: raw Gemini response (if any):\n%s", job_id, content)
        return (job_id, None, "api_error", f"{type(e).__name__}: {str(e)}")


async def _extract_chunk_async(records, concurrency, client, model_id, max_output_tokens):
    """Run extraction on a chunk of records concurrently.

    Args:
        records: List of (job_id, cleaned_description, title) tuples
        concurrency: Max number of concurrent in-flight requests
        client: Gemini genai.Client instance
        model_id: Gemini model ID
        max_output_tokens: Max tokens in Gemini response

    Returns:
        List of results: (job_id, extracted_dict) on success, or
        (job_id, None, error_type, error_message) on failure
    """
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_extract_job_async(r, semaphore, client, model_id, max_output_tokens) for r in records]
    return await asyncio.gather(*tasks)


async def _extract_jobs_async(
    db,
    run_id: int,
    chunk_size: int,
    api_key: str,
    model_id: str,
    max_retries: int = 2,
    concurrency: int = 10,
    max_output_tokens: int = 1024,
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
        max_output_tokens: Max tokens in Gemini response (default: 1024)

    Returns:
        Tuple of (processed_count, error_count)
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger = setup_logging(log_level=log_level, name="extract_jobs")

    client = genai.Client(api_key=api_key)
    logger.info("Gemini client initialized with model %s (concurrency: %d)", model_id, concurrency)

    total_processed = 0
    total_errors = 0

    while True:
        records = db.get_unextracted_jobs_chunked(chunk_size, 0)
        if not records:
            break

        logger.info("Processing chunk: %d records", len(records))

        results = list(await _extract_chunk_async(records, concurrency, client, model_id, max_output_tokens))
        successes = [r for r in results if len(r) == 2 and r[1] is not None]
        errors_in_chunk = len(records) - len(successes)

        # Accumulate all results (both successes and failures) for error persistence
        all_results = results.copy()

        # Retry failed records
        input_ids = {r[0] for r in records}
        output_ids = {r[0] for r in successes}
        missing_ids = input_ids - output_ids

        for attempt in range(1, max_retries + 1):
            if not missing_ids:
                break

            delay = 2.0 ** attempt  # 2s, 4s, 8s ...
            logger.warning(
                "%d record(s) missing (attempt %d/%d), sleeping %.1fs, retrying IDs: %s",
                len(missing_ids), attempt, max_retries, delay, sorted(missing_ids),
            )
            await asyncio.sleep(delay)
            records_to_retry = [r for r in records if r[0] in missing_ids]
            retry_results = list(await _extract_chunk_async(records_to_retry, concurrency, client, model_id, max_output_tokens))
            retry_successes = [r for r in retry_results if len(r) == 2 and r[1] is not None]
            successes.extend(retry_successes)
            all_results.extend(retry_results)
            output_ids = {r[0] for r in successes}
            missing_ids = input_ids - output_ids

        if missing_ids:
            logger.error(
                "%d record(s) permanently failed after %d retries: %s",
                len(missing_ids), max_retries, sorted(missing_ids),
            )

        # Write all errors to DB (both transient during retries and permanent failures)
        errors_to_write = []
        for r in all_results:
            if len(r) == 4 and r[1] is None:
                job_id, _, error_type, error_message = r
                errors_to_write.append((job_id, error_type, error_message, max_retries + 1))

        if errors_to_write:
            try:
                db.write_extraction_errors_batch(errors_to_write)
            except Exception as e:
                logger.error("Failed to write extraction errors to DB: %s", e)

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


def extract_jobs(
    db,
    run_id: int,
    chunk_size: int,
    api_key: str,
    model_id: str,
    max_retries: int = 2,
    concurrency: int = 10,
    max_output_tokens: int = 1024,
) -> tuple[int, int]:
    """Sync wrapper — runs entire extraction in one event loop to avoid
    'Event loop is closed' errors when the Gemini client retries."""
    return asyncio.run(
        _extract_jobs_async(db, run_id, chunk_size, api_key, model_id, max_retries, concurrency, max_output_tokens)
    )


def main() -> None:
    """CLI entrypoint called by Docker container.

    Reads config, initializes DB, runs extraction, updates pipeline_runs.
    Exits with code 0 on success, 1 on failure.
    """
    from datetime import datetime

    from src.config import load_config
    from src.database import DatabaseManager

    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

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
            max_output_tokens=config.extraction_max_output_tokens,
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
