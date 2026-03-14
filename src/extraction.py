import json
import logging
import os

from src.utils import setup_logging


"""
System prompt for fine-tuned Llama 3.2 Information Extraction Model
Aligned with job matching pipeline (data collection → extraction → embeddings → retrieval → reranking)
"""

EXTRACTION_SYSTEM_PROMPT = """Extract structured data from the job description below. Return ONLY valid JSON — no markdown, no preamble.

Required fields:
- "job_title": string — restate the title exactly as given
- "responsibilities": string[] — verb+object phrases (e.g., "Design backend APIs")
- "skills": string[] — technical and professional competencies; include explicit soft skills
- "tools_and_platforms": string[] — languages, frameworks, databases, cloud/SaaS tools
- "education": string — minimum degree if stated; else "unknown"
- "experience": {"min_years": int, "is_inferred": bool}
  - If stated: min_years=N, is_inferred=false
  - If not stated: min_years=-1, is_inferred=true

Use empty lists [] when a field has no content in the description.

Output exactly this structure:
{"job_title":"","responsibilities":[],"skills":[],"tools_and_platforms":[],"education":"","experience":{"min_years":-1,"is_inferred":true}}

Job Description:
{text}"""

# JSON Schema for Llama output validation
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


def load_model(model_path: str, n_ctx: int, n_gpu_layers: int):
    """Load the Llama model from a GGUF file.

    Args:
        model_path: Path to the GGUF model file
        n_ctx: Context window size
        n_gpu_layers: Number of layers to offload to GPU (-1 = all)

    Returns:
        Llama model instance
    """
    from llama_cpp import Llama

    return Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)


def extract_job(record: tuple[int, str | None, str], model) -> tuple[int, dict] | None:
    """Run extraction on a single job record.

    Args:
        record: (job_id, cleaned_description, title)
        model: Loaded Llama model instance

    Returns:
        (job_id, extracted_dict) on success, None on any failure
    """
    import jsonschema

    logger = logging.getLogger("extract_job")
    job_id, cleaned_description, title = record

    if not cleaned_description:
        logger.warning("Job %d has no cleaned_description, skipping", job_id)
        return None

    try:
        prompt = EXTRACTION_SYSTEM_PROMPT.replace("{text}", cleaned_description)
        response = model.create_chat_completion(
            messages=[{"role": "system", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        content = response["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        jsonschema.validate(parsed, EXTRACTION_JSON_SCHEMA)
        return (job_id, parsed)
    except json.JSONDecodeError as e:
        logger.warning("Job %d: invalid JSON from model: %s", job_id, e)
        return None
    except jsonschema.ValidationError as e:
        logger.warning("Job %d: schema validation failed: %s", job_id, e.message)
        return None
    except Exception as e:
        logger.warning("Job %d: extraction failed: %s", job_id, e)
        return None


def extract_jobs(
    db,
    run_id: int,
    chunk_size: int,
    model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    max_retries: int = 2,
) -> tuple[int, int]:
    """Run extraction over all preprocessed, unextracted jobs in chunked batches.

    Mirrors the offset-0 pattern from preprocessing: committed rows drop out of
    WHERE extracted=0, so re-querying from offset 0 correctly returns the next batch.

    Args:
        db: DatabaseManager instance
        run_id: Pipeline run ID for audit logging
        chunk_size: Number of records per batch
        model_path: Path to GGUF model file
        n_ctx: Context window size
        n_gpu_layers: GPU layers to offload (-1 = all)
        max_retries: Max retries for failed records per chunk

    Returns:
        Tuple of (processed_count, error_count)
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger = setup_logging(log_level=log_level, name="extract_jobs")

    model = load_model(model_path, n_ctx, n_gpu_layers)
    logger.info("Model loaded from %s", model_path)

    total_processed = 0
    total_errors = 0

    while True:
        records = db.get_unextracted_jobs_chunked(chunk_size, 0)
        if not records:
            break

        logger.info("Processing chunk: %d records", len(records))

        results = [extract_job(r, model) for r in records]
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
            retry_results = [extract_job(r, model) for r in records_to_retry]
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
            db.update_extraction_batch(successes)
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

        if not config.extraction_model_path:
            raise ValueError("EXTRACTION_MODEL_PATH must be set")

        db = DatabaseManager(config.db_path)
        db.initialize_schema()

        run_date = datetime.utcnow().strftime("%Y-%m-%d")
        run_id = db.create_pipeline_run(run_date, "extraction")

        processed, errors = extract_jobs(
            db,
            run_id,
            chunk_size=config.extraction_chunk_size,
            model_path=config.extraction_model_path,
            n_ctx=config.extraction_n_ctx,
            n_gpu_layers=config.extraction_n_gpu_layers,
            max_retries=config.extraction_max_retries,
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
