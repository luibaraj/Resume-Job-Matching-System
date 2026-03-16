import logging
import multiprocessing
import os
import queue
import threading

import numpy as np

from src.utils import deserialize_list, setup_logging


def build_embedding_string(record: tuple) -> tuple[int, str] | None:
    """Build the text input for embedding from extracted job fields.

    Module-level function so it is picklable by multiprocessing.Pool.

    Args:
        record: (job_id, job_title, responsibilities_json, skills_json, tools_json)

    Returns:
        (job_id, text) on success, None if resulting string is empty
    """
    job_id, job_title, responsibilities_json, skills_json, tools_json = record

    parts = []
    if job_title:
        parts.append(job_title.strip())

    responsibilities = deserialize_list(responsibilities_json)
    if responsibilities:
        parts.append(", ".join(responsibilities))

    skills = deserialize_list(skills_json)
    if skills:
        parts.append(", ".join(skills))

    tools = deserialize_list(tools_json)
    if tools:
        parts.append(", ".join(tools))

    text = "\n".join(parts).strip()
    if not text:
        return None
    return (job_id, text)


def load_model(model_id: str):
    """Load Qwen3-Embedding model via sentence-transformers on CPU in float16.

    Args:
        model_id: HuggingFace model ID or local path

    Returns:
        SentenceTransformer model instance
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device="cpu")
    model.half()
    return model


def _db_writer_thread(db, write_queue: queue.Queue,
                      writer_error: threading.Event) -> None:
    """Background thread that drains write_queue and flushes batches to DB.

    Runs until it receives the sentinel value None.
    Sets writer_error on any DB exception and drains remaining items so
    the main thread's writer.join() never deadlocks.
    """
    while True:
        item = write_queue.get()
        if item is None:
            write_queue.task_done()
            break
        try:
            db.insert_embeddings_batch(item)
        except Exception as e:
            writer_error.set()
            logging.getLogger("db_writer").error("DB write failed: %s", e, exc_info=True)
            write_queue.task_done()
            while True:
                try:
                    leftover = write_queue.get_nowait()
                    write_queue.task_done()
                    if leftover is None:
                        break
                except queue.Empty:
                    break
            return
        else:
            write_queue.task_done()


def embed_jobs(
    db,
    run_id: int,
    chunk_size: int,
    batch_size: int,
    model_id: str,
    num_workers: int,
    max_retries: int = 2,
) -> tuple[int, int]:
    """Run embedding over all extracted, unembedded jobs in chunked batches.

    Uses the offset-0 pattern: committed rows drop out of WHERE embedded=0,
    so re-querying from offset 0 correctly returns the next batch.

    Args:
        db: DatabaseManager instance
        run_id: Pipeline run ID for audit logging
        chunk_size: Number of records per batch
        batch_size: Number of texts per model.encode() call
        model_id: HuggingFace model ID or local path
        num_workers: Number of multiprocessing workers for string building
        max_retries: Max retries for records that return None from build_embedding_string

    Returns:
        Tuple of (processed_count, error_count)
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger = setup_logging(log_level=log_level, name="embed_jobs")

    model = load_model(model_id)
    logger.info("Embedding model loaded: %s", model_id)

    total_processed = 0
    total_errors = 0

    write_queue: queue.Queue = queue.Queue(maxsize=5)
    writer_error = threading.Event()
    writer = threading.Thread(
        target=_db_writer_thread,
        args=(db, write_queue, writer_error),
        daemon=True,
        name="db-writer",
    )
    writer.start()

    while True:
        records = db.get_unembedded_jobs_chunked(chunk_size, 0)
        if not records:
            break

        logger.info("Processing chunk: %d records", len(records))

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(build_embedding_string, records)

        successes = [r for r in results if r is not None]
        errors_in_chunk = len(records) - len(successes)

        # Retry records that returned None
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
            with multiprocessing.Pool(processes=num_workers) as pool:
                retry_results = pool.map(build_embedding_string, records_to_retry)
            retry_successes = [r for r in retry_results if r is not None]
            successes.extend(retry_successes)
            output_ids = {r[0] for r in successes}
            missing_ids = input_ids - output_ids

        if missing_ids:
            logger.error(
                "%d record(s) permanently failed after %d retries: %s",
                len(missing_ids), max_retries, sorted(missing_ids),
            )

        if not successes:
            logger.error(
                "All %d records in chunk failed embedding string build. "
                "Stopping to avoid infinite loop.",
                len(records),
            )
            total_errors += errors_in_chunk
            break

        # Encode in batches
        job_ids = [job_id for job_id, _ in successes]
        texts = [text for _, text in successes]
        try:
            vectors = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as e:
            logger.error("model.encode failed: %s", e, exc_info=True)
            total_errors += len(records)
            break

        # Serialize and queue
        updates = [
            (job_id, vectors[i].astype(np.float32).tobytes(), model_id)
            for i, job_id in enumerate(job_ids)
        ]

        if writer_error.is_set():
            logger.error("DB writer thread failed; stopping embedding.")
            total_errors += errors_in_chunk
            break

        write_queue.put(updates)
        db.mark_jobs_embedded(job_ids)

        total_processed += len(successes)
        total_errors += errors_in_chunk
        logger.info("Chunk complete: %d embedded, %d errors", len(successes), errors_in_chunk)

    write_queue.put(None)
    writer.join()
    if writer_error.is_set():
        logger.error("DB writer thread raised an exception during drain.")

    logger.info("Embedding complete: %d embedded, %d errors", total_processed, total_errors)
    return total_processed, total_errors


def main() -> None:
    """CLI entrypoint called by Docker container.

    Reads config, initializes DB, runs embedding, updates pipeline_runs.
    Exits with code 0 on success, 1 on failure.
    """
    from datetime import datetime

    from src.config import load_config
    from src.database import DatabaseManager

    logger = setup_logging(name="embedding_main")
    try:
        config = load_config()
        logger.setLevel(config.log_level)

        db = DatabaseManager(config.db_path)
        db.initialize_schema()

        run_date = datetime.utcnow().strftime("%Y-%m-%d")
        run_id = db.create_pipeline_run(run_date, "embedding")

        processed, errors = embed_jobs(
            db,
            run_id,
            chunk_size=config.embedding_chunk_size,
            batch_size=config.embedding_batch_size,
            model_id=config.embedding_model_id,
            num_workers=config.embedding_num_workers,
            max_retries=config.embedding_max_retries,
        )
        db.finish_pipeline_run(run_id, "success", jobs_processed=processed, jobs_skipped=errors)
        logger.info("Embedding step completed successfully")

    except Exception as e:
        logger.exception("Embedding step failed")
        try:
            from src.config import load_config
            from src.database import DatabaseManager

            config = load_config()
            db = DatabaseManager(config.db_path)
            run_date = datetime.utcnow().strftime("%Y-%m-%d")
            run_id = db.create_pipeline_run(run_date, "embedding")
            db.finish_pipeline_run(run_id, "failed", 0, 0, str(e))
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
