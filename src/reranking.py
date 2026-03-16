import logging
import os

import torch

from src.utils import setup_logging

# Qwen3-Reranker prompt template (from model card)
_RERANKER_INSTRUCTION = (
    "Given a user's professional profile, determine whether the job posting is a good match."
)
_PREFIX = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _format_input(query: str, passage: str) -> str:
    """Format a (query, passage) pair using the Qwen3-Reranker prompt template."""
    return (
        f"{_PREFIX}"
        f"<Instruct>: {_RERANKER_INSTRUCTION}\n"
        f"<Query>: {query}\n"
        f"<Document>: {passage}"
        f"{_SUFFIX}"
    )


def load_reranker(model_id: str):
    """Load Qwen3-Reranker tokenizer and model in float16 on CPU.

    Args:
        model_id: HuggingFace model ID (e.g. "Qwen/Qwen3-Reranker-0.6B")

    Returns:
        Tuple of (tokenizer, model)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.eval()
    return tokenizer, model


def build_job_text(title: str, description: str) -> str:
    """Combine job title and description into a single passage string.

    Args:
        title: Job title
        description: Cleaned job description

    Returns:
        Combined text with title on first line
    """
    return f"{title}\n{description}".strip()


def score_pairs_batched(
    tokenizer,
    model,
    query: str,
    passages: list[str],
    batch_size: int,
    max_length: int = 512,
) -> list[float]:
    """Score (query, passage) pairs in batches using Qwen3-Reranker.

    Uses the model's yes/no token logits at the last output position as relevance
    scores. Higher score = more relevant.

    Args:
        tokenizer: Qwen3-Reranker tokenizer
        model: Qwen3-Reranker causal LM model
        query: User profile text (the query)
        passages: List of job text passages to score
        batch_size: Number of pairs to process at once
        max_length: Max token length per input (truncates to fit RAM budget)

    Returns:
        List of float scores, one per passage, in the same order as passages
    """
    yes_token_id = tokenizer.convert_tokens_to_ids("yes")
    no_token_id = tokenizer.convert_tokens_to_ids("no")

    formatted = [_format_input(query, p) for p in passages]
    scores: list[float] = []

    for i in range(0, len(formatted), batch_size):
        batch = formatted[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract logits at the last token position for each item in the batch
        last_logits = outputs.logits[:, -1, :]  # [B, vocab_size]
        yes_logits = last_logits[:, yes_token_id]
        no_logits = last_logits[:, no_token_id]

        # Softmax over yes/no to get P(yes)
        pair = torch.stack([no_logits, yes_logits], dim=1)  # [B, 2]
        probs = torch.softmax(pair.float(), dim=1)[:, 1]  # P(yes)
        scores.extend(probs.tolist())

    return scores


def rerank(db, run_id: int, config) -> tuple[int, int]:
    """Rerank job_matches using Qwen3-Reranker cross-encoder scores.

    Reads candidates from job_matches, scores each (user_profile, job_text) pair,
    and writes the top-k results to job_reranked.

    Args:
        db: DatabaseManager instance
        run_id: Pipeline run ID for audit logging
        config: Config instance with reranking fields set

    Returns:
        Tuple of (matches_written, 0)
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger = setup_logging(log_level=log_level, name="rerank")

    # Load candidates
    candidates = db.get_job_matches_with_text()
    if not candidates:
        logger.warning("No job_matches found — skipping reranking")
        return 0, 0
    logger.info("Loaded %d candidates from job_matches", len(candidates))

    # Load user profile
    from src.retrieval import load_user_profile
    profile_text = load_user_profile(config.retrieval_user_profile_path).strip()
    logger.info("User profile loaded from %s", config.retrieval_user_profile_path)

    # Load reranker model
    logger.info("Loading reranker model: %s", config.reranking_model_id)
    tokenizer, model = load_reranker(config.reranking_model_id)
    logger.info("Reranker model loaded")

    # Build passage texts
    passages = [build_job_text(title, desc) for _, title, desc, _ in candidates]
    job_ids = [job_id for job_id, _, _, _ in candidates]

    # Score all pairs
    logger.info(
        "Scoring %d pairs (batch_size=%d)", len(passages), config.reranking_batch_size
    )
    scores = score_pairs_batched(
        tokenizer,
        model,
        profile_text,
        passages,
        batch_size=config.reranking_batch_size,
    )
    logger.info("Scoring complete")

    # Sort by score descending, take top_k
    ranked = sorted(zip(job_ids, scores), key=lambda x: x[1], reverse=True)
    top = ranked[: config.reranking_top_k]

    matches = [
        (job_id, score, rank + 1, config.reranking_model_id)
        for rank, (job_id, score) in enumerate(top)
    ]
    db.insert_reranked(matches)
    logger.info("Wrote %d reranked results to DB", len(matches))

    return len(matches), 0


def main() -> None:
    """CLI entrypoint called by Docker container.

    Reads config, initializes DB, runs reranking, updates pipeline_runs.
    Exits with code 0 on success, 1 on failure.
    """
    from datetime import datetime

    from src.config import load_config
    from src.database import DatabaseManager

    logger = setup_logging(name="reranking_main")
    try:
        config = load_config()
        logger.setLevel(config.log_level)

        db = DatabaseManager(config.db_path)
        db.initialize_schema()

        run_date = datetime.utcnow().strftime("%Y-%m-%d")
        run_id = db.create_pipeline_run(run_date, "reranking")

        processed, skipped = rerank(db, run_id, config)
        db.finish_pipeline_run(run_id, "success", jobs_processed=processed, jobs_skipped=skipped)
        logger.info("Reranking step completed: %d results written", processed)

    except Exception as e:
        logger.exception("Reranking step failed")
        try:
            from src.config import load_config
            from src.database import DatabaseManager

            config = load_config()
            db = DatabaseManager(config.db_path)
            run_date = datetime.utcnow().strftime("%Y-%m-%d")
            run_id = db.create_pipeline_run(run_date, "reranking")
            db.finish_pipeline_run(run_id, "failed", 0, 0, str(e))
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
