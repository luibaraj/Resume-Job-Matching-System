"""
Orchestration script: two-stage pipeline combining dense retrieval and reranking.

Reads resume from data/user_profile.txt, retrieves top 100 candidates via dense search,
reranks them to top 10 using Cohere Rerank 3, and outputs results to matched_jobs.md.
"""

import argparse
import hashlib
import logging
import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path

import chromadb
import numpy as np
from chromadb.api import ClientAPI
from dotenv import load_dotenv

# Ensure src/ can be imported from any working directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DEFAULT_DIR,
    CORPUS_LIMITATION_MESSAGE,
    DB_DEFAULT_PATH,
    HNSW_EF,
    HNSW_EF_CONSTRUCTION,
    OLLAMA_MODEL,
    RETRIEVE_TOP_K,
    RERANK_TOP_N,
)
from src.embedding import create_client, embed_batch
from src.llm_extraction import (
    extract_degree_with_llm,
    extract_seniority_with_llm,
    extract_years_with_llm,
)
from src.regex_extraction import (
    build_chroma_where_filter,
    describe_chroma_filter,
    extract_years_experience,
)
from src.retrieval import build_collection, query_collection
from src.reranking import rerank_jobs

import ollama
from src.generation import run_generation_pipeline

logger = logging.getLogger(__name__)


def load_resume(resume_path: str | None = None) -> str:
    """
    Load resume text from file.

    Args:
        resume_path: Path to resume file. Defaults to data/user_profile.txt.

    Returns:
        Stripped resume text.

    Raises:
        SystemExit: If file not found or empty.
    """
    if resume_path is None:
        resume_path = str(_PROJECT_ROOT / "data" / "user_profile.txt")

    try:
        with open(resume_path, "r", encoding="utf-8") as f:
            resume_text = f.read().strip()
    except FileNotFoundError:
        logger.error("Resume file not found: %s", resume_path)
        sys.exit(1)

    if not resume_text:
        logger.error("Resume is empty.")
        sys.exit(1)

    return resume_text


def extract_user_filters(resume_text: str):
    """
    Extract structured criteria from resume text and build a ChromaDB where filter.

    Args:
        resume_text: Full resume text from user_profile.txt.

    Returns:
        A ChromaDB where filter dict, or None if no criteria could be extracted.
    """
    user_degree = extract_degree_with_llm(resume_text, OLLAMA_MODEL)
    user_seniority = extract_seniority_with_llm(resume_text, OLLAMA_MODEL)
    user_years = extract_years_with_llm(resume_text, OLLAMA_MODEL)

    logger.info(
        "User profile — degree: %d, seniority: %d, years: %d",
        user_degree,
        user_seniority,
        user_years,
    )

    return build_chroma_where_filter(user_degree, user_seniority, user_years)


def _resume_hash(resume_text: str) -> str:
    """
    Compute MD5 hash of resume text for cache invalidation.

    Args:
        resume_text: Resume text to hash.

    Returns:
        Hexadecimal MD5 hash string.
    """
    return hashlib.md5(resume_text.encode()).hexdigest()


def load_or_embed_resume(
    client,
    resume_text: str,
    cache_path: str = "data/user_profile_embedding.npy",
    hash_path: str = "data/user_profile_embedding_hash.txt",
) -> np.ndarray:
    """
    Load cached resume embedding if available and valid, otherwise embed and cache.

    Cache is invalidated if the resume text (detected via MD5 hash) has changed.
    Uses np.save/np.load for efficient binary storage. If cache load fails, falls
    back to re-embedding and caching.

    Args:
        client: Voyage AI client.
        resume_text: Resume text to embed.
        cache_path: Path to cached embedding file (default: data/user_profile_embedding.npy).
        hash_path: Path to cached hash file (default: data/user_profile_embedding_hash.txt).

    Returns:
        Embedding vector (numpy array, float32).
    """
    current_hash = _resume_hash(resume_text)
    cache = Path(cache_path)
    hash_file = Path(hash_path)

    # Cache hit: both files exist and hash matches
    if cache.exists() and hash_file.exists():
        saved_hash = hash_file.read_text().strip()
        if saved_hash == current_hash:
            logger.info("Loading cached resume embedding...")
            try:
                return np.load(str(cache))
            except Exception as e:
                logger.warning(
                    "Failed to load cached embedding: %s. Re-embedding.",
                    e,
                )
                # Fall through to re-embed

    # Cache miss or load failure: embed, save, and cache the hash
    logger.info("Embedding resume...")
    embeddings = embed_batch(client, [resume_text])
    embedding = embeddings[0]

    # Ensure parent directory exists
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache), embedding)
    hash_file.write_text(current_hash)
    logger.info("Resume embedding cached to %s", cache_path)

    return embedding


def build_chroma_client(
    chroma_dir: str, rebuild: bool
) -> ClientAPI:
    """
    Create a persistent Chroma client, optionally clearing the collection first.

    Args:
        chroma_dir: Directory for Chroma storage.
        rebuild: If True, delete the collection before returning the client.

    Returns:
        Persistent ChromaDB client.
    """
    chroma_path = Path(chroma_dir)
    chroma_path.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(chroma_path))

    if rebuild:
        try:
            chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
            logger.info("Deleted existing collection '%s'.", CHROMA_COLLECTION_NAME)
        except ValueError:
            # Collection doesn't exist — silently continue
            pass

    return chroma_client


def write_results_markdown(results: list[dict] | list, output_path: str = "matched_jobs.md") -> None:
    """
    Write results to a Markdown file.

    Args:
        results: List of JobResult dicts with keys: id, distance, title,
                location, source_url, board_token, cleaned_description.
        output_path: Path to write the Markdown file (default: matched_jobs.md).
    """
    lines = ["# Top Matched Jobs (Reranked)\n"]

    if not results:
        lines.append(CORPUS_LIMITATION_MESSAGE)
    else:
        for i, job in enumerate(results, start=1):
            title = job.get("title", "Unknown")
            board_token = job.get("board_token", "Unknown")
            url = job.get("source_url", "No URL")
            description = job.get("cleaned_description", "")

            lines.append(f"## {i}. {title}")
            lines.append(f"- **Board Token:** `{board_token}`")

            # Extract minimum years of experience if detected
            min_years = extract_years_experience(description)
            if min_years > 0:
                lines.append(f"- **Min. Years of Experience:** {min_years}")

            lines.append(f"- **URL:** [{url}]({url})\n")

            # Render fit summary or corpus warning
            explanation = job.get("explanation")
            if explanation is not None:
                lines.append(f"- **Fit Summary:** {explanation}\n")
            elif job.get("corpus_warning"):
                lines.append(f"- **Note:** {job['corpus_warning']}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Results written to %s.", output_path)


def run_generation_for_results(
    resume_text: str,
    results: list[dict] | list,
    model: str = OLLAMA_MODEL,
    run_id: str | None = None,
) -> None:
    """
    Run generation pipeline on each reranked result and attach explanation in-place.

    Processes one pair at a time to preserve index alignment. Attaches the
    explanation string to each result dict under "explanation", or None if
    no grounded match was found. Exits early if Ollama is unreachable.

    Args:
        resume_text: User's resume text.
        results: List of job results to explain.
        model: Ollama model name.
        run_id: Optional trace ID for request tracing.
    """
    logger.info("Running generation pipeline for %d results...", len(results))

    for job in results:
        description = job.get("cleaned_description", "")
        if not description:
            job["explanation"] = None
            continue

        try:
            generation_results, corpus_message = run_generation_pipeline(
                pairs=[(resume_text, description)],
                model=model,
                run_id=run_id,
            )
        except ollama.RequestError as e:
            logger.warning("Ollama is not reachable: %s. Skipping generation.", e)
            for remaining in results:
                remaining.setdefault("explanation", None)
            return
        except ollama.ResponseError as e:
            logger.warning("Ollama model error for job '%s': %s. Skipping.", job.get("title"), e)
            job["explanation"] = None
            continue
        except Exception as e:
            logger.warning("Unexpected generation error for job '%s': %s. Skipping.", job.get("title"), e)
            job["explanation"] = None
            continue

        # Unpack result (we sent exactly one pair)
        job["explanation"] = generation_results[0]["explanation"]
        if corpus_message is not None:
            job["corpus_warning"] = corpus_message


def main() -> None:
    """Main orchestration function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Match resume to job listings using dense retrieval and reranking."
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite database path (default: DB_PATH env var or config default)",
    )
    parser.add_argument(
        "--resume-path",
        default=None,
        help="Path to resume file (default: data/user_profile.txt)",
    )
    parser.add_argument(
        "--output-path",
        default="matched_jobs.md",
        help="Output markdown file (default: matched_jobs.md)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the Chroma vector store from scratch.",
    )
    args = parser.parse_args()

    # Configure logging (after argparse)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # Load environment variables (after argparse, after logging)
    load_dotenv()

    # Validate required API keys early (before expensive operations)
    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    if not voyage_api_key:
        logger.error("VOYAGE_API_KEY environment variable is not set.")
        sys.exit(1)

    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        logger.error("COHERE_API_KEY environment variable is not set.")
        sys.exit(1)

    # Pre-flight check: ensure Ollama is reachable
    try:
        ollama.list()
        logger.info("Ollama is reachable.")
    except (ollama.RequestError, ollama.ResponseError) as e:
        logger.error("Ollama is not reachable: %s. Please start Ollama before running.", e)
        sys.exit(1)

    # Generate run_id for request tracing across all pipeline stages
    run_id = str(uuid.uuid4())
    logger.info("Pipeline run_id: %s", run_id)

    # Resolve database path: arg → env var → default
    db_path = args.db_path or os.getenv("DB_PATH", DB_DEFAULT_PATH)

    # Validate database exists
    db_path_obj = Path(db_path)
    if not db_path_obj.exists():
        logger.error(
            "Database not found at %s. Please run the pipeline scripts first: "
            "python scripts/scrape_jobs.py && "
            "python scripts/preprocess_jobs.py && "
            "python scripts/embed_jobs.py",
            db_path,
        )
        sys.exit(1)

    # Load resume
    resume_text = load_resume(args.resume_path)
    logger.info("Loaded resume (%d characters).", len(resume_text))

    # Extract user profile criteria for filtering
    query_filter = extract_user_filters(resume_text)
    logger.info("Applying filters: %s", describe_chroma_filter(query_filter))

    # Build Chroma client (rebuild only if explicitly requested via --rebuild)
    chroma_client = build_chroma_client(CHROMA_DEFAULT_DIR, rebuild=args.rebuild)

    # Connect to database and build/sync Chroma collection
    try:
        conn = sqlite3.connect(db_path)
        try:
            # Time build_collection
            logger.info("Building ChromaDB collection from embeddings...")
            start_build = time.monotonic()
            collection = build_collection(
                conn, chroma_client, CHROMA_COLLECTION_NAME, 
                ef_construction=HNSW_EF_CONSTRUCTION,
                model=OLLAMA_MODEL
            )
            elapsed_build = time.monotonic() - start_build
            logger.info("Collection ready with %d jobs indexed in %.2fs.", collection.count(), elapsed_build)

            # Sanity check: ensure we have embedded jobs
            if collection.count() == 0:
                logger.error(
                    "No embedded jobs found in the database. "
                    "Please run the pipeline scripts first: "
                    "python scripts/preprocess_jobs.py && "
                    "python scripts/embed_jobs.py."
                )
                sys.exit(1)

            # Time load_or_embed_resume
            voyage_client = create_client(voyage_api_key)
            start_resume = time.monotonic()
            query_embedding = load_or_embed_resume(voyage_client, resume_text)
            elapsed_resume = time.monotonic() - start_resume
            logger.info("Resume embedding loaded/computed in %.2fs, shape: %s", elapsed_resume, query_embedding.shape)

            # Step 1: Time dense retrieval — top 100 candidates
            logger.info("Querying for top %d candidates...", RETRIEVE_TOP_K)
            start_query = time.monotonic()
            candidates = query_collection(
                collection, query_embedding, top_k=RETRIEVE_TOP_K, ef=HNSW_EF, where=query_filter, run_id=run_id
            )
            elapsed_query = time.monotonic() - start_query
            logger.info("Queried %d candidates in %.2fs.", len(candidates), elapsed_query)

            # Step 2: Time rerank — top 10 results
            logger.info("Reranking %d candidates to top %d...", len(candidates), RERANK_TOP_N)
            start_rerank = time.monotonic()
            results = rerank_jobs(
                resume_text, candidates, top_n=RERANK_TOP_N, api_key=cohere_api_key, run_id=run_id
            )
            elapsed_rerank = time.monotonic() - start_rerank
            logger.info("Reranked to %d results in %.2fs.", len(results), elapsed_rerank)

            # Output results
            if len(results) == 0:
                logger.warning("No matching jobs found.")
            else:
                # Step 3: Time generation — attach fit explanations
                logger.info("Running generation pipeline...")
                start_gen = time.monotonic()
                run_generation_for_results(resume_text, results, run_id=run_id)
                elapsed_gen = time.monotonic() - start_gen
                logger.info("Generation completed in %.2fs.", elapsed_gen)
                write_results_markdown(results, args.output_path)

        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        logger.error("Error reading database: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Error building collection (corrupt embeddings?): %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
