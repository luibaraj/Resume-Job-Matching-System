"""
Orchestration script: two-stage pipeline combining dense retrieval and reranking.

Reads resume from data/user_profile.txt, retrieves top 100 candidates via dense search,
reranks them to top 10 using Cohere Rerank 3, and outputs results to matched_jobs.md.
"""

import hashlib
import logging
import os
import sqlite3
import sys
from pathlib import Path

import chromadb
import numpy as np
from dotenv import load_dotenv

# Ensure src/ can be imported from any working directory
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DEFAULT_DIR,
    DB_DEFAULT_PATH,
    HNSW_EF,
    HNSW_EF_CONSTRUCTION,
    RETRIEVE_TOP_K,
    RERANK_TOP_N,
)
from embedding import create_client, embed_batch
from regex_extraction import (
    build_chroma_where_filter,
    describe_chroma_filter,
    extract_user_degree,
    extract_user_seniority,
    extract_user_years_experience,
)
from retrieval import build_collection, query_collection
from reranking import rerank_jobs


def load_resume() -> str:
    """
    Load resume text from data/user_profile.txt.

    Returns:
        Stripped resume text.

    Raises:
        SystemExit: If file not found or empty.
    """
    resume_path = "data/user_profile.txt"
    try:
        with open(resume_path, "r", encoding="utf-8") as f:
            resume_text = f.read().strip()
    except FileNotFoundError:
        print(f"Error: Resume file not found: {resume_path}", file=sys.stderr)
        sys.exit(1)

    if not resume_text:
        print("Error: Resume is empty.", file=sys.stderr)
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
    user_degree = extract_user_degree(resume_text)
    user_seniority = extract_user_seniority(resume_text)
    user_years = extract_user_years_experience(resume_text)

    logging.info(
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
    Uses np.save/np.load for efficient binary storage.

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
            logging.info("Loading cached resume embedding...")
            return np.load(str(cache))

    # Cache miss: embed, save, and cache the hash
    logging.info("Embedding resume...")
    embeddings = embed_batch(client, [resume_text])
    embedding = embeddings[0]

    # Ensure parent directory exists
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache), embedding)
    hash_file.write_text(current_hash)
    logging.info(f"Resume embedding cached to {cache_path}")

    return embedding


def build_chroma_client(
    chroma_dir: str, rebuild: bool
) -> chromadb.PersistentClient:
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
            logging.info(f"Deleted existing collection '{CHROMA_COLLECTION_NAME}'.")
        except ValueError:
            # Collection doesn't exist — silently continue
            pass

    return chroma_client


def write_results_markdown(results: list[dict], output_path: str = "matched_jobs.md") -> None:
    """
    Write results to a Markdown file.

    Args:
        results: List of JobResult dicts with keys: id, distance, title,
                location, source_url, board_token, cleaned_description.
        output_path: Path to write the Markdown file (default: matched_jobs.md).
    """
    lines = ["# Top Matched Jobs (Reranked)\n"]
    for i, job in enumerate(results, start=1):
        title = job.get("title", "Unknown")
        board_token = job.get("board_token", "Unknown")
        url = job.get("source_url", "No URL")

        lines.append(f"## {i}. {title}")
        lines.append(f"- **Board Token:** `{board_token}`")
        lines.append(f"- **URL:** [{url}]({url})\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Results written to {output_path}")


def main() -> None:
    """Main orchestration function."""
    # Load environment variables
    load_dotenv()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Resolve database path: env var → default
    db_path = os.getenv("DB_PATH", DB_DEFAULT_PATH)

    # Validate database exists
    db_path_obj = Path(db_path)
    if not db_path_obj.exists():
        print(
            f"Error: Database not found at {db_path}. "
            "Please run the pipeline scripts first: "
            "python scripts/scrape_jobs.py && "
            "python scripts/preprocess_jobs.py && "
            "python scripts/embed_jobs.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load resume
    resume_text = load_resume()
    logging.info(f"Loaded resume ({len(resume_text)} characters).")

    # Extract user profile criteria for filtering
    query_filter = extract_user_filters(resume_text)
    logging.info("Applying filters: %s", describe_chroma_filter(query_filter))

    # Build Chroma client (never rebuild in non-interactive context)
    chroma_client = build_chroma_client(CHROMA_DEFAULT_DIR, rebuild=False)

    # Connect to database and build/sync Chroma collection
    try:
        conn = sqlite3.connect(db_path)
        try:
            logging.info("Building ChromaDB collection from embeddings...")
            collection = build_collection(
                conn, chroma_client, CHROMA_COLLECTION_NAME, ef_construction=HNSW_EF_CONSTRUCTION
            )
            logging.info(f"Collection ready with {collection.count()} jobs indexed.")

            # Sanity check: ensure we have embedded jobs
            if collection.count() == 0:
                print(
                    "Error: No embedded jobs found in the database. "
                    "Please run the pipeline scripts first:\n"
                    "  python scripts/preprocess_jobs.py\n"
                    "  python scripts/embed_jobs.py",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Create Voyage client and embed resume
            voyage_api_key = os.getenv("VOYAGE_API_KEY")
            if not voyage_api_key:
                print(
                    "Error: VOYAGE_API_KEY environment variable is not set.",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Validate Cohere API key
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if not cohere_api_key:
                print(
                    "Error: COHERE_API_KEY environment variable is not set.",
                    file=sys.stderr,
                )
                sys.exit(1)

            voyage_client = create_client(voyage_api_key)
            query_embedding = load_or_embed_resume(voyage_client, resume_text)
            logging.info(f"Resume embedding shape: {query_embedding.shape}")

            # Step 1: Dense retrieval — top 100 candidates
            logging.info(f"Querying for top {RETRIEVE_TOP_K} candidates...")
            candidates = query_collection(
                collection, query_embedding, top_k=RETRIEVE_TOP_K, ef=HNSW_EF, where=query_filter
            )

            # Step 2: Rerank — top 10 results
            logging.info(f"Reranking {len(candidates)} candidates to top {RERANK_TOP_N}...")
            results = rerank_jobs(
                resume_text, candidates, top_n=RERANK_TOP_N, api_key=cohere_api_key
            )

            # Output results
            if len(results) == 0:
                print("No matching jobs found.")
            else:
                write_results_markdown(results)

        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        print(f"Error reading database: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error building collection (corrupt embeddings?): {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
