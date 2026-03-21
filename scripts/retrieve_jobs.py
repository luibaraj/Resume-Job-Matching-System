"""
Orchestration script: embed a resume and retrieve top matching jobs from ChromaDB.

Reads resume from stdin or RESUME_FILE and outputs results to results.md.
"""

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
)
from embedding import create_client, embed_batch
from retrieval import build_collection, query_collection


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


def embed_resume(client, resume_text: str) -> np.ndarray:
    """
    Embed a single resume text using Voyage AI.

    The resume is a single document, so the batch size is always 1.
    Voyage's `input_type=None` is already configured in embed_batch,
    matching the vector space of job embeddings.

    Note: Very long resumes may be silently truncated by the Voyage API
    due to context window limits. This is acceptable for v1 of this tool.

    Args:
        client: Voyage AI client.
        resume_text: Resume text to embed.

    Returns:
        Embedding vector (numpy array, float32).
    """
    embeddings = embed_batch(client, [resume_text])
    return embeddings[0]


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


def write_results_markdown(results: list[dict], output_path: str = "results.md") -> None:
    """
    Write results to a Markdown file.

    Args:
        results: List of JobResult dicts with keys: id, distance, title,
                location, source_url, board_token, cleaned_description.
        output_path: Path to write the Markdown file (default: results.md).
    """
    lines = ["# Top Matching Jobs\n"]
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
            voyage_api_key = __import__("os").getenv("VOYAGE_API_KEY")
            if not voyage_api_key:
                print(
                    "Error: VOYAGE_API_KEY environment variable is not set.",
                    file=sys.stderr,
                )
                sys.exit(1)

            voyage_client = create_client(voyage_api_key)
            logging.info("Embedding resume...")
            query_embedding = embed_resume(voyage_client, resume_text)
            logging.info(f"Resume embedding shape: {query_embedding.shape}")

            # Query collection
            logging.info(f"Querying for top {RETRIEVE_TOP_K} matches...")
            results = query_collection(
                collection, query_embedding, top_k=RETRIEVE_TOP_K, ef=HNSW_EF
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
