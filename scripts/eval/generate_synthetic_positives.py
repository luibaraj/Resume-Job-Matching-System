#!/usr/bin/env python3
"""
Generate synthetic positive job descriptions for all resumes in synthetic_resume.csv.

Iterates over each resume, runs the positives pipeline (generate → validate → repair)
with target_count=5, and writes all collected job skeletons to a CSV file with:
  - A unique UUID per job
  - Resume ID reference
  - All JobSkeleton fields
  - Denormalized resume metadata (seniority, domain)
  - An embedding-ready plain-text job_description
  - ISO 8601 generation timestamp

Output: data/eval/synthetic_job_descriptions.csv
"""

import argparse
import csv
import logging
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Setup sys.path to import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from src.config import YEARS_UNKNOWN
from src.eval.positive_gen.positives_pipeline import run_pipeline
from src.eval.positive_gen.positives_validate import ResumeInfo
from src.llm_extraction import extract_years_with_llm

logger = logging.getLogger(__name__)


def extract_skills_from_resume(resume_text: str) -> list[str]:
    """
    Extract primary skills from resume text.

    Searches for a "Skills:" or "SKILLS:" section, splits on commas,
    strips whitespace, and returns up to 4 items.

    Args:
        resume_text: Full resume text.

    Returns:
        List of skill strings (may be empty).
    """
    # Find a "Skills:" section (case-insensitive)
    match = re.search(
        r"(?:skills|SKILLS|Skills)\s*:([^,\n]*(?:,[^,\n]*)*)",
        resume_text,
        re.IGNORECASE,
    )
    if not match:
        return []

    skills_str = match.group(1)
    skills = [s.strip() for s in skills_str.split(",") if s.strip()]
    return skills[:4]  # Take first 4


def build_resume_info(row: dict) -> ResumeInfo:
    """
    Build a ResumeInfo dict from a CSV row.

    Args:
        row: A single row from synthetic_resume.csv as a dict
             with keys: id, resume, seniority, domain.

    Returns:
        ResumeInfo TypedDict compatible with the validation pipeline.
    """
    resume_text = row["resume"]
    seniority_str = row["seniority"].strip()

    # Normalize seniority: capitalize first letter
    seniority = seniority_str.title()

    # Extract years of experience, with seniority-based fallback
    years = extract_years_with_llm(resume_text, model='llama3.2')
    if years == YEARS_UNKNOWN:
        # Fallback map based on seniority
        fallback_map = {
            "Junior": 1,
            "Mid": 3,
            "Senior": 6,
            "Staff": 8,
        }
        years = fallback_map.get(seniority, 3)

    # Extract primary skills
    primary_skills = extract_skills_from_resume(resume_text)
    if not primary_skills:
        # Fallback: empty list; validation/repair will handle it
        primary_skills = []

    # Domain directly from CSV
    domain = row["domain"].strip()

    return {
        "seniority": seniority,
        "years_experience": years,
        "primary_skills": primary_skills,
        "domain": domain,
        "resume_text": resume_text,
    }


def format_job_for_embedding(job: dict) -> str:
    """
    Format a JobSkeleton dict as embedding-ready plain text.

    Args:
        job: JobSkeleton dict with keys: title, seniority, years_required,
             domain, primary_skills, secondary_skills, responsibilities.

    Returns:
        Embedding-ready plain-text string.
    """
    primary_skills_str = ", ".join(job.get("primary_skills", []))
    secondary_skills_str = ", ".join(job.get("secondary_skills", []))
    responsibilities_str = "; ".join(job.get("responsibilities", []))

    return (
        f"{job['title']}. {job['seniority']} level. Domain: {job['domain']}. "
        f"Requires {job['years_required']} years of experience. "
        f"Skills: {primary_skills_str}. "
        f"Additional skills: {secondary_skills_str}. "
        f"Responsibilities: {responsibilities_str}."
    )


def main():
    """
    Main entry point: iterate resumes, run pipeline, write CSV.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate synthetic positive jobs")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Configure logging (after argparse)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent.parent
    input_csv = project_root / "data" / "eval" / "synthetic_resume.csv"
    output_csv = project_root / "data" / "eval" / "synthetic_job_descriptions.csv"

    # Verify input exists
    if not input_csv.exists():
        logger.error("Input file not found: %s", input_csv)
        sys.exit(1)

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reading resumes from: %s", input_csv)
    logger.info("Writing output to: %s", output_csv)

    all_rows = []
    total_collected = 0
    total_discarded = 0

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        resumes = list(reader)

    logger.info("Loaded %d resumes. Starting pipeline...", len(resumes))

    for idx, row in enumerate(resumes, 1):
        resume_id = row["id"]
        resume_text = row["resume"]
        resume_seniority = row["seniority"].strip()
        resume_domain = row["domain"].strip()

        logger.info(
            "[%d/%d] Processing resume_id=%s, seniority=%s, domain=%s",
            idx,
            len(resumes),
            resume_id,
            resume_seniority,
            resume_domain,
        )

        # Build ResumeInfo for the pipeline
        resume_info = build_resume_info(row)

        # Run the pipeline (target 5 jobs per resume)
        try:
            jobs = run_pipeline(
                resume_text=resume_text,
                resume_info=resume_info,
                target_count=5,
                output_path=None,  # Don't dump per-resume JSON
            )
        except Exception as e:
            logger.error(
                "[%d/%d] Pipeline failed for resume_id=%s: %s",
                idx,
                len(resumes),
                resume_id,
                e,
            )
            continue

        # Convert each job to an output row
        for job in jobs:
            output_row = {
                "id": str(uuid.uuid4()),
                "resume_id": resume_id,
                "title": job.get("title", ""),
                "seniority": job.get("seniority", ""),
                "years_required": job.get("years_required", ""),
                "domain": job.get("domain", ""),
                "primary_skills": "; ".join(job.get("primary_skills", [])),
                "secondary_skills": "; ".join(job.get("secondary_skills", [])),
                "responsibilities": "; ".join(job.get("responsibilities", [])),
                "resume_seniority": resume_seniority,
                "resume_domain": resume_domain,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "job_description": format_job_for_embedding(job),
            }
            all_rows.append(output_row)
            total_collected += 1

        logger.info(
            "[%d/%d] Collected %d jobs. Total so far: %d",
            idx,
            len(resumes),
            len(jobs),
            total_collected,
        )

    # Write CSV
    logger.info("Writing %d jobs to %s...", total_collected, output_csv)
    if all_rows:
        fieldnames = [
            "id",
            "resume_id",
            "title",
            "seniority",
            "years_required",
            "domain",
            "primary_skills",
            "secondary_skills",
            "responsibilities",
            "resume_seniority",
            "resume_domain",
            "generated_at",
            "job_description",
        ]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        logger.info("Wrote %d rows to %s", len(all_rows), output_csv)
    else:
        logger.warning("No jobs collected. CSV will be empty.")

    logger.info(
        "Pipeline complete. Total collected: %d, total discarded: %d",
        total_collected,
        total_discarded,
    )


if __name__ == "__main__":
    main()
