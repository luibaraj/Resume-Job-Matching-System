#!/usr/bin/env python3
"""
Generate synthetic negative job descriptions for all resumes in synthetic_resume.csv.

For each resume, generates intentionally mismatched job descriptions:
- 1 seniority-mismatched job
- 2 responsibility-mismatched jobs

Each job is validated and repaired if needed. Deterministic fields (title, seniority, domain,
years_required) are generated using the same approach as the positives pipeline to ensure
structural consistency. Results are written to a CSV file with:
  - A unique UUID per job
  - Resume ID reference
  - All JobSkeleton fields
  - Denormalized resume metadata (seniority, domain)
  - An embedding-ready plain-text job_description
  - The mismatch_type (seniority/responsibility)
  - ISO 8601 generation timestamp

Output: data/eval/synthetic_negative_job_descriptions.csv
"""

import argparse
import csv
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Setup sys.path to import src modules and sibling script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ollama

from src.config import OLLAMA_MODEL
from eval.negative_gen.negatives_gen import (
    generate_mismatched_skeleton,
    MismatchType,
)
from eval.negative_gen.negatives_validate import validate_mismatched_skeleton
from eval.negative_gen.negatives_repair import repair_mismatched_skeleton
from eval.positive_gen.positives_validate import ResumeInfo

# Reuse helpers from the positives script
from scripts.eval.generate_synthetic_positives import (
    extract_skills_from_resume,
    build_resume_info,
    format_job_for_embedding,
)

logger = logging.getLogger(__name__)


def run_negatives_for_resume(
    resume_info: ResumeInfo,
    mismatch_type: MismatchType,
    target_count: int,
    model: str = OLLAMA_MODEL,
) -> list[dict]:
    """
    Generate mismatched job skeletons for a single resume and mismatch type.

    Runs the generate → validate → repair loop, collecting valid skeletons
    until target_count is reached or max_attempts is exhausted.

    Args:
        resume_info: ResumeInfo dict with resume_text, seniority, domain, skills, years_experience.
        mismatch_type: Type of mismatch ("seniority", "responsibility").
        target_count: Number of valid skeletons to collect.
        model: Ollama model name.

    Returns:
        List of validated JobSkeleton dicts (may be shorter than target_count
        if max_attempts was exhausted).
    """
    collected: list[dict] = []
    total_attempts = 0
    max_attempts = target_count * 10
    discard_count = 0

    logger.info(
        f"[{mismatch_type}] Starting: target={target_count}, "
        f"max_attempts={max_attempts}, model={model}"
    )

    while len(collected) < target_count:
        if total_attempts >= max_attempts:
            logger.warning(
                f"[{mismatch_type}] Safety cap reached ({max_attempts} attempts). "
                f"Collected {len(collected)}/{target_count} jobs."
            )
            break

        total_attempts += 1
        logger.info(
            f"[{mismatch_type}] Attempt {total_attempts}/{max_attempts} — "
            f"generating skeleton..."
        )

        # Step 1: Generate
        try:
            job, mismatch_context = generate_mismatched_skeleton(
                resume_info=resume_info,
                model=model,
                mismatch_type=mismatch_type,
            )
        except ValueError as e:
            logger.warning(
                f"[{mismatch_type}] Generation parse/value error: {e}"
            )
            continue
        except (ollama.RequestError, ollama.ResponseError) as e:
            logger.error(
                f"[{mismatch_type}] Ollama error during generation: {e}"
            )
            continue

        logger.info(
            f"[{mismatch_type}] Generated: title='{job['title']}', "
            f"seniority={job['seniority']}, domain={job['domain']}"
        )

        # Step 2: Validate
        try:
            validation = validate_mismatched_skeleton(
                job=job,
                resume_info=resume_info,
                model=model,
                mismatch_type=mismatch_type,
            )
        except (ollama.RequestError, ollama.ResponseError) as e:
            logger.error(
                f"[{mismatch_type}] Ollama error during validation: {e}"
            )
            continue

        if validation["passed"]:
            collected.append(job)
            logger.info(
                f"[{mismatch_type}] Validation PASSED — "
                f"collected {len(collected)}/{target_count}"
            )
            continue

        # Step 3: Repair
        failed_check = validation["failed_check"] or "structural"
        logger.warning(
            "[%s] Validation FAILED at '%s': %s",
            mismatch_type,
            failed_check,
            validation['reason'],
        )
        logger.info("[%s] Attempting repair...", mismatch_type)

        try:
            repair_result = repair_mismatched_skeleton(
                job=job,
                failed_check=failed_check,
                reason=validation["reason"],
                resume_info=resume_info,
                mismatch_context=mismatch_context,
                model=model,
                mismatch_type=mismatch_type,
            )
        except (ollama.RequestError, ollama.ResponseError) as e:
            logger.error(
                "[%s] Ollama error during repair: %s",
                mismatch_type,
                e,
            )
            discard_count += 1
            continue

        if repair_result["success"] and repair_result["job"] is not None:
            collected.append(repair_result["job"])
            logger.info(
                "[%s] Repair SUCCEEDED (%d attempt(s)) — collected %d/%d",
                mismatch_type,
                repair_result['attempts'],
                len(collected),
                target_count,
            )
        else:
            discard_count += 1
            logger.warning(
                "[%s] Repair FAILED after %d attempt(s) — discarding. Reason: %s",
                mismatch_type,
                repair_result['attempts'],
                repair_result['discard_reason'],
            )

    logger.info(
        "[%s] Done. collected=%d, total_attempts=%d, discards=%d",
        mismatch_type,
        len(collected),
        total_attempts,
        discard_count,
    )
    return collected


def main():
    """
    Main entry point: iterate resumes, run negatives pipeline, write CSV.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate synthetic negative jobs")
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
    output_csv = (
        project_root / "data" / "eval" / "synthetic_negative_job_descriptions.csv"
    )

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

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        resumes = list(reader)

    logger.info("Loaded %d resumes. Starting negatives generation...", len(resumes))

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

        # Skip empty resumes
        if not resume_text.strip():
            logger.warning(
                "[%d/%d] Empty resume text for resume_id=%s. Skipping.",
                idx,
                len(resumes),
                resume_id,
            )
            continue

        # Build ResumeInfo for the pipeline
        resume_info = build_resume_info(row)

        # Define per-resume targets: (mismatch_type, target_count)
        generation_tasks = [
            ("seniority", 1),
            ("responsibility", 2),
        ]

        for mismatch_type, target_count in generation_tasks:
            logger.info(
                "[%d/%d] Running mismatch_type=%s, target=%d",
                idx,
                len(resumes),
                mismatch_type,
                target_count,
            )
            try:
                jobs = run_negatives_for_resume(
                    resume_info=resume_info,
                    mismatch_type=mismatch_type,
                    target_count=target_count,
                )
            except Exception as e:
                logger.error(
                    "[%d/%d] Unexpected error for resume_id=%s, mismatch_type=%s: %s",
                    idx,
                    len(resumes),
                    resume_id,
                    mismatch_type,
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
                    "secondary_skills": "; ".join(
                        job.get("secondary_skills", [])
                    ),
                    "responsibilities": "; ".join(
                        job.get("responsibilities", [])
                    ),
                    "resume_seniority": resume_seniority,
                    "resume_domain": resume_domain,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "job_description": format_job_for_embedding(job),
                    "mismatch_type": mismatch_type,
                }
                all_rows.append(output_row)
                total_collected += 1

            logger.info(
                "[%d/%d] mismatch_type=%s collected %d/%d",
                idx,
                len(resumes),
                mismatch_type,
                len(jobs),
                target_count,
            )

        logger.info(
            "[%d/%d] Resume done. Total so far: %d",
            idx,
            len(resumes),
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
            "mismatch_type",
        ]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        logger.info("Wrote %d rows to %s", len(all_rows), output_csv)
    else:
        logger.warning("No jobs collected. CSV will be empty.")

    logger.info("Generation complete. Total collected: %d", total_collected)


if __name__ == "__main__":
    main()
