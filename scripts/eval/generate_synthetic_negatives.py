#!/usr/bin/env python3
"""
Generate synthetic negative job descriptions for all resumes in synthetic_resume.csv.

For each resume, generates intentionally mismatched job descriptions:
- 1 seniority-mismatched job
- 1 domain-mismatched job
- 3 responsibility-mismatched jobs

Each job is validated and repaired if needed. Results are written to a CSV file with:
  - A unique UUID per job
  - Resume ID reference
  - All JobSkeleton fields
  - Denormalized resume metadata (seniority, domain)
  - An embedding-ready plain-text job_description
  - The mismatch_type (seniority/domain/responsibility)
  - ISO 8601 generation timestamp

Output: data/eval/synthetic_negative_job_descriptions.csv
"""

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

from config import OLLAMA_MODEL
from eval.negative_gen.negatives_gen import (
    generate_mismatched_skeleton,
    MismatchType,
)
from eval.negative_gen.negatives_validate import validate_mismatched_skeleton
from eval.negative_gen.negatives_repair import repair_mismatched_skeleton
from eval.positive_gen.positives_validate import ResumeInfo

# Reuse helpers from the positives script
from generate_synthetic_positives import (
    extract_skills_from_resume,
    build_resume_info,
    format_job_for_embedding,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_negatives_for_resume(
    resume_text: str,
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
        resume_text: Full resume text.
        resume_info: ResumeInfo dict with seniority, domain, skills, years_experience.
        mismatch_type: Type of mismatch ("seniority", "domain", "responsibility").
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
                resume_text=resume_text,
                resume_seniority=resume_info["seniority"],
                model=model,
                mismatch_type=mismatch_type,
                resume_domain=resume_info["domain"],
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
            f"[{mismatch_type}] Validation FAILED at '{failed_check}': "
            f"{validation['reason']}"
        )
        logger.info(f"[{mismatch_type}] Attempting repair...")

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
                f"[{mismatch_type}] Ollama error during repair: {e}"
            )
            discard_count += 1
            continue

        if repair_result["success"] and repair_result["job"] is not None:
            collected.append(repair_result["job"])
            logger.info(
                f"[{mismatch_type}] Repair SUCCEEDED "
                f"({repair_result['attempts']} attempt(s)) — "
                f"collected {len(collected)}/{target_count}"
            )
        else:
            discard_count += 1
            logger.warning(
                f"[{mismatch_type}] Repair FAILED after "
                f"{repair_result['attempts']} attempt(s) — discarding. "
                f"Reason: {repair_result['discard_reason']}"
            )

    logger.info(
        f"[{mismatch_type}] Done. collected={len(collected)}, "
        f"total_attempts={total_attempts}, discards={discard_count}"
    )
    return collected


def main():
    """
    Main entry point: iterate resumes, run negatives pipeline, write CSV.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    input_csv = project_root / "data" / "eval" / "synthetic_resume.csv"
    output_csv = (
        project_root / "data" / "eval" / "synthetic_negative_job_descriptions.csv"
    )

    # Verify input exists
    if not input_csv.exists():
        logger.error(f"Input file not found: {input_csv}")
        sys.exit(1)

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading resumes from: {input_csv}")
    logger.info(f"Writing output to: {output_csv}")

    all_rows = []
    total_collected = 0

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        resumes = list(reader)

    logger.info(f"Loaded {len(resumes)} resumes. Starting negatives generation...")

    for idx, row in enumerate(resumes, 1):
        resume_id = row["id"]
        resume_text = row["resume"]
        resume_seniority = row["seniority"].strip()
        resume_domain = row["domain"].strip()

        logger.info(
            f"[{idx}/{len(resumes)}] Processing resume_id={resume_id}, "
            f"seniority={resume_seniority}, domain={resume_domain}"
        )

        # Skip empty resumes
        if not resume_text.strip():
            logger.warning(
                f"[{idx}/{len(resumes)}] Empty resume text for "
                f"resume_id={resume_id}. Skipping."
            )
            continue

        # Build ResumeInfo for the pipeline
        resume_info = build_resume_info(row)

        # Define per-resume targets: (mismatch_type, target_count)
        generation_tasks = [
            ("seniority", 1),
            ("domain", 1),
            ("responsibility", 3),
        ]

        for mismatch_type, target_count in generation_tasks:
            logger.info(
                f"[{idx}/{len(resumes)}] Running mismatch_type={mismatch_type}, "
                f"target={target_count}"
            )
            try:
                jobs = run_negatives_for_resume(
                    resume_text=resume_text,
                    resume_info=resume_info,
                    mismatch_type=mismatch_type,
                    target_count=target_count,
                )
            except Exception as e:
                logger.error(
                    f"[{idx}/{len(resumes)}] Unexpected error for "
                    f"resume_id={resume_id}, mismatch_type={mismatch_type}: {e}"
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
                f"[{idx}/{len(resumes)}] mismatch_type={mismatch_type} → "
                f"collected {len(jobs)}/{target_count}"
            )

        logger.info(
            f"[{idx}/{len(resumes)}] Resume done. Total so far: {total_collected}"
        )

    # Write CSV
    logger.info(f"Writing {total_collected} jobs to {output_csv}...")
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
        logger.info(f"✓ Wrote {len(all_rows)} rows to {output_csv}")
    else:
        logger.warning("No jobs collected. CSV will be empty.")

    logger.info(f"Generation complete. Total collected: {total_collected}")


if __name__ == "__main__":
    main()
