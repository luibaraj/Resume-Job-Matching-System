"""
Synthetic positives pipeline — Orchestrator.

Drives the full generate → validate → repair → collect loop to produce
a target number of valid synthetic job skeletons from a resume. Implements
a safety cap on total generation attempts to prevent infinite loops.

This is the top-level entry point for the synthetic positives pipeline.
All LLM calls are delegated to positives_gen, positives_validate, and
positives_repair.
"""

import json
import sys
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import OLLAMA_MODEL
from .positives_gen import JobSkeleton, generate_job_skeleton
from .positives_validate import ResumeInfo, validate_job_skeleton
from .positives_repair import repair_job_skeleton


def run_pipeline(
    resume_text: str,
    resume_info: ResumeInfo,
    target_count: int = 5,
    model: str = OLLAMA_MODEL,
    output_path: str | None = "data/synthetic_positives.json",
) -> list[JobSkeleton]:
    """
    Orchestrate the full synthetic positives generation pipeline.

    Repeatedly generates, validates, and (if needed) repairs job skeletons
    until target_count valid skeletons are collected, or until max_attempts
    is exhausted.

    Args:
        resume_text: Full resume text used to generate each skeleton.
        resume_info: Structured resume metadata for validation and repair.
        target_count: Number of valid skeletons to collect (default: 5).
        model: Ollama model name (default: OLLAMA_MODEL from config).
        output_path: If not None, write collected skeletons to this JSON
                     file after the loop (default: "data/synthetic_positives.json").

    Returns:
        List of validated JobSkeleton dicts. May be shorter than target_count
        if max_attempts was exhausted before enough skeletons were collected.
    """
    # Guard against empty resume
    if not resume_text.strip():
        print("[pipeline] Error: resume_text is empty. Returning empty list.")
        return []

    collected: list[JobSkeleton] = []
    total_attempts = 0
    max_attempts = target_count * 10
    discard_count = 0
    repair_success_count = 0

    print(
        f"[pipeline] Starting: target={target_count}, "
        f"max_attempts={max_attempts}, model={model}"
    )

    while len(collected) < target_count:
        if total_attempts >= max_attempts:
            print(
                f"[pipeline] Safety cap reached ({max_attempts} attempts). "
                f"Collected {len(collected)}/{target_count} jobs."
            )
            break

        total_attempts += 1
        print(
            f"[pipeline] Attempt {total_attempts}/{max_attempts} — "
            f"generating skeleton..."
        )

        # Step 1: Generate
        try:
            job = generate_job_skeleton(resume_info, model)
        except ValueError as e:
            print(f"[pipeline] Generation parse error: {e}")
            continue
        except (ollama.RequestError, ollama.ResponseError) as e:
            print(f"[pipeline] Ollama error during generation: {e}")
            continue

        print(
            f"[pipeline] Generated: title='{job['title']}', "
            f"seniority={job['seniority']}, domain={job['domain']}"
        )

        # Step 2: Validate
        try:
            validation = validate_job_skeleton(job, resume_info, model)
        except (ollama.RequestError, ollama.ResponseError) as e:
            print(f"[pipeline] Ollama error during validation: {e}")
            continue

        if validation["passed"]:
            collected.append(job)
            print(
                f"[pipeline] Validation PASSED — "
                f"collected {len(collected)}/{target_count}"
            )
            continue

        # Step 3: Repair
        failed_check = validation["failed_check"] or "structural"
        print(
            f"[pipeline] Validation FAILED at '{failed_check}': "
            f"{validation['reason']}"
        )
        print("[pipeline] Attempting repair...")

        try:
            repair_result = repair_job_skeleton(
                job,
                failed_check,
                validation["reason"],
                resume_info,
                model,
            )
        except (ollama.RequestError, ollama.ResponseError) as e:
            print(f"[pipeline] Ollama error during repair: {e}")
            discard_count += 1
            continue

        if repair_result["success"] and repair_result["job"] is not None:
            repair_success_count += 1
            collected.append(repair_result["job"])
            print(
                f"[pipeline] Repair SUCCEEDED ({repair_result['attempts']} "
                f"attempt(s)) — collected {len(collected)}/{target_count}"
            )
        else:
            discard_count += 1
            print(
                f"[pipeline] Repair FAILED after {repair_result['attempts']} "
                f"attempt(s) — discarding. Reason: {repair_result['discard_reason']}"
            )

    # Summary
    print(
        f"[pipeline] Done. collected={len(collected)}, "
        f"total_attempts={total_attempts}, discards={discard_count}, "
        f"repairs_succeeded={repair_success_count}"
    )

    # Optional JSON output
    if output_path and collected:
        try:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(collected, f, indent=2)
            print(f"[pipeline] Wrote {len(collected)} jobs to {output_path}")
        except OSError as e:
            print(f"[pipeline] Warning: could not write to {output_path}: {e}")

    return collected
