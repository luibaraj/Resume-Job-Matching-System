#!/usr/bin/env python3
"""
Stratify and split synthetic eval data into tune/test sets at the resume level.

Reads 3 source CSVs (resumes, positives, negatives) and outputs:
  - data/eval/tune/resumes.csv, tune/positives.csv
  - data/eval/test/resumes.csv, test/positives.csv, test/negatives.csv
  - data/eval/split_summary.json

Stratification is by (seniority, domain) on resumes; split is 30 tune / 20 test using
the Hamilton (largest remainder) method for proportional allocation.
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42
TUNE_N = 30
TEST_N = 20

logger = logging.getLogger(__name__)


def load_data(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and normalize synthetic eval data from CSVs."""
    eval_dir = project_root / "data" / "eval"

    logger.info("Loading data from %s", eval_dir)
    resumes = pd.read_csv(eval_dir / "synthetic_resume.csv")
    positives = pd.read_csv(eval_dir / "synthetic_job_descriptions.csv")
    negatives = pd.read_csv(eval_dir / "synthetic_negative_job_descriptions.csv")

    # Normalize seniority: strip whitespace, lowercase
    resumes["seniority"] = resumes["seniority"].str.strip().str.lower()
    resumes["domain"] = resumes["domain"].str.strip().str.lower()

    # Build flat stratification key: "junior_backend", "senior_data", etc.
    resumes["strata_key"] = resumes["seniority"] + "_" + resumes["domain"]

    logger.info("Loaded %d resumes, %d positives, %d negatives", len(resumes), len(positives), len(negatives))
    return resumes, positives, negatives


def _largest_remainder_allocate(
    strata_counts: dict[str, int],
    total_tune: int,
) -> dict[str, int]:
    """
    Allocate total_tune items across strata proportionally using the
    Hamilton (largest remainder) method.

    Args:
        strata_counts: dict mapping stratum key to count
        total_tune: target number of items in tune set

    Returns:
        dict mapping stratum key to allocated tune count
    """
    total = sum(strata_counts.values())
    ratio = total_tune / total

    # Floor step: assign floor(count * ratio) to each stratum
    floors = {k: int(v * ratio) for k, v in strata_counts.items()}
    floor_sum = sum(floors.values())
    remainder_needed = total_tune - floor_sum

    # Remainder step: allocate remaining slots to strata with largest fractional remainders
    # Tie-break by stratum size (larger first), then alphabetical key
    remainders = {k: (v * ratio) - floors[k] for k, v in strata_counts.items()}
    sorted_keys = sorted(
        strata_counts.keys(),
        key=lambda k: (-remainders[k], -strata_counts[k], k),
    )

    allocations = dict(floors)
    for k in sorted_keys[:remainder_needed]:
        allocations[k] += 1

    # Safety: no stratum can receive more tune slots than it has resumes
    for k, v in strata_counts.items():
        allocations[k] = min(allocations[k], v)

    return allocations


def compute_stratified_split(
    resumes_df: pd.DataFrame,
    tune_n: int = TUNE_N,
    seed: int = RANDOM_SEED,
) -> tuple[set[int], set[int]]:
    """
    Compute stratified split at resume level.

    Args:
        resumes_df: resumes DataFrame with strata_key column
        tune_n: target number of resumes in tune set
        seed: random seed for within-stratum shuffle

    Returns:
        tuple of (tune_ids set, test_ids set)
    """
    # Compute allocation per stratum
    strata_groups = resumes_df.groupby("strata_key", as_index=False)
    strata_counts = {k: len(v) for k, v in resumes_df.groupby("strata_key")}
    allocations = _largest_remainder_allocate(strata_counts, tune_n)

    # Within each stratum, shuffle and split
    tune_ids = set()
    for key in strata_counts.keys():
        group = resumes_df[resumes_df["strata_key"] == key]
        shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_tune = allocations[key]
        tune_ids.update(shuffled.iloc[:n_tune]["id"].tolist())

    test_ids = set(resumes_df["id"].tolist()) - tune_ids

    logger.info("Split: %d tune, %d test", len(tune_ids), len(test_ids))
    return tune_ids, test_ids


def build_outputs(
    resumes_df: pd.DataFrame,
    positives_df: pd.DataFrame,
    negatives_df: pd.DataFrame,
    tune_ids: set[int],
    test_ids: set[int],
) -> dict[str, pd.DataFrame]:
    """
    Build output DataFrames by filtering jobs by resume_id membership.

    Args:
        resumes_df: resumes (has strata_key column to drop)
        positives_df: positive job descriptions
        negatives_df: negative job descriptions
        tune_ids: set of resume IDs in tune set
        test_ids: set of resume IDs in test set

    Returns:
        dict with keys: tune_resumes, tune_positives, test_resumes, test_positives, test_negatives
    """
    # Ensure resume_id is int for isin() comparison
    positives_df["resume_id"] = positives_df["resume_id"].astype(int)
    negatives_df["resume_id"] = negatives_df["resume_id"].astype(int)

    tune_resumes = resumes_df[resumes_df["id"].isin(tune_ids)].drop(columns=["strata_key"])
    test_resumes = resumes_df[resumes_df["id"].isin(test_ids)].drop(columns=["strata_key"])
    tune_positives = positives_df[positives_df["resume_id"].isin(tune_ids)]
    test_positives = positives_df[positives_df["resume_id"].isin(test_ids)]
    test_negatives = negatives_df[negatives_df["resume_id"].isin(test_ids)]

    return {
        "tune_resumes": tune_resumes,
        "tune_positives": tune_positives,
        "test_resumes": test_resumes,
        "test_positives": test_positives,
        "test_negatives": test_negatives,
    }


def write_outputs(outputs: dict[str, pd.DataFrame], eval_dir: Path) -> None:
    """
    Write output DataFrames to CSV files.

    Creates tune/ and test/ directories and writes all 5 CSVs (idempotent).
    """
    tune_dir = eval_dir / "tune"
    test_dir = eval_dir / "test"
    tune_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    outputs["tune_resumes"].to_csv(tune_dir / "resumes.csv", index=False)
    outputs["tune_positives"].to_csv(tune_dir / "positives.csv", index=False)
    outputs["test_resumes"].to_csv(test_dir / "resumes.csv", index=False)
    outputs["test_positives"].to_csv(test_dir / "positives.csv", index=False)
    outputs["test_negatives"].to_csv(test_dir / "negatives.csv", index=False)

    logger.info("Wrote tune/: %d resumes, %d positives", len(outputs['tune_resumes']), len(outputs['tune_positives']))
    logger.info("Wrote test/: %d resumes, %d positives, %d negatives", len(outputs['test_resumes']), len(outputs['test_positives']), len(outputs['test_negatives']))


def build_summary(
    resumes_df: pd.DataFrame,
    tune_ids: set[int],
    test_ids: set[int],
) -> dict:
    """
    Build summary report of stratified split.

    Returns:
        dict with totals, strata breakdown, and warnings
    """
    tune_resumes = resumes_df[resumes_df["id"].isin(tune_ids)]
    test_resumes = resumes_df[resumes_df["id"].isin(test_ids)]

    # Strata breakdown
    strata_info = {}
    all_strata = set(resumes_df["strata_key"].unique())

    for stratum in sorted(all_strata):
        stratum_resumes = resumes_df[resumes_df["strata_key"] == stratum]
        total = len(stratum_resumes)
        tune = len(stratum_resumes[stratum_resumes["id"].isin(tune_ids)])
        test = len(stratum_resumes[stratum_resumes["id"].isin(test_ids)])

        info = {"total": total, "tune": tune, "test": test}
        if tune == 0 or test == 0:
            side = "tune" if test == 0 else "test"
            info["warning"] = f"singleton — not represented in {side}"
        strata_info[stratum] = info

    # Warnings
    warnings = []
    for stratum, info in strata_info.items():
        if "warning" in info:
            warnings.append(f"Stratum '{stratum}' has {info['total']} resume(s) and is absent from the {['tune', 'test'][info['tune'] > 0]} set.")

    return {
        "random_seed": RANDOM_SEED,
        "generated_at": datetime.now().isoformat(),
        "totals": {
            "tune": {
                "resumes": len(tune_resumes),
            },
            "test": {
                "resumes": len(test_resumes),
            },
        },
        "strata": strata_info,
        "warnings": warnings,
    }


def write_summary(summary: dict, path: Path) -> None:
    """Write summary report to JSON file."""
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote summary to %s", path)


def print_summary(summary: dict) -> None:
    """Print human-readable summary to stdout."""
    print("\n" + "=" * 60)
    print("Split Summary")
    print("=" * 60)
    print(f"Tune: {summary['totals']['tune']['resumes']} resumes")
    print(f"Test: {summary['totals']['test']['resumes']} resumes")

    print("\nStrata Distribution:")
    print(f"{'Stratum':<30} {'Total':>6} {'Tune':>6} {'Test':>6} {'Note':<20}")
    print("-" * 70)

    for stratum in sorted(summary["strata"].keys()):
        info = summary["strata"][stratum]
        note = "  [WARN: absent from tune]" if info.get("warning") else ""
        print(f"{stratum:<30} {info['total']:>6} {info['tune']:>6} {info['test']:>6} {note}")

    if summary["warnings"]:
        print("\nWarnings:")
        for warning in summary["warnings"]:
            print(f"  ⚠️  {warning}")

    print("=" * 60 + "\n")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Stratify and split eval data into tune/test sets")
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

    # Resolve project root from script location
    project_root = Path(__file__).resolve().parent.parent.parent
    eval_dir = project_root / "data" / "eval"

    # Load, split, and write
    resumes, positives, negatives = load_data(project_root)
    tune_ids, test_ids = compute_stratified_split(resumes)
    outputs = build_outputs(resumes, positives, negatives, tune_ids, test_ids)
    write_outputs(outputs, eval_dir)

    # Build and write summary
    summary = build_summary(resumes, tune_ids, test_ids)
    write_summary(summary, eval_dir / "split_summary.json")
    print_summary(summary)


if __name__ == "__main__":
    main()
