"""
Seniority-mismatched negatives generation module.

Exports key classes and functions for generating, validating, and repairing
seniority-mismatched synthetic job descriptions from resumes.

The module is organized into three stages:
1. negatives_gen — Generate a skeleton with mismatched seniority
2. negatives_validate — Validate the skeleton against four rule sets
3. negatives_repair — Repair failed skeletons through up to 2 LLM attempts

A future negatives_pipeline module will orchestrate these stages together
and support additional negative types (skill gaps, domain mismatches, etc).
"""

from .negatives_gen import (
    SENIORITY_ORDER,
    generate_mismatched_skeleton,
    get_target_seniority,
)
from .negatives_repair import (
    RepairResult,
    repair_mismatched_skeleton,
)
from .negatives_validate import (
    validate_mismatched_skeleton,
    validate_seniority_mismatch,
    validate_skill_domain_overlap,
)

__all__ = [
    "SENIORITY_ORDER",
    "get_target_seniority",
    "generate_mismatched_skeleton",
    "validate_mismatched_skeleton",
    "validate_seniority_mismatch",
    "validate_skill_domain_overlap",
    "RepairResult",
    "repair_mismatched_skeleton",
]
