"""
Multi-type negatives generation module.

Exports key classes and functions for generating, validating, and repairing
synthetic job descriptions with intentional mismatches to resumes.
Supports three mismatch types: seniority, domain, and responsibility.

The module is organized into three stages:
1. negatives_gen — Generate a skeleton with an intentional mismatch
2. negatives_validate — Validate the skeleton against rule sets appropriate to the mismatch type
3. negatives_repair — Repair failed skeletons through up to 2 LLM attempts

A future negatives_pipeline module will orchestrate these stages together.
"""

from .negatives_gen import (
    SENIORITY_ORDER,
    DOMAIN_ORDER,
    MismatchType,
    generate_mismatched_skeleton,
    get_target_seniority,
    get_target_domain,
)
from .negatives_repair import (
    RepairResult,
    repair_mismatched_skeleton,
)
from .negatives_validate import (
    validate_mismatched_skeleton,
    validate_seniority_mismatch,
    validate_skill_domain_overlap,
    validate_domain_mismatch,
    validate_responsibility_mismatch,
)

__all__ = [
    "SENIORITY_ORDER",
    "DOMAIN_ORDER",
    "MismatchType",
    "get_target_seniority",
    "get_target_domain",
    "generate_mismatched_skeleton",
    "validate_mismatched_skeleton",
    "validate_seniority_mismatch",
    "validate_skill_domain_overlap",
    "validate_domain_mismatch",
    "validate_responsibility_mismatch",
    "RepairResult",
    "repair_mismatched_skeleton",
]
