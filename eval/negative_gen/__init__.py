"""
Negatives generation module.

Exports key classes and functions for generating, validating, and repairing
synthetic job descriptions with intentional mismatches to resumes.
Supports two mismatch types: seniority and responsibility.

The module is organized into three stages:
1. negatives_gen — Generate a skeleton with an intentional mismatch
2. negatives_validate — Validate the skeleton against rule sets appropriate to the mismatch type
3. negatives_repair — Repair failed skeletons through up to 2 LLM attempts

Deterministic fields (title, seniority, domain, years_required) are generated using
the same approach as the positives pipeline to ensure structural consistency.
"""

from .negatives_gen import (
    SENIORITY_ORDER,
    MismatchType,
    generate_mismatched_skeleton,
    get_target_seniority,
)
from .negatives_repair import (
    repair_mismatched_skeleton,
)
from .negatives_validate import (
    validate_mismatched_skeleton,
    validate_seniority_mismatch,
    validate_skill_domain_overlap,
    validate_responsibility_mismatch,
)

__all__ = [
    "SENIORITY_ORDER",
    "MismatchType",
    "get_target_seniority",
    "generate_mismatched_skeleton",
    "validate_mismatched_skeleton",
    "validate_seniority_mismatch",
    "validate_skill_domain_overlap",
    "validate_responsibility_mismatch",
    "repair_mismatched_skeleton",
]
