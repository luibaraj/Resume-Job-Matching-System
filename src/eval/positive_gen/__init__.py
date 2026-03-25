"""
Synthetic positives generation pipeline.

Exports key classes and functions for generating, validating, repairing,
and orchestrating synthetic job descriptions from resumes.
"""

from .positives_gen import (
    JobSkeleton,
    generate_job_skeleton,
    parse_skeleton_response,
)
from .positives_validate import (
    ResumeInfo,
    validate_job_skeleton,
)
from .positives_repair import (
    RepairResult,
    repair_job_skeleton,
)
from .positives_pipeline import run_pipeline

__all__ = [
    "JobSkeleton",
    "generate_job_skeleton",
    "parse_skeleton_response",
    "ResumeInfo",
    "validate_job_skeleton",
    "RepairResult",
    "repair_job_skeleton",
    "run_pipeline",
]
