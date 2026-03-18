#!/usr/bin/env python3
"""
Standalone pipeline runner: replaces Airflow DAG with a simple Python script.

Orchestrates 9 sequential pipeline steps:
  collect → preprocess → filter → extract → embed → retrieve → rerank → generate → evaluate

Features:
  - Configurable step selection (--steps / --skip-steps)
  - Automatic retries with exponential backoff (default: 1 retry, 5-min delay)
  - Env var loading from .env file
  - Proper sys.argv handling for argparse-based steps (evaluation)
  - Comprehensive logging
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Add project root to sys.path so src.* modules can be imported
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import all pipeline step functions
from src.collection import main as collection_main
from src.preprocessing import main as preprocessing_main
from src.role_filter import main as role_filter_main
from src.extraction import main as extraction_main
from src.embedding import main as embedding_main
from src.retrieval import main as retrieval_main
from src.reranking import main as reranking_main
from src.generation import main as generation_main
from src.evaluation import main as evaluation_main


# --- Env loading (replicate DAG's _load_dotenv) ---
def _load_dotenv(path: str) -> dict[str, str]:
    """Load environment variables from .env file."""
    env = {}
    try:
        with open(os.path.abspath(path)) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return env


def _apply_dotenv(env_dict: dict[str, str]) -> None:
    """Apply loaded .env vars to os.environ, with os.environ taking precedence."""
    for key, val in env_dict.items():
        os.environ.setdefault(key, val)


# --- Step registry ---
@dataclass
class Step:
    """A pipeline step."""
    name: str
    fn: Callable[[], None]
    needs_argv_patch: bool = False


STEPS: tuple[Step, ...] = (
    Step("collect", collection_main),
    Step("preprocess", preprocessing_main),
    Step("filter", role_filter_main),
    Step("extract", extraction_main),
    Step("embed", embedding_main),
    Step("retrieve", retrieval_main),
    Step("rerank", reranking_main),
    Step("generate", generation_main),
    Step("evaluate", evaluation_main, needs_argv_patch=True),
)

STEP_NAMES = [s.name for s in STEPS]


# --- Logging setup ---
def _setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging for the pipeline runner."""
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger("pipeline")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(handler)

    return logger


# --- Step invocation with SystemExit wrapping and sys.argv patching ---
def _invoke(step: Step) -> None:
    """
    Call a step's main() function with proper error handling and sys.argv patching.

    - Wraps SystemExit as RuntimeError (from DAG's _wrap)
    - Patches sys.argv for evaluation to avoid argparse collision (from DAG's _wrap_evaluation)
    """
    original_argv = sys.argv[:]
    if step.needs_argv_patch:
        sys.argv = ["evaluation"]

    try:
        step.fn()
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(
                f"Step '{step.name}' exited with code {e.code}"
            ) from e
        # Exit code 0 is treated as success; we silently accept it
    finally:
        if step.needs_argv_patch:
            sys.argv = original_argv


# --- Retry loop ---
def _run_step(
    step: Step, logger: logging.Logger, retries: int, retry_delay: float
) -> None:
    """
    Execute a step with retry logic.

    Args:
        step: The step to execute
        logger: Logger instance
        retries: Number of retries (0 means no retries, 1 attempt total)
        retry_delay: Seconds to wait between retries
    """
    for attempt in range(retries + 1):
        try:
            _invoke(step)
            return  # Success
        except Exception as exc:
            if attempt < retries:
                logger.warning(
                    f"Step '{step.name}' failed (attempt {attempt + 1}/{retries + 1}): {exc}. "
                    f"Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Step '{step.name}' failed after {retries + 1} attempt(s). Aborting pipeline."
                )
                raise


# --- CLI argument parsing ---
def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the job matching pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Run only preprocessing
  python scripts/run_pipeline.py --steps preprocess

  # Run all steps except evaluation
  python scripts/run_pipeline.py --skip-steps evaluate

  # Run with custom retry config
  python scripts/run_pipeline.py --retries 2 --retry-delay 60

  # Load from custom .env file
  python scripts/run_pipeline.py --env-file /path/to/.env
        """,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--steps",
        type=str,
        help=f"Comma-separated list of steps to run. Choices: {', '.join(STEP_NAMES)}",
    )
    group.add_argument(
        "--skip-steps",
        type=str,
        help=f"Comma-separated list of steps to skip. Choices: {', '.join(STEP_NAMES)}",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Number of retries per step (default: 1)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=300,
        help="Delay in seconds between retries (default: 300)",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file (default: <project_root>/.env)",
    )

    return parser.parse_args()


# --- Step selection logic ---
def _select_steps(args: argparse.Namespace) -> list[Step]:
    """Select which steps to run based on CLI args."""
    if args.steps:
        requested = set(s.strip() for s in args.steps.split(","))
        # Validate step names
        invalid = requested - set(STEP_NAMES)
        if invalid:
            raise ValueError(f"Invalid step names: {', '.join(invalid)}")
        # Preserve order from STEPS tuple
        return [s for s in STEPS if s.name in requested]

    if args.skip_steps:
        skipped = set(s.strip() for s in args.skip_steps.split(","))
        # Validate step names
        invalid = skipped - set(STEP_NAMES)
        if invalid:
            raise ValueError(f"Invalid step names: {', '.join(invalid)}")
        return [s for s in STEPS if s.name not in skipped]

    return list(STEPS)


# --- Main orchestration ---
def main() -> int:
    """Run the pipeline."""
    args = _parse_args()

    # Resolve .env file path
    if args.env_file:
        env_path = args.env_file
    else:
        env_path = str(Path(__file__).resolve().parent.parent / ".env")

    # Load .env file before any src.* modules read config
    dotenv = _load_dotenv(env_path)
    _apply_dotenv(dotenv)

    # Set up logging
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    logger = _setup_logging(log_level)

    # Select steps to run
    try:
        active_steps = _select_steps(args)
    except ValueError as e:
        logger.error(str(e))
        return 1

    if not active_steps:
        logger.error("No steps to run")
        return 1

    # Log pipeline start
    logger.info(f"Starting pipeline with {len(active_steps)} step(s): {', '.join(s.name for s in active_steps)}")
    logger.info(f"Retry config: {args.retries} retries, {args.retry_delay}s delay")
    logger.info(f"Env file: {env_path}")

    pipeline_start = time.monotonic()

    # Run each step
    for i, step in enumerate(active_steps, 1):
        logger.info(f"--- [{i}/{len(active_steps)}] Starting: {step.name} ---")
        step_start = time.monotonic()

        try:
            _run_step(step, logger, args.retries, args.retry_delay)
        except Exception as exc:
            logger.error(f"Pipeline aborted due to failure in step '{step.name}'")
            return 1

        elapsed = time.monotonic() - step_start
        logger.info(f"--- [{i}/{len(active_steps)}] Completed: {step.name} ({elapsed:.1f}s) ---")

    # Log pipeline completion
    total_elapsed = time.monotonic() - pipeline_start
    logger.info(f"Pipeline completed successfully in {total_elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
