"""
Service for generating fit explanations using Ollama.
"""
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import ollama

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.generation import run_generation_pipeline
from app.config import settings

logger = logging.getLogger(__name__)


class GenerationService:
    """Service for generating fit explanations."""

    def __init__(self):
        self.model = settings.OLLAMA_MODEL
        self.base_url = settings.OLLAMA_BASE_URL

    def generate_explanations(
        self,
        resume_text: str,
        jobs: List[Dict[str, Any]],
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate fit explanations for each job.

        Args:
            resume_text: User's resume text.
            jobs: List of job dicts (must contain 'cleaned_description').
            run_id: Optional trace ID.

        Returns:
            Same list of jobs with added 'explanation' field.
        """
        if not jobs:
            return []

        # Prepare pairs for generation
        pairs = []
        for job in jobs:
            description = job.get("cleaned_description", "")
            if description:
                pairs.append((resume_text, description))
            else:
                pairs.append((resume_text, ""))

        try:
            generation_output = run_generation_pipeline(
                pairs=pairs,
                model=self.model,
                run_id=run_id,
            )
        except ollama.RequestError as e:
            logger.warning("Ollama is not reachable: %s. Skipping generation.", e)
            for job in jobs:
                job["explanation"] = None
            return jobs
        except Exception as e:
            logger.warning("Generation error: %s. Skipping.", e)
            for job in jobs:
                job["explanation"] = None
            return jobs

        # Attach explanations
        if isinstance(generation_output, str):
            # CORPUS_LIMITATION_MESSAGE returned — no grounded match found
            logger.info("No grounded match found for any job.")
            for job in jobs:
                job["explanation"] = None
        else:
            # list[PairResult] with same order as pairs
            for job, pair_result in zip(jobs, generation_output):
                job["explanation"] = pair_result.get("explanation")

        return jobs
