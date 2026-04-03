"""
Service for reranking candidates using Cohere.
"""
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.reranking import rerank_jobs
from app.config import settings

logger = logging.getLogger(__name__)


class RerankingService:
    """Service for reranking job candidates."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def rerank(
        self,
        resume_text: str,
        candidates: List[Dict[str, Any]],
        top_n: int = settings.RERANK_TOP_N,
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using Cohere's rerank API.

        Args:
            resume_text: User's resume text.
            candidates: List of candidate job dicts from retrieval.
            top_n: Number of top results to return after reranking.
            run_id: Optional trace ID.

        Returns:
            Reranked list of job dicts.
        """
        if not candidates:
            return []

        try:
            results = rerank_jobs(
                resume_text=resume_text,
                candidates=candidates,
                top_n=top_n,
                api_key=self.api_key,
                run_id=run_id,
            )
            return results
        except Exception as e:
            logger.error("Reranking failed: %s", e)
            # Fallback: return original candidates limited to top_n
            return candidates[:top_n]
