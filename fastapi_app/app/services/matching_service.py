"""
Orchestration service for the full matching pipeline.
"""
import logging
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.regex_extraction import (
    extract_user_degree,
    extract_user_seniority,
    extract_user_years_experience,
    build_chroma_where_filter,
    describe_chroma_filter,
    extract_years_experience,
)

from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService
from app.services.reranking_service import RerankingService
from app.services.generation_service import GenerationService
from app.config import settings

logger = logging.getLogger(__name__)


class MatchingService:
    """Orchestrates the retrieve→rerank→generate pipeline."""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.retrieval_service = RetrievalService()
        self.reranking_service = RerankingService(api_key=settings.COHERE_API_KEY)
        self.generation_service = GenerationService()

    def match(
        self,
        resume_text: str,
        top_k: int = settings.RETRIEVE_TOP_K,
        top_n: int = settings.RERANK_TOP_N,
        use_filters: bool = True,
        include_explanations: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full matching pipeline.

        Args:
            resume_text: User's resume text.
            top_k: Number of candidates to retrieve.
            top_n: Number of top results after reranking.
            use_filters: Whether to apply user profile filters.
            include_explanations: Whether to generate fit explanations.

        Returns:
            Dictionary containing matches and metadata.
        """
        run_id = str(uuid.uuid4())
        logger.info("Starting matching pipeline with run_id: %s", run_id)

        # Step 1: Embed resume
        query_embedding = self.embedding_service.load_or_embed_resume(resume_text)

        # Step 2: Build filters
        where_filter = None
        if use_filters:
            user_degree = extract_user_degree(resume_text)
            user_seniority = extract_user_seniority(resume_text)
            user_years = extract_user_years_experience(resume_text)
            where_filter = build_chroma_where_filter(
                user_degree, user_seniority, user_years
            )
            logger.info(
                "Applied filters: %s",
                describe_chroma_filter(where_filter) if where_filter else "None",
            )

        # Step 3: Dense retrieval
        candidates = self.retrieval_service.query(
            query_embedding=query_embedding.tolist(),
            top_k=top_k,
            where_filter=where_filter,
            run_id=run_id,
        )
        logger.info("Retrieved %d candidates.", len(candidates))

        # Step 4: Reranking
        reranked = self.reranking_service.rerank(
            resume_text=resume_text,
            candidates=candidates,
            top_n=top_n,
            run_id=run_id,
        )
        logger.info("Reranked to %d results.", len(reranked))

        # Step 5: Generation (optional)
        if include_explanations:
            reranked = self.generation_service.generate_explanations(
                resume_text=resume_text,
                jobs=reranked,
                run_id=run_id,
            )

        # Step 6: Format results
        matches = []
        for job in reranked:
            min_years = extract_years_experience(job.get("cleaned_description", ""))
            matches.append({
                "id": job.get("id"),
                "title": job.get("title", "Unknown"),
                "location": job.get("location"),
                "company_name": job.get("company_name"),
                "board_token": job.get("board_token", "Unknown"),
                "source_url": job.get("source_url"),
                "min_years_experience": min_years,
                "distance": job.get("distance"),
                "rerank_score": job.get("rerank_score"),
                "explanation": job.get("explanation"),
            })

        return {
            "matches": matches,
            "total_candidates": len(candidates),
            "total_reranked": len(reranked),
            "filters_applied": where_filter,
            "run_id": run_id,
        }
