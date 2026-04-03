"""
FastAPI routes for the matching API.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from app.models.schemas import ResumeRequest, MatchResponse, HealthResponse
from app.services.matching_service import MatchingService
from app.dependencies import get_matching_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    matching_service: MatchingService = Depends(get_matching_service),
) -> HealthResponse:
    """
    Health check endpoint verifying all required services.
    """
    # Check Ollama
    try:
        import ollama
        ollama.list()
        ollama_available = True
    except Exception:
        ollama_available = False

    # Check database
    try:
        import sqlite3
        from app.config import settings
        conn = sqlite3.connect(settings.DB_PATH)
        conn.close()
        database_available = True
    except Exception:
        database_available = False

    # Check Chroma collection
    try:
        collection = matching_service.retrieval_service.get_collection()
        chroma_count = collection.count()
    except Exception:
        chroma_count = 0

    return HealthResponse(
        status="ok",
        ollama_available=ollama_available,
        database_available=database_available,
        chroma_collection_count=chroma_count,
    )


@router.post("/match", response_model=MatchResponse)
async def match_resume(
    request: ResumeRequest,
    matching_service: MatchingService = Depends(get_matching_service),
) -> MatchResponse:
    """
    Match a resume against job listings.

    Request body includes resume text and optional parameters.
    Returns a list of matched jobs with fit explanations.
    """
    if not request.resume_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Resume text cannot be empty.",
        )

    try:
        result = matching_service.match(
            resume_text=request.resume_text,
            top_k=request.top_k or 100,
            top_n=request.top_k or 10,  # Use same as top_k if not specified
            use_filters=request.use_filters,
            include_explanations=request.include_explanations,
        )
        return MatchResponse(**result)
    except Exception as e:
        logger.exception("Matching pipeline failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )
