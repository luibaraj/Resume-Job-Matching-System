from fastapi import APIRouter, Depends, HTTPException
import numpy as np
from fastapi_app.api.schemas import MatchRequest, MatchResponse, JobMatch
from fastapi_app.api.dependencies import get_voyage_client, get_chroma_collection, get_cohere_client
from src.retrieval import query_collection
from src.reranking import rerank_jobs
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/match", response_model=MatchResponse)
async def match(
    req: MatchRequest,
    voyage_client=Depends(get_voyage_client),
    collection=Depends(get_chroma_collection),
    cohere_client=Depends(get_cohere_client)
):
    # 1. Embed resume
    try:
        result = voyage_client.embed(
            [req.resume],
            model="voyage-3.5-lite",
            input_type="query"
        )
        embedding = np.array(result.embeddings[0], dtype=np.float32)
    except Exception as e:
        logger.error(f"Voyage embedding failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="embedding service unavailable"
        )

    # 2. Retrieve from Chroma
    try:
        retrieved = query_collection(collection, embedding, top_k=100)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="retrieval error"
        )
    
    if not retrieved:
        raise HTTPException(
            status_code=404,
            detail="no jobs in index"
        )

    # 3. Rerank with Cohere
    try:
        reranked = rerank_jobs(req.resume, retrieved, cohere_client, top_n=req.top_k)
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="reranking service unavailable"
        )

    # 4. Build response (skip explanation for now)
    matches = []
    for job in reranked:
        matches.append(
            JobMatch(
                job_id=job.job_id,
                title=job.title,
                score=job.score,
                explanation=None
            )
        )

    return MatchResponse(matches=matches, resume_id=None)
