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
    # 1. Check if collection is empty
    try:
        job_count = collection.count()
        if job_count == 0:
            raise HTTPException(
                status_code=404,
                detail="no jobs in index"
            )
    except HTTPException:
        # Re-raise HTTPException so it goes through the error handler
        raise
    except Exception as e:
        logger.error(f"Failed to check collection count: {e}")
        raise HTTPException(
            status_code=500,
            detail="failed to check job index"
        )

    # 2. Embed resume
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

    # 3. Retrieve from Chroma
    try:
        retrieved = query_collection(collection, embedding, top_k=100)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="retrieval error"
        )
    
    # Return empty array if no jobs found (not 404)
    if not retrieved:
        return MatchResponse(matches=[], resume_id=None)

    # 4. Rerank with Cohere - FIXED: use keyword argument for client
    try:
        reranked = rerank_jobs(
            query=req.resume, 
            jobs=retrieved, 
            client=cohere_client,  # Use keyword argument
            top_n=req.top_k
        )
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="reranking service unavailable"
        )

    # 5. Build response (skip explanation for now)
    matches = []
    # Limit to top_k matches
    for job in reranked[:req.top_k]:
        # Convert id to integer for job_id
        try:
            job_id = int(job['id'])
        except (ValueError, KeyError):
            job_id = 0
        # Use distance as score (lower distance is better, but we can invert if needed)
        # For now, use 1.0 - distance to have higher scores for more similar items
        distance = job.get('distance', 1.0)
        score = max(0.0, 1.0 - distance)
        matches.append(
            JobMatch(
                job_id=job_id,
                title=job['title'],
                score=score,
                explanation=None
            )
        )

    return MatchResponse(matches=matches, resume_id=None)
