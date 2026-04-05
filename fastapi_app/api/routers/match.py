from fastapi import APIRouter, Depends, HTTPException
import numpy as np
from fastapi_app.api.schemas import MatchRequest, MatchResponse, JobMatch
from fastapi_app.api.dependencies import get_voyage_client, get_chroma_collection, get_cohere_client, get_ollama_base_url
from src.retrieval import query_collection
from src.reranking import rerank_jobs
from src.llm_extraction import (
    extract_degree_with_llm,
    extract_seniority_with_llm,
    extract_years_with_llm,
)
from src.regex_extraction import build_chroma_where_filter
from src.generation import run_generation_pipeline
from src.config import MAX_BATCH_SIZE, CORPUS_LIMITATION_MESSAGE
import logging
import requests

logger = logging.getLogger(__name__)

router = APIRouter()

def _generate_explanation_direct(resume: str, job_title: str, job_content: str, ollama_base_url: str) -> str | None:
    """
    Generate explanation using Ollama directly (fallback).
    """
    try:
        prompt = f"""You are a career advisor. Explain why the resume is a good match for the job.

Resume summary: {resume[:500]}

Job title: {job_title}
Job description: {job_content[:500]}

Provide a concise explanation (2-3 sentences) highlighting key matches:"""
        
        response = requests.post(
            f"{ollama_base_url}/api/generate",
            json={
                "model": "llama3.2:3b-instruct-q4_K_M",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 150
                }
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        logger.error(f"Failed to generate explanation: {e}")
        return None

def generate_explanation_with_pipeline(
    resume: str, 
    job_title: str,
    job_content: str, 
    ollama_base_url: str,
    model: str = "llama3.2:3b-instruct-q4_K_M"
) -> tuple[str | None, str | None]:
    """
    Returns: (explanation, corpus_warning)
    """
    try:
        results, corpus_message = run_generation_pipeline(
            [(resume, job_content)], 
            model=model
        )
        if results and results[0]:
            return results[0]['explanation'], corpus_message
        return None, corpus_message
    except Exception as e:
        logger.error(f"Generation pipeline failed: {e}")
        # Fallback to direct Ollama call
        explanation = _generate_explanation_direct(resume, job_title, job_content, ollama_base_url)
        return explanation, None

@router.post("/match", response_model=MatchResponse)
async def match(
    req: MatchRequest,
    voyage_client=Depends(get_voyage_client),
    collection=Depends(get_chroma_collection),
    cohere_client=Depends(get_cohere_client),
    ollama_base_url=Depends(get_ollama_base_url)
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

    # 2. Extract degree, seniority, and years from resume
    try:
        user_degree = extract_degree_with_llm(req.resume, "llama3.2:3b-instruct-q4_K_M")
        user_seniority = extract_seniority_with_llm(req.resume, "llama3.2:3b-instruct-q4_K_M")
        user_years = extract_years_with_llm(req.resume, "llama3.2:3b-instruct-q4_K_M")
        where_filter = build_chroma_where_filter(user_degree, user_seniority, user_years)
        logger.info(
            "Extracted user profile — degree: %d, seniority: %d, years: %d",
            user_degree,
            user_seniority,
            user_years,
        )
    except Exception as e:
        logger.error(f"Failed to extract user filters: {e}")
        where_filter = None

    # 3. Embed resume
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

    # 4. Retrieve from Chroma with filters
    try:
        retrieved = query_collection(collection, embedding, top_k=100, where=where_filter)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="retrieval error"
        )
    
    # Return empty array if no jobs found (not 404)
    if not retrieved:
        return MatchResponse(matches=[], resume_id=None, corpus_warning=None)

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

    # 5. Generate explanations and build response
    matches = []
    corpus_warning = None  # Track warning across all jobs
    
    # Limit to top_k matches
    for job in reranked[:req.top_k]:
        # Convert id to integer for job_id
        try:
            job_id = int(job['id'])
        except (ValueError, KeyError):
            job_id = 0
        
        # Get job content for explanation
        job_content = job.get('document', '')
        if not job_content:
            job_content = job.get('title', '')
        
        # Generate explanation using pipeline
        explanation, warning = generate_explanation_with_pipeline(
            resume=req.resume,
            job_title=job.get('title', ''),
            job_content=job_content,
            ollama_base_url=ollama_base_url
        )
        
        # Capture first non-None warning
        if warning is not None and corpus_warning is None:
            corpus_warning = warning
        
        # Use distance as score (lower distance is better, but we can invert if needed)
        # For now, use 1.0 - distance to have higher scores for more similar items
        distance = job.get('distance', 1.0)
        score = max(0.0, 1.0 - distance)
        matches.append(
            JobMatch(
                job_id=job_id,
                title=job.get('title', ''),
                company_name=job.get('company_name', ''),
                score=score,
                explanation=explanation
            )
        )

    return MatchResponse(matches=matches, resume_id=None, corpus_warning=corpus_warning)
