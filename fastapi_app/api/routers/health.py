from fastapi import APIRouter, Depends, HTTPException
from fastapi_app.api.dependencies import get_voyage_client, get_chroma_collection, get_db, get_ollama_base_url, check_ollama_health
import sqlite3

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.get("/ready")
async def ready(
    voyage_client=Depends(get_voyage_client),
    collection=Depends(get_chroma_collection),
    db=Depends(get_db),
    ollama_base_url=Depends(get_ollama_base_url)
):
    checks = {}
    all_ok = True
    
    # Check DB
    try:
        cursor = db.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        checks["database"] = {"healthy": True, "message": "ok"}
    except Exception as e:
        checks["database"] = {"healthy": False, "message": f"error: {str(e)}"}
        all_ok = False
    
    # Check Chroma
    try:
        count = collection.count()
        checks["chroma"] = {"healthy": True, "message": "ok"}
    except Exception as e:
        checks["chroma"] = {"healthy": False, "message": f"error: {str(e)}"}
        all_ok = False
    
    # Check Voyage
    try:
        # Embed a dummy text to test connectivity
        result = voyage_client.embed(["test"], model="voyage-3.5-lite", input_type="query")
        checks["voyage"] = {"healthy": True, "message": "ok"}
    except Exception as e:
        checks["voyage"] = {"healthy": False, "message": f"error: {str(e)}"}
        all_ok = False
    
    # Check Ollama
    try:
        ollama_healthy, ollama_message = check_ollama_health(ollama_base_url)
        checks["ollama"] = {"healthy": ollama_healthy, "message": ollama_message}
        if not ollama_healthy:
            all_ok = False
    except Exception as e:
        checks["ollama"] = {"healthy": False, "message": f"error: {str(e)}"}
        all_ok = False
    
    # Determine overall status
    status = "ready" if all_ok else "not ready"
    
    # Return 503 if any check failed
    if not all_ok:
        raise HTTPException(
            status_code=503,
            detail={"status": status, "checks": checks}
        )
    
    return {"status": status, "checks": checks}
