from fastapi import APIRouter, Depends, HTTPException
from fastapi_app.api.dependencies import get_voyage_client, get_chroma_collection, get_db
import sqlite3

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.get("/ready")
async def ready(
    voyage_client=Depends(get_voyage_client),
    collection=Depends(get_chroma_collection),
    db=Depends(get_db)
):
    checks = {}
    all_ok = True
    
    # Check DB
    try:
        cursor = db.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        checks["db"] = "ok"
    except Exception as e:
        checks["db"] = f"error: {str(e)}"
        all_ok = False
    
    # Check Chroma
    try:
        count = collection.count()
        checks["chroma"] = "ok"
    except Exception as e:
        checks["chroma"] = f"error: {str(e)}"
        all_ok = False
    
    # Check Voyage
    try:
        # Embed a dummy text to test connectivity
        result = voyage_client.embed(["test"], model="voyage-3.5-lite", input_type="query")
        checks["voyage"] = "ok"
    except Exception as e:
        checks["voyage"] = f"error: {str(e)}"
        all_ok = False
    
    # Determine overall status
    status = "ready" if all_ok else "degraded"
    
    # Return 503 if any check failed
    if not all_ok:
        raise HTTPException(
            status_code=503,
            detail={"status": status, "checks": checks}
        )
    
    return {"status": status, "checks": checks}
