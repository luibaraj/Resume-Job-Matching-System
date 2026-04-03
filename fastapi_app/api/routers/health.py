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
    
    # Check DB
    try:
        cursor = db.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        checks["db"] = "ok"
    except Exception as e:
        checks["db"] = f"error: {str(e)}"
    
    # Check Chroma
    try:
        count = collection.count()
        checks["chroma"] = "ok"
    except Exception as e:
        checks["chroma"] = f"error: {str(e)}"
    
    # Check Voyage
    try:
        # Embed a dummy text to test connectivity
        result = voyage_client.embed(["test"], model="voyage-3.5-lite", input_type="query")
        checks["voyage"] = "ok"
    except Exception as e:
        checks["voyage"] = f"error: {str(e)}"
    
    # Determine overall status
    all_ok = all(value == "ok" for value in checks.values())
    status = "ready" if all_ok else "degraded"
    
    if all_ok:
        return {"status": status, "checks": checks}
    else:
        raise HTTPException(status_code=503, detail={"status": status, "checks": checks})
