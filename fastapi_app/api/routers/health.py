from fastapi import APIRouter
router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.get("/ready")
async def ready():
    # Phase 3: check DB, chroma, voyage connectivity
    return {"status": "ready"}
