from fastapi import Request
from fastapi.responses import JSONResponse

async def http_exception_handler(request: Request, exc):
    return JSONResponse(status_code=exc.status_code,
                        content={"error": exc.detail})

async def unhandled_exception_handler(request: Request, exc: Exception):
    # Log exc here
    return JSONResponse(status_code=500,
                        content={"error": "internal server error"})
