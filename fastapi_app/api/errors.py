from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": exc.errors()}  # Keep the array format as recommended
    )

async def http_exception_handler(request: Request, exc):
    return JSONResponse(status_code=exc.status_code,
                        content={"error": exc.detail})

async def unhandled_exception_handler(request: Request, exc: Exception):
    # Log exc here
    return JSONResponse(status_code=500,
                        content={"error": "internal server error"})
