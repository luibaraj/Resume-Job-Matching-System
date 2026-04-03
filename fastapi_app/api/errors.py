from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": exc.errors()}  # Keep the array format as recommended
    )

async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    # If the detail is already a dict (like from /ready endpoint), return it as-is
    if isinstance(exc.detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    # Otherwise, wrap string details in an error envelope for consistency
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail)}
    )

async def unhandled_exception_handler(request: Request, exc: Exception):
    # Log exc here
    return JSONResponse(status_code=500,
                        content={"error": "internal server error"})
