from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi_app.api.routers import health, match
from fastapi_app.api import errors

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # startup/shutdown hooks go here later

app = FastAPI(title="Job Matcher", lifespan=lifespan)

# Register exception handlers
app.add_exception_handler(RequestValidationError, errors.validation_exception_handler)
app.add_exception_handler(HTTPException, errors.http_exception_handler)
app.add_exception_handler(Exception, errors.unhandled_exception_handler)

# Include routers
app.include_router(health.router)
app.include_router(match.router)
