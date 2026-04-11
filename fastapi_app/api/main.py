from contextlib import asynccontextmanager
import logging
import os
import sqlite3
import chromadb
from dotenv import load_dotenv
load_dotenv()  # Loads .env from current directory

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level)


from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi_app.api.routers import health, match
from fastapi_app.api import errors
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.retrieval import build_collection

    # Sync Chroma collection from SQLite at startup
    try:
        conn = sqlite3.connect(os.environ["DB_PATH"])
        chroma_client = chromadb.PersistentClient(path=os.environ["CHROMA_DIR"])
        build_collection(conn, chroma_client)
        conn.close()
    except Exception as e:
        logger.error(f"Startup collection build failed: {e}")

    yield  # startup/shutdown hooks go here later

app = FastAPI(title="Job Matcher", lifespan=lifespan)

# Register exception handlers 
app.add_exception_handler(RequestValidationError, errors.validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, errors.http_exception_handler)
app.add_exception_handler(Exception, errors.unhandled_exception_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(match.router)

# Serve static files
app.mount("/", StaticFiles(directory="fastapi_app/static", html=True), name="static")
