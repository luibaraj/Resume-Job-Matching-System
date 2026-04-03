from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routers import health

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # startup/shutdown hooks go here later

app = FastAPI(title="Job Matcher", lifespan=lifespan)
app.include_router(health.router)
