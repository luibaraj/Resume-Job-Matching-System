# Production Deployment Design: FastAPI + Docker (Local)

## Philosophy

Minimal but real: one API service, one container, one reverse proxy. No Kubernetes, no cloud — just Docker Compose with enough structure that scaling up later is non-breaking.

---

## Target Architecture

```
[Browser / curl]
      │
      ▼
 nginx (port 80)          ← reverse proxy, serves static files if needed
      │
      ▼
 fastapi app (port 8000)  ← uvicorn, async, health checks
      │
      ├── SQLite (volume-mounted)
      ├── ChromaDB (volume-mounted)
      └── Ollama (host network or sidecar)
```

Everything runs via `docker compose up`. Secrets via `.env`. Logs via stdout → `docker logs`.

---

## Scaffold: New Files and Folders

```
project-root/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, lifespan, router registration
│   ├── dependencies.py      # Shared deps: DB conn, voyage client, cohere client
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── health.py        # GET /health, GET /ready
│   │   └── match.py         # POST /match  (core pipeline endpoint)
│   ├── schemas.py           # Pydantic request/response models
│   └── errors.py            # Exception handlers → structured JSON errors
│
├── docker/
│   ├── Dockerfile           # Multi-stage: builder + slim runtime
│   ├── nginx.conf           # Reverse proxy config
│   └── entrypoint.sh        # Wait-for-deps, then exec uvicorn
│
├── docker-compose.yml       # Services: app, nginx
├── docker-compose.override.yml  # Dev overrides: volume mounts, hot reload
├── .env.example             # Template with all required keys documented
└── tests/
    └── api/
        ├── __init__.py
        ├── test_health.py
        └── test_match.py
```

Nothing in `src/` changes — `api/` imports from `src/` directly.

---

## Phases

### Phase 1 — Skeleton (no business logic)

**Goal:** `docker compose up` starts healthy containers; `/health` returns 200.

**Files to create:**

`api/main.py`

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routers import health

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # startup/shutdown hooks go here later

app = FastAPI(title="Job Matcher", lifespan=lifespan)
app.include_router(health.router)
```

`api/routers/health.py`

```python
from fastapi import APIRouter
router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.get("/ready")
async def ready():
    # Phase 3: check DB, chroma, voyage connectivity
    return {"status": "ready"}
```

`docker/Dockerfile`

```dockerfile
# --- builder ---
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- runtime ---
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker-compose.yml`

```yaml
services:
  app:
    build:
      context: .
      dockerfile: docker/Dockerfile
    env_file: .env
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data # SQLite + ChromaDB persistence
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      app:
        condition: service_healthy
```

`docker/nginx.conf`

```nginx
server {
    listen 80;
    location / {
        proxy_pass         http://app:8000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
```

`docker-compose.override.yml` (dev only, gitignored or committed — your call)

```yaml
services:
  app:
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    volumes:
      - .:/app # hot reload
```

**Deliverable:** `curl http://localhost/health` → `{"status":"ok"}`

---

### Phase 2 — Schemas + Error Handling

**Goal:** Define request/response contracts; errors return structured JSON, never stack traces.

`api/schemas.py`

```python
from pydantic import BaseModel, Field

class MatchRequest(BaseModel):
    resume: str = Field(..., min_length=50)
    top_k: int = Field(default=10, ge=1, le=50)

class JobMatch(BaseModel):
    job_id: int
    title: str
    score: float
    explanation: str | None = None

class MatchResponse(BaseModel):
    matches: list[JobMatch]
    resume_id: str | None = None  # hash of resume for caching later
```

`api/errors.py`

```python
from fastapi import Request
from fastapi.responses import JSONResponse

async def http_exception_handler(request: Request, exc):
    return JSONResponse(status_code=exc.status_code,
                        content={"error": exc.detail})

async def unhandled_exception_handler(request: Request, exc: Exception):
    # Log exc here
    return JSONResponse(status_code=500,
                        content={"error": "internal server error"})
```

Register in `main.py`:

```python
from fastapi.exceptions import HTTPException
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)
```

**Deliverable:** Bad requests return `{"error": "..."}` with correct status codes.

---

### Phase 3 — Dependencies + Readiness

**Goal:** Inject real clients; `/ready` checks actual connectivity.

`api/dependencies.py`

```python
import os
import sqlite3
from functools import lru_cache
import voyageai
import chromadb
from src.embedding import create_client

@lru_cache
def get_voyage_client():
    return create_client(os.environ["VOYAGE_API_KEY"])

@lru_cache
def get_chroma_collection():
    client = chromadb.PersistentClient(path=os.environ["CHROMA_DIR"])
    return client.get_collection(os.environ["CHROMA_COLLECTION"])

def get_db():
    conn = sqlite3.connect(os.environ["DB_PATH"])
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
```

Update `/ready` to call `get_voyage_client()` and `get_chroma_collection()`, catch exceptions, return `503` if anything fails.

**Deliverable:** `docker compose up` → `/ready` validates all dependencies are reachable.

---

### Phase 4 — Match Endpoint (Core Pipeline)

**Goal:** Wire existing `src/` pipeline into `POST /match`.

`api/routers/match.py`

```python
from fastapi import APIRouter, Depends, HTTPException
from api.schemas import MatchRequest, MatchResponse, JobMatch
from api.dependencies import get_voyage_client, get_chroma_collection
from src.retrieval import query_collection
from src.reranking import rerank_jobs
from src.generation import generate_explanation
import numpy as np

router = APIRouter()

@router.post("/match", response_model=MatchResponse)
async def match(req: MatchRequest,
                voyage=Depends(get_voyage_client),
                collection=Depends(get_chroma_collection)):
    # 1. embed
    result = voyage.embed([req.resume], model="voyage-3.5-lite", input_type="query")
    embedding = np.array(result.embeddings[0])

    # 2. retrieve
    retrieved = query_collection(collection, embedding, top_k=100)
    if not retrieved:
        raise HTTPException(status_code=404, detail="no jobs in index")

    # 3. rerank
    reranked = rerank_jobs(req.resume, retrieved, top_n=req.top_k)

    # 4. explain (optional — skip if Ollama unavailable)
    matches = [JobMatch(job_id=j.job_id, title=j.title, score=j.score)
               for j in reranked]
    return MatchResponse(matches=matches)
```

Register in `main.py`: `app.include_router(match.router)`

**Deliverable:** `curl -X POST http://localhost/match -d '{"resume":"..."}' -H 'Content-Type: application/json'` returns ranked jobs.

---

### Phase 5 — Tests + CI Gate

**Goal:** pytest covers API layer; `docker compose run app pytest` passes.

`tests/api/test_health.py`

```python
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    assert client.get("/health").status_code == 200

def test_ready_structure():
    r = client.get("/ready")
    assert r.status_code in (200, 503)
    assert "status" in r.json()
```

`tests/api/test_match.py` — mock `get_voyage_client` and `get_chroma_collection` with `pytest` fixtures + `app.dependency_overrides`.

Add to `docker-compose.yml` app service:

```yaml
command: >
  sh -c "pytest tests/ && uvicorn api.main:app ..."
```

Or run tests separately: `docker compose run --rm app pytest tests/`

**Deliverable:** All tests green in container. This is your CI gate before any deploy.

---

### Phase 6 — Hardening (Production Checklist)

These are small, independent tasks — do in any order:

| Task              | What to do                                                                               |
| ----------------- | ---------------------------------------------------------------------------------------- |
| Secrets           | Never bake `.env` into image; use `env_file` (already done) or Docker secrets            |
| Logging           | Add `structlog` or stdlib JSON logging; log request ID, latency, status                  |
| Request timeout   | `uvicorn --timeout-keep-alive 75`; nginx `proxy_read_timeout 120s` (already in template) |
| Rate limiting     | Add `slowapi` middleware to FastAPI app                                                  |
| CORS              | `app.add_middleware(CORSMiddleware, ...)` if browser clients needed                      |
| Image size        | Audit `requirements.txt`; add `.dockerignore` to exclude `data/`, `.git/`, `outputs/`    |
| Restart policy    | `restart: unless-stopped` already in compose template                                    |
| Graceful shutdown | Lifespan handles cleanup; uvicorn handles SIGTERM by default                             |

---

## Aider Workflow

Run aider pointed at the relevant files for each phase:

```bash
# Phase 1
aider api/main.py api/routers/health.py docker/Dockerfile docker-compose.yml docker/nginx.conf

# Phase 2
aider api/schemas.py api/errors.py api/main.py

# Phase 3
aider api/dependencies.py api/routers/health.py

# Phase 4
aider api/routers/match.py api/main.py src/retrieval.py src/reranking.py

# Phase 5
aider tests/api/test_health.py tests/api/test_match.py

# Phase 6 — one task at a time
aider api/main.py  # logging / rate limiting / CORS
```

Always complete + test one phase before starting the next.

---

## .env.example

```
VOYAGE_API_KEY=
COHERE_API_KEY=
DB_PATH=/app/data/jobs.db
CHROMA_DIR=/app/data/chroma
CHROMA_COLLECTION=jobs
```
