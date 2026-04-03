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
from fastapi_app.api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}

def test_ready():
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ready"}
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

---

## API Contracts

These contracts are the ground truth for test-first development. Write tests against these before implementing any handler. Implementation must not deviate without updating this section first.

---

### `GET /health`

**Purpose:** Liveness check — is the process running?

**Auth:** None

**Response — always 200:**

```json
{ "status": "ok" }
```

**Never returns non-200.** If the server is up, this is 200. No dependency checks here.

**Tests to write:**

- Returns 200
- Body is exactly `{"status": "ok"}`
- Response time < 50ms (no I/O)

---

### `GET /ready`

**Purpose:** Readiness check — are all dependencies reachable?

**Auth:** None

**Response — 200 when all deps healthy:**

```json
{
  "status": "ready",
  "checks": {
    "db": "ok",
    "chroma": "ok",
    "voyage": "ok"
  }
}
```

**Response — 503 when any dep fails:**

```json
{
  "status": "degraded",
  "checks": {
    "db": "ok",
    "chroma": "error: collection not found",
    "voyage": "ok"
  }
}
```

**Contract rules:**

- Always returns one of: `200` or `503`
- `checks` always contains exactly the keys: `db`, `chroma`, `voyage`
- Each check value is either the string `"ok"` or a string starting with `"error:"`
- `status` is `"ready"` iff all checks are `"ok"`, otherwise `"degraded"`

**Tests to write:**

- All healthy → 200, `status == "ready"`, all checks `"ok"`
- DB unreachable (override dep) → 503, `status == "degraded"`, `db` starts with `"error:"`
- Chroma unreachable → 503, `chroma` starts with `"error:"`
- Voyage unreachable → 503, `voyage` starts with `"error:"`
- Partial failure → only failing check shows error, others show `"ok"`

---

### `POST /match`

**Purpose:** Core pipeline — embed resume, retrieve + rerank jobs, return ranked matches.

**Auth:** None (add API key header in Phase 6 hardening if needed)

**Content-Type:** `application/json`

**Request body:**

```json
{
  "resume": "<resume text>",
  "top_k": 10
}
```

| Field    | Type    | Required | Constraints         | Default |
| -------- | ------- | -------- | ------------------- | ------- |
| `resume` | string  | yes      | min length 50 chars | —       |
| `top_k`  | integer | no       | 1–50 inclusive      | 10      |

**Response — 200:**

```json
{
  "matches": [
    {
      "job_id": 42,
      "title": "Senior Backend Engineer",
      "score": 0.91,
      "explanation": "Strong Python and AWS experience aligns with..."
    }
  ],
  "resume_id": "a3f9c2b1"
}
```

| Field                   | Type    | Nullable | Notes                                                         |
| ----------------------- | ------- | -------- | ------------------------------------------------------------- |
| `matches`               | array   | no       | ordered by `score` descending; length ≤ `top_k`               |
| `matches[].job_id`      | integer | no       | corresponds to `jobs.id` in SQLite                            |
| `matches[].title`       | string  | no       |                                                               |
| `matches[].score`       | float   | no       | range [0.0, 1.0]; reranker relevance score                    |
| `matches[].explanation` | string  | yes      | `null` if Ollama unavailable or skipped                       |
| `resume_id`             | string  | yes      | first 8 chars of SHA-256 of resume text; `null` until Phase 6 |

**Error responses:**

| Status | `error` value                     | Condition                                            |
| ------ | --------------------------------- | ---------------------------------------------------- |
| 422    | Pydantic detail array             | `resume` missing, too short, or `top_k` out of range |
| 404    | `"no jobs in index"`              | ChromaDB collection is empty                         |
| 503    | `"embedding service unavailable"` | Voyage API call fails                                |
| 503    | `"reranking service unavailable"` | Cohere API call fails                                |
| 500    | `"internal server error"`         | Any other unhandled exception                        |

All error responses use this envelope:

```json
{ "error": "<message>" }
```

Never expose stack traces, internal paths, or dependency names in error messages returned to clients.

**Contract rules:**

- `matches` is always an array (empty array `[]` is valid if index has jobs but none score above threshold — not a 404)
- 404 is only raised when the index is completely empty
- `matches` length is always ≤ `top_k`; may be less if fewer jobs exist
- `score` is always a float, never `null`
- Response is always valid against the schema even when `explanation` is null

**Tests to write:**

_Validation:_

- Missing `resume` → 422
- `resume` length 49 chars → 422
- `resume` length 50 chars → 200 (boundary)
- `top_k = 0` → 422
- `top_k = 51` → 422
- `top_k = 50` → 200 (boundary)
- `top_k` omitted → 200, defaults to 10

_Happy path (mock voyage + chroma + cohere):_

- Valid resume → 200
- `matches` is a list
- Each match has `job_id` (int), `title` (str), `score` (float)
- `matches` length ≤ `top_k`
- `matches` sorted by `score` descending
- `explanation` is str or null (not missing key)

_Error paths (override deps to raise):_

- Empty chroma collection → 404, `{"error": "no jobs in index"}`
- Voyage raises exception → 503, `{"error": "embedding service unavailable"}`
- Cohere raises exception → 503, `{"error": "reranking service unavailable"}`
- Unhandled exception in handler → 500, `{"error": "internal server error"}` (no traceback)

---

### Error Envelope Contract

All non-2xx responses across **all** endpoints conform to:

```json
{ "error": "<human-readable string>" }
```

FastAPI's default 422 validation error format is **overridden** to also use this envelope. The `detail` array from Pydantic is collapsed into a single readable string or kept as-is under `"error"` — pick one and be consistent. Recommended: keep Pydantic's `detail` array as the value of `"error"` for 422s since it's machine-readable and useful for clients.

**Tests to write:**

- 404 on unknown route → check body has `"error"` key (FastAPI default is `{"detail":"Not Found"}` — override this)
- 422 on bad body → body has `"error"` key

---

### Dependency Override Pattern (for all tests)

All tests use `app.dependency_overrides` to avoid real I/O:

```python
# conftest.py for tests/api/
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies import get_voyage_client, get_chroma_collection, get_db
from unittest.mock import MagicMock

@pytest.fixture
def mock_voyage():
    client = MagicMock()
    client.embed.return_value = MagicMock(embeddings=[[0.1] * 1024])
    return client

@pytest.fixture
def mock_collection():
    col = MagicMock()
    col.count.return_value = 5
    col.query.return_value = {
        "ids": [["1", "2", "3"]],
        "documents": [["Job A desc", "Job B desc", "Job C desc"]],
        "metadatas": [[{"title": "Eng A", "job_id": 1},
                       {"title": "Eng B", "job_id": 2},
                       {"title": "Eng C", "job_id": 3}]],
        "distances": [[0.1, 0.2, 0.3]],
    }
    return col

@pytest.fixture
def api_client(mock_voyage, mock_collection):
    app.dependency_overrides[get_voyage_client] = lambda: mock_voyage
    app.dependency_overrides[get_chroma_collection] = lambda: mock_collection
    yield TestClient(app)
    app.dependency_overrides.clear()
```

Use `api_client` fixture in all `test_match.py` tests. For error path tests, override the dep to raise the relevant exception inline per test.
