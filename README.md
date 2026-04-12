# Resume-Job Matching System

A RAG pipeline that matches resumes to job postings. Given a resume, the system retrieves the most semantically relevant jobs from a corpus, reranks them with cross-encoders, and generates grounded explanations for why each match works.

> **Scale note:** The data collection pipeline and FastAPI app are at minimum viable scale—enough to operationally test and validate the RAG system end-to-end. The corpus may not cover every role or industry, but the RAG pipeline itself will reliably surface the best matches within whatever jobs have been collected. Poor results are a corpus coverage problem, not a pipeline problem.

## Why This Approach?

Matching resumes to jobs is a retrieval problem: you need to find relevant jobs in a large corpus quickly. But raw semantic similarity alone isn't enough—a "Senior" job and a "Junior" resume might score high together despite the mismatch. This system uses a three-stage refinement:

1. **Dense retrieval** (VoyageAI embeddings + ChromaDB HNSW) finds top 100 semantically similar jobs
2. **Cross-encoder reranking** (Cohere) re-scores those 100 using a trained model optimized for ranking
3. **Local LLM generation** (Ollama) produces human-readable explanations grounded in the actual job description and resume

The result is fast, scalable, and interpretable. Benchmarks on held-out test data show **90% precision@5** and **88% recall@10**—meaning 90% of top-5 matches are valid, and we catch 88% of true positives in top-10.

## System Architecture

The system is split into two pipelines: one that builds the job corpus offline (cron-based), and one that handles live resume matching.

```
┌─────────────────────────────────────────────────────────────────────┐
│ DATA PIPELINE (Offline, runs on cron)                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  [8 Job APIs] ──→ collect_jobs.py ──→ jobs.db                      │
│  • Greenhouse         (company-specific boards)                      │
│  • JSearch            (RapidAPI, search-based)                      │
│  • JobSearch          (RapidAPI, search-based)                      │
│  • JobsAPI            (RapidAPI, Bing aggregator)                   │
│  • SerpAPI            (Google Jobs)                                  │
│  • Arbeitnow          (free, no auth)                               │
│  • Jobicy             (free remote focus)                            │
│  • Jooble             (direct API, 500/month cap)                   │
│                                                                       │
│           ↓ preprocess_jobs.py                                       │
│       (HTML clean + metadata extract)                                │
│           ↓ embed_jobs.py                                            │
│  (VoyageAI embeddings → SQLite BLOB)                                │
│           ↓ ChromaDB                                                 │
│      (HNSW index, persistent)                                        │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ INFERENCE PIPELINE (FastAPI, runs online)                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  [Resume text]                                                       │
│       ↓ embed (VoyageAI)                                            │
│       ↓ retrieve (ChromaDB, top-100)                                │
│       ↓ rerank (Cohere, top-10)                                     │
│       ↓ generate (Ollama LLaMA, grounded explanation)               │
│       ↓ /match endpoint                                              │
│   [Matched jobs + explanations]                                      │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component               | Technology                                   | Role                            |
| ----------------------- | -------------------------------------------- | ------------------------------- |
| **Job Collection**      | 8 APIs (Greenhouse, SerpAPI, RapidAPI, etc.) | Aggregate postings              |
| **Database**            | SQLite                                       | Raw jobs + embeddings storage   |
| **Text Cleaning**       | Custom regex + HTML parsing                  | HTML → plain text               |
| **Metadata Extraction** | Regex + Ollama LLM fallback                  | Degree, seniority, years        |
| **Embeddings**          | VoyageAI `voyage-3.5-lite`                   | Dense 1024-dim vectors          |
| **Vector Store**        | ChromaDB (HNSW)                              | Fast semantic search            |
| **Reranking**           | Cohere `rerank-english-v3.0`                 | Cross-encoder scoring           |
| **Local LLM**           | Ollama + LLaMA 3.2 3B                        | Grounded explanation generation |
| **Web Framework**       | FastAPI                                      | REST API & health checks        |
| **Reverse Proxy**       | Nginx (Alpine)                               | HTTP gateway                    |
| **Containerization**    | Docker + Docker Compose                      | Production deployment           |

## How It Works: The Pipeline

### Stage 1: Collecting Jobs

The system aggregates jobs from 8 different sources via `data_pipeline/collect_jobs.py`. The `JobCollectionOrchestrator` class coordinates between them, handling the complexity of different rate limits and API patterns.

Each collector extends a shared `BaseCollector` base class, which provides exponential backoff retry logic (max 3 attempts), budget pooling (e.g., JSearch gets 38 requests per run, Jooble gets 38 due to its 500/month cap), and automatic 429 (rate-limit) detection with adaptive pagination.

Jobs are deduplicated by checking the `(external_id, source_system, source_board)` tuple—if a job was already seen, it gets marked as `active` and its timestamp updates; otherwise it's inserted fresh. Every 3 days, the system marks jobs it hasn't re-seen as `removed`.

The whole collection process runs on a schedule via GitHub Actions 3 times a week (Mon/Wed/Fri at 6AM UTC). A `collection_log` table in SQLite tracks requests, errors, and jobs found per source for debugging.

### Stage 2: Cleaning & Extracting Metadata

Raw job descriptions come in as messy HTML with embedded scripts, iframes, and encoded entities. The preprocessing step in `scripts/pipeline/preprocess_jobs.py` cleans them up:

1. Unescape HTML entities (`&#39;` becomes `'`)
2. Strip out iframes, images, and scripts
3. Extract plain text from the remaining HTML
4. Collapse excess whitespace and normalize newlines
5. Standardize unicode punctuation (smart quotes become straight quotes)

Once cleaned, `src/regex_extraction.py` pulls out key metadata using simple pattern matching:

- **Degree requirement:** Searches for "BS", "Master's", "PhD" and normalizes to 0–3
- **Seniority level:** Looks for "junior", "senior", "lead" and bins to 0–3
- **Years of experience:** Regex for "X years"; if not found, falls back to Ollama LLM via `src/llm_extraction.py`

The cleaned text gets stored in the `cleaned_description` column, and the extracted metadata fills `required_degree`, `seniority_level`, and `min_years_experience` columns in the database.

### Stage 3: Embedding into Vector Space

Now we convert the cleaned job descriptions into dense vectors so we can do similarity search. `scripts/pipeline/embed_jobs.py` uses VoyageAI's `voyage-3.5-lite` model, which produces 1024-dimensional vectors.

To keep costs down, the system batches embedding requests in groups of 128 (the API's hard limit) and includes retry logic with exponential backoff in case of transient failures. Vectors are serialized as float32 numpy arrays and stored directly as BLOBs in the SQLite database—no external vector database at this stage.

The system tracks completion with an `embedded=1` flag, making the process resume-friendly: if embedding gets interrupted, re-running it skips jobs that were already vectorized. This matters when you're embedding millions of jobs.

### Stage 4: Retrieving Candidate Jobs

When a resume comes in, `src/retrieval.py` queries a ChromaDB instance that holds the job vectors in an HNSW (Hierarchical Navigable Small World) index. HNSW is a graph-based approximate nearest-neighbor algorithm—incredibly fast even with millions of vectors.

The retrieval step is deliberately over-inclusive: it pulls the top 100 jobs by cosine distance. Why not just keep the top 10? Because filtering on metadata (degree, seniority, years) happens _before_ the dense search. If a resume specifies "senior + 5+ years experience required", we filter the corpus first, then search within that subset. This avoids retrieving mismatches upfront.

The HNSW index is tuned with `ef_construction=400` (effort during indexing) and `ef=400` (effort during search), trading a bit of speed for higher recall—you want to catch all the good matches now, not miss them.

### Stage 5: Reranking with a Cross-Encoder

Top-100 jobs are still too many to show. The system now calls Cohere's `rerank-english-v3.0`, a cross-encoder model optimized for ranking relevance. Unlike embedding-based retrieval (which scores jobs in isolation), reranking scores each job _in context of the resume_, using both the resume and job description together.

`src/reranking.py` narrows the top-100 down to a final top-10 (or whatever `top_k` the user requested, up to 50). The batch API includes rate-limit awareness: the system throttles to 7 seconds between requests and detects 429 (rate limit) responses, retrying immediately rather than backing off.

For the cross-encoder to work well, we format each job compactly: title + location + seniority label. This gives the model enough context without overwhelming it.

### Stage 6: Generating Grounded Explanations

The final piece is explainability. Instead of just returning a score, the system runs an LLM to explain _why_ each match is good. This uses Ollama with a quantized `llama3.2:3b-instruct-q4_K_M` model (runs locally, ~1.6GB).

The full generation pipeline in `src/generation.py` works like this:

1. **Extract job requirements:** LLM reads the job description and pulls out a list of required skills and domain
2. **Find matches:** LLM compares the resume skills against the job requirements
3. **Filter hallucinations:** Verify that any mentioned skill actually appears in the job description (span-in-text check)
4. **Generate explanation:** LLM summarizes the matches in plain English

If the LLM times out or errors, the system falls back to a direct prompt that returns a generic explanation based on the job title and description.

For safety, resumes longer than 8000 characters get truncated before passing to the LLM. The model uses temperature 0.7 (enough variation for naturalness, but still deterministic) and nucleus sampling (top-p 0.9).

## API Reference

The FastAPI server exposes three endpoints: health, readiness, and matching.

### `GET /health`

Liveness probe—always returns 200 if the server is running.

```bash
curl http://localhost:8000/health
```

### `GET /ready`

Deep readiness check. Returns 200 only if all dependencies are healthy. Returns 503 if any fail. Useful for load balancers and orchestration systems.

Checks:

- SQLite database is reachable and contains the `jobs` table
- ChromaDB vector index is reachable
- VoyageAI API key is set and the service is responsive
- Ollama is running and has the required LLM model loaded

```bash
curl http://localhost:8000/ready
```

### `POST /match`

The main endpoint. Takes a resume and returns the best job matches with explanations.

**Request:**

```json
{
  "resume": "string (min 50 chars, required)",
  "top_k": "int (default 10, range 1–50, optional)",
  "min_years_experience": "int | null (optional, -1=unknown)",
  "seniority_level": "int | null (optional, 0–3, where 1=entry, 3=senior)",
  "required_degree": "int | null (optional, 0–3, where 1=bachelor, 3=phd)"
}
```

**Response:**

```json
{
  "matches": [
    {
      "job_id": 42,
      "title": "Senior Backend Engineer",
      "company_name": "Example Corp",
      "score": 0.95,
      "explanation": "Your 5+ years of Python and Kubernetes experience matches...",
      "absolute_url": "https://example.com/jobs/42"
    }
  ],
  "resume_id": "abc123-...",
  "corpus_warning": null
}
```

The `corpus_warning` field is set when no matches were found. This is a corpus coverage issue—the collected jobs may be too few or too narrow for certain roles—not a signal that the resume is a poor fit. The RAG pipeline will reliably return the best available matches; if results feel thin, expanding the corpus is the fix.

**Example:**

```bash
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{
    "resume": "Senior Python engineer with 7 years experience building microservices...",
    "top_k": 5,
    "seniority_level": 3
  }'
```

## Getting Started

### What You Need

For production, you'll need Docker, Docker Compose, and Ollama running on your host machine. You'll also need API keys from three services:

- **VoyageAI** ([console.voyageai.com](https://console.voyageai.com)) — for embeddings
- **Cohere** ([dashboard.cohere.com](https://dashboard.cohere.com)) — for reranking
- **RapidAPI, SerpAPI, Jooble** — for job collection (optional if you just want to test with existing data)

First, pull the Ollama model locally:

```bash
ollama pull llama3.2:3b-instruct-q4_K_M
```

### Production Setup (Docker Compose)

1. **Build the job corpus.** This is a one-time step (or re-run as needed):

   ```bash
   # Collect jobs from 8 sources
   python -m data_pipeline.collect_jobs

   # Clean them
   python scripts/pipeline/preprocess_jobs.py

   # Embed them
   python scripts/pipeline/embed_jobs.py
   ```

2. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Start the services:**

   ```bash
   docker-compose up -d
   ```

   This spins up:
   - FastAPI app on port 8000 (with SQLite + ChromaDB mounted to `./data`)
   - Nginx reverse proxy on port 80

4. **Verify everything is healthy:**

   ```bash
   curl http://localhost/ready
   ```

5. **Try a request:**
   ```bash
   curl -X POST http://localhost/match \
     -H "Content-Type: application/json" \
     -d '{
       "resume": "I am a senior Python engineer with 7 years of experience...",
       "top_k": 5
     }'
   ```

### Development Setup (Local)

For testing or development without Docker:

1. **Install Python dependencies:**

   ```bash
   pip install -r fastapi_app/requirements.txt
   ```

2. **Run the data pipeline:**

   ```bash
   python -m data_pipeline.collect_jobs
   python scripts/pipeline/preprocess_jobs.py
   python scripts/pipeline/embed_jobs.py
   ```

3. **Start the FastAPI server:**

   ```bash
   cd fastapi_app
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

4. **Test locally:**
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/ready
   ```

## Environment Variables

| Var                       | Required | Description                                                      |
| ------------------------- | -------- | ---------------------------------------------------------------- |
| `VOYAGE_API_KEY`          | ✓        | VoyageAI API key for embeddings                                  |
| `COHERE_API_KEY`          | ✓        | Cohere API key for reranking                                     |
| `CHROMA_DIR`              | ✓        | Path to ChromaDB persistent directory (e.g., `/app/data/chroma`) |
| `CHROMA_COLLECTION`       | ✓        | ChromaDB collection name (e.g., `jobs`)                          |
| `DB_PATH`                 | ✓        | Path to SQLite jobs database (e.g., `/app/data/jobs.db`)         |
| `OLLAMA_BASE_URL`         | -        | Ollama endpoint (default: `http://localhost:11434`)              |
| `LOG_LEVEL`               | -        | Logging verbosity: `INFO`, `DEBUG`, etc. (default: `INFO`)       |
| `SCRAPE_DAYS`             | -        | Job collection window in days (default: `7`)                     |
| **Data Pipeline Only**    | -        | -                                                                |
| `X_RAPID_API`             | -        | RapidAPI key (for JSearch, JobSearch, JobsAPI)                   |
| `SERPAPI_KEY`             | -        | SerpAPI key (for Google Jobs)                                    |
| `JOOBLE_API_KEY`          | -        | Jooble API key (500/month cap)                                   |
| `GREENHOUSE_BOARD_TOKENS` | -        | Comma-separated Greenhouse board tokens                          |
| `ADZUNA_API`              | -        | Adzuna API token                                                 |
| `ADZUNA_APP`              | -        | Adzuna app token                                                 |

## Keeping the Job Corpus Fresh

A GitHub Actions workflow (`.github/workflows/collect.yml`) runs the job collection pipeline automatically on a schedule: Monday, Wednesday, Friday at 6AM UTC. It:

1. Pulls the previous database as an artifact
2. Runs `python -m data_pipeline.collect_jobs` to fetch new jobs
3. Uploads the updated database back to GitHub as an artifact (kept for 90 days)

This means the corpus accumulates jobs over time without manual intervention. However, preprocessing and embedding are not yet automated—you currently run those locally when you deploy a fresh corpus. If the corpus grows large, you might want to add those as separate workflow jobs or run them on a faster machine.

## Evaluating Quality

The system includes a built-in evaluation harness for measuring retrieval and ranking quality. Here's how we benchmark:

### Results

Real test results on a held-out evaluation set:

| Dataset | Precision@5 | Recall@10 | Resumes |
| ------- | ----------- | --------- | ------- |
| Tune    | **1.00**    | **0.94**  | 30      |
| Test    | **0.90**    | **0.88**  | 20      |

Precision@5 means: of the top-5 results, how many were valid matches? Recall@10 means: of all known good matches, what fraction did we catch in the top-10? These metrics come from evaluating the system on a corpus of 1000 real jobs plus synthetic ground truth.

### How Evaluation Works

Evaluating a matching system is tricky without ground truth. We solve this by synthetically generating known-good and known-bad job postings aligned to test resumes, then measuring whether the system finds the good ones and rejects the bad ones.

The evaluation pipeline has these steps:

1. **Generate synthetic positives:** Ollama creates 5 realistic job postings for each test resume (aligned to the resume's skills and experience level)
2. **Generate synthetic negatives:** Ollama creates 3 intentionally mismatched jobs per resume (e.g., junior resume + senior job)
3. **Stratify and split:** Allocate resumes into tune (30) and test (20) sets, stratified by seniority and domain to ensure balanced coverage
4. **Evaluate on tune set:** Measure and optimize hyperparameters
5. **Evaluate on test set:** Final benchmark on unseen resumes using the same corpus

For each resume, the evaluation swaps its synthetic positives and negatives into the ChromaDB collection, runs the matching pipeline, and checks:

- How many known-good jobs appeared in the top-100 retrieval? (**embedding hit**)
- How many of those made it to the top-10 after reranking? (**rerank hit**)
- Where did we lose them? (embedding vs reranking)

This breakdown helps identify whether retrieval or reranking needs tuning.

### Synthetic Data

**Positives:** Each test resume gets 5 synthetic job postings generated by Ollama. The LLM reads the resume and creates realistic job descriptions (title, seniority, required years, domain, skills) that would actually match it. It then validates these (e.g., seniority levels must be valid) and repairs any that fail validation.

**Negatives:** 3 intentional mismatches per resume—typically a seniority mismatch (junior candidate, senior job) plus two responsibility/domain mismatches. This tests whether the system correctly rejects bad fits.

## Database Schema (SQLite)

**`jobs` table:**

```sql
CREATE TABLE jobs (
  id INTEGER PRIMARY KEY,
  external_id TEXT,
  source_system TEXT,
  source_board TEXT,
  title TEXT,
  location TEXT,
  description TEXT,                    -- raw HTML/text
  cleaned_description TEXT,            -- preprocessed plain text
  company_name TEXT,
  source_url TEXT,
  absolute_url TEXT,

  embedding BLOB,                      -- float32 numpy array (serialized)
  embedded INTEGER DEFAULT 0,          -- 0 or 1 flag

  required_degree INTEGER DEFAULT 0,   -- 0=unknown, 1=bachelor, 2=master, 3=phd
  seniority_level INTEGER DEFAULT 0,   -- 0=unknown, 1=entry, 2=mid, 3=senior
  min_years_experience INTEGER DEFAULT -1,  -- -1=unknown

  job_status TEXT DEFAULT 'active',    -- active, removed
  scraped_at TEXT,
  updated_date TEXT
);

CREATE TABLE collection_log (
  id INTEGER PRIMARY KEY,
  run_at TEXT,
  source TEXT,
  jobs_found INTEGER,
  jobs_new INTEGER,
  jobs_error INTEGER,
  requests INTEGER,
  errors TEXT
);
```

## Docker Deployment

### Multi-Stage Build

**Dockerfile:** `docker/Dockerfile`

- **Builder stage:** Python 3.11 + dependencies → `/install`
- **Runtime stage:** Minimal Python 3.11-slim + copy installed packages
- **Exposes:** Port 8000
- **CMD:** `uvicorn fastapi_app.api.main:app --host 0.0.0.0 --port 8000`

### Docker Compose

**docker-compose.yml:**

- **`app` service:** FastAPI container
  - Mounts `./data:/app/data` (persistent SQLite + ChromaDB)
  - Health check on `/health` endpoint
  - Environment: `OLLAMA_BASE_URL=http://host.docker.internal:11434` (reaches host Ollama)
  - Restart: unless-stopped
- **`nginx` service:** Reverse proxy (Alpine)
  - Port 80 → app:8000
  - Configuration: `docker/nginx.conf`
  - Restart: unless-stopped

**Note:** Ollama runs **on host** (not in container), exposed via `host.docker.internal:11434` on Docker Desktop.

## Troubleshooting

### Health Checks

Use the health endpoints to debug issues:

```bash
# Quick liveness check
curl http://localhost/health

# Full dependency check (shows detailed errors)
curl http://localhost/ready
```

### Viewing Logs

```bash
# FastAPI app output
docker-compose logs -f app

# Nginx reverse proxy output
docker-compose logs -f nginx

# Ollama (on host—check it's running)
ollama list
```

### Common Problems

**503 from /ready**
→ Ollama is not running. Start it: `ollama serve`

**Ollama model not found**
→ Pull the model: `ollama pull llama3.2:3b-instruct-q4_K_M`

**ChromaDB connection error**
→ Check that `./data/chroma` exists and is writable

**VoyageAI API key invalid**
→ Verify your key in `.env` is correct and not expired

**Cohere API returns 429 (rate limit)**
→ This is handled automatically. The system throttles to 7 seconds between batch rerank requests

### Inspecting the Database

```bash
sqlite3 data/jobs.db

# How many jobs do we have?
SELECT job_status, COUNT(*) FROM jobs GROUP BY job_status;

# How many jobs still need embedding?
SELECT COUNT(*) FROM jobs WHERE embedded = 0;

# What's the collection history?
SELECT source, jobs_found, requests, errors FROM collection_log ORDER BY run_at DESC LIMIT 10;
```

## Code Organization

The codebase is organized by pipeline stage and responsibility:

- **`data_pipeline/`** — Orchestrator and 8 job collectors. Handles collection, deduplication, and logging.
- **`scripts/pipeline/`** — Preprocessing, embedding, and matching. Run these in sequence to build the corpus.
- **`scripts/eval/`** — Evaluation suite: synthetic data generation, stratification, and benchmarking.
- **`src/`** — Core logic: embeddings, retrieval, reranking, generation, text cleaning, metadata extraction.
- **`fastapi_app/`** — REST API server, health checks, request validation, tests.
- **`docker/`** — Dockerfile (multi-stage) and Nginx configuration.
- **`data/`** — Persistent storage: SQLite database, ChromaDB vectors, evaluation outputs.

Each stage is modular—you can run parts independently or swap components as needed.

## Contact

Questions? Open an issue or contact the team.
