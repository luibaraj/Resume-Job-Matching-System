# Embedding Model

## Voyage AI `voyage-3.5-lite`

My main constraints are good performance for symmetric retrieval (query is approximately the same length as the documents) and low cost. This Voyage AI model provides competitive performance for retrieval tasks, a generous free tier, and still presents low cost after the free tier is diminished. The main caveat is that this model has an asymmetric design and prepends text based on if the input is a query, document, or none. Since we are comparing content-to-content and not question-to-answer, we can simply set input type to none so that prepending is skipped entirely.

# Preprocessing Pipeline

These steps were curated based on the data quality report provided by inspect_raw_jobs.py.

1. Unescape HTML entities for downstream steps to work on real HTML.
2. Strip iframe and image tags entirely.
3. Explain plain text from clean HTML.
4. Normalize whitespace.
5. Normalize unicode puncutation with ASCII equivalents.

# Vector Database

## Chroma

The Chroma vector database is lightweight and great for prototyping, and it can still reach high recall accuracy if tuned properly.

# Generation

## Model Selection

Becuase the main priority of this system is cost, the model of choice is a locally hosted LLAMA 3.2 with 3 billion parameters. Optimized latency with Ollama by quantizing the model to 4 bit.

## Generation Pipeline (Batch Processing)

**Input**: Batch of up to 10 (resume, job_posting) pairs + reranker scores

**Processing**:

1. **Extract Job Requirements**
   - LLM extracts top 3-5 required skills from job posting
   - Output: exact text spans from job posting
   - Validate: Regex verify each span exists in job posting
   - Discard invalid spans

2. **Search Resume for Matches**
   - LLM searches resume for text matching each validated requirement
   - Output: exact resume text spans (minimal, regex-unique)
   - Validate: Normalize whitespace, regex verify each span in resume
   - Mark hallucinations if span not found; exclude from results

3. **Filter Pairs**
   - Scrap any (resume, job_posting) pair with zero validated matches
   - **Branch: Check if all 10 pairs scrapped**
     - **Yes**: Output message "No strongly matching jobs found in current corpus. This indicates corpus limitations, not poor fit. Recommend expanding job database or checking back later."
     - **No**: Continue to step 4

4. **Generate Explanations**
   - LLM explains fit in 1-2 sentences using only validated (requirement, resume_match) pairs as context
   - Output: explanation string

5. **Log Results**
   - Track: [explanation, num_validated_pairs, hallucination_count]
   - Flag pairs with hallucinations for manual review

**Output**: Personalized explanations for each user-job pair, or corpus-limitation message

# Synthetic Positives Pipeline

## Purpose

Synthetic positives are (resume, job_description) pairs where the job description
is constructed to match a real resume. Used in eval to test retrieval and reranking
can surface a known positive.

## Step 1: Skeleton Generation (`src/eval/synthetic_positives_generation.py`)

**Input**: Resume text

**Output**: `JobSkeleton` dict with fields:

- `title`: Job title with seniority prefix (e.g., "Senior Backend Engineer")
- `seniority`: Junior / Mid / Senior / Staff
- `years_required`: Raw string, may be a range (e.g., "4-6")
- `domain`: backend / frontend / fullstack / data
- `primary_skills`: List of core required skills
- `secondary_skills`: List of secondary/nice-to-have skills

**Processing**:

1. Build a structured prompt asking the LLM to generate one job skeleton
2. Call Ollama (LLaMA 3.2 3B) with bounded token output (`SKELETON_MAX_TOKENS = 200`)
3. Parse the response into a Python dict by iterating lines, splitting on first `:`,
   normalizing field names (case-insensitive, spaces removed), and extracting comma-separated
   skill lists

**Error handling**:

- Empty or unparseable LLM output raises `ValueError`
- Ollama connectivity failures propagate as `ollama.RequestError`
- Each field defaults to empty string or empty list if missing; only raises if zero
  recognized fields are found (completely malformed response)

## Step 2: Validation (`src/eval/synthetic_posititves_validation.py`)

**Input**: `JobSkeleton` dict from Step 1, plus `ResumeInfo` (seniority, years_experience, primary_skills, domain)

**Output**: Validation outcome dict with keys `passed` (bool), `failed_check` (str | None), `reason` (str | None)

**Processing**: Four rule sets are executed in sequence; the pipeline short-circuits on the first failure.

### Rule Set 1 — Structural Validation

Verifies format compliance: title is non-empty, seniority is one of Junior/Mid/Senior/Staff,
years_required is 0–20, domain is one of backend/frontend/fullstack/data, and primary_skills
has 2–4 items. Uses a single LLM call with a structured checklist prompt.

### Rule Set 2 — Seniority-Years Alignment

Verifies the years_required value is consistent with the seniority level:
Junior ≤ 2, Mid 2–5, Senior 4–8, Staff ≥ 6.
For range strings (e.g., "4-6"), the maximum is taken before comparison.

### Rule Set 3 — Resume-Job Alignment

Verifies that at least 2 of the resume's primary skills appear in the job's primary skills,
job seniority is within ±1 level of resume seniority, and job years required ≤ resume
years_experience + 2.

### Rule Set 4 — Domain Consistency

Verifies the job domain aligns with the resume domain (exact or adjacent match) and that
the domain is consistent with the job title.

### Validation Flow

```
validate_structural(job)                              → PASS/FAIL
    ↓ PASS
validate_seniority_years(job)                         → PASS/FAIL
    ↓ PASS
validate_resume_job_alignment(job, resume_info)       → PASS/FAIL
    ↓ PASS
validate_domain_consistency(job, resume_info)         → PASS/FAIL
    ↓ PASS → accepted skeleton → Step 3: Expansion

Any FAIL → Step 3: Fix/Discard
```

**Error handling**:

- Unparseable LLM responses (neither "PASS" nor "FAIL: ...") are treated as failures
  with reason "Unparseable LLM response: <raw>" and routed to fix/discard
- `ollama.RequestError` and `ollama.ResponseError` propagate to the caller

**Token budget**: `VALIDATION_MAX_TOKENS = 50` — validation responses are at most
one short line ("PASS" or "FAIL: <brief reason>"), so a low token cap is safe and
prevents the model from emitting unsolicited commentary.

## Step 3: Expansion (Future)

The validated skeleton from Step 2 will be expanded into a full synthetic job description
by providing the structured fields as context for a second LLM call.
