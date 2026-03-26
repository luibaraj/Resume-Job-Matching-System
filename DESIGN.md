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

## Step 3: Failure Recovery (`src/eval/positives_repair.py`)

**Input**: Failed `JobSkeleton`, `failed_check` string, `reason` string, `ResumeInfo`

**Output**: `RepairResult` dict: `success` (bool), `job` (JobSkeleton | None), `attempts` (int), `discard_reason` (str | None)

**Retry strategy**: Up to 2 repair attempts. Each attempt calls the LLM with a targeted
fix prompt, parses via `parse_skeleton_response`, and re-runs all 4 validations.

- **Attempt 1**: Only the fields relevant to the failing check are sent to the LLM. The model
  outputs only those fields back. The repaired values are merged into the original skeleton
  (unchanged fields preserved). Temperature `GENERATION_TEMPERATURE = 0.7`.
- **Attempt 2**: Same targeted approach but stricter — explicit enum hints inline in the
  format template, imperative close ("You MUST correct this."). Temperature lowered to `0.3`
  for more deterministic output. If attempt 1 shifted the failure to a different check,
  attempt 2 targets the new failing fields.
- **Discard**: After 2 failed attempts, returns `RepairResult(success=False, job=None)`.

| `failed_check`         | Fields sent to LLM                                                 | Fix instruction                                                         |
| ---------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| `structural`           | `seniority`, `domain`, `years_required`, `primary_skills`, `title` | Fix enum values, years range 1–20, skills count 2–4                     |
| `seniority_years`      | `seniority`, `years_required`                                      | Fix YearsRequired to bracket: Junior 0–2, Mid 2–5, Senior 4–8, Staff 6+ |
| `resume_job_alignment` | `primary_skills`, `seniority`                                      | Fix skills (≥2 overlap with resume), seniority (±1 of resume level)     |
| `domain_consistency`   | `domain`                                                           | Fix domain to match resume domain + title                               |

**Error handling**:

- `ValueError` from `parse_skeleton_response` is caught and counted as a failed attempt
- `ollama.RequestError` / `ollama.ResponseError` propagate to the caller

**Token budget**: `REPAIR_MAX_TOKENS = 200` — same output format as skeleton generation.

## Step 4: Expansion (Future)

The validated (or repaired) skeleton from Steps 2–3 will be expanded into a full synthetic
job description by providing the structured fields as context for a second LLM call.

## Step 5: Orchestration (`src/eval/positives_pipeline.py`)

**Input**: Resume text + `ResumeInfo`, target skeleton count

**Output**: List of validated `JobSkeleton` dicts (up to target count)

**Processing**: Orchestrates Steps 1–3 in a loop with automatic retry and discard logic.

### Pipeline Loop

```
while collected < target_count:
    ↓ (check safety cap)
    Generate skeleton from resume
        → on parse/Ollama error: continue (count attempt, restart)
    ↓
    Validate against all 4 rules
        → PASS: collect and continue
        → FAIL: proceed to repair
    ↓
    Repair (max 2 attempts internally)
        → success: collect and continue
        → failure: discard and restart from scratch
```

### Safety Cap

- Max total attempts = `target_count * 10`
- If exhausted before collecting `target_count`, return partial list (no exception)
- Prevents infinite loops if Ollama is unstable or model quality is poor

### Entry Point

```python
def run_pipeline(
    resume_text: str,
    resume_info: ResumeInfo,
    target_count: int = 5,
    model: str = OLLAMA_MODEL,
    output_path: str | None = "data/synthetic_positives.json",
) -> list[JobSkeleton]:
```

- `target_count`: Number of valid skeletons to collect (default: 5)
- `output_path`: If not None, write collected skeletons to JSON after loop (default: `"data/synthetic_positives.json"`)
- Returns: List of validated skeletons (may be shorter than `target_count` if safety cap hit)

### Error Handling

- **Ollama connectivity failure** (generation, validation, or repair): caught, logged, counted as 1 attempt, loop continues
- **LLM parse failure** (generation): caught, logged, counted as 1 attempt, loop continues
- **Repair exhausts attempts**: skeleton discarded, loop restarts from step 1
- **Safety cap hit**: loop exits, returns partial result

### Observability

- Prints at each stage: generation, validation pass/fail, repair pass/fail, discard reason
- Final summary: total collected, total attempts, discard count, repair success count
- Optional JSON write with timestamps/metadata (for inspection of generated skeletons)

---

# Synthetic Negatives Pipeline

## Purpose

Synthetic negatives are (r) pairs where the job description is intentionally mismatched to the resume in specific ways. Used in eval to test retrieval and reranking correctly reject non-matching jobs.

Three mismatch types are fully implemented: **seniority**, **domain**, and **responsibility**.

## Mismatch Types

Each negative type creates a different dimension of incompatibility while maintaining plausibility. The following table summarizes what differs and what remains consistent:

| Mismatch Type  | What differs                            | What stays the same                                            |
| -------------- | --------------------------------------- | -------------------------------------------------------------- |
| seniority      | seniority level, years_required         | domain, primary skills, secondary skills, responsibilities     |
| domain         | domain, job title                       | seniority level, years_required, skills are domain-appropriate |
| responsibility | responsibilities, specific skills focus | seniority level, domain, years_required                        |

### Seniority Mismatch

A job description that is realistic in domain and skills but targets a mismatched seniority level:

- **Junior candidate** → Senior or Staff job (overqualification, cannot handle complexity)
- **Mid candidate** → Junior or Staff job (extreme under/over qualification)
- **Senior candidate** → Junior job (underqualification, would be bored)
- **Staff candidate** → Junior or Mid job (underqualification)

The seniority gap must be ≥1 level for Mid-level candidates, ≥2 levels for Junior/Senior/Staff (to ensure true mismatch).

### Domain Mismatch

A job description with the same seniority and years as the resume, but in a completely different engineering domain:

- **backend engineer** → frontend or data role
- **frontend engineer** → backend or data role
- **fullstack engineer** → data role
- **data engineer** → frontend or backend role

A domain shift is valid only if it is not "adjacent" (backend ↔ fullstack and frontend ↔ fullstack are too similar to count as true mismatches). Data has no adjacency and can mismatch with any other domain.

### Responsibility Mismatch

A job description with the same seniority, domain, and years as the resume, but with a different sub-role or specialization within that domain:

- **backend engineer focused on API design** → backend engineer focused on database optimization
- **frontend engineer focused on component libraries** → frontend engineer focused on mobile web
- **data engineer focused on warehousing** → data engineer focused on real-time streaming

The responsibilities must describe work the candidate demonstrably does not do, verified by the validation step.

## Architecture

Follows the same **generate → validate → repair** pattern as positives.

### Step 1: Skeleton Generation (`src/eval/negative_gen/negatives_gen.py`)

**Input**: Resume text, resume seniority level, mismatch type, resume domain (for domain/responsibility types)

**Output**: `(JobSkeleton dict, mismatch_context dict)` tuple

- Tuple is used because target selection uses `random.choice`; repair needs the exact target that was chosen

**Processing**:

1. Dispatch based on `mismatch_type`:
   - **seniority**: `get_target_seniority(resume_seniority)` → pick a mismatched target seniority
     - Junior → Senior or Staff (random)
     - Mid → Junior or Staff (random)
     - Senior → Junior
     - Staff → Junior or Mid (random)
   - **domain**: `get_target_domain(resume_domain)` → pick a mismatched target domain (excluding adjacent)
   - **responsibility**: same seniority/domain, diverge sub-role within domain
2. `_years_range_for_seniority(target_seniority)` → get canonical years for target (e.g., "4-7" for Senior)
3. Build prompt with hard constraints for the mismatch dimension; resume text provides context for non-mismatch dimensions
4. Call Ollama with same config as positives_gen (`SKELETON_MAX_TOKENS = 200`)
5. Parse response into JobSkeleton dict
6. Return tuple: `(skeleton dict, mismatch_context)` where context carries `target_seniority`, `target_domain`, and/or `resume_domain` depending on type

Three prompt builders:

- `_build_mismatched_skeleton_prompt(resume_text, target_seniority, years_range)` — seniority hard-constrained; domain/skills from resume context
- `_build_domain_mismatch_prompt(resume_text, resume_seniority, target_domain)` — domain hard-constrained; seniority matches resume
- `_build_responsibility_mismatch_prompt(resume_text, resume_seniority, resume_domain)` — seniority+domain match resume; sub-role diverges

Canonical years ranges by seniority: Junior (0-2), Mid (2-4), Senior (4-7), Staff (7-10)

**Error handling**:

- Same as positives_gen: ValueError on unparseable response, Ollama errors propagate

### Step 2: Validation (`src/eval/negative_gen/negatives_validate.py`)

**Input**: `JobSkeleton` dict from Step 1, `ResumeInfo`, mismatch_type string

**Output**: Validation outcome dict with keys `passed` (bool), `failed_check` (str | None), `reason` (str | None)

**Processing**: Entry point is `validate_mismatched_skeleton(job, resume_info, model, mismatch_type)`, which dispatches to a type-specific validation chain:

| mismatch_type    | Validation chain                                                         |
| ---------------- | ------------------------------------------------------------------------ |
| `seniority`      | structural → seniority_years → seniority_mismatch → skill_domain_overlap |
| `domain`         | structural → seniority_years → domain_mismatch                           |
| `responsibility` | structural → seniority_years → responsibility_mismatch                   |

Each check runs in sequence; pipeline short-circuits on the first failure.

All checks and their logic:

| Check                     | Type          | Logic                                                                                                                                                        |
| ------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `structural`              | LLM           | All 7 fields present, well-formed, in ranges (same as positives)                                                                                             |
| `seniority_years`         | Deterministic | Skeleton's seniority and years_required are internally consistent (same as positives)                                                                        |
| `seniority_mismatch`      | Deterministic | Job seniority gap must meet minimum: Mid resume ≥1 level gap, Junior/Senior/Staff ≥2 level gap                                                               |
| `domain_mismatch`         | Deterministic | Job domain ≠ resume domain AND job domain is not adjacent to resume domain (backend ↔ fullstack and frontend ↔ fullstack are adjacent)                       |
| `responsibility_mismatch` | LLM           | Job responsibilities describe work the candidate demonstrably does not do, verified against resume text                                                      |
| `skill_domain_overlap`    | LLM           | ≥2 shared skills, same/adjacent domain, realistic responsibilities for the seniority level — ignoring seniority distance (used only for seniority negatives) |

**Key differences from positives**:

- `validate_resume_job_alignment` NOT reused (it enforces seniority proximity ±1, which negatives intentionally violate)
- `validate_domain_consistency` is NOT reused; `domain_mismatch` is deterministic and checks adjacency instead
- New `responsibility_mismatch` check verifies responsibilities are plausibly unrelated to the resume
- `skill_domain_overlap` replaces positives' alignment logic; verifies plausibility without enforcing seniority proximity

**Token budget**: `VALIDATION_MAX_TOKENS = 50` (same as positives)

### Step 3: Failure Recovery (`src/eval/negative_gen/negatives_repair.py`)

**Input**: Failed `JobSkeleton`, `failed_check` string, `reason` string, `ResumeInfo`, **`mismatch_context`** dict (from generation), **`mismatch_type`** string

**Output**: `RepairResult` dict: `success` (bool), `job` (JobSkeleton | None), `attempts` (int), `discard_reason` (str | None)

**Retry strategy**: Up to 2 repair attempts, same as positives_repair.

**Critical difference**: `mismatch_context` is passed as a parameter (not re-computed via `get_target_seniority` or `get_target_domain`) to ensure repair targets the same mismatch goal that generation chose.

Field targeting by failed check (which fields the LLM regenerates):

| `failed_check`            | Fields sent to LLM                                                         | Fix instruction                                                                       |
| ------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `structural`              | seniority, domain, years_required, primary_skills, title, responsibilities | Fix enum values, years range, skills count, responsibilities count                    |
| `seniority_years`         | seniority, years_required                                                  | Fix YearsRequired to bracket for target seniority (same as positives)                 |
| `seniority_mismatch`      | seniority, years_required, title                                           | Set seniority to target_seniority from mismatch_context and years to matching bracket |
| `domain_mismatch`         | domain, title                                                              | Set domain to target_domain from mismatch_context; update title to match              |
| `responsibility_mismatch` | responsibilities, primary_skills, secondary_skills                         | Rewrite responsibilities for a different sub-role within same domain                  |
| `skill_domain_overlap`    | primary_skills, secondary_skills, domain, responsibilities                 | Fix skills/domain to overlap with resume; do not change seniority                     |

**Temperature and attempt strategy**:

- Attempt 1: `GENERATION_TEMPERATURE = 0.7`, targeted fields
- Attempt 2: `0.3`, explicit enum hints in prompt, targeted fields; may shift to new failing check if first attempt changed the failure signature
- After 2 failures: discard

**Token budget**: `REPAIR_MAX_TOKENS = 200` (same as positives)

---

## Shared Types and Utilities

**Owned by positives_validate, imported by negatives**:

- `JobSkeleton` — TypedDict (7 fields)
- `ResumeInfo` — TypedDict
- `ValidationResult` — TypedDict for check results
- `_normalize_skeleton(job)` — Canonicalizes aliases
- `_parse_validation_response(response)` — Parses "PASS" or "FAIL: reason"
- `validate_structural(job, model)` — Reused directly
- `validate_seniority_years(job, model)` — Reused directly
- `parse_skeleton_response(response)` — Parses raw LLM response into dict
- Seniority brackets: Junior (0-2), Mid (0-5), Senior (2-8), Staff (6+)

**Owned by negatives, unique to negatives**:

- `SENIORITY_ORDER` — ["Junior", "Mid", "Senior", "Staff"]
- `get_target_seniority(resume_seniority)` — Pick mismatched target
- `_years_range_for_seniority(seniority)` — Map to canonical range
- `validate_seniority_mismatch(job, resume_info)` — Deterministic gap check
- `validate_skill_domain_overlap(job, resume_info, model)` — LLM alignment check (no seniority)
- Min gap dict: Junior 2, Mid 1, Senior 2, Staff 2

---

## Future Negative Types

Additional negative types can be plugged into `src/eval/negative_gen/` as separate modules (e.g., `negatives_skill_gap.py`, `negatives_industry_mismatch.py`), each following the same generate → validate → repair pattern with type-specific validators.

A future `negatives_pipeline.py` would orchestrate multiple negative types together, collecting a balanced set (e.g., 5 seniority-mismatch + 5 skill-gap negatives) in a single call.
