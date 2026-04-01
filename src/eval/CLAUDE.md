# src/eval/CLAUDE.md

Eval sub-package for synthetic data generation and pipeline evaluation. Modular 10-file structure:

## Module Dependency Chain

**types.py** → **eval_config.py** → **data_loading.py**, **embedding_cache.py** → **collection.py**, **metrics.py**, **reporting.py**, **eval_utils.py**

## Key Non-Obvious Details

**types.py**
- `PositiveRetrievalStatus` and `ResumeEvalResult` TypedDicts
- `reranker_hit` is `Optional[bool]` to distinguish "reranker skipped" from "reranker missed"

**eval_config.py**
- Eval-only constants (paths, k-values, MLflow URIs)
- Separate from root `src/config.py`
- Mirrored tune/test path constants (swap args to run either dataset)

**data_loading.py**
- `chunked_select()` batches large `IN (?)` queries in 500-ID chunks (SQLite param limit)
- `sample_jobs()` deterministic sampling (seed=42) cached to CSV
- Template placeholder is `{}` in query, not `%s`

**embedding_cache.py**
- `.npz` + `.hash` caching for VoyageAI calls
- Numpy constraint: npz keys are strings; int resume IDs cast to/from str at boundaries

**collection.py**
- Per-eval ChromaDB collection (separate from production)
- ChromaDB IDs prefixed: `"pos_"` for positives, `"job_"` for jobs (prevents collisions)
- `swap_positives()` atomically injects ground truth per-resume before retrieval query
- Stores regex-extracted metadata (degree, seniority, years) as document metadata

**metrics.py**
- Pure, framework-free `precision_at_k`, `recall_at_k` functions
- When `k > len(retrieved)`, denominator = actual length (no padding)

**reporting.py**
- Writes JSON + logs to MLflow; metric names sanitized (spaces/slashes → underscores)
- Distributional stats (std, min, max) only logged if active MLflow run exists

**eval_utils.py**
- Shared LLM wrappers + prompt builders for repair/validate modules
- `merge_repaired_fields()` partial merge: only non-empty repaired values applied

## Sub-Packages
- **positive_gen/** — 4-file pipeline (gen → validate → repair → orchestrate)
- **negative_gen/** — 3-file pipeline (gen → validate → repair)

See their own CLAUDE.md files.
