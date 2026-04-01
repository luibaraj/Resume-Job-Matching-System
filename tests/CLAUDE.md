# tests/CLAUDE.md

19 test files using `pytest`. External services (Ollama, VoyageAI, Cohere, ChromaDB) are mocked except noted.

## Test Coverage Map

**Core Pipeline** (6 files)
- `test_greenhouse_scraper.py` — scraper classes, board discovery, job extraction
- `test_preprocess.py` — preprocessing steps 1–5, full pipeline
- `test_regex_extraction.py` — all regex extraction functions (degree, seniority, years, etc.)
- `test_reranking.py` — `rerank_jobs()`, `_format_document()` (Cohere mocked)
- `test_retrieval.py` — ChromaDB `query_collection()` (uses in-memory temp collections, NOT mocked)
- `test_generation.py` — LLM explanation generation (Ollama mocked)

**Eval Infrastructure** (6 files)
- `test_eval_collection.py` — collection caching, build/rebuild logic, `swap_positives()`
- `test_eval_data_loading.py` — `compute_hash()`, `chunked_select()` batching, `sample_jobs()` CSV cache
- `test_eval_embedding_cache.py` — cache hit/miss, `.npz` key casting, empty text handling
- `test_eval_reporting.py` — JSON structure, per-domain/seniority metrics, MLflow logging
- `test_eval_types.py` — field presence and type checks for TypedDicts
- `test_metrics.py` — edge cases: empty inputs, k > len, empty ground truth, batch lengths

**Positive Generation** (4 files)
- `test_positives_gen.py` — skeleton generation, response parsing, skill/responsibility generation
- `test_positives_validate.py` — all 4 rule sets, normalization, orchestration
- `test_positives_repair.py` — 2-attempt loop, field targeting, temperature drop
- `test_positives_pipeline.py` — safety cap, error handling, JSON output, empty resume guard

**Negative Generation** (3 files)
- `test_negatives_gen.py` — seniority/responsibility mismatch generation
- `test_negatives_validate.py` — seniority gap checks, skill/domain overlap, responsibility mismatch
- `test_negatives_repair.py` — 2-attempt repair loop, `mismatch_context` preservation

## Key Non-Obvious
- `test_retrieval.py` uses real in-memory ChromaDB (ephemeral collections) — only test bypassing mock architecture
- Backwards-compat test aliases in `negatives_repair.py` (`_format_fields_for_prompt`, `_merge_repaired_fields`) — do NOT remove (tests import them)

## Run Tests
```bash
pytest                  # all
pytest --cov=src       # with coverage
```
