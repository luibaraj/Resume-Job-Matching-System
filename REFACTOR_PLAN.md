# Production Refactor Plan: scripts/pipeline/

Refactor all 4 pipeline scripts to meet the 8 production standards. No changes to `src/`. No new shared utility modules.

---

## Step 1: Fix Structural Issues

### scrape_jobs.py
- Move `logging.basicConfig()` from module level into `async def main()` (Standards #4, #7)
- Change all `logger.*` f-string calls to `%s`-style lazy formatting (Standard #5, #7)

### preprocess_jobs.py
- Replace `sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))` with `str(Path(__file__).resolve().parent.parent.parent)` (Standard #4)
- Add `from pathlib import Path` to imports

### embed_jobs.py
- Fix misleading log message `"after all retries failed"` → `"embed_batch raised after exhausting retries — {type}: {exc}"` (Standard #4, #7)
- Add inline comment on `DB_CHUNK_SIZE = 512` explaining power-of-two alignment. Also change `CHUNK_SIZE = 500` in `preprocess_jobs.py` to `512` for consistency. (Standard #4)

### match_jobs.py
- Add `_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent` at module level; consolidate existing `src_path` to use it (Standard #4)
- Move late import `from config import CORPUS_LIMITATION_MESSAGE` (line 196 in `write_results_markdown`) into top-level `from config import (...)` block (Standard #4)
- Add `EMBEDDING_DIM` to top-level config import (needed for Step 2 cache validation)
- Add `logger = logging.getLogger(__name__)` at module level; replace all bare `logging.*` calls in function bodies with `logger.*` (Standard #7)

---

## Step 2: Fix Logic Errors

### scrape_jobs.py
- Add retry loop in `scrape_board_safe()` for `requests.exceptions.ConnectionError` and `requests.exceptions.Timeout` (HTTP-layer retries already handled by `GreenhouseScraper`). Up to 3 attempts, exponential back-off `delay = 2.0 * 2**(attempt-1)`. Add named constants:
  ```python
  _SCRAPE_MAX_RETRIES: int = 3
  _SCRAPE_RETRY_BASE_DELAY: float = 2.0
  ```
  Non-retryable exceptions exit immediately. (Standard #3, #4)

### preprocess_jobs.py
- Replace list comprehension building `updates` with explicit per-row loop; wrap each `preprocess_description()` in `try/except Exception`; log `logger.warning(...)` per failure; fall back to `cleaned = ""`. Prevents bad row from aborting entire batch + infinite loop. (Standard #1, #3)

### match_jobs.py
- Move `VOYAGE_API_KEY` and `COHERE_API_KEY` validation to immediately after `load_dotenv()`, before `build_collection()` (which is expensive). Remove existing validation blocks at lines 335 and 344. (Standard #3, #2)
- Add `try/except` around `np.load()` in `load_or_embed_resume()` cache-hit branch. On failure, log `logger.warning(...)` and fall through to re-embed. (Standard #1)

---

## Step 3: Add Observability

### scrape_jobs.py
- Add `run_id = uuid.uuid4().hex[:8]`; include in initial and summary log lines. Add `import uuid`. (Standard #7)
- Time each stage in `main()`: `init_db`, `scrape_all_boards`, `write_jobs_to_db`. (Standard #7)

### preprocess_jobs.py
- Add jobs/sec throughput to per-batch progress log (Standard #7)
- Add total elapsed + average throughput to final `"Done."` log (Standard #7)

### embed_jobs.py
- Add `DEBUG`-level per-sub-batch timing log: job count, duration, id range (Standard #7)
- Add total elapsed + throughput to final `"Done."` log (Standard #7)

### match_jobs.py
- Time each stage in `main()`: `build_collection`, `load_or_embed_resume`, `query_collection`, `rerank_jobs`, `run_generation_for_results`. Add `import time`. (Standard #7)

---

## Step 4: Add argparse + Standardize Interfaces

### All 4 scripts — add to `main()`:
```python
parser = argparse.ArgumentParser(...)
parser.add_argument("--db-path", default=None)
parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
args = parser.parse_args()

logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
load_dotenv()
db_path = args.db_path or os.getenv("DB_PATH", DB_DEFAULT_PATH)
```
(Standard #4, #6)

### match_jobs.py — additional args:
```
--resume-path   Override resume file path
--output-path   Override output .md path (default: matched_jobs.md)
--rebuild       Rebuild ChromaDB collection (default: False; use only when embeddings change)
```

### match_jobs.py — eliminate `print()`:
Replace all `print(..., file=sys.stderr)` and status `print(...)` with `logger.error/warning/info`. (Standard #7)

| Location | Before | After |
|---|---|---|
| `load_resume()` errors | `print(..., file=sys.stderr)` | `logger.error(...)` |
| `main()` DB not found | `print(..., file=sys.stderr)` | `logger.error(...)` |
| `main()` no embedded jobs | `print(..., file=sys.stderr)` | `logger.error(...)` |
| `main()` no matches | `print("No matching jobs found.")` | `logger.warning(...)` |
| `write_results_markdown()` | `print(f"Results written to {output_path}")` | `logger.info(...)` |
| `main()` except handlers | `print(..., file=sys.stderr)` | `logger.error(...)` |

---

## Step 5: Add Tests

### New: tests/test_scrape_jobs.py
- `test_init_db_idempotent`
- `test_write_jobs_skips_exceptions`
- `test_scrape_board_safe_success`
- `test_scrape_board_safe_retries_connection_error` (fails 2x, succeeds 3rd)
- `test_scrape_board_safe_non_retryable_no_retry`

### Extend: tests/test_preprocess.py
- `test_bad_row_isolated`: mock `preprocess_description` to raise for 1/3 rows; all 3 reach `preprocessed=1`
- `test_main_db_path_arg`: patch `sys.argv`; verify `run_preprocessing` called with correct path

### New: tests/test_embed_jobs.py
- `test_run_embedding_adds_columns`
- `test_run_embedding_skips_empty_descriptions`
- `test_run_embedding_bad_batch_skipped`
- `test_run_embedding_idempotent`
- `test_main_voyage_key_missing` → `sys.exit(1)`

### New: tests/test_match_jobs.py
- `test_load_resume_resolved_from_project_root` (not CWD)
- `test_load_resume_empty_file` → `sys.exit(1)`
- `test_load_or_embed_resume_cache_hit`
- `test_load_or_embed_resume_corrupted_cache` → re-embed
- `test_api_keys_validated_before_build_collection`
- `test_write_results_markdown_no_explanations` → limitation message

---

## Files Modified
- `scripts/pipeline/scrape_jobs.py`
- `scripts/pipeline/preprocess_jobs.py`
- `scripts/pipeline/embed_jobs.py`
- `scripts/pipeline/match_jobs.py`
- `tests/test_preprocess.py`

## Files Created
- `tests/test_scrape_jobs.py`
- `tests/test_embed_jobs.py`
- `tests/test_match_jobs.py`

---

## Verification
```bash
python scripts/pipeline/scrape_jobs.py --help
python scripts/pipeline/preprocess_jobs.py --help
python scripts/pipeline/embed_jobs.py --help
python scripts/pipeline/match_jobs.py --help

pytest tests/test_scrape_jobs.py tests/test_embed_jobs.py tests/test_match_jobs.py -v
pytest --cov=src --cov=scripts -v
```

---

## Resolved Decisions
- Both `CHUNK_SIZE` (preprocess) and `DB_CHUNK_SIZE` (embed) → `512` (power-of-two, consistent)
- `--rebuild` defaults to `False`; must be explicitly passed to clear ChromaDB collection
