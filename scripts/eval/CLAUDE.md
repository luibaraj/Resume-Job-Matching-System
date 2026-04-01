# scripts/eval/CLAUDE.md

5 scripts for evaluation pipeline: generate synthetic data, stratify, split, run evals on tune/test sets.

## Run Order

`generate_synthetic_positives.py` → `generate_synthetic_negatives.py` → `stratify_and_split.py` → `run_tuning_eval.py` → `run_test_eval.py`

## Per-File Details

**generate_synthetic_positives.py**
- Reads `data/eval/synthetic_resume.csv`
- Runs full positive pipeline (target 5 valid jobs per resume)
- Outputs: `data/eval/synthetic_job_descriptions.csv` (UUID, job skeleton fields, timestamp)

**generate_synthetic_negatives.py**
- Reads same resume CSV; imports `build_resume_info()` and `format_job_for_embedding()` from positives script via `sys.path`
- Generates 3 negatives per resume: 1 seniority mismatch + 2 responsibility mismatches
- Inline generate→validate→repair loop (no separate pipeline module like positives)
- Outputs: `data/eval/synthetic_negative_job_descriptions.csv`

**stratify_and_split.py**
- Loads 3 CSVs (resumes, jobs, negatives); normalizes seniority/domain
- Stratified split by `(seniority, domain)` pair: 30 tune / 20 test
- Hamilton method for proportional allocation (solves rounding via largest remainder)
- Emits `data/eval/split_summary.json` with per-stratum counts + singleton warnings

**run_tuning_eval.py**
- 3-phase loop: (1) retrieval, (2) batch Cohere reranking, (3) scoring
- Per-resume: swap positives into ChromaDB collection, dense retrieval (top 100), classify hits/misses, rerank batch, compute metrics@k
- Wraps entire run in `mlflow.start_run()` for centralized logging
- Output: JSON + CSV to `data/eval/results/`

**run_test_eval.py**
- Near-duplicate of `run_tuning_eval.py` for held-out test set
- Uses `TEST_*` constants from `eval_config`, separate ChromaDB collection (`CHROMA_TEST_EVAL_COLLECTION`), test-specific cache paths
- Contains its own `write_test_results_json()` and `write_test_missed_positives_csv()` (deliberate duplication to isolate test vs tune paths and outputs)
- Output: JSON + CSV to `data/eval/results/test/`

## Key Non-Obvious
- Negatives script imports from positives script via `sys.path` (not a shared module)
- Test eval duplication is intentional — isolated paths prevent accidental cross-contamination
