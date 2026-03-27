"""
Evaluation-specific configuration constants.

These constants define the tuning and evaluation protocol parameters,
distinct from the project-wide config in src/config.py.
"""

# Sampling
TUNE_SAMPLE_N: int = 1000
TEST_SAMPLE_N: int = 1000
SAMPLE_SEED: int = 42

# Metrics
K_PRECISION: int = 5
K_RECALL: int = 10

# ChromaDB (eval-specific collection, separate from production)
CHROMA_TUNE_EVAL_DIR: str = "data/chroma_tune_eval"
CHROMA_TUNE_EVAL_COLLECTION: str = "tune_eval"

# Paths
TUNE_RESUMES_PATH: str = "data/eval/tune/resumes.csv"
TUNE_POSITIVES_PATH: str = "data/eval/tune/positives.csv"
TUNE_SAMPLED_JOBS_PATH: str = "data/eval/tune/sampled_jobs.csv"
TEST_SAMPLED_JOBS_PATH: str = "data/eval/test/sampled_jobs.csv"
RESULTS_DIR: str = "data/eval/results"
TUNE_RESULTS_JSON: str = "data/eval/results/tune_eval_results.json"
TUNE_MISSED_CSV: str = "data/eval/results/tune_missed_positives.csv"

# Cache paths for embeddings (to avoid recomputing on re-runs)
TUNE_POSITIVE_EMBEDDINGS_CACHE: str = "data/eval/tune/positive_embeddings.npz"
TUNE_POSITIVE_EMBEDDINGS_HASH: str = "data/eval/tune/positive_embeddings.hash"
TUNE_RESUME_EMBEDDINGS_CACHE: str = "data/eval/tune/resume_embeddings.npz"
TUNE_RESUME_EMBEDDINGS_HASH: str = "data/eval/tune/resume_embeddings.hash"
TUNE_SAMPLED_JOBS_HASH: str = "data/eval/tune/sampled_jobs.hash"
