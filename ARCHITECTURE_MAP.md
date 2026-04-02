# Repository Architecture Map

## Core Components

### Data Processing
- `src/preprocess.py`: HTML cleaning and text normalization
- `src/regex_extraction.py`: Pattern matching for job requirements
- `scripts/pipeline/preprocess_jobs.py`: Job preprocessing pipeline

### Database
- `src/db_utils.py`: Database utility functions
- `src/embedding.py`: Embedding serialization/deserialization

### Evaluation System
#### Positive Generation
- `src/eval/positive_gen/`: Modules for generating positive job matches
  - `positives_gen.py`: Core job skeleton generation
  - `positives_repair.py`: Fixing invalid job skeletons
  - `positives_validate.py`: Validation logic
  - `positives_pipeline.py`: End-to-end pipeline

#### Negative Generation  
- `src/eval/negative_gen/`: Modules for generating negative job matches
  - `negatives_gen.py`: Core mismatch generation
  - `negatives_repair.py`: Fixing invalid mismatches
  - `negatives_validate.py`: Validation logic

#### Evaluation Infrastructure
- `src/eval/collection.py`: ChromaDB collection management
- `src/eval/data_loading.py`: Data sampling and loading
- `src/eval/embedding_cache.py`: Embedding caching system
- `src/eval/metrics.py`: Evaluation metrics calculation
- `src/eval/reporting.py`: Results reporting
- `src/eval/types.py`: Type definitions

### Job Matching
- `src/generation.py`: Core matching pipeline
- `src/retrieval.py`: Job retrieval from ChromaDB
- `src/reranking.py`: Cohere-based reranking

### Scraping
- `src/greenhouse_scraper.py`: Greenhouse job board scraping

## Scripts
- `scripts/eval/`: Evaluation scripts
  - `run_test_eval.py`: Test evaluation runner
  - `run_tuning_eval.py`: Tuning evaluation runner

## Tests
- `tests/`: Unit tests
  - `test_eval_types.py`: Type validation tests
  - `test_generation.py`: Generation pipeline tests
  - `test_reranking.py`: Reranking tests

## Configuration
- `.gitignore`: Git ignore rules
