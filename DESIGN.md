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
