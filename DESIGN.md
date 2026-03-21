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

# Retrieval Tuning Experiment

The main objective of this experiment is to find the lowest parameter for `ef` that passes a pre-defined threshold for recall.

## Recall

As apart of the business problem, the system should not miss any potential resume-job matches. To frame this mathematically, this means to reduce the number of false negatives. The best metric to measure the impact of false negatives in performance is recall. Let Recall = TP / (TP + FN) where TP = retrieved documents that are relevant and FN = relevant documents that were NOT retrieved.

## Benchmark

The system will only be tested using a singlular resume/query. The reasoning behind this is to keep developement simple, cost-efficient, and allow the system to be optimized differently per-user while using their own profile (e.g. to let users decide the latency-recall trade-off themselves).

The experiment should be orchestrated as follows:

1. Extract key details from the user profile
   - 3-5 top skills
   - 3-5 top responsbilities
   - job titles
   - experience level (new grad/junior/mid/senior)
2. Create job title variations
   - 3-4 job titles
3. Create a job description template with 90-80% skill overlap
4. Generate 50 varying job descrpitons using the template from step 3 (these are the relevant documents to the user query in this experiment)
5. Embed each job description from step 4 using the embedding module
6. Retrieve 1000 random embeddings and their meta data from jobs.db
7. Create a vector databse and query the user profile while iterating through values of `ef` (200-600), and break the loop once recall performance surpasses a specified threshold
8. Store the optimized value for `ef` in config.py
