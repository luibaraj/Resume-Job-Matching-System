# Embedding Model

## Voyage AI `voyage-3.5-lite`

My main constraints are good performance for symmetric retrieval (query is approximately the same length as the documents) and low cost. This Voyage AI model provides competitive performance for retrieval tasks, a generous free tier, and still presents low cost after the free tier is diminished. The main caveat is that this model has an asymmetric design and prepends text based on if the input is a query, document, or none. Since we are comparing content-to-content and not question-to-answer, we can simply set input type to none so that prepending is skipped entirely.
