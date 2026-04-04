# --- builder ---
FROM python:3.11-slim AS builder
WORKDIR /app
COPY fastapi_app/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- runtime ---
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY fastapi_app /app/fastapi_app
COPY src/__init__.py src/config.py src/embedding.py src/retrieval.py src/reranking.py src/regex_extraction.py /app/src/
EXPOSE 8000
CMD ["uvicorn", "fastapi_app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]                                                                                                                             
