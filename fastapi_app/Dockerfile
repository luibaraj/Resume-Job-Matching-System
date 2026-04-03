FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-fastapi.txt .
RUN pip install --no-cache-dir -r requirements-fastapi.txt

# Copy the FastAPI app
COPY ./app ./app

# Copy the parent project (src, config, etc.)
COPY ../src ../src
COPY ../data ../data
COPY ../.env ../.env
COPY ../requirements.txt ../requirements.txt

# Install project dependencies
RUN pip install --no-cache-dir -r ../requirements.txt

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
