#!/bin/bash

# Start the FastAPI application with uvicorn

set -e

cd "$(dirname "$0")/.."

# Check if .env exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Please create one from .env.example"
fi

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
