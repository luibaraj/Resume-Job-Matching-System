#!/bin/bash
set -e

echo "Starting services..."

# Start Ollama on host
ollama serve &

# Wait for Ollama to be ready
echo "Waiting for Ollama..."
until curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do sleep 1; done
echo "Ollama ready"

# Start Docker Compose services
docker-compose up -d

# Wait for app to be healthy
echo "Waiting for app..."
until curl -sf http://localhost:8000/health >/dev/null 2>&1; do sleep 2; done
echo "App ready"

# Open the UI in the default browser
open http://localhost:8000 2>/dev/null || xdg-open http://localhost:8000 2>/dev/null || true

echo "Services started successfully!"
echo "App available at http://localhost:8000"
echo "Nginx available at http://localhost:80"
