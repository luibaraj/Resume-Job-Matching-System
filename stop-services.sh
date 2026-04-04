#!/bin/bash
# chmod +x stop-services.sh
set -e

echo "Stopping Ollama container..."
docker stop ollama 2>/dev/null || true
docker rm ollama 2>/dev/null || true

echo "Stopping docker-compose services..."
docker-compose down

echo "✓ All services stopped"