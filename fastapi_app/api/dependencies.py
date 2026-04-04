import os
import sqlite3
from functools import lru_cache
import voyageai
import chromadb
import cohere
import requests
from src.embedding import create_client

@lru_cache
def get_voyage_client():
    return create_client(os.environ["VOYAGE_API_KEY"])

@lru_cache
def get_chroma_collection():
    client = chromadb.PersistentClient(path=os.environ["CHROMA_DIR"])
    return client.get_collection(os.environ["CHROMA_COLLECTION"])

@lru_cache
def get_cohere_client():
    return cohere.Client(os.environ["COHERE_API_KEY"])

@lru_cache
def get_ollama_base_url():
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

def get_db():
    conn = sqlite3.connect(os.environ["DB_PATH"], check_same_thread=False)   
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def check_ollama_health(base_url: str) -> tuple[bool, str]:
    """
    Check if Ollama server is healthy.
    
    Args:
        base_url: Ollama server URL (e.g., "http://localhost:11434")
    
    Returns:
        Tuple of (is_healthy: bool, message: str)
    """
    try:
        # Check if server is reachable
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        
        # Check if required model exists
        data = response.json()
        models = data.get("models", [])
        required_model = "llama3.2:3b-instruct-q4_K_M"
        
        if not any(model.get("name") == required_model for model in models):
            return False, f"Required model '{required_model}' not found in Ollama"
        
        return True, "Ollama is healthy"
    
    except requests.exceptions.ConnectionError as e:
        return False, f"Ollama connection failed: {str(e)}"
    except requests.exceptions.Timeout as e:
        return False, f"Ollama request timed out: {str(e)}"
    except requests.exceptions.HTTPError as e:
        return False, f"Ollama server error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error checking Ollama: {str(e)}"
