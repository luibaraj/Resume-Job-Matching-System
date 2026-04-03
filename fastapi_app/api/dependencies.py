import os
import sqlite3
from functools import lru_cache
import voyageai
import chromadb
from src.embedding import create_client

@lru_cache
def get_voyage_client():
    return create_client(os.environ["VOYAGE_API_KEY"])

@lru_cache
def get_chroma_collection():
    client = chromadb.PersistentClient(path=os.environ["CHROMA_DIR"])
    return client.get_collection(os.environ["CHROMA_COLLECTION"])

def get_db():
    conn = sqlite3.connect(os.environ["DB_PATH"])
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
