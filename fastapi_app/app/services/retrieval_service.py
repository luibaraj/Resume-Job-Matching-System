"""
Service for dense retrieval from ChromaDB.
"""
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import chromadb
from chromadb.api import ClientAPI, Collection

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.retrieval import query_collection
from src.regex_extraction import build_chroma_where_filter, describe_chroma_filter
from app.config import settings

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for querying ChromaDB collection."""

    def __init__(self):
        self.chroma_client: ClientAPI = chromadb.PersistentClient(
            path=settings.CHROMA_DIR
        )
        self.collection: Optional[Collection] = None

    def get_collection(self) -> Collection:
        """Get or create the Chroma collection."""
        if self.collection is None:
            try:
                self.collection = self.chroma_client.get_collection(
                    settings.CHROMA_COLLECTION_NAME
                )
            except ValueError:
                logger.error(
                    "Collection '%s' not found. Please run the embedding pipeline first.",
                    settings.CHROMA_COLLECTION_NAME,
                )
                raise
        return self.collection

    def query(
        self,
        query_embedding: List[float],
        top_k: int = settings.RETRIEVE_TOP_K,
        where_filter: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the collection for similar jobs.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of candidates to retrieve.
            where_filter: ChromaDB where filter dict.
            run_id: Optional trace ID.

        Returns:
            List of candidate job dicts.
        """
        collection = self.get_collection()
        candidates = query_collection(
            collection=collection,
            query_embedding=query_embedding,
            top_k=top_k,
            ef=settings.HNSW_EF,
            where=where_filter,
            run_id=run_id,
        )
        return candidates
