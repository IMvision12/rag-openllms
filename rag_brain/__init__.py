"""RAG pipeline (ingestion → embeddings → retrieval → generation) with vector or Neo4j backends."""

from rag_brain.config import RetrievalBackend
from rag_brain.pipeline import RAGPipeline

__all__ = ["RAGPipeline", "RetrievalBackend"]
