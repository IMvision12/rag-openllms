"""RAG pipeline with dual vector (Chroma) + graph (Neo4j) backend support."""

import logging
import os
import warnings

# Suppress HuggingFace / sentence-transformers noise before any HF imports
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=".*unauthenticated.*HF Hub.*")
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
warnings.filterwarnings("ignore", message=".*position_ids.*")
for _name in (
    "sentence_transformers",
    "sentence_transformers.SentenceTransformer",
    "huggingface_hub",
    "huggingface_hub.utils",
    "transformers",
):
    logging.getLogger(_name).setLevel(logging.ERROR)

from rag_brain.config import ChunkingStrategy, RetrievalBackend
from rag_brain.pipeline import RAGPipeline

__all__ = ["RAGPipeline", "RetrievalBackend", "ChunkingStrategy"]
