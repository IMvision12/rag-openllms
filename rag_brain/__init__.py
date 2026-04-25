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

# Our own logger — prints to the terminal that launched streamlit / CLI.
# Override level via `RAG_LOG_LEVEL=DEBUG` for very verbose output (per-chunk
# extraction details, raw LLM responses, etc.).
_rag_logger = logging.getLogger("rag_brain")
_rag_logger.setLevel(os.environ.get("RAG_LOG_LEVEL", "INFO").upper())
if not _rag_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    _rag_logger.addHandler(_h)
_rag_logger.propagate = False

from rag_brain.config import ChunkingStrategy, RetrievalBackend
from rag_brain.pipeline import RAGPipeline

__all__ = ["RAGPipeline", "RetrievalBackend", "ChunkingStrategy"]
