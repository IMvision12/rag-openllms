import logging
import os
import warnings

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from rag_brain.config import Settings


def _suppress_hf_noise() -> None:
    """Silence noisy HuggingFace / sentence-transformers loading output."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore", message=".*unauthenticated.*HF Hub.*")
    warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
    for name in (
        "sentence_transformers",
        "sentence_transformers.SentenceTransformer",
        "huggingface_hub",
        "transformers",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    _suppress_hf_noise()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )
