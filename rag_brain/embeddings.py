import torch
from langchain_huggingface import HuggingFaceEmbeddings

from rag_brain.config import Settings


def get_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
