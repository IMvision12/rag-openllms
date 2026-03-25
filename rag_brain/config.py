from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalBackend(str, Enum):
    vector = "vector"
    neo4j = "neo4j"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- Retrieval store ---
    retrieval_backend: RetrievalBackend = Field(
        default=RetrievalBackend.vector,
        alias="RAG_BACKEND",
        description="vector = Chroma; neo4j = Neo4jVector + Neo4jGraph",
    )

    # --- Chroma (vector) ---
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma_db"),
        alias="CHROMA_PERSIST_DIR",
    )
    chroma_collection: str = Field(default="rag_docs", alias="CHROMA_COLLECTION")

    # --- Neo4j ---
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="", alias="NEO4J_PASSWORD")
    neo4j_database: str | None = Field(default=None, alias="NEO4J_DATABASE")
    neo4j_vector_index: str = Field(default="rag_chunk_vectors", alias="NEO4J_VECTOR_INDEX")

    # --- Embeddings (local sentence-transformers; aligns with proposal) ---
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )

    # --- Chunking ---
    chunk_size: int = Field(default=1200, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    # --- Retrieval ---
    top_k: int = Field(default=4, alias="TOP_K")

    # --- LLM: Ollama (local OSS) or OpenAI-compatible ---
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")  # ollama | openai
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3", alias="OLLAMA_MODEL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")


def load_settings() -> Settings:
    return Settings()
