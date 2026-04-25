from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class RetrievalBackend(str, Enum):
    vector = "vector"
    neo4j = "neo4j"
    both = "both"


class ChunkingStrategy(str, Enum):
    fixed = "fixed"
    semantic = "semantic"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- Retrieval store ---
    retrieval_backend: RetrievalBackend = Field(
        default=RetrievalBackend.both,
        alias="RAG_BACKEND",
        description="vector = Chroma; neo4j = Neo4jVector; both = Chroma + Neo4j hybrid",
    )

    # --- Chroma (vector) ---
    chroma_persist_dir: Path = Field(
        default=_PROJECT_ROOT / "data" / "chroma_db",
        alias="CHROMA_PERSIST_DIR",
    )
    chroma_collection: str = Field(default="rag_docs", alias="CHROMA_COLLECTION")

    # --- Neo4j ---
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="", alias="NEO4J_PASSWORD")
    neo4j_database: str | None = Field(default=None, alias="NEO4J_DATABASE")

    # --- Embeddings (local sentence-transformers) ---
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )

    # --- Chunking ---
    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.fixed,
        alias="CHUNKING_STRATEGY",
    )
    chunk_size: int = Field(default=1200, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    # --- Retrieval ---
    top_k: int = Field(default=4, alias="TOP_K")

    # --- LLM ---
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")  # ollama | huggingface

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3", alias="OLLAMA_MODEL")

    # HuggingFace (local transformers pipeline)
    hf_model: str = Field(default="Qwen/Qwen2.5-1.5B-Instruct", alias="HF_MODEL")
    # Access token for gated HF repos (Llama, Gemma, Mistral, etc.). Empty
    # string = anonymous; the UI supplies this per-session and doesn't
    # persist it to disk.
    hf_token: str = Field(default="", alias="HF_TOKEN")

    # --- Graph-extraction LLM (for Neo4j knowledge-graph building) ---
    # Entity/relation extraction calls an LLM per chunk — the main answer
    # LLM is often a small local model too slow for this. A cloud API LLM
    # is 10–100× faster and runs many chunks in parallel cheaply. "none"
    # skips extraction entirely (Neo4j stores vector-only flat chunks).
    graph_llm_provider: str = Field(
        default="none", alias="GRAPH_LLM_PROVIDER"
    )  # none | anthropic | openai | gemini | openrouter
    graph_llm_model: str = Field(default="", alias="GRAPH_LLM_MODEL")
    graph_llm_api_key: str = Field(default="", alias="GRAPH_LLM_API_KEY")
    # Parallel chunk workers for extraction. Cloud APIs tolerate 4–8 concurrent
    # requests; raise carefully to avoid rate limits.
    graph_llm_workers: int = Field(default=4, alias="GRAPH_LLM_WORKERS")

    # --- Graph retrieval (neo4j backend uses pure graph traversal) ---
    # Hop depth for entity expansion at query time. 0 = matched entities only;
    # 1 = include direct neighbors; clamped to [0, 3] in the Cypher.
    graph_hops: int = Field(default=1, alias="GRAPH_HOPS")
    # Max seed entities to expand from per query (prevents huge subgraphs
    # when the question is broad and many entities match).
    graph_max_entities: int = Field(default=10, alias="GRAPH_MAX_ENTITIES")


def load_settings() -> Settings:
    return Settings()
