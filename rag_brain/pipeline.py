from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_ollama import ChatOllama

from rag_brain.config import RetrievalBackend, Settings, load_settings
from rag_brain.embeddings import get_embeddings
from rag_brain.ingestion import load_documents, split_documents


def _neo4j_conn_kwargs(settings: Settings) -> dict[str, Any]:
    out: dict[str, Any] = {
        "url": settings.neo4j_uri,
        "username": settings.neo4j_user,
        "password": settings.neo4j_password,
    }
    if settings.neo4j_database:
        out["database"] = settings.neo4j_database
    return out


def _format_context(docs: list[Document]) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source", "?")
        page = meta.get("page", "")
        page_str = f", page {page}" if page != "" else ""
        parts.append(f"[{i}] (source: {src}{page_str})\n{d.page_content}")
    return "\n\n".join(parts)


def _deduplicate_docs(docs: list[Document]) -> list[Document]:
    """Remove duplicate chunks based on page_content."""
    seen: set[str] = set()
    unique: list[Document] = []
    for d in docs:
        key = d.page_content.strip()
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def _build_hf_llm(settings: Settings):
    """Build a HuggingFace transformers pipeline wrapped as a chat model for LangChain."""
    import torch
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }

    if settings.hf_quantize == "4bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif settings.hf_quantize == "8bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(settings.hf_model)
    model = AutoModelForCausalLM.from_pretrained(settings.hf_model, **model_kwargs)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=settings.hf_max_new_tokens,
        temperature=0.1,
        do_sample=True,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return ChatHuggingFace(llm=llm)


def _build_llm(settings: Settings):
    if settings.llm_provider.lower() == "huggingface":
        return _build_hf_llm(settings)
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.1,
    )


_RAG_SYSTEM_PROMPT = (
    "You are a careful research assistant for question answering over the user's documents. "
    "Answer using ONLY the provided context. For every claim you make, cite the chunk number "
    "in square brackets (e.g. [1], [3]). If the context does not contain enough information "
    "to answer, say so explicitly. Do not make up information."
)


class RAGPipeline:
    """End-to-end RAG with support for Chroma, Neo4j, or both simultaneously."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self._embeddings = get_embeddings(self.settings)
        self._chroma_store: Chroma | None = None
        self._neo4j_store: Neo4jVector | None = None
        self._neo4j_graph: Neo4jGraph | None = None
        self._llm = None

    def _ensure_llm(self):
        if self._llm is None:
            self._llm = _build_llm(self.settings)
        return self._llm

    @property
    def _use_chroma(self) -> bool:
        return self.settings.retrieval_backend in (RetrievalBackend.vector, RetrievalBackend.both)

    @property
    def _use_neo4j(self) -> bool:
        return self.settings.retrieval_backend in (RetrievalBackend.neo4j, RetrievalBackend.both)

    # ── Ingestion ──────────────────────────────────────────────────────

    def ingest(self, file_path: Path | str, *, recreate: bool = True) -> int:
        """Load a document (PDF/DOCX), chunk, embed, and store. Returns chunk count."""
        raw = load_documents(file_path)
        chunks = split_documents(raw, self.settings)

        if self._use_chroma:
            self._ingest_chroma(chunks, recreate=recreate)

        if self._use_neo4j:
            self._ingest_neo4j(chunks, recreate=recreate)

        return len(chunks)

    def _ingest_chroma(self, chunks: list[Document], *, recreate: bool) -> None:
        persist = self.settings.chroma_persist_dir
        if recreate and persist.exists():
            shutil.rmtree(persist)
        persist.mkdir(parents=True, exist_ok=True)
        if recreate:
            self._chroma_store = Chroma.from_documents(
                documents=chunks,
                embedding=self._embeddings,
                persist_directory=str(persist),
                collection_name=self.settings.chroma_collection,
            )
        else:
            store = Chroma(
                persist_directory=str(persist),
                embedding_function=self._embeddings,
                collection_name=self.settings.chroma_collection,
            )
            store.add_documents(chunks)
            self._chroma_store = store

    def _ingest_neo4j(self, chunks: list[Document], *, recreate: bool) -> None:
        if not self.settings.neo4j_password:
            raise ValueError("NEO4J_PASSWORD is required for neo4j backend.")
        conn = _neo4j_conn_kwargs(self.settings)
        self._neo4j_graph = Neo4jGraph(**conn, refresh_schema=False)
        if recreate:
            self._neo4j_store = Neo4jVector.from_documents(
                chunks,
                self._embeddings,
                index_name=self.settings.neo4j_vector_index,
                **conn,
            )
        else:
            store = self._ensure_neo4j()
            store.add_documents(chunks)
            self._neo4j_store = store

    # ── Loading existing stores ────────────────────────────────────────

    def _ensure_chroma(self) -> Chroma:
        if self._chroma_store is None:
            s = self.settings
            s.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
            self._chroma_store = Chroma(
                persist_directory=str(s.chroma_persist_dir),
                embedding_function=self._embeddings,
                collection_name=s.chroma_collection,
            )
        return self._chroma_store

    def _ensure_neo4j(self) -> Neo4jVector:
        if self._neo4j_store is None:
            s = self.settings
            if not s.neo4j_password:
                raise ValueError("NEO4J_PASSWORD is required for neo4j backend.")
            self._neo4j_graph = Neo4jGraph(
                **_neo4j_conn_kwargs(s), refresh_schema=False
            )
            self._neo4j_store = Neo4jVector.from_existing_index(
                self._embeddings,
                index_name=s.neo4j_vector_index,
                **_neo4j_conn_kwargs(s),
            )
        return self._neo4j_store

    # ── Retrieval ──────────────────────────────────────────────────────

    def _retrieve(self, question: str) -> list[Document]:
        """Retrieve from configured backends, merge and deduplicate."""
        k = self.settings.top_k
        all_docs: list[Document] = []

        if self._use_chroma:
            store = self._ensure_chroma()
            all_docs.extend(store.similarity_search(question, k=k))

        if self._use_neo4j:
            store = self._ensure_neo4j()
            all_docs.extend(store.similarity_search(question, k=k))

        deduped = _deduplicate_docs(all_docs)
        # When using both backends we retrieve 2*k candidates; trim back to top_k
        return deduped[: k]

    # ── Query ──────────────────────────────────────────────────────────

    def query(self, question: str) -> dict[str, Any]:
        """Run retrieval + generation. Returns answer and retrieved chunks."""
        docs = self._retrieve(question)
        if not docs:
            import sys
            print("Warning: no documents retrieved. Did you ingest files first?", file=sys.stderr)
        context = _format_context(docs)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _RAG_SYSTEM_PROMPT),
                ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
            ]
        )

        answer = (prompt | self._ensure_llm() | StrOutputParser()).invoke(
            {"context": context, "question": question}
        )
        return {
            "answer": answer,
            "retrieved": [
                {"content": d.page_content, "metadata": dict(d.metadata or {})} for d in docs
            ],
        }

    # ── Legacy alias ───────────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: Path | str, *, recreate: bool = True) -> int:
        return self.ingest(pdf_path, recreate=recreate)


def run_cli() -> None:
    import argparse
    import json
    import os
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="RAG brain — vector, Neo4j, or both backends.")
    parser.add_argument(
        "--backend",
        choices=["vector", "neo4j", "both"],
        default=os.environ.get("RAG_BACKEND", "both"),
        help="vector = Chroma; neo4j = Neo4jVector; both = hybrid retrieval from Chroma + Neo4j.",
    )
    parser.add_argument(
        "--ingest",
        type=str,
        default=None,
        help="Path to a PDF or DOCX file to ingest.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Question to ask after retrieval.",
    )
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="For Chroma: add to existing collection instead of rebuilding.",
    )
    parser.add_argument(
        "--chunking",
        choices=["fixed", "semantic"],
        default=os.environ.get("CHUNKING_STRATEGY", "fixed"),
        help="Chunking strategy: fixed-size or semantic (sentence-boundary aware).",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Print retrieved chunks as JSON after the answer.",
    )
    args = parser.parse_args()

    os.environ["RAG_BACKEND"] = args.backend
    os.environ["CHUNKING_STRATEGY"] = args.chunking
    pipe = RAGPipeline(load_settings())

    if args.ingest:
        n = pipe.ingest(args.ingest, recreate=not args.no_recreate)
        print(f"Ingested {n} chunks into backend={args.backend} (chunking={args.chunking}).")

    if args.query:
        out = pipe.query(args.query)
        print(out["answer"])
        if args.show_chunks:
            print("\n--- retrieved chunks ---")
            print(json.dumps(out["retrieved"], indent=2, ensure_ascii=False))
    elif not args.ingest:
        parser.print_help()


if __name__ == "__main__":
    run_cli()
