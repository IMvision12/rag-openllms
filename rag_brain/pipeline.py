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
from langchain_openai import ChatOpenAI

from rag_brain.config import RetrievalBackend, Settings, load_settings
from rag_brain.embeddings import get_embeddings
from rag_brain.ingestion import load_pdf_as_documents, split_documents


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
        parts.append(f"[{i}] (source: {src})\n{d.page_content}")
    return "\n\n".join(parts)


def _build_llm(settings: Settings):
    if settings.llm_provider.lower() == "openai":
        kwargs: dict[str, Any] = {"model": settings.openai_model}
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return ChatOpenAI(**kwargs)
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.1,
    )


class RAGPipeline:
    """End-to-end RAG: PDF → chunks → embeddings → (Chroma | Neo4jVector) → retrieve → LLM."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self._embeddings = get_embeddings(self.settings)
        self._vectorstore: Chroma | Neo4jVector | None = None
        self._neo4j_graph: Neo4jGraph | None = None
        self._llm = _build_llm(self.settings)

    def ingest_pdf(self, pdf_path: Path | str, *, recreate: bool = True) -> int:
        """Load PDF, chunk, embed, and upsert into the configured backend. Returns chunk count."""
        raw = load_pdf_as_documents(pdf_path)
        chunks = split_documents(raw, self.settings)
        backend = self.settings.retrieval_backend

        if backend == RetrievalBackend.vector:
            persist = self.settings.chroma_persist_dir
            if recreate and persist.exists():
                shutil.rmtree(persist)
            persist.mkdir(parents=True, exist_ok=True)
            if recreate:
                self._vectorstore = Chroma.from_documents(
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
                self._vectorstore = store
            return len(chunks)

        if backend == RetrievalBackend.neo4j:
            if not self.settings.neo4j_password:
                raise ValueError("NEO4J_PASSWORD is required for neo4j backend.")

            # Graph handle for Neo4j (connection + schema); retrieval uses Neo4jVector below.
            self._neo4j_graph = Neo4jGraph(**_neo4j_conn_kwargs(self.settings))

            self._vectorstore = Neo4jVector.from_documents(
                chunks,
                self._embeddings,
                index_name=self.settings.neo4j_vector_index,
                **_neo4j_conn_kwargs(self.settings),
            )
            return len(chunks)

        raise ValueError(f"Unknown backend: {backend}")

    def _get_retriever(self):
        if self._vectorstore is None:
            self._load_existing_store()
        assert self._vectorstore is not None
        return self._vectorstore.as_retriever(search_kwargs={"k": self.settings.top_k})

    def _load_existing_store(self) -> None:
        s = self.settings
        if s.retrieval_backend == RetrievalBackend.vector:
            s.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
            self._vectorstore = Chroma(
                persist_directory=str(s.chroma_persist_dir),
                embedding_function=self._embeddings,
                collection_name=s.chroma_collection,
            )
            return
        if s.retrieval_backend == RetrievalBackend.neo4j:
            if not s.neo4j_password:
                raise ValueError("NEO4J_PASSWORD is required for neo4j backend.")
            self._neo4j_graph = Neo4jGraph(**_neo4j_conn_kwargs(s))
            self._vectorstore = Neo4jVector.from_existing_index(
                self._embeddings,
                index_name=s.neo4j_vector_index,
                **_neo4j_conn_kwargs(s),
            )
            return
        raise ValueError(f"Unknown backend: {s.retrieval_backend}")

    def query(self, question: str) -> dict[str, Any]:
        """Run retrieval + generation; returns answer and retrieved chunk texts."""
        retriever = self._get_retriever()
        docs = retriever.invoke(question)
        context = _format_context(docs)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a careful assistant for question answering over the user's documents. "
                    "Answer using only the provided context. If the context does not contain the answer, "
                    "say that you do not have enough information. Cite chunk numbers when helpful.",
                ),
                ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
            ]
        )

        answer = (prompt | self._llm | StrOutputParser()).invoke(
            {"context": context, "question": question}
        )
        return {
            "answer": answer,
            "retrieved": [
                {"content": d.page_content, "metadata": dict(d.metadata or {})} for d in docs
            ],
        }


def run_cli() -> None:
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="RAG brain — vector or Neo4j backend (no UI).")
    parser.add_argument(
        "--backend",
        choices=["vector", "neo4j"],
        default=os.environ.get("RAG_BACKEND", "vector"),
        help="vector = Chroma; neo4j = Neo4jVector (+ Neo4jGraph for connection).",
    )
    parser.add_argument(
        "--ingest",
        type=str,
        default=None,
        help="Path to PDF to ingest (loads, chunks, embeds, writes to store).",
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
        help="For Chroma: add to existing collection instead of rebuilding from_documents.",
    )
    args = parser.parse_args()

    os.environ["RAG_BACKEND"] = args.backend
    pipe = RAGPipeline(load_settings())

    if args.ingest:
        n = pipe.ingest_pdf(args.ingest, recreate=not args.no_recreate)
        print(f"Ingested {n} chunks into backend={args.backend}.")

    if args.query:
        out = pipe.query(args.query)
        print(out["answer"])
        print("\n--- retrieved (debug) ---")
        print(json.dumps(out["retrieved"], indent=2, ensure_ascii=False))
    elif not args.ingest:
        parser.print_help()


if __name__ == "__main__":
    run_cli()
