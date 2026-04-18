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


def _rmtree_windows_safe(path: Path) -> None:
    """rmtree with retries — Windows file handles can release a beat later."""
    import time
    for attempt in range(8):
        try:
            shutil.rmtree(path)
            return
        except (PermissionError, OSError):
            if attempt == 7:
                raise
            time.sleep(0.25 * (attempt + 1))


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


_HF_MAX_NEW_TOKENS = 512


def _build_hf_llm(settings: Settings):
    """Build a HuggingFace transformers pipeline wrapped as a chat model for LangChain."""
    import os
    import torch
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    # Mirror the token into HF_TOKEN / HUGGINGFACE_HUB_TOKEN so any HF
    # utility (sentence-transformers, hub downloads) invoked in this
    # process also authenticates — not just the explicit token= kwargs below.
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = settings.hf_token

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }

    token = settings.hf_token or None
    tokenizer = AutoTokenizer.from_pretrained(settings.hf_model, token=token)
    model = AutoModelForCausalLM.from_pretrained(settings.hf_model, token=token, **model_kwargs)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=_HF_MAX_NEW_TOKENS,
        temperature=0.1,
        do_sample=True,
        return_full_text=False,
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

_CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question naturally and concisely from "
    "your own general knowledge. Do not mention any documents, do not fabricate citations, "
    "and keep replies conversational."
)

# Cross-encoder reranker. Unlike bi-encoder cosine similarity (which drifts
# with query length and phrasing — short queries against long chunks hover
# at 0.1–0.3 even for highly relevant hits), cross-encoder scores are
# calibrated: a fixed threshold reliably separates "this chunk answers
# the question" from "it doesn't". This is the standard production
# approach (LlamaIndex / LangChain / Haystack all recommend it).
#
# bge-reranker-base is a ~280 MB multilingual model. Scores are raw
# logits; we apply sigmoid to land in [0, 1]. Threshold 0.3 is what
# Cohere / BGE / ZeroEntropy reranker guides recommend as a usable "good
# enough" cutoff for relevance — sigmoid(0) = 0.5 is already "likely
# relevant"; below 0.3 is confidently irrelevant.
_RERANKER_MODEL = "BAAI/bge-reranker-base"
_RERANK_SCORE_THRESHOLD = 0.3
# Over-retrieve this many candidates from each backend before reranking.
# The reranker is O(k) per query, so 20 is cheap and gives the reranker
# enough recall even when the bi-encoder's top-k misses the best chunk.
_RERANK_CANDIDATE_POOL = 20

# Force Chroma to use cosine distance so similarity_search_with_score returns
# values in [0, 2] where distance = 1 - cos_sim. Gives us a clean
# 1 - distance → cos_sim mapping and matches Neo4jVector's native metric.
# Older collections created without this metadata (default L2) are auto-
# wiped in _ensure_chroma — user just re-ingests.
_CHROMA_COLLECTION_META = {"hnsw:space": "cosine"}


def _sigmoid(x: float) -> float:
    """Safe scalar sigmoid (avoids overflow on large negatives)."""
    import math
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _normalize_chroma_distance(dist: float) -> float:
    """Chroma cosine distance (= 1 - cos_sim) → cos_sim, clamped to [0, 1]."""
    return max(0.0, min(1.0, 1.0 - float(dist)))


def _normalize_neo4j_score(sim: float) -> float:
    """Neo4j returns cosine similarity directly; clamp defensively."""
    return max(0.0, min(1.0, float(sim)))


class RAGPipeline:
    """End-to-end RAG with support for Chroma, Neo4j, or both simultaneously."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self._embeddings = get_embeddings(self.settings)
        self._chroma_store: Chroma | None = None
        self._neo4j_store: Neo4jVector | None = None
        self._neo4j_graph: Neo4jGraph | None = None
        self._llm = None
        self._reranker = None
        self._reranker_load_failed = False

    def _ensure_llm(self):
        if self._llm is None:
            self._llm = _build_llm(self.settings)
        return self._llm

    def _get_reranker(self):
        """Lazy-load the cross-encoder reranker. Returns None if loading
        fails (e.g. no network on first use, OOM); _retrieve_scored falls
        back to bi-encoder cosine so the pipeline still works."""
        if self._reranker is not None or self._reranker_load_failed:
            return self._reranker
        try:
            import torch
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            token = (self.settings.hf_token or None)
            kwargs: dict[str, Any] = {"max_length": 512, "device": device}
            # sentence-transformers accepts `token=` for gated repos; wrap
            # in try/except because older versions use `use_auth_token`.
            try:
                self._reranker = CrossEncoder(_RERANKER_MODEL, token=token, **kwargs)
            except TypeError:
                self._reranker = CrossEncoder(_RERANKER_MODEL, use_auth_token=token, **kwargs)
        except Exception:
            self._reranker_load_failed = True
            self._reranker = None
        return self._reranker

    @property
    def _use_chroma(self) -> bool:
        return self.settings.retrieval_backend in (RetrievalBackend.vector, RetrievalBackend.both)

    @property
    def _use_neo4j(self) -> bool:
        return self.settings.retrieval_backend in (RetrievalBackend.neo4j, RetrievalBackend.both)

    # ── Ingestion ──────────────────────────────────────────────────────

    def ingest(self, file_path: Path | str, *, recreate: bool = True) -> int:
        """Load a document (PDF/DOCX), chunk, embed, and store. Returns chunk count.

        When both backends are enabled, a failure in one is recorded in
        `self.last_ingest_warnings` instead of aborting — so a missing Neo4j
        server doesn't block Chroma from working (and vice versa).
        """
        raw = load_documents(file_path)
        chunks = split_documents(raw, self.settings)

        self.last_ingest_warnings: list[str] = []
        successes = 0
        backends_tried = int(self._use_chroma) + int(self._use_neo4j)

        if self._use_chroma:
            try:
                self._ingest_chroma(chunks, recreate=recreate)
                successes += 1
            except Exception as e:
                if backends_tried > 1:
                    self.last_ingest_warnings.append(f"Chroma ingest failed: {e}")
                else:
                    raise

        if self._use_neo4j:
            try:
                self._ingest_neo4j(chunks, recreate=recreate)
                successes += 1
            except Exception as e:
                if backends_tried > 1:
                    self.last_ingest_warnings.append(f"Neo4j ingest failed: {e}")
                else:
                    raise

        if successes == 0:
            raise RuntimeError(
                "All backends failed: " + " | ".join(self.last_ingest_warnings)
            )

        return len(chunks)

    def _release_chroma(self) -> None:
        """Release file locks held by the Chroma client (needed on Windows before rmtree)."""
        # Try to stop the underlying chromadb System so it releases SQLite handles.
        store = self._chroma_store
        if store is not None:
            for attr_path in (("_client", "_system"), ("_client", "_server")):
                obj = store
                try:
                    for a in attr_path:
                        obj = getattr(obj, a, None)
                        if obj is None:
                            break
                    if obj is not None and hasattr(obj, "stop"):
                        obj.stop()
                except Exception:
                    pass
        self._chroma_store = None
        try:
            from chromadb.api.client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass
        import gc
        gc.collect()

    def _ingest_chroma(self, chunks: list[Document], *, recreate: bool) -> None:
        persist = self.settings.chroma_persist_dir

        if recreate:
            # Fully wipe: release all cached chromadb clients, then physically
            # delete the persist dir. Using delete_collection alone leaves
            # orphaned UUID subdirs and can leave the in-memory client holding
            # a stale collection ref across pipeline rebuilds.
            self._release_chroma()
            if persist.exists():
                try:
                    _rmtree_windows_safe(persist)
                except Exception:
                    # If rmtree still fails after our retries, drop the
                    # collection via the API as a fallback so ingest doesn't
                    # bomb entirely.
                    try:
                        import chromadb
                        client = chromadb.PersistentClient(path=str(persist))
                        try:
                            client.delete_collection(name=self.settings.chroma_collection)
                        except Exception:
                            pass
                        del client
                        self._release_chroma()
                    except Exception:
                        pass

        persist.mkdir(parents=True, exist_ok=True)

        if recreate:
            self._chroma_store = Chroma.from_documents(
                documents=chunks,
                embedding=self._embeddings,
                persist_directory=str(persist),
                collection_name=self.settings.chroma_collection,
                collection_metadata=_CHROMA_COLLECTION_META,
            )
        else:
            store = Chroma(
                persist_directory=str(persist),
                embedding_function=self._embeddings,
                collection_name=self.settings.chroma_collection,
                collection_metadata=_CHROMA_COLLECTION_META,
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
                collection_metadata=_CHROMA_COLLECTION_META,
            )
            # Defensive: probe the collection. If chromadb's SQLite + on-disk
            # state is inconsistent (stale UUID cached, orphan subdirs, etc.),
            # OR the collection was created with a different hnsw:space than
            # what we need now (older builds defaulted to L2, which broke
            # our relevance scoring), wipe and re-open cleanly.
            needs_rebuild = False
            try:
                self._chroma_store._collection.count()
                existing_meta = self._chroma_store._collection.metadata or {}
                if existing_meta.get("hnsw:space") != _CHROMA_COLLECTION_META["hnsw:space"]:
                    needs_rebuild = True
            except Exception:
                needs_rebuild = True
            if needs_rebuild:
                self._release_chroma()
                if s.chroma_persist_dir.exists():
                    try:
                        _rmtree_windows_safe(s.chroma_persist_dir)
                    except Exception:
                        pass
                s.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
                self._chroma_store = Chroma(
                    persist_directory=str(s.chroma_persist_dir),
                    embedding_function=self._embeddings,
                    collection_name=s.chroma_collection,
                    collection_metadata=_CHROMA_COLLECTION_META,
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

    def _retrieve_candidates(self, question: str, pool: int) -> list[Document]:
        """Retrieve up to `pool` unscored candidates per backend, dedupe.

        This is the first stage of the two-stage retrieve-then-rerank
        pipeline. We pull more than top_k here so the reranker has recall
        room; it will pick the truly relevant ones.
        """
        all_docs: list[Document] = []
        self.last_query_warnings: list[str] = []
        backends_tried = int(self._use_chroma) + int(self._use_neo4j)

        if self._use_chroma:
            try:
                store = self._ensure_chroma()
                all_docs.extend(store.similarity_search(question, k=pool))
            except Exception as e:
                if backends_tried > 1:
                    self.last_query_warnings.append(f"Chroma retrieval failed: {e}")
                else:
                    raise

        if self._use_neo4j:
            try:
                store = self._ensure_neo4j()
                all_docs.extend(store.similarity_search(question, k=pool))
            except Exception as e:
                if backends_tried > 1:
                    self.last_query_warnings.append(f"Neo4j retrieval failed: {e}")
                else:
                    raise

        return _deduplicate_docs(all_docs)

    def _retrieve_scored_cosine(self, question: str) -> list[tuple[Document, float]]:
        """Bi-encoder cosine fallback: used when the reranker is unavailable.

        Scores are uncalibrated — don't expect the threshold tuned for the
        cross-encoder to behave well here. Better than nothing.
        """
        k = self.settings.top_k
        scored: list[tuple[Document, float]] = []
        self.last_query_warnings: list[str] = []
        backends_tried = int(self._use_chroma) + int(self._use_neo4j)

        if self._use_chroma:
            try:
                store = self._ensure_chroma()
                for d, dist in store.similarity_search_with_score(question, k=k):
                    scored.append((d, _normalize_chroma_distance(dist)))
            except Exception as e:
                if backends_tried > 1:
                    self.last_query_warnings.append(f"Chroma retrieval failed: {e}")
                else:
                    raise
        if self._use_neo4j:
            try:
                store = self._ensure_neo4j()
                for d, sim in store.similarity_search_with_score(question, k=k):
                    scored.append((d, _normalize_neo4j_score(sim)))
            except Exception as e:
                if backends_tried > 1:
                    self.last_query_warnings.append(f"Neo4j retrieval failed: {e}")
                else:
                    raise
        best: dict[str, tuple[Document, float]] = {}
        for d, s in scored:
            key = d.page_content.strip()
            if key not in best or best[key][1] < s:
                best[key] = (d, float(s))
        return sorted(best.values(), key=lambda t: -t[1])[:k]

    def _retrieve_scored(self, question: str) -> list[tuple[Document, float]]:
        """Two-stage retrieve-and-rerank returning (doc, relevance) pairs.

        Stage 1: pull `_RERANK_CANDIDATE_POOL` candidates per backend using
        the bi-encoder (Chroma / Neo4j), dedupe by chunk text.
        Stage 2: score each (query, chunk) pair with a cross-encoder —
        scores are calibrated, so the threshold in `query()` actually
        means "relevant" rather than "lexically similar to long text".

        Falls back to uncalibrated cosine if the reranker can't load.
        """
        k_return = self.settings.top_k
        candidates = self._retrieve_candidates(question, _RERANK_CANDIDATE_POOL)
        if not candidates:
            return []

        reranker = self._get_reranker()
        if reranker is None:
            return self._retrieve_scored_cosine(question)

        pairs = [(question, d.page_content) for d in candidates]
        try:
            raw = reranker.predict(pairs, show_progress_bar=False)
            # raw is numpy array or list of floats; apply sigmoid → [0, 1]
            scores = [_sigmoid(float(s)) for s in raw]
        except Exception:
            # Reranker call failed at runtime (CUDA OOM, corrupt weights,
            # etc.) — one-shot fall back rather than crashing the query.
            self._reranker_load_failed = True
            self._reranker = None
            return self._retrieve_scored_cosine(question)

        scored = sorted(zip(candidates, scores), key=lambda t: -t[1])
        return scored[:k_return]

    def _retrieve(self, question: str) -> list[Document]:
        """Backward-compatible retrieval without scores."""
        return [d for d, _ in self._retrieve_scored(question)]

    # ── Query ──────────────────────────────────────────────────────────

    def query(self, question: str) -> dict[str, Any]:
        """Retrieve + rerank + threshold on calibrated reranker score.

        Standard industry pattern (LangChain / LlamaIndex / Haystack all
        recommend it): cross-encoder scores are calibrated, so a fixed
        threshold on the top reranked chunk reliably tells us whether the
        question is actually answerable from the docs.

        - top_score >= _RERANK_SCORE_THRESHOLD → RAG mode (strict prompt,
          cite chunk numbers)
        - top_score <  _RERANK_SCORE_THRESHOLD → chat mode (answer from
          general knowledge, no citations, no doc context passed to LLM)
        """
        scored = self._retrieve_scored(question)
        top_score = scored[0][1] if scored else 0.0
        docs = [d for d, _ in scored]
        use_rag = bool(docs) and top_score >= _RERANK_SCORE_THRESHOLD

        if use_rag:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", _RAG_SYSTEM_PROMPT),
                    ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
                ]
            )
            answer = (prompt | self._ensure_llm() | StrOutputParser()).invoke(
                {"context": _format_context(docs), "question": question}
            )
            mode = "rag"
            retrieved_docs = docs
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", _CHAT_SYSTEM_PROMPT),
                    ("human", "{question}"),
                ]
            )
            answer = (prompt | self._ensure_llm() | StrOutputParser()).invoke(
                {"question": question}
            )
            mode = "chat"
            retrieved_docs = []

        return {
            "answer": answer,
            "mode": mode,
            "top_score": top_score,
            "reranker_used": self._reranker is not None,
            "retrieved": [
                {"content": d.page_content, "metadata": dict(d.metadata or {})}
                for d in retrieved_docs
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
