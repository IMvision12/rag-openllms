from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama

from rag_brain.config import RetrievalBackend, Settings, load_settings
from rag_brain.embeddings import get_embeddings
from rag_brain.ingestion import load_documents, split_documents

log = logging.getLogger("rag_brain.pipeline")


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


def _build_hf_chat(*, model_name: str, hf_token: str = "", temperature: float = 0.1):
    """Build a HuggingFace transformers chat model.

    Shared by the answer LLM and the graph-extraction LLM — they pick
    different models and temperatures but the loading path is identical.
    """
    import os
    import torch
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

    # Mirror the token into HF_TOKEN / HUGGINGFACE_HUB_TOKEN so any HF
    # utility (sentence-transformers, hub downloads) invoked in this
    # process also authenticates — not just the explicit token= kwargs below.
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    token = hf_token or None
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token, **model_kwargs)

    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=_HF_MAX_NEW_TOKENS,
        temperature=temperature,
        do_sample=temperature > 0,
        return_full_text=False,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return ChatHuggingFace(llm=llm)


def _build_hf_llm(settings: Settings):
    return _build_hf_chat(
        model_name=settings.hf_model,
        hf_token=settings.hf_token,
        temperature=0.1,
    )


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
# 1 - distance → cos_sim mapping. Neo4j now uses pure graph retrieval, so
# this only affects the Chroma path.
# Older collections created without this metadata (default L2) are auto-
# wiped in _ensure_chroma — user just re-ingests.
_CHROMA_COLLECTION_META = {"hnsw:space": "cosine"}

# Cosine similarity threshold for embedding-based entity merging.
# 0.92 is conservative — catches near-paraphrases ("Apple" ≈ "Apple Inc.")
# without false-merging weakly related concepts. Tighten if you see bad
# merges in the logs; loosen if too many obvious aliases stay separate.
_ENTITY_MERGE_COSINE_THRESHOLD = 0.92


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


_QUERY_ENTITY_PROMPT = (
    "You extract entity mentions for a knowledge-graph search. Given the "
    "user's question, identify named entities, proper nouns, and concrete "
    "concepts likely to appear as graph nodes. Skip stopwords and common "
    "words.\n\n"
    "Respond with a JSON array of strings on a single line. Nothing else — "
    "no prose, no markdown fencing, no explanations, no thinking out loud.\n\n"
    "Examples:\n"
    'Q: who founded Acme Corp?\n'
    'A: ["Acme Corp"]\n\n'
    'Q: what did Alice say about Bob\'s project?\n'
    'A: ["Alice", "Bob"]\n\n'
    "Q: how does it work?\n"
    "A: []"
)


class RAGPipeline:
    """End-to-end RAG with support for Chroma, Neo4j, or both simultaneously.

    Backends:
      - vector: Chroma (vector similarity)
      - neo4j:  pure knowledge-graph retrieval (entity extract → match →
                hop expand → fetch source chunks via :MENTIONS)
      - both:   union of the two, deduplicated and reranked together
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self._embeddings = get_embeddings(self.settings)
        self._chroma_store: Chroma | None = None
        self._neo4j_graph: Neo4jGraph | None = None
        self._llm = None
        self._graph_query_llm = None
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
        """Pure-graph ingest. No vector index — retrieval traverses the
        entity graph at query time via Cypher.

        `LLMGraphTransformer` extracts (entity, relation, entity) triples
        from each chunk; `add_graph_documents(..., include_source=True)`
        writes them along with `:Document` nodes that retain the full
        chunk text and link to every entity they mention via `[:MENTIONS]`.
        Those `:Document` nodes are what `_graph_retrieve` returns.

        Requires a configured graph-extraction LLM — without it the graph
        would be empty and the backend would return nothing at query time.
        """
        s = self.settings
        if not s.neo4j_password:
            raise ValueError("NEO4J_PASSWORD is required for neo4j backend.")
        if (s.graph_llm_provider or "none").lower() == "none":
            raise ValueError(
                "neo4j backend requires graph extraction. Set GRAPH_LLM_PROVIDER, "
                "GRAPH_LLM_MODEL, and GRAPH_LLM_API_KEY in .env."
            )

        self._neo4j_graph = Neo4jGraph(**_neo4j_conn_kwargs(s), refresh_schema=False)

        if recreate:
            # Wipe everything: prior entities, source docs, leftover chunks.
            self._neo4j_graph.query("MATCH (n) DETACH DELETE n")

        extracted = self._extract_graph(chunks)
        if extracted == 0:
            warnings = getattr(self, "last_ingest_warnings", [])
            detail = "; ".join(warnings) if warnings else "no graph documents produced"
            raise RuntimeError(f"Graph extraction failed: {detail}")

        # Post-extraction graph hygiene. Both are best-effort — a failure
        # here doesn't undo a successful chunk extraction.
        try:
            self._link_chunks_to_files()
        except Exception as e:
            log.warning("file-hierarchy: skipped due to error: %s", e)
            if hasattr(self, "last_ingest_warnings"):
                self.last_ingest_warnings.append(f"File-hierarchy step skipped: {e}")
        try:
            self._merge_similar_entities()
        except Exception as e:
            log.warning("entity-merge: skipped due to error: %s", e)
            if hasattr(self, "last_ingest_warnings"):
                self.last_ingest_warnings.append(f"Entity-merge step skipped: {e}")

    def _build_graph_llm(self):
        """Build the LLM used for knowledge-graph extraction.

        Open-source providers only:
          - ollama:      local Ollama server (any model pulled with `ollama pull`).
                         Single in-memory model, so workers > 1 doesn't actually
                         parallelize — Ollama queues concurrent calls.
          - huggingface: local Transformers pipeline. Same single-model bottleneck;
                         needs HF token for gated repos (Llama, Gemma, Mistral).

        Local extraction is meaningfully slower than cloud APIs (per-chunk
        forward pass on one GPU/CPU) — budget minutes-to-hours for thesis
        corpora, not seconds. Recommend `GRAPH_LLM_WORKERS=1`.

        Returns None if provider is 'none' or model is missing.
        """
        s = self.settings
        provider = (s.graph_llm_provider or "none").lower()
        if provider == "none":
            return None
        model = (s.graph_llm_model or "").strip()
        if not model:
            return None

        if provider == "ollama":
            return ChatOllama(
                base_url=s.ollama_base_url,
                model=model,
                temperature=0,
            )
        if provider == "huggingface":
            return _build_hf_chat(
                model_name=model,
                hf_token=s.hf_token,
                temperature=0,
            )
        raise ValueError(
            f"Unknown graph_llm_provider: {provider!r}. Use one of: none, ollama, huggingface."
        )

    def _extract_graph(self, chunks: list[Document]) -> int:
        """Run LLMGraphTransformer over chunks → entities + relationships.

        Returns the count of chunks successfully written to the graph.
        Zero means the caller should treat the ingest as a hard failure
        (pure-graph backend has nothing to retrieve from otherwise).

        Uses the dedicated graph-extraction LLM. Chunks run concurrently
        via a thread pool to hide per-request latency (cloud APIs handle
        this trivially; local models bottleneck on the single weight copy).
        """
        try:
            from langchain_experimental.graph_transformers import LLMGraphTransformer
        except Exception as e:
            if hasattr(self, "last_ingest_warnings"):
                self.last_ingest_warnings.append(
                    f"Graph extraction skipped (langchain-experimental missing): {e}"
                )
            return 0

        try:
            llm = self._build_graph_llm()
        except Exception as e:
            if hasattr(self, "last_ingest_warnings"):
                self.last_ingest_warnings.append(f"Graph LLM init failed: {e}")
            return 0
        if llm is None:
            if hasattr(self, "last_ingest_warnings"):
                self.last_ingest_warnings.append(
                    "Graph extraction skipped — no graph-extraction LLM configured."
                )
            return 0

        # `ignore_tool_usage=True` forces prompt-based extraction, which
        # works with any chat model. For tool-calling-capable models
        # (Anthropic/OpenAI/Gemini) the default path is faster + cleaner.
        try:
            transformer = LLMGraphTransformer(llm=llm)
        except TypeError:
            transformer = LLMGraphTransformer(llm=llm, ignore_tool_usage=True)

        # Fan out chunks across a thread pool. LLMGraphTransformer exposes
        # `process_response` for per-doc processing; we call it directly
        # so we can parallelize without depending on langchain's async stack.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        workers = max(1, int(self.settings.graph_llm_workers or 4))
        graph_docs = []
        errors: list[str] = []
        log.info("graph-extract: starting %d chunks (workers=%d)", len(chunks), workers)

        def _run_one(doc):
            return transformer.process_response(doc)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, c): c for c in chunks}
            for fut in as_completed(futures):
                try:
                    graph_docs.append(fut.result())
                except Exception as e:
                    errors.append(str(e))
        log.info(
            "graph-extract: %d/%d chunks produced GraphDocuments (%d errors)",
            len(graph_docs), len(chunks), len(errors),
        )

        if errors and hasattr(self, "last_ingest_warnings"):
            self.last_ingest_warnings.append(
                f"Graph extraction had {len(errors)} chunk failure(s); first: {errors[0]}"
            )
        if not graph_docs:
            return 0

        # `baseEntityLabel=True` adds a shared `:__Entity__` label so you
        # can query across all entity types; `include_source=True` creates
        # `:Document` nodes linked to each entity via `[:MENTIONS]`.
        self._neo4j_graph.add_graph_documents(
            graph_docs,
            baseEntityLabel=True,
            include_source=True,
        )
        return len(graph_docs)

    # ── Post-extraction graph hygiene ──────────────────────────────────

    def _link_chunks_to_files(self) -> int:
        """Group `:Document` chunks under a `:File` parent node.

        Each :Document keeps its `source` property (filename); we MERGE one
        :File node per unique source and link via `[:HAS_CHUNK]`. Idempotent —
        re-runs on the same data don't duplicate nodes or edges.

        This gives the graph a clean two-level hierarchy
        (File → Document → __Entity__) so the visualization matches user
        intuition ("I uploaded 2 PDFs, I expect 2 file-level clusters").
        """
        if self._neo4j_graph is None:
            return 0
        rows = self._neo4j_graph.query(
            """
            MATCH (d:Document)
            WHERE d.source IS NOT NULL
            MERGE (f:File {name: d.source})
            MERGE (f)-[:HAS_CHUNK]->(d)
            RETURN count(d) AS linked
            """
        )
        linked = int(rows[0].get("linked", 0)) if rows else 0
        log.info("file-hierarchy: linked %d Document(s) under :File parents", linked)
        return linked

    def _merge_similar_entities(self) -> int:
        """Cluster `:__Entity__` nodes by ID similarity and merge them.

        The biggest weakness of LLMGraphTransformer is per-chunk entity-name
        drift — "Gitesh", "Gitesh Chawda", "G. Chawda" stay as 3 separate
        nodes because LangChain's MERGE is exact-string. This step fixes
        that by clustering similar entity IDs and merging each cluster
        into a single canonical node (the longest, most informative ID).

        Two entities cluster together if EITHER:
          - One's *words* are a subset of the other's (catches
            "Gitesh" ⊂ "Gitesh Chawda" but NOT "Java" ⊂ "JavaScript"
            since we match whole words, not substrings)
          - Their normalized embeddings have cosine similarity ≥ 0.92
            (catches paraphrases the substring rule misses)

        Uses `apoc.refactor.mergeNodes` for the actual merge — same APOC
        already required by `add_graph_documents`. Returns the number of
        nodes that were merged into canonical ones.
        """
        import re as _re

        if self._neo4j_graph is None:
            return 0
        rows = self._neo4j_graph.query(
            "MATCH (e:__Entity__) WHERE e.id IS NOT NULL RETURN DISTINCT e.id AS id"
        )
        ids = [r["id"] for r in rows if r.get("id")]
        if len(ids) < 2:
            return 0

        log.info("entity-merge: scanning %d entities", len(ids))

        # Embed all IDs in one batch; cheap on CPU at this scale.
        embeddings = self._embeddings.embed_documents(ids)
        import numpy as np
        arr = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        sim = arr @ arr.T

        # Union-Find for clustering.
        n = len(ids)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Pre-compute word sets for the substring rule. \w matches across
        # alnum + underscore which is what we want; punctuation is ignored.
        word_sets = [set(_re.findall(r"\w+", i.lower())) for i in ids]
        lower_ids = [i.lower() for i in ids]

        def _is_url(s_lower: str) -> bool:
            return s_lower.startswith(("http://", "https://")) or "://" in s_lower

        def _word_subset_safe(a: int, b: int) -> bool:
            """Conservative word-subset rule with three guards against the
            false merges we saw in practice (URLs, form-field labels, and
            wildly different specificity)."""
            wa, wb = word_sets[a], word_sets[b]
            if not (wa and wb):
                return False
            if not (wa <= wb or wb <= wa):
                return False
            # URL guard — "Fbi" must NOT merge into "https://www.fbi.gov/...".
            # Treat URLs as unique atoms.
            if _is_url(lower_ids[a]) != _is_url(lower_ids[b]):
                return False
            # Form-field guard — "Sex" vs "Sex: (Check One)" are a real entity
            # and a form-field label. Don't collapse.
            if (":" in ids[a]) != (":" in ids[b]):
                return False
            # Specificity guard — "Date" vs "Date Of Birth (Mm/Dd/Yy)" share
            # a word but mean different things. Only merge when the longer
            # side adds at most ~one extra word per shorter-side word.
            longer = max(len(wa), len(wb))
            shorter = min(len(wa), len(wb))
            if longer > 2 * shorter + 1:
                return False
            return True

        for i in range(n):
            for j in range(i + 1, n):
                if _word_subset_safe(i, j) or sim[i, j] >= _ENTITY_MERGE_COSINE_THRESHOLD:
                    union(i, j)

        clusters: dict[int, list[int]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)

        merge_count = 0
        for members in clusters.values():
            if len(members) < 2:
                continue
            # Canonical = longest ID (most descriptive); tie-break alphabetical
            # so the choice is deterministic across runs.
            canonical_idx = max(members, key=lambda k: (len(ids[k]), ids[k]))
            canonical_id = ids[canonical_idx]
            for k in members:
                if k == canonical_idx:
                    continue
                alias_id = ids[k]
                try:
                    self._neo4j_graph.query(
                        """
                        MATCH (canonical:__Entity__ {id: $canonical})
                        MATCH (alias:__Entity__ {id: $alias})
                        WHERE canonical <> alias
                        CALL apoc.refactor.mergeNodes(
                            [canonical, alias],
                            {mergeRels: true, properties: 'discard'}
                        )
                        YIELD node RETURN node
                        """,
                        params={"canonical": canonical_id, "alias": alias_id},
                    )
                    merge_count += 1
                    log.info("entity-merge: %r → %r", alias_id, canonical_id)
                except Exception as e:
                    log.warning("entity-merge: failed %r → %r: %s", alias_id, canonical_id, e)

        log.info("entity-merge: %d node(s) merged into canonicals", merge_count)
        return merge_count

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

    def _ensure_neo4j_graph(self) -> Neo4jGraph:
        if self._neo4j_graph is None:
            s = self.settings
            if not s.neo4j_password:
                raise ValueError("NEO4J_PASSWORD is required for neo4j backend.")
            self._neo4j_graph = Neo4jGraph(
                **_neo4j_conn_kwargs(s), refresh_schema=False
            )
        return self._neo4j_graph

    # ── Graph retrieval (pure graph; no vector index involved) ─────────

    def _ensure_graph_query_llm(self):
        """Lazy-build the LLM used to extract entities from a question.
        Reuses the same provider config as graph ingest extraction."""
        if self._graph_query_llm is None:
            self._graph_query_llm = self._build_graph_llm()
        return self._graph_query_llm

    def _extract_query_entities(self, question: str) -> list[str]:
        """Ask the graph LLM for a JSON list of entity mentions in `question`.

        Returns [] if the LLM is unavailable or output can't be parsed.
        Failures are appended to `last_query_warnings` so the UI can surface
        them — silent empties are the worst footgun for thesis demos.

        Robustness for local models:
          - strips <think>...</think> blocks (qwen3, deepseek-r1 emit these)
          - strips ```json ...``` markdown fences
          - falls back to the LAST [...] block if the model wraps in prose
        """
        import json as _json
        import re as _re

        llm = self._ensure_graph_query_llm()
        if llm is None:
            log.warning("entity-extract: no graph LLM configured; returning []")
            return []
        prompt = ChatPromptTemplate.from_messages(
            [("system", _QUERY_ENTITY_PROMPT), ("human", "{question}")]
        )
        log.info("entity-extract: question=%r", question)
        try:
            raw = (prompt | llm | StrOutputParser()).invoke({"question": question})
        except Exception as e:
            log.warning("entity-extract: LLM call failed: %s", e)
            if hasattr(self, "last_query_warnings"):
                self.last_query_warnings.append(f"Entity extraction LLM call failed: {e}")
            return []
        log.debug("entity-extract: raw output (first 400 chars): %r", (raw or "")[:400])

        text = (raw or "").strip()
        # 1. Strip reasoning blocks. qwen3/deepseek-r1 etc. emit these by
        # default; the JSON we want comes after.
        text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
        # 2. Strip ```json ... ``` markdown fences.
        if text.startswith("```"):
            text = _re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=_re.MULTILINE).strip()
        # 3. Coerce Python-set syntax `{"a", "b"}` → JSON array `["a", "b"]`.
        # Small local models like qwen3 occasionally emit set literals where
        # we asked for a list; both convey the same data.
        def _coerce_sets(s: str) -> str:
            def repl(m: _re.Match) -> str:
                inner = m.group(1)
                # If it looks like a real JSON object (has `"key":` pattern),
                # leave it alone — only convert when it's clearly a set.
                if _re.search(r'"\s*:\s*', inner):
                    return m.group(0)
                return "[" + inner + "]"
            prev = None
            while s != prev:
                prev = s
                s = _re.sub(r"\{([^{}]*)\}", repl, s)
            return s
        text = _coerce_sets(text)

        items: list | None = None
        # 4. Try strict JSON first.
        try:
            parsed = _json.loads(text)
            if isinstance(parsed, list):
                items = parsed
        except Exception:
            pass
        # 5. Fall back to the LAST [...] block in the text — useful when the
        # model narrates before/after the JSON. We try last-first because
        # any inline example list would appear earlier.
        if items is None:
            for block in reversed(_re.findall(r"\[[^\[\]]*\]", text, flags=_re.DOTALL)):
                try:
                    parsed = _json.loads(block)
                    if isinstance(parsed, list):
                        items = parsed
                        break
                except Exception:
                    continue

        if items is None:
            preview = (raw or "")[:200].replace("\n", " ")
            log.warning("entity-extract: unparseable output: %r", preview)
            if hasattr(self, "last_query_warnings"):
                self.last_query_warnings.append(
                    f"Entity extractor returned unparseable output: {preview!r}"
                )
            return []

        # Flatten arbitrary nesting. Models occasionally emit
        # `[["Alice", "Bob"]]` (list-of-lists) or mix scalars with sub-lists;
        # collect every leaf string and discard structure.
        out: list[str] = []
        def _collect(x):
            if isinstance(x, list):
                for y in x:
                    _collect(y)
            elif x is not None:
                s = str(x).strip()
                if s:
                    out.append(s)
        _collect(items)

        # An empty (but valid) extraction is just as fatal as a parse failure
        # for graph retrieval — surface it so the user can see the LLM ran but
        # found nothing.
        if not out:
            preview = (raw or "")[:120].replace("\n", " ")
            log.warning("entity-extract: 0 entities; raw=%r", preview)
            if hasattr(self, "last_query_warnings"):
                self.last_query_warnings.append(
                    f"Graph LLM extracted 0 entities from the question. Raw output: {preview!r}"
                )
        else:
            log.info("entity-extract: %d entities: %r", len(out), out)
        return out

    def _graph_retrieve(self, question: str, pool: int) -> list[Document]:
        """Pure graph retrieval.

        1. Extract entity terms from the question.
        2. Match `:__Entity__` nodes whose `id` substring-matches a term.
        3. Expand `graph_hops` away through any relationship.
        4. Return the source `:Document` chunk text linked via `[:MENTIONS]`.

        No vector similarity anywhere in this path. Recall depends entirely
        on whether the question's entities exist in the graph — that's the
        intended tradeoff for this backend.
        """
        terms = self._extract_query_entities(question)
        # Snapshot the entities for the query-time diagnostic surface (UI
        # shows "what did the graph LLM extract?" so silent failures aren't
        # mysterious).
        self.last_extracted_entities = list(terms)
        if not terms:
            return []

        graph = self._ensure_neo4j_graph()
        # Clamp hops to a sane literal range — Cypher requires variable-
        # length path bounds to be literals, so we format it into the query
        # only after validating.
        hops = max(0, min(int(self.settings.graph_hops or 0), 3))
        max_entities = max(1, int(self.settings.graph_max_entities or 10))

        cypher = f"""
        UNWIND $terms AS term
        MATCH (e:__Entity__)
        WHERE toLower(e.id) CONTAINS toLower(term)
        WITH e, count(*) AS hits
        ORDER BY hits DESC
        LIMIT $max_entities
        WITH collect(e) AS seeds
        UNWIND seeds AS m
        OPTIONAL MATCH (m)-[*0..{hops}]-(n:__Entity__)
        WITH collect(DISTINCT m) + collect(DISTINCT n) AS allEntities
        UNWIND allEntities AS ent
        MATCH (ent)<-[:MENTIONS]-(d:Document)
        RETURN DISTINCT d.text AS text,
                        coalesce(d.source, '?') AS source
        LIMIT $pool
        """
        log.info(
            "graph-retrieve: cypher seeds=%r hops=%d max_entities=%d pool=%d",
            terms, hops, max_entities, pool,
        )
        try:
            rows = graph.query(
                cypher,
                params={"terms": terms, "max_entities": max_entities, "pool": pool},
            )
        except Exception as e:
            log.warning("graph-retrieve: Cypher failed: %s", e)
            return []
        out: list[Document] = []
        for r in rows:
            text = r.get("text")
            if not text:
                continue
            out.append(Document(page_content=text, metadata={"source": r.get("source") or "?"}))
        log.info("graph-retrieve: %d row(s) → %d Document(s)", len(rows), len(out))
        return out

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
                all_docs.extend(self._graph_retrieve(question, pool))
            except Exception as e:
                if backends_tried > 1:
                    self.last_query_warnings.append(f"Neo4j graph retrieval failed: {e}")
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
                # Pure graph hits have no native cosine score. When the
                # cross-encoder reranker is unavailable we treat any chunk
                # whose entity matched as presumed-relevant (score 1.0) so
                # it isn't filtered out by the threshold gate downstream.
                for d in self._graph_retrieve(question, k):
                    scored.append((d, 1.0))
            except Exception as e:
                if backends_tried > 1:
                    self.last_query_warnings.append(f"Neo4j graph retrieval failed: {e}")
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
        # Reset per-query diagnostics so the UI never shows stale info.
        self.last_extracted_entities: list[str] = []
        scored = self._retrieve_scored(question)
        top_score = scored[0][1] if scored else 0.0
        docs = [d for d, _ in scored]
        use_rag = bool(docs) and top_score >= _RERANK_SCORE_THRESHOLD
        log.info(
            "query: q=%r → mode=%s top_score=%.3f docs=%d",
            question, "rag" if use_rag else "chat", top_score, len(docs),
        )

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
            "extracted_entities": list(getattr(self, "last_extracted_entities", []) or []),
            "graph_backend_used": self._use_neo4j,
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
        help="vector = Chroma; neo4j = pure knowledge-graph retrieval; both = hybrid (Chroma vector + Neo4j graph).",
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
