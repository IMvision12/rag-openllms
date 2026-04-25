"""Streamlit UI for the RAG OpenLLMs pipeline."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import streamlit as st

from rag_brain.config import load_settings
from rag_brain.pipeline import RAGPipeline


# ─── Presets ──────────────────────────────────────────────────────────────

HF_LLM_PRESETS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "tiiuae/falcon-7b-instruct",
    "Custom…",
]

OLLAMA_LLM_PRESETS = [
    # Cloud-hosted (no local download; require `ollama run <name>` with a
    # signed-in Ollama account and sufficient credits)
    "kimi-k2.5:cloud",
    "glm-5:cloud",
    "minimax-m2.7:cloud",
    "gemma4:31b-cloud",
    "qwen3.5:397b-cloud",
    "gpt-oss:120b-cloud",
    "gpt-oss:20b-cloud",
    # Local (pull via `ollama pull <name>`)
    "gpt-oss:120b",
    "gpt-oss:20b",
    "gemma4:31b",
    "gemma4:26b",
    "gemma4:e4b",
    "gemma4:e2b",
    "deepseek-r1:8b",
    "Custom…",
]

EMBEDDING_PRESETS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "intfloat/e5-base-v2",
    "Custom…",
]

# Canonical defaults. _hard_reset_session restores these explicitly so the
# header chips and widgets both render fresh values without having to rely
# on Streamlit's internal widget cache being cleared by session_state wipe.
#
# Keys set to None are *user choices* with no hardcoded default — the UI
# renders them with no pre-selection and requires the user to pick before
# Next/Ingest/Chat is enabled. Only environment-neutral knobs (chunking,
# chunk size, top-k, quantization, max tokens, embedding preset) keep a
# baked-in default.
_DEFAULT_CFG: dict = {
    "cfg_backend": None,          # user picks: vector | both | neo4j
    "cfg_chunking": "fixed",
    "cfg_top_k": 4,
    "cfg_chunk_size": 1200,
    "cfg_chunk_overlap": 200,
    "cfg_emb_sel": EMBEDDING_PRESETS[0],
    "cfg_emb_custom": EMBEDDING_PRESETS[0],
    "cfg_llm_provider": None,     # user picks: huggingface | ollama
    "cfg_ollama_sel": None,       # user picks when provider=ollama
    "cfg_ollama_custom": "",
    "cfg_hf_sel": None,           # user picks when provider=huggingface
    "cfg_hf_custom": "",
    "cfg_hf_token": "",           # optional, required for gated HF models
    # Graph-extraction LLM (only relevant when backend includes Neo4j).
    # "none" = skip entity/relation extraction; Neo4j stores only flat
    # Chunk nodes with embeddings (disconnected graph).
    "cfg_graph_llm_provider": "none",  # none | anthropic | openai | gemini | openrouter
    "cfg_graph_llm_model": "",
    "cfg_graph_llm_api_key": "",
}


# ─── Page config + styling ────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG OpenLLMs",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root {
        --accent: #7c8aff;
        --accent-soft: rgba(124, 138, 255, 0.14);
        --accent-border: rgba(124, 138, 255, 0.35);
        --surface: #141824;
        --surface-2: #1a1f2d;
        --border: rgba(255, 255, 255, 0.08);
        --text-muted: rgba(230, 231, 238, 0.62);
      }
      section[data-testid="stSidebar"], button[kind="header"] { display: none !important; }
      #MainMenu, footer { visibility: hidden; }

      .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1180px;
      }

      h1, h2, h3, h4 { letter-spacing: -0.02em; font-weight: 600; }
      h1 { margin-top: 0; margin-bottom: 0.25rem; font-size: 2.25rem; }
      h2 { font-size: 1.5rem; margin-top: 0.5rem; }
      h3 { font-size: 1.1rem; margin-top: 1rem; color: rgba(230, 231, 238, 0.88); }
      hr { margin: 1.25rem 0; border: none; border-top: 1px solid var(--border); }

      .muted { color: var(--text-muted); font-size: 0.92rem; }

      /* Status chips */
      .status-bar { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 0.75rem; }
      .chip {
        display: inline-flex; align-items: center;
        padding: 4px 12px;
        border-radius: 999px;
        background: var(--surface);
        border: 1px solid var(--border);
        font-size: 0.78rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        white-space: nowrap;
        color: rgba(230, 231, 238, 0.9);
      }
      .chip b { font-weight: 500; margin-right: 6px; color: var(--text-muted); }

      /* Custom status banners (replace st.info/warning/success/error) */
      .banner {
        display: flex; align-items: center; gap: 10px;
        padding: 12px 16px;
        border-radius: 10px;
        font-size: 0.92rem;
        border: 1px solid var(--border);
        background: var(--surface);
        color: rgba(230, 231, 238, 0.92);
      }
      .banner.info    { border-left: 3px solid var(--accent); }
      .banner.success { border-left: 3px solid #4ade80; }
      .banner.warning { border-left: 3px solid #fbbf24; }
      .banner.error   { border-left: 3px solid #f87171; }
      .banner .dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--accent); flex-shrink: 0;
      }
      .banner.success .dot { background: #4ade80; }
      .banner.warning .dot { background: #fbbf24; }
      .banner.error .dot { background: #f87171; }

      /* Override native Streamlit alerts to match */
      div[data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid var(--border);
        background: var(--surface) !important;
        padding: 10px 14px !important;
      }
      div[data-testid="stAlert"] > div { background: transparent !important; }

      /* Navigation pills */
      .nav-wrap div[role="radiogroup"] {
        gap: 4px;
        background: var(--surface);
        padding: 5px;
        border-radius: 12px;
        border: 1px solid var(--border);
      }
      .nav-wrap div[role="radiogroup"] > label {
        background: transparent;
        padding: 7px 18px;
        border-radius: 8px;
        margin: 0 !important;
        transition: background 0.15s, color 0.15s;
        color: var(--text-muted);
      }
      .nav-wrap div[role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.04);
        color: rgba(230, 231, 238, 0.95);
      }
      .nav-wrap div[role="radiogroup"] > label:has(input:checked) {
        background: var(--accent-soft);
        color: #fff;
      }
      .nav-wrap div[role="radiogroup"] > label > div:first-child { display: none; }

      /* Metrics */
      div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 16px;
      }
      div[data-testid="stMetricLabel"] { color: var(--text-muted); font-size: 0.8rem; }
      div[data-testid="stMetricValue"] { font-weight: 600; font-size: 1.4rem; }

      /* Buttons */
      .stButton > button, .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 500;
        border: 1px solid var(--border);
        transition: transform 0.05s, background 0.15s;
      }
      .stButton > button:hover { transform: translateY(-1px); }
      .stButton > button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
        color: #0a0d14;
      }
      .stButton > button[kind="primary"]:hover {
        background: #9aa5ff;
        border-color: #9aa5ff;
      }

      /* Inputs */
      .stTextInput > div > div > input,
      .stNumberInput > div > div > input,
      .stTextArea textarea,
      div[data-baseweb="select"] > div {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
      }

      /* Chat */
      .stChatMessage {
        border-radius: 14px;
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 14px 18px;
      }
      .stChatInput { border-radius: 12px; }

      /* File uploader */
      section[data-testid="stFileUploaderDropzone"] {
        background: var(--surface);
        border: 1.5px dashed var(--border);
        border-radius: 12px;
      }

      /* Progress */
      .stProgress > div > div > div { background: var(--accent); }

      /* Stepper */
      .stepper {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        margin: 1.25rem 0 2rem;
      }
      .step-item {
        display: flex;
        align-items: center;
        gap: 10px;
        color: var(--text-muted);
        font-size: 0.9rem;
        font-weight: 500;
      }
      .step-item .num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: var(--surface);
        border: 1px solid var(--border);
        font-size: 0.8rem;
        font-weight: 600;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      }
      .step-item.active { color: #fff; }
      .step-item.active .num {
        background: var(--accent);
        border-color: var(--accent);
        color: #0a0d14;
      }
      .step-item.done { color: rgba(230, 231, 238, 0.8); }
      .step-item.done .num {
        background: var(--accent-soft);
        border-color: var(--accent-border);
        color: var(--accent);
      }
      .step-line {
        flex: 0 0 48px;
        height: 1px;
        background: var(--border);
        margin: 0 12px;
      }
      .step-line.done { background: var(--accent-border); }
    </style>
    """,
    unsafe_allow_html=True,
)


def banner(kind: str, text: str) -> None:
    """Render a custom themed banner (info/success/warning/error)."""
    st.markdown(
        f'<div class="banner {kind}"><span class="dot"></span><span>{text}</span></div>',
        unsafe_allow_html=True,
    )


# ─── Fresh-load detection ─────────────────────────────────────────────────
# Generate a token that lives for the life of the Streamlit server process.
# Compare it to a token in the URL: mismatch means the user just reloaded
# (or hit the app fresh, or the server restarted) — so wipe everything.

@st.cache_resource
def _server_boot_token() -> str:
    import secrets
    return secrets.token_hex(6)

_BOOT_TOKEN = _server_boot_token()


def _hard_reset_session() -> None:
    """Wipe every key in session_state, release resources, clear caches.

    Explicitly re-stamps all cfg_* defaults and bumps _ui_nonce so widgets
    that Streamlit tracks internally (position-based, no key=) are forced to
    re-instantiate on the next rerun — otherwise their cached state leaks
    back and silently overwrites the reset values after widgets render.
    """
    pipe = st.session_state.get("pipeline")
    if pipe is not None:
        try:
            pipe._release_chroma()
        except Exception:
            pass
    # Preserve and bump the nonce before wiping; widget keys suffixed with
    # this nonce become fresh widget identities for Streamlit.
    next_nonce = int(st.session_state.get("_ui_nonce", 0)) + 1
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.session_state["_ui_nonce"] = next_nonce
    # Re-stamp cfg defaults + non-cfg state so the header chips (which read
    # cfg_* directly) and widgets that render after a reset both see correct
    # values even within the *same* render pass.
    for k, v in _DEFAULT_CFG.items():
        st.session_state[k] = v
    st.session_state["pipeline"] = None
    st.session_state["pipeline_signature"] = None
    st.session_state["chat_history"] = []
    st.session_state["ingested_files"] = []
    st.session_state["step"] = 1
    import gc
    gc.collect()


def _wk(base: str) -> str:
    """Widget key namespaced by the reset nonce — each Reset bumps the
    nonce, giving every widget a fresh identity so Streamlit can't carry
    cached widget state across a reset."""
    return f"{base}__v{st.session_state.get('_ui_nonce', 0)}"


# URL token drives the reset. On a fresh browser load (or reload) the URL
# has no `s` param OR a stale one — we wipe state and stamp the current boot
# token, then rerun.
_url_token = st.query_params.get("s")
if _url_token != _BOOT_TOKEN:
    _hard_reset_session()
    st.query_params["s"] = _BOOT_TOKEN
    st.rerun()


# ─── Session state ────────────────────────────────────────────────────────

st.session_state.setdefault("_ui_nonce", 0)
st.session_state.setdefault("pipeline", None)
st.session_state.setdefault("pipeline_signature", None)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("ingested_files", [])
st.session_state.setdefault("step", 1)

# Config widget defaults. Keys set to None are user picks with no default —
# the UI forces an explicit choice (disabled Next until set) and chips show
# em-dashes. Keep in sync with _DEFAULT_CFG above.
for _k, _v in _DEFAULT_CFG.items():
    st.session_state.setdefault(_k, _v)


def _effective(sel_key: str, custom_key: str) -> str:
    choice = st.session_state.get(sel_key)
    if choice == "Custom…":
        return (st.session_state.get(custom_key) or "").strip()
    return choice


def current_config() -> dict:
    return {
        "backend": st.session_state.cfg_backend,
        "chunking": st.session_state.cfg_chunking,
        "top_k": st.session_state.cfg_top_k,
        "chunk_size": st.session_state.cfg_chunk_size,
        "chunk_overlap": st.session_state.cfg_chunk_overlap,
        "embedding_model": _effective("cfg_emb_sel", "cfg_emb_custom"),
        "llm_provider": st.session_state.cfg_llm_provider,
        "ollama_model": _effective("cfg_ollama_sel", "cfg_ollama_custom"),
        "hf_model": _effective("cfg_hf_sel", "cfg_hf_custom"),
        "hf_token": (st.session_state.cfg_hf_token or "").strip(),
        "graph_llm_provider": (st.session_state.cfg_graph_llm_provider or "none"),
        "graph_llm_model": (st.session_state.cfg_graph_llm_model or "").strip(),
        "graph_llm_api_key": (st.session_state.cfg_graph_llm_api_key or "").strip(),
    }


def _enum_val(x) -> str:
    """Read .value off an enum, or return the str — model_copy(update=...) can
    leave a field as a raw string if pydantic skipped re-validation."""
    return getattr(x, "value", x) if not isinstance(x, str) else x


def _pipe_matches_ui(pipe: RAGPipeline, c: dict) -> bool:
    """Hard check: does the pipeline's actual settings match what the UI shows?

    Last line of defense against stale cached pipelines. If any key field
    mismatches, we force a rebuild regardless of the signature cache.
    If the UI has required picks unset, the pipeline can't match by
    construction — return False so callers invalidate the stale pipeline.
    """
    s = pipe.settings
    if not c.get("llm_provider") or not c.get("backend"):
        return False
    if s.llm_provider.lower() != c["llm_provider"].lower():
        return False
    if _enum_val(s.retrieval_backend) != c["backend"]:
        return False
    if _enum_val(s.chunking_strategy) != c["chunking"]:
        return False
    if s.embedding_model != c["embedding_model"]:
        return False
    if s.llm_provider.lower() == "ollama" and s.ollama_model != (c.get("ollama_model") or ""):
        return False
    if s.llm_provider.lower() == "huggingface" and s.hf_model != (c.get("hf_model") or ""):
        return False
    if (s.graph_llm_provider or "none").lower() != (c.get("graph_llm_provider") or "none").lower():
        return False
    if (s.graph_llm_model or "") != (c.get("graph_llm_model") or ""):
        return False
    if (s.graph_llm_api_key or "") != (c.get("graph_llm_api_key") or ""):
        return False
    return True


def _invalidate_pipeline() -> None:
    pipe = st.session_state.pipeline
    if pipe is not None:
        try:
            pipe._release_chroma()
        except Exception:
            pass
    st.session_state.pipeline = None
    st.session_state.pipeline_signature = None
    import gc
    gc.collect()


def ensure_pipeline() -> RAGPipeline | None:
    """Auto-build the pipeline if missing or out of sync with current config."""
    c = current_config()
    pipe = st.session_state.pipeline

    # Hard sanity check: if the cached pipeline doesn't match what the UI
    # shows, wipe it. This is stricter than the signature comparison and
    # catches any drift that sig caching might miss.
    if pipe is not None and not _pipe_matches_ui(pipe, c):
        _invalidate_pipeline()
        pipe = None

    if pipe is not None:
        return pipe

    is_first_load = st.session_state.pipeline_signature is None
    spinner_msg = (
        "Loading pipeline (first run downloads models — may take a minute)…"
        if is_first_load
        else "Applying new settings…"
    )
    try:
        with st.spinner(spinner_msg):
            pipe = build_pipeline()
            # Verify what we actually built matches what the UI asked for —
            # if not, something is structurally broken and we should surface it.
            if not _pipe_matches_ui(pipe, c):
                raise RuntimeError(
                    f"Built pipeline has llm_provider={pipe.settings.llm_provider!r} "
                    f"but UI requested {c['llm_provider']!r} — settings override failed."
                )
            st.session_state.pipeline = pipe
            st.session_state.pipeline_signature = tuple(sorted(c.items()))
        return pipe
    except Exception as e:
        banner("error", f"Pipeline initialization failed: {e}")
        return None


def _missing_config_fields(c: dict) -> list[str]:
    """Names of required user picks that haven't been made yet."""
    missing: list[str] = []
    if not c.get("backend"):
        missing.append("backend (Step 1)")
    p = c.get("llm_provider")
    if not p:
        missing.append("LLM provider (Step 2)")
    elif p == "ollama" and not c.get("ollama_model"):
        missing.append("Ollama model (Step 2)")
    elif p == "huggingface" and not c.get("hf_model"):
        missing.append("HuggingFace model (Step 2)")
    return missing


def build_pipeline() -> RAGPipeline:
    """Build a RAGPipeline with explicit settings — no env-var dance, no .env fallback."""
    from rag_brain.config import Settings, RetrievalBackend, ChunkingStrategy

    c = current_config()
    missing = _missing_config_fields(c)
    if missing:
        raise ValueError("Please make a selection for: " + ", ".join(missing) + ".")
    # Start from file-based settings so Neo4j credentials, paths, etc. still come
    # from .env, then override every UI-controlled field explicitly.
    base = load_settings()
    # Re-validate everything so enum fields stay real enums after the override
    # (model_copy with raw strings keeps them as strings, which breaks .value).
    data = base.model_dump()
    data.update({
        "retrieval_backend": RetrievalBackend(c["backend"]),
        "chunking_strategy": ChunkingStrategy(c["chunking"]),
        "top_k": int(c["top_k"]),
        "chunk_size": int(c["chunk_size"]),
        "chunk_overlap": int(c["chunk_overlap"]),
        "embedding_model": c["embedding_model"],
        "llm_provider": c["llm_provider"],
        "ollama_model": c["ollama_model"] or "",
        "hf_model": c["hf_model"] or "",
        "hf_token": c.get("hf_token") or "",
        "graph_llm_provider": (c.get("graph_llm_provider") or "none"),
        "graph_llm_model": c.get("graph_llm_model") or "",
        "graph_llm_api_key": c.get("graph_llm_api_key") or "",
    })
    settings = Settings(**data)
    return RAGPipeline(settings)


# ─── Config-drift detection ───────────────────────────────────────────────
# If the cached pipeline's actual settings don't match what the UI shows
# right now, drop it. The next chat/ingest will rebuild.

if (
    st.session_state.pipeline is not None
    and not _pipe_matches_ui(st.session_state.pipeline, current_config())
):
    _invalidate_pipeline()


# ─── Header ───────────────────────────────────────────────────────────────

_hdr_l, _hdr_r = st.columns([5, 1])
with _hdr_l:
    st.markdown("# ◆ RAG OpenLLMs")
    st.markdown(
        '<span class="muted">Hybrid retrieval over Chroma + Neo4j with open-source LLMs.</span>',
        unsafe_allow_html=True,
    )
with _hdr_r:
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    if st.button("⟲ Reset", use_container_width=True, key="header_reset",
                 help="Wipe all selections, uploads, chat history, and cached pipeline."):
        _hard_reset_session()
        st.rerun()

pipe: RAGPipeline | None = st.session_state.pipeline
_cfg_now = current_config()


def _llm_chip(cfg: dict) -> str:
    p = cfg.get("llm_provider")
    if not p:
        return "—"
    model = cfg.get("ollama_model") if p == "ollama" else cfg.get("hf_model")
    if not model:
        return f"{p}:—"
    return f"{p}:{model.split('/')[-1]}"


chips = [
    ("backend", _cfg_now["backend"] or "—"),
    ("chunking", _cfg_now["chunking"] or "—"),
    ("top-k", str(_cfg_now["top_k"]) if _cfg_now["top_k"] is not None else "—"),
    ("llm", _llm_chip(_cfg_now)),
    ("status", "active" if pipe is not None else "not initialized"),
]
_chips_html = "".join(f'<span class="chip"><b>{k}</b>{v}</span>' for k, v in chips)
st.markdown(f'<div class="status-bar">{_chips_html}</div>', unsafe_allow_html=True)

st.markdown('<hr>', unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────

def is_hf_model_cached(repo_id: str) -> bool:
    """Check if a HuggingFace model has at least its config cached locally."""
    try:
        from huggingface_hub import try_to_load_from_cache
        # Embedding models use config.json; LLMs also have config.json
        return try_to_load_from_cache(repo_id=repo_id, filename="config.json") is not None
    except Exception:
        return False


def human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def get_hf_cache_size(repo_id: str) -> int | None:
    """Return total bytes cached for a given repo_id, or None if not found."""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        for repo in info.repos:
            if repo.repo_id == repo_id:
                return int(repo.size_on_disk)
    except Exception:
        return None
    return None


try:
    # huggingface_hub's tqdm wrapper knows about HF-specific kwargs like `name=`
    from huggingface_hub.utils import tqdm as _BaseTqdm
except ImportError:
    from tqdm.auto import tqdm as _BaseTqdm


import threading as _threading


class StreamlitTqdm(_BaseTqdm):
    """Thread-safe byte counter for snapshot_download.

    HF parallelizes downloads across worker threads, so calling Streamlit
    widgets directly from tqdm.update() (which runs on those threads) silently
    no-ops. Instead we just update class-level counters under a lock; the main
    thread polls them and renders the Streamlit progress bar.
    """
    _lock = _threading.Lock()
    _done = 0
    _total = 0

    @classmethod
    def reset_counters(cls) -> None:
        with cls._lock:
            cls._done = 0
            cls._total = 0

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("leave", False)
        kwargs.setdefault("mininterval", 0.05)
        kwargs["disable"] = False
        unit = kwargs.get("unit") or ""
        self._sl_is_bytes = isinstance(unit, str) and unit.startswith("B")
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            safe = {k: v for k, v in kwargs.items() if k != "name"}
            super().__init__(*args, **safe)
        self.disable = False

    def update(self, n=1):
        try:
            super().update(n)
        except Exception:
            pass
        if getattr(self, "_sl_is_bytes", False):
            with StreamlitTqdm._lock:
                StreamlitTqdm._done = int(getattr(self, "n", 0) or 0)
                StreamlitTqdm._total = int(getattr(self, "total", 0) or 0)

    def refresh(self, *args, **kwargs):
        try:
            super().refresh(*args, **kwargs)
        except Exception:
            pass
        if getattr(self, "_sl_is_bytes", False):
            with StreamlitTqdm._lock:
                StreamlitTqdm._done = int(getattr(self, "n", 0) or 0)
                StreamlitTqdm._total = int(getattr(self, "total", 0) or 0)

    def display(self, msg=None, pos=None):
        return True


def download_hf_model(repo_id: str, token: str | None = None) -> None:
    """Download a HuggingFace model with a live Streamlit progress bar.

    `token`, when supplied, authorizes access to gated repos (Llama, Gemma,
    Mistral, etc.). Without it, snapshot_download hits 401 for gated models.
    """
    import importlib
    import inspect
    import logging
    import threading
    import time

    with st.status(f"Downloading {repo_id}…", expanded=True) as status:
        progress = st.progress(0.0, text="Connecting to HuggingFace Hub…")
        StreamlitTqdm.reset_counters()

        # --- HF progress bar plumbing (run on main thread, before worker starts) ---
        hf_logger = logging.getLogger("huggingface_hub")
        original_level = hf_logger.level
        hf_logger.setLevel(logging.INFO)
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        original_flag = None
        _hf_tqdm_mod = None
        try:
            _hf_tqdm_mod = importlib.import_module("huggingface_hub.utils.tqdm")
            original_flag = _hf_tqdm_mod.HF_HUB_DISABLE_PROGRESS_BARS
            _hf_tqdm_mod.HF_HUB_DISABLE_PROGRESS_BARS = False
            _hf_tqdm_mod.progress_bar_states["_global"] = True
        except Exception:
            pass

        # --- Run snapshot_download on a worker thread ---
        result: dict = {}

        def _worker():
            try:
                from huggingface_hub import snapshot_download
                kwargs: dict = {"repo_id": repo_id}
                sig = inspect.signature(snapshot_download)
                if "tqdm_class" in sig.parameters:
                    kwargs["tqdm_class"] = StreamlitTqdm
                if token:
                    kwargs["token"] = token
                result["path"] = snapshot_download(**kwargs)
            except Exception as e:
                result["error"] = e

        t0 = time.time()
        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        try:
            # Poll counters from the main thread — only the main thread can
            # push widget updates to the Streamlit frontend.
            while worker.is_alive():
                with StreamlitTqdm._lock:
                    done = StreamlitTqdm._done
                    total = StreamlitTqdm._total
                if total > 0:
                    frac = min(max(done / total, 0.0), 1.0)
                    text = f"{human_size(done)} / {human_size(total)}  ·  {frac * 100:.1f}%"
                else:
                    frac = 0.0
                    text = f"Fetching file list…  {human_size(done)} downloaded"
                try:
                    progress.progress(frac, text=text)
                except Exception:
                    pass
                time.sleep(0.15)
            worker.join(timeout=2)

            if "error" in result:
                raise result["error"]
            path = result.get("path", "?")
            elapsed = time.time() - t0
            total = get_hf_cache_size(repo_id) or StreamlitTqdm._total or 0
            progress.progress(1.0, text=f"✓ {human_size(total)} in {elapsed:.1f}s")
            status.write(f"Local path: `{path}`")
            status.update(label=f"✓ {repo_id} ready ({human_size(total)})", state="complete")
        except Exception as e:
            status.update(label=f"Download failed: {e}", state="error")
        finally:
            hf_logger.setLevel(original_level)
            if _hf_tqdm_mod is not None and original_flag is not None:
                _hf_tqdm_mod.HF_HUB_DISABLE_PROGRESS_BARS = original_flag


def list_hf_cached_repos() -> list[dict]:
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        return sorted(
            [{"id": r.repo_id, "size": r.size_on_disk, "type": r.repo_type} for r in info.repos],
            key=lambda x: -x["size"],
        )
    except Exception:
        return []


def delete_hf_repo(repo_id: str) -> int:
    """Delete all cached revisions for a single repo. Returns bytes freed."""
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir()
    for repo in info.repos:
        if repo.repo_id == repo_id:
            revisions = [rev.commit_hash for rev in repo.revisions]
            if revisions:
                strategy = info.delete_revisions(*revisions)
                freed = strategy.expected_freed_size
                strategy.execute()
                return freed
    return 0


def delete_all_hf_cache() -> int:
    """Delete every cached HuggingFace repo. Returns bytes freed."""
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir()
    revisions = []
    for repo in info.repos:
        for rev in repo.revisions:
            revisions.append(rev.commit_hash)
    if not revisions:
        return 0
    strategy = info.delete_revisions(*revisions)
    freed = strategy.expected_freed_size
    strategy.execute()
    return freed


def render_hf_model_card(repo_id: str, key: str, kind: str) -> None:
    """Show cache status + download button for a HF repo."""
    if not repo_id:
        return
    cached = is_hf_model_cached(repo_id)
    size = get_hf_cache_size(repo_id) if cached else None

    col_info, col_btn = st.columns([3, 2])
    with col_info:
        if cached:
            size_str = human_size(size) if size else ""
            banner("success", f"<b>{kind}</b> · <code>{repo_id}</code> cached locally ({size_str}).")
        else:
            banner("info", f"<b>{kind}</b> · <code>{repo_id}</code> not downloaded yet.")
    with col_btn:
        label = "Re-verify" if cached else "Download now"
        btn_type = "secondary" if cached else "primary"
        if st.button(label, key=f"dl_btn_{key}", use_container_width=True, type=btn_type):
            token = (st.session_state.get("cfg_hf_token") or "").strip() or None
            download_hf_model(repo_id, token=token)
            st.rerun()


# Reasoning-model outputs (Qwen, DeepSeek-R1, QwQ, …) wrap chain-of-thought
# in <think>…</think>. Strip those before rendering so the chat shows only
# the final answer; surface the reasoning in a collapsed expander instead.
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def _split_thinking(text: str) -> tuple[str, str]:
    """Return (thinking, answer) with all <think>…</think> blocks extracted."""
    if not text:
        return "", text or ""
    thinking_parts = _THINK_RE.findall(text)
    answer = _THINK_RE.sub("", text)
    # Handle truncated generation where </think> never arrived.
    if "<think>" in answer.lower():
        answer = _OPEN_THINK_RE.sub("", answer)
    thinking = "\n\n".join(p.strip() for p in thinking_parts if p.strip())
    return thinking, answer.strip()


def _render_assistant_message(content: str, mode: str | None = None, top_score: float | None = None) -> None:
    thinking, answer = _split_thinking(content)
    if thinking:
        with st.expander("Reasoning", expanded=False):
            st.markdown(thinking)
    st.markdown(answer or "_(empty answer)_")
    if mode == "rag":
        st.caption(f"📄 Answered from your documents · top relevance **{top_score:.2f}**")
    elif mode == "chat":
        s = f" · top relevance **{top_score:.2f}**" if top_score is not None else ""
        st.caption(f"💬 Chat mode — no document context used{s}")


def _render_retrieved(retrieved: list[dict]) -> None:
    if not retrieved:
        return
    with st.expander(f"Retrieved chunks ({len(retrieved)})"):
        for i, d in enumerate(retrieved, start=1):
            meta = d.get("metadata", {}) or {}
            src = meta.get("source", "?")
            page = meta.get("page", "")
            header = f"[{i}] {src}" + (f" — page {page}" if page != "" else "")
            st.markdown(f"**{header}**")
            st.text(d.get("content", "")[:1200])
            st.divider()


# ─── Wizard stepper ───────────────────────────────────────────────────────

STEPS = [
    (1, "Configuration"),
    (2, "Models"),
    (3, "Documents"),
    (4, "Chat"),
]


def render_stepper(current: int) -> None:
    parts = []
    for i, (num, label) in enumerate(STEPS):
        if num < current:
            cls = "done"
        elif num == current:
            cls = "active"
        else:
            cls = ""
        parts.append(
            f'<div class="step-item {cls}"><span class="num">{num}</span><span>{label}</span></div>'
        )
        if i < len(STEPS) - 1:
            line_cls = "done" if num < current else ""
            parts.append(f'<div class="step-line {line_cls}"></div>')
    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_nav(*, can_back: bool = True, next_label: str = "Next →", next_disabled: bool = False) -> None:
    st.markdown('<hr>', unsafe_allow_html=True)
    left, _, right = st.columns([1, 3, 1])
    step = st.session_state.step
    with left:
        if can_back and step > 1:
            if st.button("← Back", use_container_width=True, key=f"back_{step}"):
                st.session_state.step -= 1
                st.rerun()
    with right:
        if step < len(STEPS):
            if st.button(
                next_label,
                type="primary",
                use_container_width=True,
                disabled=next_disabled,
                key=f"next_{step}",
            ):
                st.session_state.step += 1
                st.rerun()


# ─── Step 1: Retrieval ────────────────────────────────────────────────────

def step_retrieval() -> None:
    st.markdown("## Step 1 — Configuration")
    st.markdown(
        '<span class="muted">Choose how documents are chunked and how retrieval runs.</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    st.markdown("### Backend")
    backend_options = ["vector", "both", "neo4j"]
    backend_current = st.session_state.cfg_backend
    _b_idx = backend_options.index(backend_current) if backend_current in backend_options else None
    chosen_backend = st.radio(
        "Retrieval backend",
        backend_options,
        index=_b_idx,
        horizontal=True,
        label_visibility="collapsed",
        help="vector = Chroma only (no external server) · both = Chroma + Neo4j graph · neo4j = pure knowledge-graph retrieval",
        key=_wk("cfg_backend"),
    )
    st.session_state.cfg_backend = chosen_backend
    if chosen_backend:
        st.caption(f"Selected backend: **{chosen_backend}**")
    else:
        st.caption("_Pick a backend to continue._")
    if chosen_backend in ("neo4j", "both"):
        banner("info", "Neo4j credentials are read from <code>.env</code> (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD).")

    st.markdown("### Chunking")
    c1, c2 = st.columns(2)
    chunking_opts = ["fixed", "semantic"]
    with c1:
        chosen_chunking = st.radio(
            "Strategy",
            chunking_opts,
            index=chunking_opts.index(st.session_state.cfg_chunking),
            horizontal=True,
            key=_wk("cfg_chunking"),
            help="fixed = RecursiveCharacterTextSplitter · semantic = embedding-based breakpoints",
        )
        st.session_state.cfg_chunking = chosen_chunking
    with c2:
        chosen_top_k = st.slider(
            "Top-k retrieved chunks", 1, 20,
            value=int(st.session_state.cfg_top_k),
            key=_wk("cfg_top_k"),
        )
        st.session_state.cfg_top_k = int(chosen_top_k)

    if st.session_state.cfg_chunking == "fixed":
        c3, c4 = st.columns(2)
        with c3:
            chosen_size = st.number_input(
                "Chunk size", 200, 4000, step=100,
                value=int(st.session_state.cfg_chunk_size),
                key=_wk("cfg_chunk_size"),
            )
            st.session_state.cfg_chunk_size = int(chosen_size)
        with c4:
            chosen_overlap = st.number_input(
                "Chunk overlap", 0, 1000, step=50,
                value=int(st.session_state.cfg_chunk_overlap),
                key=_wk("cfg_chunk_overlap"),
            )
            st.session_state.cfg_chunk_overlap = int(chosen_overlap)


# ─── Step 2: Models ───────────────────────────────────────────────────────

def step_models() -> None:
    st.markdown("## Step 2 — Models")
    st.markdown(
        '<span class="muted">Pick the embedding model and the LLM that generates answers.</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    st.markdown("### Embedding model")
    e1, e2 = st.columns([3, 2])
    with e1:
        emb_current = st.session_state.cfg_emb_sel
        if emb_current not in EMBEDDING_PRESETS:
            emb_current = EMBEDDING_PRESETS[0]
            st.session_state.cfg_emb_sel = emb_current
        chosen_emb = st.selectbox(
            "Preset",
            EMBEDDING_PRESETS,
            index=EMBEDDING_PRESETS.index(emb_current),
            key=_wk("cfg_emb_sel"),
        )
        st.session_state.cfg_emb_sel = chosen_emb
    with e2:
        if chosen_emb == "Custom…":
            custom_emb = st.text_input(
                "Custom HF model ID",
                value=st.session_state.cfg_emb_custom,
                key=_wk("cfg_emb_custom"),
            )
            st.session_state.cfg_emb_custom = custom_emb

    emb_id = _effective("cfg_emb_sel", "cfg_emb_custom")
    st.caption(f"Selected embedding model: **{emb_id}**")
    render_hf_model_card(emb_id, key="emb", kind="Embedding")

    st.markdown("### LLM")
    provider_options = ["huggingface", "ollama"]
    provider_current = st.session_state.cfg_llm_provider
    _p_idx = provider_options.index(provider_current) if provider_current in provider_options else None
    chosen_provider = st.radio(
        "Provider",
        provider_options,
        index=_p_idx,
        horizontal=True,
        help="huggingface = local transformers pipeline (no server needed) · ollama = local Ollama server",
        key=_wk("cfg_llm_provider"),
    )
    st.session_state.cfg_llm_provider = chosen_provider
    if chosen_provider:
        st.caption(f"Selected LLM provider: **{chosen_provider}**")
    else:
        st.caption("_Pick a provider to continue._")

    if chosen_provider == "ollama":
        l1, l2 = st.columns([3, 2])
        with l1:
            ollama_current = st.session_state.cfg_ollama_sel
            _o_idx = OLLAMA_LLM_PRESETS.index(ollama_current) if ollama_current in OLLAMA_LLM_PRESETS else None
            chosen_ollama = st.selectbox(
                "Model preset",
                OLLAMA_LLM_PRESETS,
                index=_o_idx,
                placeholder="Choose an Ollama model…",
                key=_wk("cfg_ollama_sel"),
            )
            st.session_state.cfg_ollama_sel = chosen_ollama
        with l2:
            if chosen_ollama == "Custom…":
                custom = st.text_input(
                    "Custom Ollama model",
                    value=st.session_state.cfg_ollama_custom,
                    key=_wk("cfg_ollama_custom"),
                )
                st.session_state.cfg_ollama_custom = custom
        eff = _effective("cfg_ollama_sel", "cfg_ollama_custom") if chosen_ollama else ""
        if eff:
            st.caption(f"Selected Ollama model: **{eff}**")
            banner("info", f"Ollama models are pulled via the Ollama CLI. Run <code>ollama pull {eff}</code> in your terminal.")
        else:
            st.caption("_Pick a model to continue._")
    elif chosen_provider == "huggingface":
        l1, l2 = st.columns([3, 2])
        with l1:
            hf_current = st.session_state.cfg_hf_sel
            _h_idx = HF_LLM_PRESETS.index(hf_current) if hf_current in HF_LLM_PRESETS else None
            chosen_hf = st.selectbox(
                "Model preset",
                HF_LLM_PRESETS,
                index=_h_idx,
                placeholder="Choose a HuggingFace model…",
                key=_wk("cfg_hf_sel"),
            )
            st.session_state.cfg_hf_sel = chosen_hf
        with l2:
            if chosen_hf == "Custom…":
                custom_hf = st.text_input(
                    "Custom HF model ID",
                    value=st.session_state.cfg_hf_custom,
                    key=_wk("cfg_hf_custom"),
                )
                st.session_state.cfg_hf_custom = custom_hf

        # HuggingFace access token — required for gated repos (Llama 3,
        # Gemma, Mistral Instruct, etc.). Stored in session_state only,
        # never persisted to disk.
        hf_token = st.text_input(
            "HuggingFace access token (optional)",
            value=st.session_state.cfg_hf_token,
            type="password",
            placeholder="hf_...",
            help=(
                "Required only for gated models (Llama, Gemma, Mistral, etc.). "
                "Create one at https://huggingface.co/settings/tokens — a "
                "read-only token is enough. Stored only in this browser session."
            ),
            key=_wk("cfg_hf_token"),
        )
        st.session_state.cfg_hf_token = (hf_token or "").strip()

        hf_id = _effective("cfg_hf_sel", "cfg_hf_custom") if chosen_hf else ""
        if hf_id:
            st.caption(f"Selected HF model: **{hf_id}**")
            render_hf_model_card(hf_id, key="hf_llm", kind="LLM")
        else:
            st.caption("_Pick a model to continue._")

    # ─── Graph-extraction LLM (only when Neo4j is in the backend) ─────
    if st.session_state.cfg_backend in ("neo4j", "both"):
        st.markdown("")
        st.markdown("### Graph-extraction LLM")
        st.markdown(
            '<span class="muted">Neo4j\'s knowledge graph is built by calling an LLM on every chunk '
            'to extract entities and relationships. A fast cloud API (Anthropic/OpenAI/Gemini) '
            'is strongly recommended — local models take minutes per document.</span>',
            unsafe_allow_html=True,
        )

        GRAPH_PROVIDERS = ["none", "anthropic", "openai", "gemini", "openrouter"]
        GRAPH_MODEL_PRESETS: dict[str, list[str]] = {
            "anthropic": [
                "claude-haiku-4-5-20251001",
                "claude-sonnet-4-6",
                "claude-opus-4-7",
                "Custom…",
            ],
            "openai": [
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4.1-mini",
                "Custom…",
            ],
            "gemini": [
                "gemini-2.0-flash",
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "Custom…",
            ],
            "openrouter": [
                "anthropic/claude-haiku-4-5",
                "openai/gpt-4o-mini",
                "google/gemini-2.0-flash",
                "meta-llama/llama-3.3-70b-instruct",
                "Custom…",
            ],
        }
        PROVIDER_HELP = {
            "none": "Skip extraction. Neo4j stores only flat :Chunk nodes (disconnected graph, vector search still works).",
            "anthropic": "Claude API. Get a key at console.anthropic.com.",
            "openai": "OpenAI API. Get a key at platform.openai.com.",
            "gemini": "Google AI Studio. Get a key at aistudio.google.com/apikey.",
            "openrouter": "Unified API for many providers. Get a key at openrouter.ai.",
        }

        gp_current = st.session_state.cfg_graph_llm_provider or "none"
        if gp_current not in GRAPH_PROVIDERS:
            gp_current = "none"
        chosen_gp = st.radio(
            "Provider",
            GRAPH_PROVIDERS,
            index=GRAPH_PROVIDERS.index(gp_current),
            horizontal=True,
            key=_wk("cfg_graph_llm_provider"),
            help="Used only for building the knowledge graph. Answer generation still uses the LLM above.",
        )
        st.session_state.cfg_graph_llm_provider = chosen_gp
        st.caption(PROVIDER_HELP[chosen_gp])

        if chosen_gp != "none":
            presets = GRAPH_MODEL_PRESETS[chosen_gp]
            current_model = (st.session_state.cfg_graph_llm_model or "").strip()
            if current_model in presets:
                sel_default = current_model
            elif current_model:
                sel_default = "Custom…"
            else:
                sel_default = presets[0]
            g1, g2 = st.columns([3, 2])
            with g1:
                chosen_model = st.selectbox(
                    "Model",
                    presets,
                    index=presets.index(sel_default),
                    key=_wk(f"cfg_graph_llm_model_sel_{chosen_gp}"),
                )
            with g2:
                if chosen_model == "Custom…":
                    custom_model = st.text_input(
                        "Custom model ID",
                        value=current_model if current_model not in presets else "",
                        key=_wk(f"cfg_graph_llm_model_custom_{chosen_gp}"),
                        placeholder="provider-specific model name",
                    )
                    effective_model = (custom_model or "").strip()
                else:
                    effective_model = chosen_model
            st.session_state.cfg_graph_llm_model = effective_model

            api_key = st.text_input(
                f"{chosen_gp.title()} API key",
                value=st.session_state.cfg_graph_llm_api_key,
                type="password",
                placeholder="sk-...",
                help="Stored only in this browser session. Never written to disk.",
                key=_wk(f"cfg_graph_llm_api_key_{chosen_gp}"),
            )
            st.session_state.cfg_graph_llm_api_key = (api_key or "").strip()

            if effective_model and st.session_state.cfg_graph_llm_api_key:
                st.caption(f"Graph-extraction LLM: **{chosen_gp}:{effective_model}**")
            else:
                st.caption("_Pick a model and enter an API key to enable graph extraction._")
        else:
            # Reset model/key when provider is none so we don't send stale values through.
            st.session_state.cfg_graph_llm_model = ""
            st.session_state.cfg_graph_llm_api_key = ""

    # Cache management
    st.markdown("")
    with st.expander("🗂 Cache management — downloaded HuggingFace models", expanded=False):
        repos = list_hf_cached_repos()
        if not repos:
            st.caption("No HuggingFace models cached yet.")
        else:
            total = sum(r["size"] for r in repos)
            top = st.columns([3, 2])
            with top[0]:
                st.markdown(
                    f"<b>{len(repos)}</b> model(s) cached · <b>{human_size(total)}</b> total on disk",
                    unsafe_allow_html=True,
                )
            with top[1]:
                if st.button(
                    "Delete ALL cached models",
                    key="del_all_trigger",
                    use_container_width=True,
                ):
                    st.session_state["confirm_del_all"] = True

            if st.session_state.get("confirm_del_all"):
                banner(
                    "warning",
                    "This will delete every cached HuggingFace model on this machine. "
                    "Any model used again afterwards must be re-downloaded. This cannot be undone.",
                )
                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button(
                        "Yes, delete all",
                        type="primary",
                        key="del_all_confirm",
                        use_container_width=True,
                    ):
                        with st.spinner("Deleting cached models…"):
                            freed = delete_all_hf_cache()
                        st.session_state["confirm_del_all"] = False
                        banner("success", f"Freed <b>{human_size(freed)}</b> of disk space.")
                        st.rerun()
                with cc2:
                    if st.button("Cancel", key="del_all_cancel", use_container_width=True):
                        st.session_state["confirm_del_all"] = False
                        st.rerun()

            st.markdown("")
            st.markdown("#### Individual models")
            for r in repos:
                row = st.columns([4, 1, 1])
                with row[0]:
                    st.markdown(f"`{r['id']}` · <span class='muted'>{r['type']}</span>", unsafe_allow_html=True)
                with row[1]:
                    st.markdown(f"<span class='muted'>{human_size(r['size'])}</span>", unsafe_allow_html=True)
                with row[2]:
                    if st.button("Delete", key=f"del_{r['id']}", use_container_width=True):
                        with st.spinner(f"Deleting {r['id']}…"):
                            freed = delete_hf_repo(r["id"])
                        banner("success", f"Deleted <code>{r['id']}</code> · freed {human_size(freed)}")
                        st.rerun()


# ─── Step 3: Documents ────────────────────────────────────────────────────

def step_documents() -> None:
    st.markdown("## Step 3 — Documents")
    st.markdown(
        '<span class="muted">Upload your documents and ingest them into the configured stores.</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    files = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key=_wk("main_uploader"),
    )

    u1, u2 = st.columns([3, 2])
    with u1:
        recreate = st.checkbox(
            "Wipe existing stores before ingesting",
            value=not st.session_state.ingested_files,
            help="On first ingest this should be on; subsequent ingests should append.",
        )
    with u2:
        ingest_clicked = st.button(
            "Ingest documents",
            disabled=not files,
            type="primary",
            use_container_width=True,
            key="main_ingest_btn",
        )

    if ingest_clicked and files:
        pipe = ensure_pipeline()
        if pipe is not None:
            progress = st.progress(0.0, text="Starting…")
            total = 0
            for i, uf in enumerate(files):
                suffix = Path(uf.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.getbuffer())
                    tmp_path = tmp.name
                try:
                    progress.progress(i / len(files), text=f"Ingesting {uf.name}…")
                    n = pipe.ingest(tmp_path, recreate=(recreate and i == 0))
                    total += n
                    st.session_state.ingested_files.append({"name": uf.name, "chunks": n})
                    for w in getattr(pipe, "last_ingest_warnings", []) or []:
                        banner("warning", w)
                except Exception as e:
                    banner("error", f"Failed on <code>{uf.name}</code>: {e}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                progress.progress((i + 1) / len(files), text=f"Done {i + 1}/{len(files)}")
            progress.empty()
            banner("success", f"Ingested {total} chunk(s) from {len(files)} file(s).")

    # Library summary
    if st.session_state.ingested_files:
        st.markdown("### Ingested so far")
        lib = st.session_state.ingested_files
        total_chunks = sum(x["chunks"] for x in lib)
        m1, m2, m3 = st.columns(3)
        m1.metric("Files", len(lib))
        m2.metric("Total chunks", total_chunks)
        m3.metric("Avg / file", f"{total_chunks / len(lib):.1f}")
        for item in reversed(lib[-10:]):
            st.markdown(f"• `{item['name']}` — **{item['chunks']}** chunks")


# ─── Step 4: Chat ─────────────────────────────────────────────────────────

def step_chat() -> None:
    h1, h2 = st.columns([4, 1])
    with h1:
        st.markdown("## Step 4 — Chat")
        st.markdown(
            '<span class="muted">Ask questions grounded in your ingested documents.</span>',
            unsafe_allow_html=True,
        )
    with h2:
        if st.session_state.chat_history and st.button(
            "Clear chat", use_container_width=True, key="clear_chat_btn"
        ):
            st.session_state.chat_history = []
            st.rerun()

    # Show exactly what the next query will use — prevents confusion between
    # UI selections and cached pipeline state.
    pipe_active = st.session_state.pipeline
    cfg = current_config()
    missing = _missing_config_fields(cfg)
    ui_llm = _llm_chip(cfg)
    s1, s2 = st.columns([3, 1])
    with s1:
        if missing:
            banner("warning", "Missing selection: " + ", ".join(missing) + ". Go back and complete them before chatting.")
        elif pipe_active is None:
            banner("info", f"Will build pipeline on send → <b>LLM:</b> <code>{ui_llm}</code>")
        else:
            ps = pipe_active.settings
            active_llm = (
                f"{ps.llm_provider}:{(ps.ollama_model if ps.llm_provider == 'ollama' else ps.hf_model).split('/')[-1]}"
            )
            if active_llm == ui_llm:
                banner("success", f"<b>Active LLM:</b> <code>{active_llm}</code>")
            else:
                banner(
                    "warning",
                    f"Config changed — will rebuild on send. Active now: <code>{active_llm}</code> → next: <code>{ui_llm}</code>",
                )
    with s2:
        if st.button("⟲ Force rebuild", use_container_width=True, key="force_rebuild"):
            try:
                if pipe_active is not None:
                    pipe_active._release_chroma()
            except Exception:
                pass
            st.session_state.pipeline = None
            st.session_state.pipeline_signature = None
            import gc
            gc.collect()
            st.rerun()

    if not st.session_state.ingested_files:
        banner("warning", "You haven't ingested any documents — answers will have no context. Go back to <b>Step 3</b>.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_assistant_message(
                    msg["content"],
                    mode=msg.get("mode"),
                    top_score=msg.get("top_score"),
                )
            else:
                st.markdown(msg["content"])
            if msg.get("retrieved"):
                _render_retrieved(msg["retrieved"])

    prompt = st.chat_input("Ask a question about your documents…")
    if prompt:
        pipe = ensure_pipeline()
        if pipe is not None:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Retrieving + generating…"):
                    try:
                        out = pipe.query(prompt)
                    except Exception as e:
                        banner("error", f"Query failed: {e}")
                        out = None
                if out is not None:
                    mode = out.get("mode")
                    top_score = out.get("top_score")
                    _render_assistant_message(out["answer"], mode=mode, top_score=top_score)
                    retrieved = out.get("retrieved", [])
                    _render_retrieved(retrieved)
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": out["answer"],
                            "retrieved": retrieved,
                            "mode": mode,
                            "top_score": top_score,
                        }
                    )


# ─── Render current step ──────────────────────────────────────────────────

render_stepper(st.session_state.step)

step = st.session_state.step
if step == 1:
    step_retrieval()
    render_nav(next_disabled=not st.session_state.cfg_backend)
elif step == 2:
    step_models()
    render_nav(next_disabled=bool(_missing_config_fields(current_config())))
elif step == 3:
    step_documents()
    render_nav(
        next_label="Next → Chat",
        next_disabled=not st.session_state.ingested_files,
    )
elif step == 4:
    step_chat()
    # No next button on final step; just back
    st.markdown('<hr>', unsafe_allow_html=True)
    bl, _, br = st.columns([1, 3, 1])
    with bl:
        if st.button("← Back", use_container_width=True, key="back_final"):
            st.session_state.step -= 1
            st.rerun()
    with br:
        if st.button("⟲ Reset everything", use_container_width=True, key="restart"):
            _hard_reset_session()
            st.rerun()
