"""Streamlit UI for the RAG OpenLLMs pipeline."""

from __future__ import annotations

import json
import os
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
    "llama3", "llama3.1", "llama3.2", "mistral", "phi3",
    "gemma2", "qwen2.5", "Custom…",
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


# ─── Session state ────────────────────────────────────────────────────────

st.session_state.setdefault("pipeline", None)
st.session_state.setdefault("pipeline_signature", None)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("ingested_files", [])
st.session_state.setdefault("step", 1)

# Config widget defaults (persisted via widget keys)
st.session_state.setdefault("cfg_backend", "both")
st.session_state.setdefault("cfg_chunking", "fixed")
st.session_state.setdefault("cfg_top_k", 4)
st.session_state.setdefault("cfg_chunk_size", 1200)
st.session_state.setdefault("cfg_chunk_overlap", 200)
st.session_state.setdefault("cfg_emb_sel", EMBEDDING_PRESETS[0])
st.session_state.setdefault("cfg_emb_custom", EMBEDDING_PRESETS[0])
st.session_state.setdefault("cfg_llm_provider", "ollama")
st.session_state.setdefault("cfg_ollama_sel", OLLAMA_LLM_PRESETS[0])
st.session_state.setdefault("cfg_ollama_custom", "llama3")
st.session_state.setdefault("cfg_hf_sel", HF_LLM_PRESETS[0])
st.session_state.setdefault("cfg_hf_custom", "Qwen/Qwen2.5-1.5B-Instruct")
st.session_state.setdefault("cfg_hf_quantize", "none")
st.session_state.setdefault("cfg_hf_max_tokens", 512)


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
        "hf_quantize": st.session_state.cfg_hf_quantize,
        "hf_max_new_tokens": st.session_state.cfg_hf_max_tokens,
    }


def ensure_pipeline() -> RAGPipeline | None:
    """Auto-build the pipeline if missing or out of sync with current config."""
    current_sig = tuple(sorted(current_config().items()))
    active_sig = st.session_state.pipeline_signature
    pipe = st.session_state.pipeline

    if pipe is not None and active_sig == current_sig:
        return pipe

    is_first_load = pipe is None
    spinner_msg = (
        "Loading pipeline (first run downloads models — may take a minute)…"
        if is_first_load
        else "Applying new settings…"
    )
    try:
        with st.spinner(spinner_msg):
            pipe = build_pipeline()
            st.session_state.pipeline = pipe
            st.session_state.pipeline_signature = current_sig
        return pipe
    except Exception as e:
        banner("error", f"Pipeline initialization failed: {e}")
        return None


def build_pipeline() -> RAGPipeline:
    c = current_config()
    os.environ["RAG_BACKEND"] = c["backend"]
    os.environ["CHUNKING_STRATEGY"] = c["chunking"]
    os.environ["TOP_K"] = str(c["top_k"])
    os.environ["CHUNK_SIZE"] = str(c["chunk_size"])
    os.environ["CHUNK_OVERLAP"] = str(c["chunk_overlap"])
    os.environ["EMBEDDING_MODEL"] = c["embedding_model"]
    os.environ["LLM_PROVIDER"] = c["llm_provider"]
    if c["llm_provider"] == "ollama":
        os.environ["OLLAMA_MODEL"] = c["ollama_model"]
        os.environ.pop("HF_QUANTIZE", None)
    else:
        os.environ["HF_MODEL"] = c["hf_model"]
        os.environ["HF_MAX_NEW_TOKENS"] = str(c["hf_max_new_tokens"])
        if c["hf_quantize"] != "none":
            os.environ["HF_QUANTIZE"] = c["hf_quantize"]
        else:
            os.environ.pop("HF_QUANTIZE", None)
    return RAGPipeline(load_settings())


# ─── Header ───────────────────────────────────────────────────────────────

st.markdown("# ◆ RAG OpenLLMs")
st.markdown(
    '<span class="muted">Hybrid retrieval over Chroma + Neo4j with open-source LLMs.</span>',
    unsafe_allow_html=True,
)

pipe: RAGPipeline | None = st.session_state.pipeline
if pipe is None:
    st.markdown(
        '<div class="status-bar"><span class="chip"><b>status</b>not initialized</span></div>',
        unsafe_allow_html=True,
    )
else:
    s = pipe.settings
    llm_label = s.ollama_model if s.llm_provider == "ollama" else s.hf_model
    chips = [
        ("backend", s.retrieval_backend.value),
        ("chunking", s.chunking_strategy.value),
        ("top-k", str(s.top_k)),
        ("llm", f"{s.llm_provider}:{llm_label.split('/')[-1]}"),
    ]
    html = "".join(f'<span class="chip"><b>{k}</b>{v}</span>' for k, v in chips)
    st.markdown(f'<div class="status-bar">{html}</div>', unsafe_allow_html=True)

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


class StreamlitTqdm(_BaseTqdm):
    """tqdm subclass that pipes byte-download progress into a Streamlit progress bar.

    snapshot_download creates ONE parent bar (unit="B", total=0 initially), then
    mutates `self.total` as each file's size is discovered, and calls update(n)
    as bytes arrive. We read `self.n` and `self.total` on every tick so the
    Streamlit bar always reflects current progress.
    """
    _st_placeholder = None  # class-level handle to the st.progress widget

    @classmethod
    def reset_state(cls, placeholder) -> None:
        cls._st_placeholder = placeholder

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("leave", False)
        kwargs.setdefault("mininterval", 0.05)
        unit = kwargs.get("unit") or ""
        self._sl_is_bytes = isinstance(unit, str) and unit.startswith("B")
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            safe = {k: v for k, v in kwargs.items() if k != "name"}
            super().__init__(*args, **safe)
        self._sl_render()

    def update(self, n=1):
        try:
            super().update(n)
        except Exception:
            pass
        self._sl_render()

    def refresh(self, *args, **kwargs):
        try:
            super().refresh(*args, **kwargs)
        except Exception:
            pass
        self._sl_render()

    def display(self, msg=None, pos=None):
        return True

    def _sl_render(self) -> None:
        if not getattr(self, "_sl_is_bytes", False):
            return
        ph = StreamlitTqdm._st_placeholder
        if ph is None:
            return
        total = getattr(self, "total", 0) or 0
        current = getattr(self, "n", 0) or 0
        if total > 0:
            frac = min(max(current / total, 0.0), 1.0)
            text = f"{human_size(current)} / {human_size(total)}"
        else:
            frac = 0.0
            text = f"Fetching file list…  {human_size(current)} downloaded"
        try:
            ph.progress(frac, text=text)
        except Exception:
            pass


def download_hf_model(repo_id: str) -> None:
    """Download a HuggingFace model with a live Streamlit progress bar."""
    import inspect
    import logging
    import time

    with st.status(f"Downloading {repo_id}…", expanded=True) as status:
        progress = st.progress(0.0, text="Connecting to HuggingFace Hub…")
        StreamlitTqdm.reset_state(progress)

        # HF disables its tqdm when the huggingface_hub logger is at ERROR level.
        # rag_brain/__init__.py sets it to ERROR for quietness — override here
        # so progress updates actually flow through.
        hf_logger = logging.getLogger("huggingface_hub")
        original_level = hf_logger.level
        hf_logger.setLevel(logging.INFO)

        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        try:
            from huggingface_hub.utils import enable_progress_bars
            enable_progress_bars()
        except Exception:
            pass

        try:
            from huggingface_hub import snapshot_download

            kwargs: dict = {"repo_id": repo_id}
            sig = inspect.signature(snapshot_download)
            if "tqdm_class" in sig.parameters:
                kwargs["tqdm_class"] = StreamlitTqdm
            else:
                status.write(
                    "Note: your `huggingface_hub` is too old for live progress. "
                    "Run `pip install -U huggingface_hub` to enable the bar. "
                    "Download will still proceed — check the terminal for progress."
                )

            t0 = time.time()
            path = snapshot_download(**kwargs)
            elapsed = time.time() - t0
            total = get_hf_cache_size(repo_id) or 0
            progress.progress(1.0, text=f"✓ {human_size(total)} in {elapsed:.1f}s")
            status.write(f"Local path: `{path}`")
            status.update(label=f"✓ {repo_id} ready ({human_size(total)})", state="complete")
        except Exception as e:
            status.update(label=f"Download failed: {e}", state="error")
        finally:
            hf_logger.setLevel(original_level)
            StreamlitTqdm._st_placeholder = None


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
            download_hf_model(repo_id)
            st.rerun()


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
    st.radio(
        "Retrieval backend",
        ["both", "vector", "neo4j"],
        horizontal=True,
        key="cfg_backend",
        label_visibility="collapsed",
        help="both = hybrid Chroma + Neo4j · vector = Chroma only · neo4j = Neo4jVector only",
    )
    if st.session_state.cfg_backend in ("neo4j", "both"):
        banner("info", "Neo4j credentials are read from <code>.env</code> (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD).")

    st.markdown("### Chunking")
    c1, c2 = st.columns(2)
    with c1:
        st.radio(
            "Strategy",
            ["fixed", "semantic"],
            horizontal=True,
            key="cfg_chunking",
            help="fixed = RecursiveCharacterTextSplitter · semantic = embedding-based breakpoints",
        )
    with c2:
        st.slider("Top-k retrieved chunks", 1, 20, key="cfg_top_k")

    if st.session_state.cfg_chunking == "fixed":
        c3, c4 = st.columns(2)
        with c3:
            st.number_input("Chunk size", 200, 4000, step=100, key="cfg_chunk_size")
        with c4:
            st.number_input("Chunk overlap", 0, 1000, step=50, key="cfg_chunk_overlap")


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
        st.selectbox("Preset", EMBEDDING_PRESETS, key="cfg_emb_sel")
    with e2:
        if st.session_state.cfg_emb_sel == "Custom…":
            st.text_input("Custom HF model ID", key="cfg_emb_custom")

    emb_id = _effective("cfg_emb_sel", "cfg_emb_custom")
    render_hf_model_card(emb_id, key="emb", kind="Embedding")

    st.markdown("### LLM")
    st.radio(
        "Provider",
        ["ollama", "huggingface"],
        horizontal=True,
        key="cfg_llm_provider",
        help="ollama = local inference via Ollama server · huggingface = direct transformers pipeline",
    )

    if st.session_state.cfg_llm_provider == "ollama":
        l1, l2 = st.columns([3, 2])
        with l1:
            st.selectbox("Model preset", OLLAMA_LLM_PRESETS, key="cfg_ollama_sel")
        with l2:
            if st.session_state.cfg_ollama_sel == "Custom…":
                st.text_input("Custom Ollama model", key="cfg_ollama_custom")
        eff = _effective("cfg_ollama_sel", "cfg_ollama_custom")
        banner("info", f"Ollama models are pulled via the Ollama CLI. Run <code>ollama pull {eff}</code> in your terminal.")
    else:
        l1, l2 = st.columns([3, 2])
        with l1:
            st.selectbox("Model preset", HF_LLM_PRESETS, key="cfg_hf_sel")
        with l2:
            if st.session_state.cfg_hf_sel == "Custom…":
                st.text_input("Custom HF model ID", key="cfg_hf_custom")
        q1, q2 = st.columns(2)
        with q1:
            st.selectbox(
                "Quantization",
                ["none", "4bit", "8bit"],
                key="cfg_hf_quantize",
                help="4/8-bit uses bitsandbytes — requires CUDA.",
            )
        with q2:
            st.number_input("Max new tokens", 64, 4096, step=64, key="cfg_hf_max_tokens")

        hf_id = _effective("cfg_hf_sel", "cfg_hf_custom")
        render_hf_model_card(hf_id, key="hf_llm", kind="LLM")

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
        key="main_uploader",
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

    if not st.session_state.ingested_files:
        banner("warning", "You haven't ingested any documents — answers will have no context. Go back to <b>Step 3</b>.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
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
                    st.markdown(out["answer"])
                    retrieved = out.get("retrieved", [])
                    _render_retrieved(retrieved)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": out["answer"], "retrieved": retrieved}
                    )


# ─── Render current step ──────────────────────────────────────────────────

render_stepper(st.session_state.step)

step = st.session_state.step
if step == 1:
    step_retrieval()
    render_nav()
elif step == 2:
    step_models()
    render_nav()
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
        if st.button("⟲ Start over", use_container_width=True, key="restart"):
            st.session_state.step = 1
            st.rerun()
