# RAG OpenLLMs

Retrieval-Augmented Generation framework using open-source Large Language Models.

Built as a master's project for CPSC 597 at California State University, Fullerton.

## Features

- **Dual backend support** — Chroma (vector DB) + Neo4j (graph DB), usable individually or simultaneously via hybrid retrieval
- **Multi-format ingestion** — PDF and DOCX documents with metadata preservation
- **Multiple chunking strategies** — fixed-size or embedding-based semantic (via `SemanticChunker`)
- **Hybrid retrieval** — queries both backends, deduplicates overlapping chunks, returns top-k results
- **Source attribution** — LLM answers cite chunk numbers for traceability
- **Open-source LLMs only** — local inference via Ollama (Llama 3, Mistral, etc.) or HuggingFace Transformers (Qwen, Gemma, Falcon, etc.)
- **GPU auto-detection** — uses CUDA for embeddings when available, falls back to CPU
- **Evaluation framework** — retrieval metrics (Precision@k, Recall@k, MRR) and generation metrics (ROUGE-L, BERTScore)

## Architecture

```
Document (PDF/DOCX)
    │
    ▼
Document Loader ──► Text Chunking ───────────► Embedding Generation
                    (fixed/SemanticChunker)    (sentence-transformers)
                                            │
                        ┌───────────────────┼───────────────────┐
                        ▼                                       ▼
                   Chroma (vector DB)                   Neo4j (graph DB)
                        │                                       │
                        └───────────────┬───────────────────────┘
                                        ▼
                              Hybrid Retrieval
                            (deduplicate + top-k)
                                        │
                                        ▼
                                LLM Generation
                         (Ollama / HuggingFace — local)
                                        │
                                        ▼
                              Answer with citations
```

## Quickstart

### 1. Configure environment

```bash
cd rag-openllms
# Edit .env with your settings (see Configuration below)
```

For `RAG_BACKEND=neo4j` or `RAG_BACKEND=both`, `NEO4J_PASSWORD` must be set.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start services

- **Ollama** — if using `LLM_PROVIDER=ollama`, ensure it's running with your model pulled (e.g. `ollama pull llama3`)
- **HuggingFace** — if using `LLM_PROVIDER=huggingface`, models are downloaded automatically on first run
- **Neo4j** — ensure it's running if using `neo4j` or `both` backend

## Usage

### Streamlit UI (recommended)

A 4-step wizard: **Configuration → Models → Documents → Chat**. Select backends and chunking, pick and download embedding + LLM models (with a live progress bar), ingest PDFs/DOCX, then chat with citations.

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The pipeline auto-initializes when you first ingest or send a message — no separate init step.

### CLI

### Chroma (vector) backend

```bash
# Ingest a PDF
python -m rag_brain --backend vector --ingest "document.pdf"

# Ingest a DOCX
python -m rag_brain --backend vector --ingest "report.docx"

# Ingest with semantic chunking
python -m rag_brain --backend vector --ingest "document.pdf" --chunking semantic

# Append to existing collection instead of rebuilding
python -m rag_brain --backend vector --ingest "document.pdf" --no-recreate

# Query
python -m rag_brain --backend vector --query "What are the key findings?"

# Query with retrieved chunks visible
python -m rag_brain --backend vector --query "What are the key findings?" --show-chunks
```

### Neo4j (graph) backend

```bash
# Ingest a PDF
python -m rag_brain --backend neo4j --ingest "document.pdf"

# Ingest a DOCX
python -m rag_brain --backend neo4j --ingest "report.docx"

# Ingest with semantic chunking
python -m rag_brain --backend neo4j --ingest "document.pdf" --chunking semantic

# Append to existing index instead of rebuilding
python -m rag_brain --backend neo4j --ingest "document.pdf" --no-recreate

# Query
python -m rag_brain --backend neo4j --query "What are the key findings?"

# Query with retrieved chunks visible
python -m rag_brain --backend neo4j --query "What are the key findings?" --show-chunks
```

### Hybrid (both backends)

```bash
# Ingest into Chroma + Neo4j simultaneously
python -m rag_brain --backend both --ingest "document.pdf"

# Ingest with semantic chunking
python -m rag_brain --backend both --ingest "report.docx" --chunking semantic

# Append to both stores
python -m rag_brain --backend both --ingest "document.pdf" --no-recreate

# Query (retrieves from both, deduplicates, returns top-k)
python -m rag_brain --backend both --query "What are the key findings?"

# Query with retrieved chunks visible
python -m rag_brain --backend both --query "What are the key findings?" --show-chunks
```

### Explore the knowledge graph

After ingesting with `neo4j` or `both`, browse the extracted entity-relation graph in **Neo4j Browser**:

🔗 **[http://localhost:7474](http://localhost:7474)** (default Neo4j HTTP port — same machine as the bolt URL in `.env`)

Log in with your `NEO4J_USERNAME` / `NEO4J_PASSWORD`, then paste this into the query bar to see the whole graph with all connections:

```cypher
MATCH (n)-[r]-(m) RETURN n, r, m
```

Drag nodes to rearrange, double-click to expand neighbors, scroll to zoom. For larger graphs add `LIMIT 500` at the end to keep the renderer snappy.

### Evaluate

Use the evaluation module programmatically:

```python
from rag_brain.evaluation import evaluate

result = evaluate(
    question="What is RAG?",
    prediction="RAG combines retrieval with generation...",
    reference="RAG is a technique that augments LLM generation with retrieved documents.",
    retrieved_sources=["paper.pdf", "notes.pdf"],
    relevant_sources=["paper.pdf"],
)
print(result.retrieval)   # Precision@k, Recall@k, MRR
print(result.generation)  # ROUGE-L, BERTScore
```

## CLI Reference

```
python -m rag_brain [OPTIONS]

Options:
  --backend {vector,neo4j,both}   Retrieval backend (default: both)
  --ingest PATH                   Path to PDF or DOCX file to ingest
  --query TEXT                    Question to ask
  --chunking {fixed,semantic}     Chunking strategy (default: fixed)
  --no-recreate                   Add to existing collection instead of rebuilding
  --show-chunks                   Print retrieved chunks as JSON after the answer
```

## Configuration

All settings are read from `.env` in the project root:

| Variable | Default | Description |
|---|---|---|
| `RAG_BACKEND` | `both` | `vector`, `neo4j`, or `both` |
| `CHUNKING_STRATEGY` | `fixed` | `fixed` or `semantic` |
| `CHUNK_SIZE` | `1200` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `4` | Number of chunks to retrieve |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Chroma storage path |
| `CHROMA_COLLECTION` | `rag_docs` | Chroma collection name |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | _(required)_ | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `NEO4J_VECTOR_INDEX` | `rag_chunk_vectors` | Neo4j vector index name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `LLM_PROVIDER` | `ollama` | `ollama` or `huggingface` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `HF_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model ID |

## Project Structure

```
rag-openllms/
├── .env                        # Configuration
├── .streamlit/config.toml      # Streamlit theme
├── app.py                      # Streamlit UI (4-step wizard)
├── requirements.txt            # Dependencies
├── rag_brain/
│   ├── __init__.py             # Package exports + HF noise suppression
│   ├── __main__.py             # CLI entry point
│   ├── config.py               # Settings (Pydantic) + enums
│   ├── ingestion.py            # PDF/DOCX loading + chunking strategies
│   ├── embeddings.py           # Embedding model init (GPU auto-detect)
│   ├── pipeline.py             # RAGPipeline: ingest, retrieve, query
│   └── evaluation.py           # Retrieval + generation metrics
```

## Tech Stack

- **LangChain + LangChain Experimental** — orchestration framework + `SemanticChunker`
- **ChromaDB** — local vector database
- **Neo4j** — graph database with vector index
- **Sentence Transformers** — embedding models (MiniLM, MPNet, BGE)
- **Ollama** — local LLM inference (Llama 3, Mistral, etc.)
- **HuggingFace Transformers** — direct model loading (Qwen, Gemma, Falcon, etc.) via `ChatHuggingFace`
- **PyPDF / python-docx** — document processing
- **ROUGE-score / BERTScore** — evaluation metrics
