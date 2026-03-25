# RAG OpenLLMs

Retrieval-Augmented Generation framework using open-source Large Language Models.

Built as a master's project for CPSC 597 at California State University, Fullerton.

## Features

- **Dual backend support** — Chroma (vector DB) + Neo4j (graph DB), usable individually or simultaneously via hybrid retrieval
- **Multi-format ingestion** — PDF and DOCX documents with metadata preservation
- **Multiple chunking strategies** — fixed-size or semantic (sentence-boundary aware)
- **Hybrid retrieval** — queries both backends, deduplicates overlapping chunks, returns top-k results
- **Source attribution** — LLM answers cite chunk numbers for traceability
- **Open-source LLMs only** — local inference via Ollama (Llama 3, Mistral, etc.) or HuggingFace Transformers (Qwen, Gemma, Falcon, etc.) with optional 4-bit/8-bit quantization
- **GPU auto-detection** — uses CUDA for embeddings when available, falls back to CPU
- **Evaluation framework** — retrieval metrics (Precision@k, Recall@k, MRR) and generation metrics (ROUGE-L, BERTScore)

## Architecture

```
Document (PDF/DOCX)
    │
    ▼
Document Loader ──► Text Chunking ──► Embedding Generation
                    (fixed/semantic)   (sentence-transformers)
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

### Ingest documents

Hybrid (both backends):

```bash
python -m rag_brain --backend both --ingest "path/to/document.pdf"
```

With semantic chunking:

```bash
python -m rag_brain --backend both --ingest "report.docx" --chunking semantic
```

Single backend:

```bash
python -m rag_brain --backend vector --ingest "document.pdf"
python -m rag_brain --backend neo4j --ingest "document.pdf"
```

Add to an existing Chroma collection instead of rebuilding:

```bash
python -m rag_brain --backend vector --ingest "document.pdf" --no-recreate
```

### Query

```bash
python -m rag_brain --backend both --query "What are the key findings?"
```

The CLI prints the answer with source citations, plus a JSON dump of retrieved chunks.

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
  --no-recreate                   Add to existing Chroma collection
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
| `HF_MAX_NEW_TOKENS` | `512` | Max tokens to generate |
| `HF_QUANTIZE` | _(none)_ | `4bit`, `8bit`, or empty for full precision |

## Project Structure

```
rag-openllms/
├── .env                        # Configuration
├── requirements.txt            # Dependencies
├── rag_brain/
│   ├── __init__.py             # Package exports
│   ├── __main__.py             # CLI entry point
│   ├── config.py               # Settings (Pydantic) + enums
│   ├── ingestion.py            # PDF/DOCX loading + chunking strategies
│   ├── embeddings.py           # Embedding model init (GPU auto-detect)
│   ├── pipeline.py             # RAGPipeline: ingest, retrieve, query
│   └── evaluation.py           # Retrieval + generation metrics
```

## Tech Stack

- **LangChain** — orchestration framework
- **ChromaDB** — local vector database
- **Neo4j** — graph database with vector index
- **Sentence Transformers** — embedding models (MiniLM, MPNet, BGE)
- **Ollama** — local LLM inference (Llama 3, Mistral, etc.)
- **HuggingFace Transformers** — direct model loading (Qwen, Gemma, Falcon, etc.) with 4-bit/8-bit quantization
- **Sentence Transformers** — embedding models (MiniLM, MPNet, BGE)
- **PyPDF / python-docx** — document processing
- **ROUGE-score / BERTScore** — evaluation metrics
