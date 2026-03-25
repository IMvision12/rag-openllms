# RAG OpenLLMs (RAG brain)

End-to-end “RAG brain” pipeline for **PDF question-answering**:

- Ingest PDF → chunk text → create embeddings
- Store embeddings in either **Chroma** (vector) or **Neo4jVector** (neo4j)
- Retrieve the most relevant chunks and answer with an LLM (**Ollama** or **OpenAI-compatible**)

## Quickstart

### 1) Configure environment

From the `rag-openllms/` directory:

- Copy `.env.example` to `.env`
- Set the variables you need (see `.env.example` for defaults)

Notes:

- The app reads environment variables from `./.env` (so make sure you run commands with `rag-openllms/` as your working directory).
- For `RAG_BACKEND=neo4j`, `NEO4J_PASSWORD` must be set.

### 2) Install dependencies

Use your preferred method. Example:

```bash
pip install -r requirements.txt
```

## Run

### Ingest a PDF

Vector backend (Chroma):

```bash
python -m rag_brain --backend vector --ingest "path/to/document.pdf"
```

If you want to add to an existing Chroma collection instead of rebuilding it:

```bash
python -m rag_brain --backend vector --ingest "path/to/document.pdf" --no-recreate
```

Neo4j backend:

```bash
python -m rag_brain --backend neo4j --ingest "path/to/document.pdf"
```

### Query

```bash
python -m rag_brain --backend vector --query "Your question here"
```

For neo4j:

```bash
python -m rag_brain --backend neo4j --query "Your question here"
```

The CLI prints the answer, plus a debug dump of the retrieved chunk contents.

## What to configure (most important)

From `.env`:

- `RAG_BACKEND` = `vector` or `neo4j`
- Chroma:
  - `CHROMA_PERSIST_DIR`
  - `CHROMA_COLLECTION`
- Neo4j:
  - `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
  - `NEO4J_VECTOR_INDEX` (index name for vectors)
- Embeddings:
  - `EMBEDDING_MODEL`
  - `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`
- LLM:
  - `LLM_PROVIDER` = `ollama` or `openai`
  - Ollama: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
  - OpenAI: `OPENAI_API_KEY`, `OPENAI_MODEL`, optional `OPENAI_BASE_URL`

