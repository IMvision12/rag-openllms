from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_brain.config import Settings


def load_pdf_as_documents(path: Path | str, *, source_name: str | None = None) -> list[Document]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    loader = PyPDFLoader(str(p))
    docs = loader.load()
    src = source_name or p.name
    for d in docs:
        d.metadata.setdefault("source", src)
    return docs


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)
