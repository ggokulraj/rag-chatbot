import pytest
from llama_index.core.embeddings import MockEmbedding
import config
from ingestion import get_chroma_collection, ingest_files, build_index

MOCK_EMBED = MockEmbedding(embed_dim=384)


def test_get_chroma_collection_creates_collection():
    """get_chroma_collection returns a usable ChromaDB collection."""
    col = get_chroma_collection()
    assert col is not None
    assert col.name == "test_collection"


def test_ingest_txt_file_adds_documents(tmp_path):
    """ingest_files ingests a plain text file and stores vectors in ChromaDB."""
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("The sky is blue. Roses are red. Python is great.")

    index = ingest_files([str(txt_file)], embed_model=MOCK_EMBED)

    assert index is not None
    assert get_chroma_collection().count() > 0


def test_build_index_returns_index_after_ingestion(tmp_path):
    """build_index returns a VectorStoreIndex from an existing ChromaDB store."""
    txt_file = tmp_path / "doc.txt"
    txt_file.write_text("FastAPI is a modern web framework for Python.")

    ingest_files([str(txt_file)], embed_model=MOCK_EMBED)
    index = build_index(embed_model=MOCK_EMBED)

    assert index is not None
