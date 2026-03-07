import os
import tempfile
import pytest
import config

# Patch config paths to use temp dirs for tests
@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setattr(config, "COLLECTION_NAME", "test_collection")
    (tmp_path / "data").mkdir()


def test_get_chroma_collection_creates_collection(tmp_path):
    """get_chroma_collection returns a usable ChromaDB collection."""
    from ingestion import get_chroma_collection
    col = get_chroma_collection()
    assert col is not None
    assert col.name == "test_collection"


def test_ingest_txt_file_adds_documents(tmp_path, monkeypatch):
    """ingest_files ingests a plain text file and returns an index."""
    monkeypatch.setattr(config, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "COLLECTION_NAME", "test_ingest")

    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("The sky is blue. Roses are red. Python is great.")

    from ingestion import ingest_files
    index = ingest_files([str(txt_file)])
    assert index is not None


def test_build_index_returns_index_after_ingestion(tmp_path, monkeypatch):
    """build_index returns a VectorStoreIndex from an existing ChromaDB store."""
    monkeypatch.setattr(config, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "COLLECTION_NAME", "test_build")

    txt_file = tmp_path / "doc.txt"
    txt_file.write_text("FastAPI is a modern web framework for Python.")

    from ingestion import ingest_files, build_index
    ingest_files([str(txt_file)])
    index = build_index()
    assert index is not None
