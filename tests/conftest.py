import pytest
import config


@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    """Redirect all config paths to temp dirs for test isolation."""
    monkeypatch.setattr(config, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setattr(config, "COLLECTION_NAME", "test_collection")
    # data/ must pre-exist as the SimpleDirectoryReader scan root; chroma/ is auto-created by ChromaDB
    (tmp_path / "data").mkdir()
