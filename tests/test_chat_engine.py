import pytest
import config


def test_create_chat_engine_returns_engine(tmp_path, monkeypatch, mock_embed):
    """create_chat_engine returns a CondensePlusContextChatEngine when an index exists."""
    monkeypatch.setattr(config, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "COLLECTION_NAME", "test_chat_engine")

    txt_file = tmp_path / "doc.txt"
    txt_file.write_text("LlamaIndex is a data framework for LLM applications.")

    from ingestion import ingest_files
    index = ingest_files([str(txt_file)], embed_model=mock_embed)

    from llama_index.core.llms import MockLLM
    from llama_index.core.chat_engine import CondensePlusContextChatEngine
    from chat_engine import create_chat_engine
    engine = create_chat_engine(index, llm=MockLLM())
    assert isinstance(engine, CondensePlusContextChatEngine)


def test_chat_engine_returns_response(tmp_path, monkeypatch, mock_embed):
    """chat engine produces a non-empty response string."""
    monkeypatch.setattr(config, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "COLLECTION_NAME", "test_chat_response")

    txt_file = tmp_path / "facts.txt"
    txt_file.write_text(
        "The Eiffel Tower is located in Paris, France. "
        "It was built in 1889 by Gustave Eiffel."
    )

    from ingestion import ingest_files
    index = ingest_files([str(txt_file)], embed_model=mock_embed)

    from llama_index.core.llms import MockLLM
    from chat_engine import create_chat_engine
    engine = create_chat_engine(index, llm=MockLLM())
    response = engine.chat("Where is the Eiffel Tower?")

    assert response is not None
    assert len(response.response) > 0
