from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import config
from ingestion import build_index


def create_chat_engine(index=None, llm=None) -> CondensePlusContextChatEngine:
    """
    Create a multi-turn chat engine backed by the given index.
    If no index is provided, loads from the persisted ChromaDB store.
    An optional llm can be injected (e.g. MockLLM for tests); defaults to Ollama.
    """
    if llm is None:
        llm = Ollama(model=config.OLLAMA_LLM_MODEL, request_timeout=180.0)
    Settings.llm = llm

    if index is None:
        index = build_index()

    retriever = index.as_retriever(similarity_top_k=config.TOP_K)
    memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

    return CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        llm=llm,
        verbose=False,
    )
