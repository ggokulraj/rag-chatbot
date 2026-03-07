"""
Document ingestion pipeline using LlamaIndex + ChromaDB.

Requires a running Ollama instance (http://localhost:11434) with the
`nomic-embed-text` model pulled, unless an explicit `embed_model` is
injected into `ingest_files` or `build_index`.
"""

from pathlib import Path
import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
import config

__all__ = ["get_chroma_collection", "build_index", "ingest_files"]


def _get_embed_model() -> OllamaEmbedding:
    return OllamaEmbedding(model_name=config.OLLAMA_EMBED_MODEL)


def _configure_ingestion_settings(embed_model) -> None:
    """Apply LlamaIndex global settings for ingestion. Not thread-safe for concurrent calls."""
    Settings.embed_model = embed_model
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP


def get_chroma_collection() -> chromadb.Collection:
    """Return (or create) the ChromaDB collection for document storage."""
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return client.get_or_create_collection(config.COLLECTION_NAME)


def build_index(embed_model=None) -> VectorStoreIndex:
    """Load an existing index from the persisted ChromaDB store."""
    if embed_model is None:
        embed_model = _get_embed_model()

    # chunk_size/overlap are ingestion-time only; embed_model is required
    # at query-time for similarity search against stored vectors
    Settings.embed_model = embed_model

    collection = get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )


def ingest_files(file_paths: list[str], embed_model=None) -> VectorStoreIndex:
    """Ingest new files into the ChromaDB vector store and return the index."""
    if not file_paths:
        raise ValueError("file_paths must not be empty")
    missing = [p for p in file_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Files not found: {missing}")

    if embed_model is None:
        embed_model = _get_embed_model()

    _configure_ingestion_settings(embed_model)

    documents = SimpleDirectoryReader(input_files=file_paths).load_data()

    collection = get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
    )
