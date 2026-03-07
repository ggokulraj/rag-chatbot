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


def _get_embed_model():
    return OllamaEmbedding(model_name=config.OLLAMA_EMBED_MODEL)


def get_chroma_collection():
    """Return (or create) the ChromaDB collection for document storage."""
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return client.get_or_create_collection(config.COLLECTION_NAME)


def build_index(embed_model=None) -> VectorStoreIndex:
    """Load an existing index from the persisted ChromaDB store."""
    if embed_model is None:
        embed_model = _get_embed_model()

    Settings.embed_model = embed_model
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP

    collection = get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )


def ingest_files(file_paths: list[str], embed_model=None) -> VectorStoreIndex:
    """Ingest new files into the ChromaDB vector store and return the index."""
    if embed_model is None:
        embed_model = _get_embed_model()

    Settings.embed_model = embed_model
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP

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
