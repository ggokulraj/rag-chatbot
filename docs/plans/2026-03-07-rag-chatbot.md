# RAG Chatbot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a local Streamlit RAG chatbot that ingests mixed document types, answers questions with source citations, and supports multi-turn conversation — all via Ollama with no cloud API keys.

**Architecture:** Users upload files through Streamlit; LlamaIndex ingests them into a disk-persisted ChromaDB vector store using Ollama embeddings. A `CondensePlusContextChatEngine` condenses chat history before retrieval so follow-up questions resolve correctly, then streams answers with source filenames cited below each response.

**Tech Stack:** Python 3.12, Streamlit, LlamaIndex (llama-index-core + plugins), ChromaDB, Ollama (`qwen3.5:4b` LLM + `nomic-embed-text` embeddings), pytest (unit tests for non-UI modules)

---

## Prerequisites (run before starting)

```bash
# Verify Ollama is installed and running
ollama serve &
ollama pull qwen3.5:4b
ollama pull nomic-embed-text
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `config.py`
- Create: `.gitignore`
- Create: `data/.gitkeep`
- Create: `chroma_db/.gitkeep`

**Step 1: Create `.gitignore`**

```
chroma_db/
data/
__pycache__/
*.pyc
.env
*.egg-info/
dist/
.venv/
```

**Step 2: Create `requirements.txt`**

```
streamlit>=1.35.0
llama-index-core>=0.10.0
llama-index-vector-stores-chroma>=0.1.0
llama-index-llms-ollama>=0.1.0
llama-index-embeddings-ollama>=0.1.0
llama-index-readers-file>=0.1.0
chromadb>=0.5.0
pypdf>=4.0.0
python-docx>=1.1.0
pytest>=8.0.0
```

**Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: All packages install without error. If `llama-index-readers-file` fails, install `pypdf` and `python-docx` separately — they're the file parsers.

**Step 4: Create `config.py`**

```python
OLLAMA_LLM_MODEL = "qwen3.5:4b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
CHROMA_PATH = "./chroma_db"
DATA_PATH = "./data"
COLLECTION_NAME = "rag_docs"
```

**Step 5: Create placeholder directories**

```bash
mkdir -p data chroma_db
# Windows:
New-Item -ItemType Directory -Force data, chroma_db
```

**Step 6: Commit**

```bash
git add requirements.txt config.py .gitignore data chroma_db
git commit -m "feat: project scaffolding and config

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Ingestion Pipeline

**Files:**
- Create: `ingestion.py`
- Create: `tests/test_ingestion.py`

**Step 1: Create `tests/` directory and write failing tests**

Create `tests/__init__.py` (empty) and `tests/test_ingestion.py`:

```python
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
```

**Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_ingestion.py -v
```

Expected: `ImportError: cannot import name 'get_chroma_collection' from 'ingestion'` (module doesn't exist yet).

**Step 3: Create `ingestion.py`**

```python
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
```

**Step 4: Run tests — verify they pass**

```bash
pytest tests/test_ingestion.py -v
```

Expected output:
```
tests/test_ingestion.py::test_get_chroma_collection_creates_collection PASSED
tests/test_ingestion.py::test_ingest_txt_file_adds_documents PASSED
tests/test_ingestion.py::test_build_index_returns_index_after_ingestion PASSED
3 passed
```

Note: These tests call Ollama for real embeddings. Make sure `ollama serve` is running and `nomic-embed-text` is pulled. If Ollama is not available in CI, mock `OllamaEmbedding` — but for local dev, the real call is fine.

**Step 5: Commit**

```bash
git add ingestion.py tests/
git commit -m "feat: add document ingestion pipeline with ChromaDB

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Chat Engine

**Files:**
- Create: `chat_engine.py`
- Create: `tests/test_chat_engine.py`

**Step 1: Write failing tests**

Create `tests/test_chat_engine.py`:

```python
import pytest
import config


@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "COLLECTION_NAME", "test_chat_col")
    (tmp_path / "data").mkdir(exist_ok=True)


def test_create_chat_engine_returns_engine(tmp_path, monkeypatch):
    """create_chat_engine returns a chat engine when an index exists."""
    monkeypatch.setattr(config, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "COLLECTION_NAME", "test_chat_engine")

    txt_file = tmp_path / "doc.txt"
    txt_file.write_text("LlamaIndex is a data framework for LLM applications.")

    from ingestion import ingest_files
    index = ingest_files([str(txt_file)])

    from chat_engine import create_chat_engine
    engine = create_chat_engine(index)
    assert engine is not None


def test_chat_engine_returns_response(tmp_path, monkeypatch):
    """chat engine produces a non-empty response string."""
    monkeypatch.setattr(config, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "COLLECTION_NAME", "test_chat_response")

    txt_file = tmp_path / "facts.txt"
    txt_file.write_text(
        "The Eiffel Tower is located in Paris, France. "
        "It was built in 1889 by Gustave Eiffel."
    )

    from ingestion import ingest_files
    index = ingest_files([str(txt_file)])

    from chat_engine import create_chat_engine
    engine = create_chat_engine(index)
    response = engine.chat("Where is the Eiffel Tower?")

    assert response is not None
    assert len(response.response) > 0
```

**Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_chat_engine.py -v
```

Expected: `ImportError: cannot import name 'create_chat_engine' from 'chat_engine'`

**Step 3: Create `chat_engine.py`**

```python
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import config
from ingestion import build_index


def create_chat_engine(index=None) -> CondensePlusContextChatEngine:
    """
    Create a multi-turn chat engine backed by the given index.
    If no index is provided, loads from the persisted ChromaDB store.
    """
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
```

**Step 4: Run tests — verify they pass**

```bash
pytest tests/test_chat_engine.py -v
```

Expected:
```
tests/test_chat_engine.py::test_create_chat_engine_returns_engine PASSED
tests/test_chat_engine.py::test_chat_engine_returns_response PASSED
2 passed
```

Note: `test_chat_engine_returns_response` calls the real LLM. It may take 15–60 seconds on CPU-only. This is expected.

**Step 5: Commit**

```bash
git add chat_engine.py tests/test_chat_engine.py
git commit -m "feat: add multi-turn chat engine with source retrieval

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Streamlit UI

**Files:**
- Create: `app.py`

There are no automated tests for the Streamlit UI layer. Testing is manual (see smoke test below).

**Step 1: Create `app.py`**

```python
import streamlit as st
from pathlib import Path
import config
from ingestion import ingest_files, build_index
from chat_engine import create_chat_engine

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 RAG Chatbot")

# ── Session state initialisation ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None
if "index" not in st.session_state:
    st.session_state.index = None

# ── Sidebar: file upload & ingestion ──────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Supported: PDF, DOCX, TXT, MD, HTML, CSV",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "md", "html", "csv"],
    )

    if uploaded_files:
        new_paths = []
        for f in uploaded_files:
            if f.name not in st.session_state.ingested_files:
                dest = Path(config.DATA_PATH) / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(f.read())
                new_paths.append(str(dest))
                st.session_state.ingested_files.add(f.name)

        if new_paths:
            with st.spinner(f"Ingesting {len(new_paths)} file(s) — this may take a minute…"):
                try:
                    st.session_state.index = ingest_files(new_paths)
                    st.session_state.chat_engine = create_chat_engine(
                        st.session_state.index
                    )
                    names = ", ".join(Path(p).name for p in new_paths)
                    st.success(f"✅ Ingested: {names}")
                except Exception as exc:
                    st.error(
                        f"Ingestion failed: {exc}\n\n"
                        "Make sure Ollama is running: `ollama serve`"
                    )

    if st.session_state.ingested_files:
        st.subheader("Loaded Documents")
        for name in sorted(st.session_state.ingested_files):
            st.write(f"• {name}")

    st.divider()
    st.caption(
        f"LLM: `{config.OLLAMA_LLM_MODEL}`  \n"
        f"Embeddings: `{config.OLLAMA_EMBED_MODEL}`"
    )

# ── Main chat area ─────────────────────────────────────────────────────────────
if not st.session_state.ingested_files:
    st.info("👈 Upload one or more documents in the sidebar to get started.")
else:
    # Display prior messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for src in msg["sources"]:
                        st.write(f"• {src}")

    # Accept new user input
    if prompt := st.chat_input("Ask a question about your documents…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Lazily initialise chat engine (e.g. after app restart with existing DB)
            if st.session_state.chat_engine is None:
                try:
                    index = build_index()
                    st.session_state.chat_engine = create_chat_engine(index)
                except Exception as exc:
                    st.error(
                        f"Could not load knowledge base: {exc}\n\n"
                        "Is Ollama running? Try: `ollama serve`"
                    )
                    st.stop()

            with st.spinner("Thinking…"):
                try:
                    response = st.session_state.chat_engine.chat(prompt)
                    answer = response.response
                    sources = []
                    if response.source_nodes:
                        for node in response.source_nodes:
                            fname = node.metadata.get("file_name", "Unknown source")
                            if fname not in sources:
                                sources.append(fname)

                    st.markdown(answer)
                    if sources:
                        with st.expander("📚 Sources"):
                            for src in sources:
                                st.write(f"• {src}")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as exc:
                    st.error(
                        f"Error generating response: {exc}\n\n"
                        "Make sure Ollama is running: `ollama serve`"
                    )
```

**Step 2: Run all unit tests to make sure nothing is broken**

```bash
pytest tests/ -v
```

Expected: all 5 tests pass.

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit chat UI with file upload and source citations

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Smoke Test (Manual)

**Step 1: Ensure Ollama is running with required models**

```bash
ollama serve
# In another terminal:
ollama list  # verify qwen3.5:4b and nomic-embed-text appear
```

**Step 2: Launch the app**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**Step 3: Upload a PDF and a TXT covering different topics**

- Upload any PDF (e.g. a product manual, research paper)
- Upload a `.txt` file with clearly different content (e.g. a recipe, a short story)

Wait for the spinner to finish. The sidebar should list both file names.

**Step 4: Ask a question answered by the PDF**

Type a specific question whose answer is only in the PDF. Verify:
- [ ] The answer is factually grounded in the PDF
- [ ] The sources expander shows the PDF filename

**Step 5: Ask a question answered by the TXT**

Ask a question whose answer is only in the TXT. Verify:
- [ ] The answer is grounded in the TXT
- [ ] The sources expander shows the TXT filename

**Step 6: Test multi-turn memory**

Ask a follow-up that references your previous question (e.g. "Can you expand on that?"). Verify:
- [ ] The answer makes sense in the context of the conversation (not a cold retrieval)

**Step 7: Test error case — empty knowledge base**

Refresh the browser (clears `st.session_state`) and try typing a question before uploading. Verify:
- [ ] "Please upload documents first" message (or the info banner)

**Step 8: Final commit**

```bash
git add .
git commit -m "chore: verify smoke test passes

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Summary

| Task | Files Created | Tests |
|---|---|---|
| 1. Scaffolding | `requirements.txt`, `config.py`, `.gitignore` | — |
| 2. Ingestion | `ingestion.py` | `tests/test_ingestion.py` (3 tests) |
| 3. Chat Engine | `chat_engine.py` | `tests/test_chat_engine.py` (2 tests) |
| 4. Streamlit UI | `app.py` | Manual smoke test |

**Run the app:** `streamlit run app.py`  
**Run unit tests:** `pytest tests/ -v`
