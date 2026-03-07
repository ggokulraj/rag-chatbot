# RAG Chatbot — Design Document

**Date:** 2026-03-07  
**Status:** Approved

---

## Problem Statement

Build a local Retrieval-Augmented Generation (RAG) chatbot as a Streamlit prototype. Users upload mixed document types (PDF, DOCX, TXT, Markdown, HTML, CSV), ask questions in a chat interface, and receive answers grounded in those documents with source citations. The chatbot supports multi-turn conversation within a session. Everything runs locally via Ollama — no cloud API keys required.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI                       │
│  ┌──────────────┐   ┌───────────────────────────┐  │
│  │ File Uploader│   │  Chat Interface           │  │
│  │ (drag & drop)│   │  (Q&A + source citations) │  │
│  └──────┬───────┘   └──────────┬────────────────┘  │
└─────────┼──────────────────────┼────────────────────┘
          │ ingest                │ query
          ▼                      ▼
┌─────────────────┐    ┌──────────────────────────┐
│  Ingestion      │    │  Chat Engine             │
│  Pipeline       │    │  (LlamaIndex             │
│  - Load docs    │    │   CondensePlusContext)    │
│  - Chunk        │    │  - Condenses history +    │
│  - Embed        │    │    new query              │
│  - Store        │    │  - Retrieves top-k chunks │
└──────┬──────────┘    │  - Generates answer       │
       │               └──────────┬────────────────┘
       ▼                          │
┌─────────────────┐               │
│  ChromaDB       │◄──────────────┘
│  (local, disk)  │  similarity search
└─────────────────┘
       │ embeddings/LLM via
       ▼
┌─────────────────┐
│  Ollama (local) │
│  LLM:   qwen3.5:4b       │
│  Embed: nomic-embed-text │
└─────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| RAG Framework | LlamaIndex |
| Vector Store | ChromaDB (disk-persisted) |
| LLM | Ollama — `qwen3.5:4b` |
| Embeddings | Ollama — `nomic-embed-text` |
| Language | Python 3.11+ |

**Rationale:** LlamaIndex's `SimpleDirectoryReader` handles multi-document types with zero boilerplate. `CondensePlusContextChatEngine` provides accurate multi-turn retrieval by condensing history before searching. Everything runs locally — no API keys needed.

---

## Project Structure

```
rag-chatbot/
├── app.py                  # Streamlit entry point
├── ingestion.py            # Document loading, chunking, embedding → ChromaDB
├── chat_engine.py          # LlamaIndex chat engine setup & query
├── config.py               # All tunable settings
├── docs/
│   └── plans/              # Design docs
├── data/                   # Uploaded docs stored temporarily
├── chroma_db/              # ChromaDB persisted vector store
└── requirements.txt
```

---

## Data Flow

### Ingestion
1. User uploads file(s) in Streamlit → saved to `data/`
2. `SimpleDirectoryReader` loads & parses by file extension
3. `SentenceSplitter` chunks at 512 tokens with 50-token overlap
4. `nomic-embed-text` via Ollama embeds each chunk
5. Chunks stored in ChromaDB named collection on disk

### Query
1. User types question in chat
2. `CondensePlusContextChatEngine` condenses history + question → standalone query
3. Top-5 similar chunks retrieved from ChromaDB
4. Chunks + query sent to `qwen3.5:4b` via Ollama
5. Response streamed back; source filenames shown below the answer

---

## Configuration (`config.py`)

| Setting | Default |
|---|---|
| Ollama LLM model | `qwen3.5:4b` |
| Ollama embedding model | `nomic-embed-text` |
| Chunk size (tokens) | `512` |
| Chunk overlap (tokens) | `50` |
| Top-k retrieval | `5` |
| ChromaDB path | `./chroma_db` |
| Data upload path | `./data` |

---

## Error Handling

- **Ollama not running** → clear error in Streamlit sidebar with startup instructions
- **Unsupported file type** → warn and skip; don't crash
- **Empty knowledge base** → "Please upload documents first" message on query
- **Duplicate ingestion** → track ingested filenames in `st.session_state`; skip re-embedding

---

## Testing

Manual smoke test procedure:
1. `ollama pull qwen3.5:4b && ollama pull nomic-embed-text`
2. Launch app: `streamlit run app.py`
3. Upload a PDF + a TXT file covering different topics
4. Ask questions spanning both documents
5. Verify answers are grounded and source citations appear
6. Ask a follow-up question referencing prior answer to verify multi-turn memory

No automated test suite in scope for this prototype.
