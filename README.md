# RAG Chatbot

A local Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LlamaIndex, ChromaDB, and Ollama. Upload documents, ask questions, and get answers grounded in your documents — fully local, no cloud API keys required.

## Features

- 📄 Multi-format document ingestion: PDF, DOCX, TXT, Markdown, HTML, CSV
- 🧠 Multi-turn conversation with memory
- 📚 Source citations for every answer
- 🔒 Fully local — powered by Ollama (no API keys)

## Prerequisites

1. **Install Ollama:** https://ollama.com/download

2. **Pull required models:**
   ```bash
   ollama pull qwen3.5:4b
   ollama pull nomic-embed-text
   ```

3. **Python 3.11+**

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama (leave running in background)
ollama serve
```

## Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Usage

1. Upload documents via the sidebar (PDF, DOCX, TXT, MD, HTML, CSV)
2. Wait for ingestion to complete
3. Ask questions in the chat box
4. Source citations appear below each answer

> **Note:** The `data/` directory stores uploaded files. It may accumulate files if ingestion fails; delete them manually if needed.

## Configuration

Edit `config.py` to change models, chunk size, or retrieval settings:

| Setting | Default | Description |
|---|---|---|
| `OLLAMA_LLM_MODEL` | `qwen3.5:4b` | LLM for answer generation |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Token overlap between chunks |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `CHROMA_PATH` | `./chroma_db` | Vector store location |

## Tests

```bash
pytest tests/ -v
```

> **Note:** Unit tests use mock LLM and embeddings — Ollama does not need to be running.

## System Requirements

| Component | Recommended |
|---|---|
| RAM | 8GB minimum, 16GB recommended |
| GPU | Optional (CPU-only works with 3B–4B models) |
| Disk | ~3GB for models, varies for ChromaDB |
