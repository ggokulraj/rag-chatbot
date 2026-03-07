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
