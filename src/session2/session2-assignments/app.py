import streamlit as st
import rag_engine

st.set_page_config(page_title="Multi-PDF Advanced RAG Assistant", layout="wide")

# Sidebar: File Management & Document Routing
with st.sidebar:
    st.header("📂 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Index Uploaded Files", use_container_width=True):
            with st.spinner("Processing, embedding, and indexing chunks..."):
                for uploaded_file in uploaded_files:
                    bytes_data = uploaded_file.read()
                    chunk_count = rag_engine.ingest_document(bytes_data, uploaded_file.name)
                    st.success(f"Indexed '{uploaded_file.name}' ({chunk_count} chunks)")

    # Inspect current database state
    collection = rag_engine.get_chroma_collection()
    count = collection.count()
    st.metric("Total Indexed Chunks in DB", count)

    # Fetch unique document names from indexed chunks
    indexed_sources = set()
    if count > 0:
        all_data = collection.get(include=["metadatas"])
        for meta in all_data.get("metadatas", []):
            if meta and "source" in meta:
                indexed_sources.add(meta["source"])

    st.subheader("🎯 Retrieval Routing")
    source_options = ["All"] + sorted(list(indexed_sources))
    selected_source = st.selectbox(
        "Filter queries by document:",
        options=source_options,
        help="Select 'All' to query across all PDFs, or isolate a single topic."
    )

# Main UI: Chat & Generation
st.title("📄 Multi-PDF Advanced RAG System")
st.caption("Engineered with OpenRouter Embeddings, ChromaDB Metadata Filtering, and Ollama LLM.")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"Sources used: {', '.join(msg['sources'])}")

# User query input
user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching vectors and generating answer..."):
            try:
                answer, sources = rag_engine.query_rag(
                    question=user_question,
                    selected_source=selected_source
                )
                st.markdown(answer)
                if sources:
                    st.caption(f"Sources used: {', '.join(sources)}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"Error during query execution: {e}")