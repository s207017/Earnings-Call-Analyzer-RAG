"""Q&A — ask questions about the uploaded transcript using RAG."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import apply_theme

apply_theme()

st.header("Ask Questions About This Transcript")

# ── Gate: require uploaded data ──
if st.session_state.get("upload_analyzed_df") is None:
    st.info("Upload an earnings call transcript on the **Home** page to use Q&A.")
    st.stop()

chunks_df = st.session_state.get("upload_chunks_df")
ticker = st.session_state.get("_analyzed_ticker", "")
quarter = st.session_state.get("_analyzed_quarter", "")

st.caption(f"{ticker} {quarter} | Powered by Llama 3.2 (local via Ollama)")

# ── Initialize RAG engine ──
if "upload_qa_messages" not in st.session_state or st.session_state.upload_qa_messages is None:
    st.session_state.upload_qa_messages = []
if "upload_rag_engine" not in st.session_state:
    st.session_state.upload_rag_engine = None

if st.session_state.upload_rag_engine is None and chunks_df is not None and not chunks_df.empty:
    with st.spinner("Building search index for this transcript... (embedding chunks)"):
        try:
            from src.agents.retrieval import RetrievalEngine
            from src.agents.rag_qa import RAGEngine

            engine = RetrievalEngine()
            engine.build_index(chunks_df)
            st.session_state.upload_rag_engine = RAGEngine(engine)
            st.success("Q&A engine ready!")
        except Exception as e:
            st.warning(f"Could not initialize Q&A: {e}")

# ── Chat history ──
for msg in st.session_state.upload_qa_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{i}]** {src.get('speaker', 'Unknown')} ({src.get('role', '')}) | "
                                f"Score: {src.get('relevance_score', 0):.3f}")
                    st.markdown(f"> {src.get('text', '')[:200]}...")

# ── Chat input ──
if question := st.chat_input("Ask about this earnings call...", key="qa_chat"):
    st.session_state.upload_qa_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.upload_rag_engine is not None:
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                try:
                    result = st.session_state.upload_rag_engine.answer(question, top_k=5)
                    st.markdown(result["answer"])
                    msg = {"role": "assistant", "content": result["answer"],
                           "sources": result.get("sources", [])}
                    st.session_state.upload_qa_messages.append(msg)

                    if result.get("sources"):
                        with st.expander(f"Sources ({len(result['sources'])})"):
                            for i, src in enumerate(result["sources"], 1):
                                st.markdown(f"**[{i}]** {src.get('speaker', 'Unknown')} "
                                            f"({src.get('role', '')}) | "
                                            f"Score: {src.get('relevance_score', 0):.3f}")
                                st.markdown(f"> {src.get('text', '')[:200]}...")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        with st.chat_message("assistant"):
            st.markdown("Q&A engine not available. Make sure Ollama is running.")
            st.session_state.upload_qa_messages.append(
                {"role": "assistant", "content": "Q&A engine not available."})

# ── Clear chat button ──
if st.session_state.upload_qa_messages:
    if st.button("Clear Chat"):
        st.session_state.upload_qa_messages = []
        if st.session_state.upload_rag_engine:
            st.session_state.upload_rag_engine.clear_history()
        st.rerun()
