"""Q&A — ask questions about the uploaded transcript using RAG."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import apply_theme, require_upload

apply_theme()

st.header("Ask Questions About This Transcript")

require_upload("Q&A")

from utils import is_multi_mode

# In multi-quarter mode, combine all chunks for the RAG index
if is_multi_mode():
    import pandas as pd
    multi_quarters = st.session_state.get("multi_quarters", {})
    quarter_order = st.session_state.get("multi_quarter_order", [])
    ticker = st.session_state.get("multi_ticker", "")
    quarter = ", ".join(quarter_order)
    all_chunks = []
    for q in quarter_order:
        cdf = multi_quarters[q].get("chunks_df")
        if cdf is not None:
            all_chunks.append(cdf)
    chunks_df = pd.concat(all_chunks, ignore_index=True) if all_chunks else None
else:
    chunks_df = st.session_state.get("upload_chunks_df")
    ticker = st.session_state.get("_analyzed_ticker", "")
    quarter = st.session_state.get("_analyzed_quarter", "")

st.caption(f"{ticker} {quarter} | Powered by Llama 3.2 (local via Ollama)")

# ── Sidebar filters ──
with st.sidebar:
    st.markdown("### Q&A Settings")

    top_k = st.slider("Number of sources", 3, 15, 5, key="qa_top_k")

    st.markdown("#### Filter sources by")

    # Section filter
    section_options = ["All", "Prepared Remarks", "Q&A"]
    section_filter = st.selectbox("Section", section_options, key="qa_section_filter")

    # Role filter
    role_options = ["All"]
    if chunks_df is not None and "role" in chunks_df.columns:
        roles = sorted(chunks_df["role"].dropna().unique().tolist())
        role_options += roles
    role_filter = st.selectbox("Speaker Role", role_options, key="qa_role_filter")

    # Speaker filter
    speaker_options = ["All"]
    if chunks_df is not None and "speaker" in chunks_df.columns:
        speakers = sorted(chunks_df["speaker"].dropna().unique().tolist())
        speaker_options += [s for s in speakers if s != "Unknown"]
    speaker_filter = st.selectbox("Speaker", speaker_options, key="qa_speaker_filter")

    rerank_enabled = st.checkbox("Cross-encoder re-ranking", value=True, key="qa_rerank",
                                 help="Re-rank results with a cross-encoder for better precision. Slightly slower.")

# Build metadata filters dict
filters = {}
if section_filter == "Prepared Remarks":
    filters["section"] = "prepared_remarks"
elif section_filter == "Q&A":
    filters["section"] = "qa"
if role_filter != "All":
    filters["role"] = role_filter
if speaker_filter != "All":
    filters["speaker"] = speaker_filter

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
            engine.build_index(chunks_df, use_speaker_turns=True)
            st.session_state.upload_rag_engine = RAGEngine(engine)
            st.success(f"Q&A engine ready! ({engine.index.ntotal} chunks indexed)")
        except Exception as e:
            st.warning(f"Could not initialize Q&A: {e}")

# ── Chat history ──
for msg in st.session_state.upload_qa_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], 1):
                    score_label = f"Score: {src.get('rerank_score', src.get('relevance_score', 0)):.3f}"
                    st.markdown(
                        f"**[{i}]** {src.get('speaker', 'Unknown')} "
                        f"({src.get('role', '')}) | "
                        f"{src.get('section', '').replace('_', ' ').title()} | "
                        f"{score_label}")
                    st.markdown(f"> {src.get('text', '')[:300]}...")
        if "queries_used" in msg and len(msg["queries_used"]) > 1:
            st.caption(f"Query expanded to: {', '.join(msg['queries_used'][1:])}")

# ── Suggested questions ──
if not st.session_state.upload_qa_messages:
    st.markdown("**Try one of these questions to see RAG in action:**")
    suggestions = [
        "What did management say about future revenue growth guidance?",
        "How did the CFO explain changes in operating margins?",
        "What risks or challenges were discussed during the Q&A session?",
        "Did analysts express concerns about any specific business segment?",
        "What capital allocation or buyback plans were mentioned?",
        "How does management view the competitive landscape?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(suggestions):
        if cols[i % 2].button(q, key=f"suggest_{i}", use_container_width=True):
            st.session_state.upload_qa_messages.append({"role": "user", "content": q})
            st.session_state["_pending_qa_question"] = q
            st.rerun()

# Handle pending suggestion
_pending = st.session_state.pop("_pending_qa_question", None)

# ── Chat input ──
question = _pending or st.chat_input("Ask about this earnings call...", key="qa_chat")

if question:
    # Add user message if not already added (pending suggestions are pre-added)
    if not _pending:
        st.session_state.upload_qa_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.upload_rag_engine is not None:
        # Update rerank setting
        st.session_state.upload_rag_engine.retrieval.rerank_enabled = rerank_enabled

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                try:
                    result = st.session_state.upload_rag_engine.answer(
                        question,
                        filters=filters if filters else None,
                        top_k=top_k,
                    )
                    st.markdown(result["answer"])
                    msg = {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                        "queries_used": result.get("queries_used", []),
                    }
                    st.session_state.upload_qa_messages.append(msg)

                    if result.get("sources"):
                        with st.expander(f"Sources ({len(result['sources'])})"):
                            for i, src in enumerate(result["sources"], 1):
                                score_label = f"Score: {src.get('rerank_score', src.get('relevance_score', 0)):.3f}"
                                st.markdown(
                                    f"**[{i}]** {src.get('speaker', 'Unknown')} "
                                    f"({src.get('role', '')}) | "
                                    f"{src.get('section', '').replace('_', ' ').title()} | "
                                    f"{score_label}")
                                st.markdown(f"> {src.get('text', '')[:300]}...")

                    if result.get("queries_used") and len(result["queries_used"]) > 1:
                        st.caption(f"Query expanded to: {', '.join(result['queries_used'][1:])}")

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
