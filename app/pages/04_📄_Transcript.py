"""Transcript Explorer — browse parsed chunks with sentiment and risk highlighting."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import apply_theme, fmt_risk_categories, highlight_transcript_chunk, require_upload, is_multi_mode

apply_theme()

st.header("Transcript Explorer")

require_upload("the transcript")

# ── Multi-quarter mode: quarter selector tabs ──
if is_multi_mode():
    ticker = st.session_state.get("multi_ticker", "")
    quarter_order = st.session_state.get("multi_quarter_order", [])
    multi_quarters = st.session_state.get("multi_quarters", {})

    st.caption(f"{ticker} | {len(quarter_order)} quarters")

    selected_q = st.selectbox("Select Quarter", quarter_order,
                              index=len(quarter_order) - 1, key="transcript_quarter_select")
    qdata = multi_quarters.get(selected_q, {})
    analyzed_df = qdata.get("analyzed_df")
    if analyzed_df is None:
        st.warning(f"No data for {selected_q}")
        st.stop()
    quarter = selected_q
else:
    analyzed_df = st.session_state.upload_analyzed_df
    ticker = st.session_state.get("_analyzed_ticker", "")
    quarter = st.session_state.get("_analyzed_quarter", "")

st.caption(f"{ticker} {quarter} | {len(analyzed_df)} chunks")

# ── Filters ──
col1, col2, col3 = st.columns(3)

with col1:
    search = st.text_input("Search text", placeholder="Search within transcript...")

with col2:
    sections = ["All"] + sorted(analyzed_df["section"].unique().tolist())
    section_filter = st.selectbox("Section", sections)

with col3:
    speakers = ["All"] + sorted(analyzed_df["speaker"].unique().tolist())
    speaker_filter = st.selectbox("Speaker", speakers)

display_df = analyzed_df.copy()

if search:
    display_df = display_df[display_df["text"].str.contains(search, case=False, na=False)]
if section_filter != "All":
    display_df = display_df[display_df["section"] == section_filter]
if speaker_filter != "All":
    display_df = display_df[display_df["speaker"] == speaker_filter]

st.markdown(f"Showing **{len(display_df)}** of {len(analyzed_df)} chunks")

# Legend
st.markdown(
    '<span style="background:rgba(52,211,153,0.15);padding:2px 6px;border-radius:4px">'
    'green bg = positive sentence (FinBERT)</span> &nbsp; '
    '<span style="background:rgba(239,68,68,0.15);padding:2px 6px;border-radius:4px">'
    'red bg = negative sentence (FinBERT)</span> &nbsp; '
    '<span style="color:#ef4444;font-weight:700;text-decoration:underline">underline</span>'
    ' = risk keyword',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Display chunks ──
for _, row in display_df.iterrows():
    score = row["lm_net_score"]
    if score > 0.01:
        color = "green"
    elif score < -0.01:
        color = "red"
    else:
        color = "gray"

    risk_tag = ""
    if row["risk_count"] > 0:
        risk_tag = f" | Risk: {fmt_risk_categories(row['risk_categories'])}"

    section_label = "Prepared Remarks" if row["section"] == "prepared_remarks" else "Q&A"

    color_hex = {"green": "#34d399", "red": "#ef4444", "gray": "#94a3b8"}[color]
    border_color = {"green": "rgba(52,211,153,0.4)", "red": "rgba(239,68,68,0.4)", "gray": "rgba(148,163,184,0.2)"}[color]

    st.markdown(f"**{row['speaker']}** ({row['role']}) — _{section_label}_ | "
                f"Sentiment: :{color}[{score:.4f}]{risk_tag}")
    # Highlight sentiment words (font color) and risk keywords (background)
    risk_details = row.get("risk_details", [])
    if isinstance(risk_details, float):
        risk_details = []
    finbert_sents = row.get("finbert_sentences", [])
    if not isinstance(finbert_sents, list):
        finbert_sents = []
    highlighted_text = highlight_transcript_chunk(row["text"], risk_details, finbert_sents)
    st.markdown(
        f'<div style="border-left:3px solid {border_color};'
        f'background:#1e1e2e;border-radius:0 8px 8px 0;'
        f'padding:12px 16px;color:#c6cdd5;margin:8px 0;'
        f'font-family:Inter,-apple-system,sans-serif;font-size:14px;line-height:1.7;'
        f'white-space:normal;word-wrap:break-word;overflow-wrap:break-word">'
        f'{highlighted_text}</div>',
        unsafe_allow_html=True,
    )
    st.divider()
