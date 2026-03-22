"""Transcript Explorer — browse parsed chunks with sentiment and risk highlighting."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import apply_theme, fmt_risk_categories

apply_theme()

st.header("Transcript Explorer")

# ── Gate: require uploaded data ──
if st.session_state.get("upload_analyzed_df") is None:
    st.info("Upload an earnings call transcript on the **Home** page to browse the transcript.")
    st.stop()

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

    st.markdown(f"**{row['speaker']}** ({row['role']}) — _{section_label}_ | "
                f"Sentiment: :{color}[{score:.4f}]{risk_tag}")
    st.markdown(f"> {row['text']}")
    st.divider()
