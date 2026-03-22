"""Sentiment Analysis — detailed sentiment breakdown of the uploaded transcript."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import apply_theme, PLOTLY_TEMPLATE, styled_plotly, highlight_sentiment_words

apply_theme()

st.header("Sentiment Analysis")

# ── Gate: require uploaded data ──
if st.session_state.get("upload_analyzed_df") is None:
    st.info("Upload an earnings call transcript on the **Home** page to see sentiment analysis.")
    st.stop()

analyzed_df = st.session_state.upload_analyzed_df
ticker = st.session_state.get("_analyzed_ticker", "")
quarter = st.session_state.get("_analyzed_quarter", "")

st.caption(f"{ticker} {quarter} | {len(analyzed_df)} chunks")

# ── Score distribution (side by side) ──
col_lm, col_vader = st.columns(2)
with col_lm:
    fig_lm = go.Figure()
    fig_lm.add_trace(go.Histogram(x=analyzed_df["lm_net_score"], marker_color="#4285f4", nbinsx=20))
    fig_lm.update_layout(template=PLOTLY_TEMPLATE, title="LM Net Score Distribution",
                         xaxis_title="LM Net Score", yaxis_title="Count")
    st.plotly_chart(styled_plotly(fig_lm), use_container_width=True)

with col_vader:
    fig_vader = go.Figure()
    fig_vader.add_trace(go.Histogram(x=analyzed_df["vader_compound"], marker_color="#ff9900", nbinsx=20))
    fig_vader.update_layout(template=PLOTLY_TEMPLATE, title="VADER Compound Distribution",
                            xaxis_title="VADER Compound", yaxis_title="Count")
    st.plotly_chart(styled_plotly(fig_vader), use_container_width=True)

# ── By section and role ──
col1, col2 = st.columns(2)
with col1:
    section_avg = analyzed_df.groupby("section")[["lm_net_score", "vader_compound"]].mean()
    st.markdown("**By Section**")
    st.dataframe(section_avg.style.format("{:.4f}"), use_container_width=True)

with col2:
    role_avg = analyzed_df.groupby("role")[["lm_net_score", "vader_compound"]].mean()
    st.markdown("**By Speaker Role**")
    st.dataframe(role_avg.style.format("{:.4f}"), use_container_width=True)

# ── By speaker bar chart (main speakers on top, analysts/others below) ──
MAIN_ROLES = {"CEO", "CFO", "COO", "CTO", "VP", "IR", "Operator"}
speaker_data = analyzed_df.groupby(["speaker", "role"])["lm_net_score"].mean().reset_index()
speaker_data["is_main"] = speaker_data["role"].isin(MAIN_ROLES)

# Sort: main speakers first (sorted by score desc), then others (sorted by score desc)
main = speaker_data[speaker_data["is_main"]].sort_values("lm_net_score", ascending=True)
others = speaker_data[~speaker_data["is_main"]].sort_values("lm_net_score", ascending=True)

# Build ordered list: others at bottom (plotly renders bottom-up), then main on top
ordered = pd.concat([others, main])

if len(ordered) > 1:
    # Use plain speaker names on y-axis; role badges rendered separately
    labels_main = [r["speaker"] for _, r in main.iterrows()]
    labels_others = [r["speaker"] for _, r in others.iterrows()]
    all_labels = labels_others + labels_main

    ROLE_COLORS = {
        "CEO": "#ef4444", "CFO": "#22d3ee", "COO": "#a78bfa",
        "CTO": "#34d399", "VP": "#fbbf24", "IR": "#8b5cf6",
        "Operator": "#6ee7b7", "Analyst": "#ff9900", "Management": "#4285f4",
        "Other": "#64748b", "Unknown": "#64748b",
    }

    fig_speaker = go.Figure()
    # Company speakers (top)
    if len(main) > 0:
        fig_speaker.add_trace(go.Bar(
            x=main["lm_net_score"].values,
            y=labels_main,
            orientation="h",
            name="Company",
            marker=dict(color="#4285f4"),
            showlegend=True,
        ))
    # External speakers (bottom)
    if len(others) > 0:
        fig_speaker.add_trace(go.Bar(
            x=others["lm_net_score"].values,
            y=labels_others,
            orientation="h",
            name="External (Analysts)",
            marker=dict(color="#ff9900"),
            showlegend=True,
        ))
    # Add colored role annotations next to each bar
    all_rows = pd.concat([others, main])
    for i, (_, r) in enumerate(all_rows.iterrows()):
        role = r["role"]
        color = ROLE_COLORS.get(role, "#94a3b8")
        score = r["lm_net_score"]
        # Place annotation at the end of the bar
        x_pos = score + 0.002 if score >= 0 else score - 0.002
        anchor = "left" if score >= 0 else "right"
        fig_speaker.add_annotation(
            x=x_pos, y=i,
            text=f"<b>{role}</b>",
            showarrow=False,
            font=dict(color=color, size=11),
            xanchor=anchor,
            yanchor="middle",
        )
    # Add dotted divider line
    if len(main) > 0 and len(others) > 0:
        divider_pos = len(others) - 0.5
        fig_speaker.add_hline(y=divider_pos, line_dash="dot", line_color="rgba(255,255,255,0.5)",
                              line_width=2)
    fig_speaker.update_layout(template=PLOTLY_TEMPLATE, title="Sentiment by Speaker (LM Net)",
                              xaxis_title="LM Net Score", yaxis_title="Speaker",
                              yaxis=dict(categoryorder="array",
                                         categoryarray=all_labels))
    st.plotly_chart(styled_plotly(fig_speaker), use_container_width=True)

# ── Sentiment over transcript flow (side by side) ──
st.markdown("**Sentiment Flow (chunk order)**")
qa_start_idx = None
for i, section in enumerate(analyzed_df["section"].values):
    if section == "qa":
        qa_start_idx = i
        break

col_lm_flow, col_vader_flow = st.columns(2)
with col_lm_flow:
    fig_lm_flow = go.Figure()
    fig_lm_flow.add_trace(go.Scatter(
        x=list(range(len(analyzed_df))),
        y=analyzed_df["lm_net_score"].values,
        mode="lines+markers",
        marker=dict(size=4, color="#4285f4"),
        line=dict(width=1, color="#4285f4", shape="spline", smoothing=1.3),
    ))
    if qa_start_idx is not None:
        fig_lm_flow.add_vline(x=qa_start_idx, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                              annotation_text="Q&A")
    fig_lm_flow.update_layout(template=PLOTLY_TEMPLATE, title="LM Net Score Across Transcript",
                              xaxis_title="Chunk Index", yaxis_title="LM Net Score")
    st.plotly_chart(styled_plotly(fig_lm_flow), use_container_width=True)

with col_vader_flow:
    fig_vader_flow = go.Figure()
    fig_vader_flow.add_trace(go.Scatter(
        x=list(range(len(analyzed_df))),
        y=analyzed_df["vader_compound"].values,
        mode="lines+markers",
        marker=dict(size=4, color="#ff9900"),
        line=dict(width=1, color="#ff9900", shape="spline", smoothing=1.3),
    ))
    if qa_start_idx is not None:
        fig_vader_flow.add_vline(x=qa_start_idx, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                                annotation_text="Q&A")
    fig_vader_flow.update_layout(template=PLOTLY_TEMPLATE, title="VADER Compound Across Transcript",
                                xaxis_title="Chunk Index", yaxis_title="VADER Compound")
    st.plotly_chart(styled_plotly(fig_vader_flow), use_container_width=True)

# ── Most positive / negative chunks ──
st.markdown('<span style="color:#34d399">green = positive</span> &nbsp; '
            '<span style="color:#ef4444">red = negative</span> &nbsp; '
            '<span style="color:#fbbf24">yellow = uncertainty</span> &nbsp; '
            '(Loughran-McDonald lexicon)', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Most Positive Chunks**")
    top_pos = analyzed_df.nlargest(3, "lm_net_score")
    for _, row in top_pos.iterrows():
        st.markdown(f"Score: **{row['lm_net_score']:.4f}** | {row['speaker']} ({row['role']})")
        highlighted = highlight_sentiment_words(row["text"])
        st.markdown(f"> {highlighted}", unsafe_allow_html=True)
        st.divider()

with col2:
    st.markdown("**Most Negative Chunks**")
    top_neg = analyzed_df.nsmallest(3, "lm_net_score")
    for _, row in top_neg.iterrows():
        st.markdown(f"Score: **{row['lm_net_score']:.4f}** | {row['speaker']} ({row['role']})")
        highlighted = highlight_sentiment_words(row["text"])
        st.markdown(f"> {highlighted}", unsafe_allow_html=True)
        st.divider()
