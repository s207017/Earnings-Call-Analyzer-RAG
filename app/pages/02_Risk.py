"""Risk Detection — risk signal breakdown of the uploaded transcript."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import apply_theme, PLOTLY_TEMPLATE, styled_plotly, fmt_risk_category, fmt_risk_categories

apply_theme()

st.header("Risk Signal Detection")

# ── Gate: require uploaded data ──
if st.session_state.get("upload_analyzed_df") is None:
    st.info("Upload an earnings call transcript on the **Home** page to see risk analysis.")
    st.stop()

analyzed_df = st.session_state.upload_analyzed_df
ticker = st.session_state.get("_analyzed_ticker", "")
quarter = st.session_state.get("_analyzed_quarter", "")

st.caption(f"{ticker} {quarter} | {int(analyzed_df['risk_count'].sum())} risk signals detected")

# ── Category bar chart ──
all_cats = []
for cats in analyzed_df["risk_categories"]:
    all_cats.extend(cats)

if not all_cats:
    st.success("No risk signals detected in this transcript.")
    st.stop()

cat_counts = pd.Series(all_cats).value_counts()
cat_labels = [fmt_risk_category(c) for c in cat_counts.index]
fig_risk = go.Figure(go.Bar(
    x=cat_labels, y=cat_counts.values,
    marker=dict(color=cat_counts.values, colorscale="YlOrRd"),
    width=0.4,
))
fig_risk.update_layout(template=PLOTLY_TEMPLATE, title="Risk Categories Detected",
                       xaxis_title="Category", yaxis_title="Count",
                       yaxis=dict(dtick=1))
st.plotly_chart(styled_plotly(fig_risk), use_container_width=True)

# ── Severity breakdown ──
all_severities = []
for details in analyzed_df["risk_details"]:
    for d in details:
        all_severities.append(d.get("severity", "medium"))

if all_severities:
    sev_counts = pd.Series(all_severities).value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("High Severity", int(sev_counts.get("high", 0)))
    col2.metric("Medium Severity", int(sev_counts.get("medium", 0)))
    col3.metric("Low Severity", int(sev_counts.get("low", 0)))

# ── Risk by section ──
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Risk by Section**")
    section_risk = analyzed_df.groupby("section")["risk_count"].sum()
    for section, count in section_risk.items():
        label = "Prepared Remarks" if section == "prepared_remarks" else "Q&A"
        st.markdown(f"- {label}: {int(count)} signals")

with col2:
    st.markdown("**Risk by Speaker Role**")
    role_risk = analyzed_df.groupby("role")["risk_count"].sum().sort_values(ascending=False)
    for role, count in role_risk.items():
        if count > 0:
            st.markdown(f"- {role}: {int(count)} signals")

# ── Risk flow across transcript ──
st.markdown("---")
st.markdown("**Risk Intensity Across Transcript**")
fig_flow = go.Figure()
fig_flow.add_trace(go.Bar(
    x=list(range(len(analyzed_df))),
    y=analyzed_df["risk_intensity"].values,
    marker_color=["#ef4444" if v > 4 else "#f97316" if v > 0 else "#1e293b"
                  for v in analyzed_df["risk_intensity"].values],
    name="Risk Intensity",
))
qa_start_idx = None
for i, section in enumerate(analyzed_df["section"].values):
    if section == "qa":
        qa_start_idx = i
        break
if qa_start_idx is not None:
    fig_flow.add_vline(x=qa_start_idx, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                       annotation_text="Q&A starts")
fig_flow.update_layout(template=PLOTLY_TEMPLATE, title="Risk Intensity per Chunk",
                       xaxis_title="Chunk Index", yaxis_title="Intensity")
st.plotly_chart(styled_plotly(fig_flow), use_container_width=True)

# ── Top risk excerpts ──
st.markdown("---")
st.markdown("**Top Risk Excerpts**")
risk_chunks = analyzed_df[analyzed_df["risk_count"] > 0].nlargest(5, "risk_intensity")
for _, row in risk_chunks.iterrows():
    st.markdown(f"**Categories:** {fmt_risk_categories(row['risk_categories'])} | Intensity: {row['risk_intensity']}")
    st.markdown(f"_{row['speaker']} ({row['role']})_")
    st.markdown(f"> {row['text'][:300]}...")
    st.divider()
