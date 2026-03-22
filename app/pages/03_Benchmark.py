"""Benchmark — compare uploaded transcript against historical dataset."""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import apply_theme, PLOTLY_TEMPLATE, styled_plotly, load_sentiment, load_risk, load_features

apply_theme()

st.header("Benchmark Against Historical Dataset")

# ── Gate: require uploaded data ──
if st.session_state.get("upload_analyzed_df") is None:
    st.info("Upload an earnings call transcript on the **Home** page to see benchmark comparisons.")
    st.stop()

analyzed_df = st.session_state.upload_analyzed_df
ticker = st.session_state.get("_analyzed_ticker", "")
quarter = st.session_state.get("_analyzed_quarter", "")

st.caption(f"Comparing {ticker} {quarter} against the historical dataset")

# ── Sentiment benchmark ──
hist_sentiment = load_sentiment()

if hist_sentiment.empty:
    st.warning("No historical data available for benchmarking. Run the full pipeline first.")
    st.stop()

hist_quarterly = hist_sentiment.groupby(["ticker", "quarter"]).agg(
    lm_net=("lm_net_score", "mean"),
    vader=("vader_compound", "mean"),
).reset_index()

uploaded_lm = analyzed_df["lm_net_score"].mean()
uploaded_vader = analyzed_df["vader_compound"].mean()

lm_pct = (hist_quarterly["lm_net"] < uploaded_lm).mean() * 100
vader_pct = (hist_quarterly["vader"] < uploaded_vader).mean() * 100

col1, col2 = st.columns(2)
col1.metric("LM Net Score", f"{uploaded_lm:.4f}", delta=f"Percentile: {lm_pct:.0f}%")
col2.metric("VADER Compound", f"{uploaded_vader:.4f}", delta=f"Percentile: {vader_pct:.0f}%")

# ── Distribution plot ──
fig_bench = go.Figure()
fig_bench.add_trace(go.Histogram(x=hist_quarterly["lm_net"], name="Historical Calls",
                                  marker_color="#cccccc", opacity=0.7, nbinsx=40))
fig_bench.add_vline(x=uploaded_lm, line_dash="dash", line_color="red",
                    annotation_text=f"{ticker} {quarter}")
fig_bench.update_layout(template=PLOTLY_TEMPLATE,
                        title="Uploaded Call vs Historical Sentiment Distribution (LM Net)",
                        xaxis_title="LM Net Score", yaxis_title="Count")
st.plotly_chart(styled_plotly(fig_bench), use_container_width=True)

# ── Sector comparison ──
companies_path = Path(__file__).parent.parent.parent / "configs" / "companies.json"
if companies_path.exists():
    with open(companies_path) as f:
        companies_list = json.load(f)
    ticker_to_sector = {c["ticker"]: c["sector"] for c in companies_list}
    uploaded_sector = ticker_to_sector.get(ticker, None)

    if uploaded_sector:
        sector_tickers = [t for t, s in ticker_to_sector.items() if s == uploaded_sector]
        sector_data = hist_quarterly[hist_quarterly["ticker"].isin(sector_tickers)]
        if not sector_data.empty:
            sector_pct = (sector_data["lm_net"] < uploaded_lm).mean() * 100
            st.markdown(f"**Within {uploaded_sector} sector:** this call is at the "
                        f"**{sector_pct:.0f}th percentile** (out of {len(sector_data)} calls)")

# ── Risk benchmark ──
hist_risk = load_risk()
if not hist_risk.empty:
    st.markdown("---")
    st.subheader("Risk Benchmark")
    uploaded_risk_count = analyzed_df["risk_count"].sum()
    hist_risk_quarterly = hist_risk.groupby(["ticker", "quarter"])["risk_count"].sum().reset_index()
    risk_pct = (hist_risk_quarterly["risk_count"] < uploaded_risk_count).mean() * 100
    st.metric("Total Risk Signals", int(uploaded_risk_count),
              delta=f"Percentile: {risk_pct:.0f}%")

# ── Expected market reaction ──
features = load_features()
if not features.empty and "abnormal_ret_1d" in features.columns:
    st.markdown("---")
    st.subheader("Expected Market Reaction (Based on Historical Model)")
    similar = features.dropna(subset=["mean_lm_net_score", "abnormal_ret_1d"])
    if not similar.empty:
        q_low = similar["mean_lm_net_score"].quantile(0.33)
        q_high = similar["mean_lm_net_score"].quantile(0.67)
        if uploaded_lm < q_low:
            bucket = "Low Sentiment"
            bucket_data = similar[similar["mean_lm_net_score"] < q_low]
        elif uploaded_lm > q_high:
            bucket = "High Sentiment"
            bucket_data = similar[similar["mean_lm_net_score"] > q_high]
        else:
            bucket = "Mid Sentiment"
            bucket_data = similar[(similar["mean_lm_net_score"] >= q_low) &
                                   (similar["mean_lm_net_score"] <= q_high)]

        avg_ret = bucket_data["abnormal_ret_1d"].mean()
        st.markdown(f"This call falls in the **{bucket}** bucket "
                    f"(based on LM net score of {uploaded_lm:.4f})")
        st.markdown(f"Historical average 1-day abnormal return for this bucket: "
                    f"**{avg_ret*100:.2f}%** (n={len(bucket_data)})")
