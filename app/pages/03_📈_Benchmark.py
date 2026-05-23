"""Benchmark — compare uploaded transcript against historical dataset."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import (apply_theme, PLOTLY_TEMPLATE, styled_plotly, load_sentiment,
                   load_risk, load_features, load_finance_results, require_upload,
                   is_multi_mode)


def extract_uploaded_features(analyzed_df: pd.DataFrame, features_df: pd.DataFrame = None,
                              ticker: str = None, quarter: str = None) -> dict:
    """Extract features from the uploaded transcript, including momentum if prior quarter exists."""
    feat = {}

    # Sentiment features
    for col in ["lm_net_score", "lm_positive_score", "lm_negative_score",
                "lm_uncertainty_score", "vader_compound",
                "finbert_positive", "finbert_negative", "finbert_neutral"]:
        if col in analyzed_df.columns:
            feat[f"mean_{col}"] = analyzed_df[col].mean()

    # Management vs analyst sentiment
    if "role" in analyzed_df.columns and "lm_net_score" in analyzed_df.columns:
        mgmt = analyzed_df[analyzed_df["role"].isin(["CEO", "CFO", "COO", "CTO"])]
        analysts = analyzed_df[analyzed_df["role"] == "Analyst"]
        feat["mgmt_sentiment"] = mgmt["lm_net_score"].mean() if not mgmt.empty else np.nan
        feat["analyst_sentiment"] = analysts["lm_net_score"].mean() if not analysts.empty else np.nan

    # Prepared vs Q&A sentiment
    if "section" in analyzed_df.columns and "lm_net_score" in analyzed_df.columns:
        prepared = analyzed_df[analyzed_df["section"] == "prepared_remarks"]
        qa = analyzed_df[analyzed_df["section"] == "qa"]
        feat["prepared_sentiment"] = prepared["lm_net_score"].mean() if not prepared.empty else np.nan
        feat["qa_sentiment"] = qa["lm_net_score"].mean() if not qa.empty else np.nan
        if not np.isnan(feat.get("prepared_sentiment", np.nan)) and not np.isnan(feat.get("qa_sentiment", np.nan)):
            feat["sentiment_divergence"] = feat["prepared_sentiment"] - feat["qa_sentiment"]

    # Risk features
    if "risk_count" in analyzed_df.columns:
        feat["total_risk_count"] = analyzed_df["risk_count"].sum()
    if "risk_intensity" in analyzed_df.columns:
        feat["avg_risk_intensity"] = analyzed_df["risk_intensity"].mean()
    if "risk_categories" in analyzed_df.columns:
        all_cats = []
        for cats in analyzed_df["risk_categories"]:
            if isinstance(cats, list):
                all_cats.extend(cats)
        feat["num_risk_categories"] = len(set(all_cats))

    # ── Research-backed engineered features ──

    # Q&A-weighted sentiment (Price et al. 2012)
    qa_s = feat.get("qa_sentiment", np.nan)
    pr_s = feat.get("prepared_sentiment", np.nan)
    if not np.isnan(qa_s) and not np.isnan(pr_s):
        feat["qa_weighted_sentiment"] = 0.65 * qa_s + 0.35 * pr_s
    elif not np.isnan(qa_s):
        feat["qa_weighted_sentiment"] = qa_s

    # Negative emphasis (LM 2011)
    neg = feat.get("mean_lm_negative_score", 0)
    pos = feat.get("mean_lm_positive_score", 0)
    feat["negative_emphasis"] = neg * 1.5
    feat["neg_pos_ratio"] = neg / (pos + 1e-8)

    # Uncertainty-adjusted sentiment
    unc = feat.get("mean_lm_uncertainty_score", 0)
    lm_net = feat.get("mean_lm_net_score", 0)
    feat["uncertainty_adj_sentiment"] = lm_net - (unc * 0.5)

    # Analyst-management divergence (Brockman et al. 2015)
    mgmt_s = feat.get("mgmt_sentiment", np.nan)
    analyst_s = feat.get("analyst_sentiment", np.nan)
    if not np.isnan(mgmt_s) and not np.isnan(analyst_s):
        feat["mgmt_analyst_divergence"] = mgmt_s - analyst_s

    # Risk-sentiment interaction
    feat["risk_sentiment_interaction"] = lm_net * feat.get("total_risk_count", 0)

    # Text features (readability, FLS, specificity)
    try:
        from src.agents.text_features import extract_call_features
        chunk_texts = analyzed_df["text"].tolist() if "text" in analyzed_df.columns else []
        if chunk_texts:
            text_feats = extract_call_features(chunk_texts)
            feat.update(text_feats)
    except Exception:
        pass

    # Q&A dynamics features
    try:
        from src.agents.qa_features import qa_dynamics_features
        qa_feats = qa_dynamics_features(analyzed_df)
        feat.update(qa_feats)
    except Exception:
        pass

    # ── Momentum features (look up prior quarter in historical data) ──
    if features_df is not None and ticker and quarter:
        prior_q = _get_prior_quarter(quarter)
        if prior_q:
            prior = features_df[(features_df["ticker"] == ticker) & (features_df["quarter"] == prior_q)]
            if not prior.empty:
                prior_row = prior.iloc[0]
                if "mean_lm_net_score" in prior_row and not np.isnan(prior_row["mean_lm_net_score"]):
                    feat["mean_lm_net_score_momentum"] = feat.get("mean_lm_net_score", 0) - prior_row["mean_lm_net_score"]
                if "mean_vader_compound" in prior_row and not np.isnan(prior_row["mean_vader_compound"]):
                    feat["mean_vader_compound_momentum"] = feat.get("mean_vader_compound", 0) - prior_row["mean_vader_compound"]
                if "total_risk_count" in prior_row and not np.isnan(prior_row["total_risk_count"]):
                    feat["risk_delta"] = feat.get("total_risk_count", 0) - prior_row["total_risk_count"]

    return feat


def _get_prior_quarter(quarter: str) -> str:
    """Get the previous quarter string. '2020Q2' → '2020Q1', '2020Q1' → '2019Q4'."""
    import re
    m = re.match(r"(\d{4})Q(\d)", quarter)
    if not m:
        return None
    year, q = int(m.group(1)), int(m.group(2))
    if q == 1:
        return f"{year-1}Q4"
    return f"{year}Q{q-1}"


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add research-backed engineered features to a feature matrix."""
    df = df.copy()

    if "qa_sentiment" in df.columns and "prepared_sentiment" in df.columns:
        df["qa_weighted_sentiment"] = df["qa_sentiment"].fillna(0) * 0.65 + df["prepared_sentiment"].fillna(0) * 0.35

    if "mean_lm_negative_score" in df.columns:
        df["negative_emphasis"] = df["mean_lm_negative_score"] * 1.5
    if "mean_lm_negative_score" in df.columns and "mean_lm_positive_score" in df.columns:
        df["neg_pos_ratio"] = df["mean_lm_negative_score"] / (df["mean_lm_positive_score"] + 1e-8)

    if "mean_lm_net_score" in df.columns and "mean_lm_uncertainty_score" in df.columns:
        df["uncertainty_adj_sentiment"] = df["mean_lm_net_score"] - (df["mean_lm_uncertainty_score"] * 0.5)

    if "mgmt_sentiment" in df.columns and "analyst_sentiment" in df.columns:
        df["mgmt_analyst_divergence"] = df["mgmt_sentiment"] - df["analyst_sentiment"]

    if "mean_lm_net_score" in df.columns and "total_risk_count" in df.columns:
        df["risk_sentiment_interaction"] = df["mean_lm_net_score"] * df["total_risk_count"]

    return df


def _get_feature_cols(enhanced_df: pd.DataFrame) -> list:
    """Get feature columns with sufficient coverage."""
    candidate_cols = [c for c in enhanced_df.columns
                      if c.startswith(("mean_", "mgmt_", "analyst_", "prepared_", "qa_",
                                       "sentiment_", "total_risk", "avg_risk", "num_risk",
                                       "negative_emphasis", "neg_pos", "uncertainty_adj",
                                       "mgmt_analyst", "risk_sentiment", "qa_weighted",
                                       "call_", "risk_delta"))]
    threshold = len(enhanced_df) * 0.7
    return [c for c in candidate_cols if enhanced_df[c].notna().sum() >= threshold]


@st.cache_resource
def _load_direction_model():
    """Load pre-trained direction model from disk."""
    import pickle
    model_path = Path(__file__).parent.parent.parent / "models" / "direction_model.pkl"
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_direction(uploaded_features: dict, model_state: dict) -> dict:
    """Predict stock direction using pre-trained models (instant)."""
    feature_cols = model_state["feature_cols"]
    scaler = model_state["scaler"]
    medians = model_state["medians"]

    # Build feature vector
    x_uploaded = np.array([[uploaded_features.get(col, np.nan) for col in feature_cols]])
    for j in range(x_uploaded.shape[1]):
        if np.isnan(x_uploaded[0, j]):
            x_uploaded[0, j] = medians[j]

    x_uploaded_scaled = scaler.transform(x_uploaded)

    predictions = {}
    probabilities = {}
    for name, model in model_state["models"].items():
        pred = int(model.predict(x_uploaded_scaled)[0])
        prob = model.predict_proba(x_uploaded_scaled)[0]
        predictions[name] = pred
        probabilities[name] = {"down": float(prob[0]), "up": float(prob[1])}

    up_votes = sum(1 for p in predictions.values() if p == 1)
    ensemble_direction = 1 if up_votes > len(predictions) / 2 else 0
    ensemble_prob_up = np.mean([p["up"] for p in probabilities.values()])

    return {
        "direction": ensemble_direction,
        "direction_label": "UP" if ensemble_direction == 1 else "DOWN",
        "confidence": float(max(ensemble_prob_up, 1 - ensemble_prob_up)),
        "prob_up": float(ensemble_prob_up),
        "prob_down": float(1 - ensemble_prob_up),
        "individual": predictions,
        "probabilities": probabilities,
        "n_train": model_state["n_train"],
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "feature_importances": model_state["feature_importances"],
        "baseline_accuracy": model_state["baseline_accuracy"],
        "cv_results": model_state.get("cv_results", {}),
    }


apply_theme()

st.header("Benchmark Against Historical Dataset")

require_upload("benchmark comparisons")

# ── Multi-quarter mode: per-quarter predictions with real momentum ──
if is_multi_mode():
    ticker = st.session_state.get("multi_ticker", "")
    quarter_order = st.session_state.get("multi_quarter_order", [])
    multi_quarters = st.session_state.get("multi_quarters", {})

    st.caption(f"Comparing {ticker} across {len(quarter_order)} quarters")

    features = load_features()
    model_state = _load_direction_model()

    # Extract features for each quarter, using prior quarter from uploaded data for momentum
    quarter_features = {}
    quarter_predictions = {}
    prior_analyzed = None

    for i, q in enumerate(quarter_order):
        adf = multi_quarters[q].get("analyzed_df")
        if adf is None:
            continue

        # Extract base features
        feat = extract_uploaded_features(adf, features, ticker, q)

        # Compute real momentum from prior uploaded quarter (if consecutive)
        if prior_analyzed is not None:
            prior_feat = extract_uploaded_features(prior_analyzed, None, ticker, quarter_order[i - 1])
            if "mean_lm_net_score" in prior_feat and "mean_lm_net_score" in feat:
                feat["mean_lm_net_score_momentum"] = feat["mean_lm_net_score"] - prior_feat["mean_lm_net_score"]
            if "mean_vader_compound" in prior_feat and "mean_vader_compound" in feat:
                feat["mean_vader_compound_momentum"] = feat["mean_vader_compound"] - prior_feat["mean_vader_compound"]
            if "total_risk_count" in prior_feat and "total_risk_count" in feat:
                feat["risk_delta"] = feat["total_risk_count"] - prior_feat["total_risk_count"]

        quarter_features[q] = feat
        prior_analyzed = adf

        # Predict
        if model_state is not None:
            pred = predict_direction(feat, model_state)
            quarter_predictions[q] = pred

    # Show prediction trend
    if quarter_predictions:
        st.subheader("Direction Prediction Across Quarters")

        pred_rows = []
        for q in quarter_order:
            if q in quarter_predictions:
                p = quarter_predictions[q]
                pred_rows.append({
                    "Quarter": q,
                    "Direction": p["direction_label"],
                    "Confidence": p["confidence"],
                    "P(Up)": p["prob_up"],
                    "P(Down)": p["prob_down"],
                })

        pred_df = pd.DataFrame(pred_rows)

        # Direction cards
        cols = st.columns(len(pred_rows))
        for col_w, row in zip(cols, pred_rows):
            color = "#34d399" if row["Direction"] == "UP" else "#ef4444"
            col_w.markdown(
                f'<div style="background:#181825;border:1px solid {color}40;'
                f'border-radius:12px;padding:16px;text-align:center">'
                f'<div style="color:#94a3b8;font-size:12px">{row["Quarter"]}</div>'
                f'<div style="color:{color};font-size:28px;font-weight:800">{row["Direction"]}</div>'
                f'<div style="color:#94a3b8;font-size:12px">{row["Confidence"]:.1%} conf</div>'
                f'</div>', unsafe_allow_html=True)

        # P(Up) trend line
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(
            x=pred_df["Quarter"], y=pred_df["P(Up)"],
            mode="lines+markers", name="P(Up)",
            marker=dict(size=10, color="#34d399"),
            line=dict(color="#34d399", width=2.5),
        ))
        fig_prob.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                           annotation_text="50%")
        fig_prob.update_layout(template=PLOTLY_TEMPLATE, title="P(Up) Trend Across Quarters",
                               yaxis_title="Probability", yaxis=dict(range=[0, 1]))
        st.plotly_chart(styled_plotly(fig_prob), use_container_width=True)

        # Momentum indicator
        has_real_momentum = any(
            q in quarter_features and any(
                k.endswith("_momentum") or k == "risk_delta"
                for k in quarter_features[q]
                if quarter_features[q].get(k) is not None
                and not (isinstance(quarter_features[q].get(k), float) and np.isnan(quarter_features[q].get(k)))
            )
            for q in quarter_order[1:]
        )
        if has_real_momentum:
            st.caption("Real sentiment momentum computed from uploaded consecutive quarters")

    st.markdown("---")
    st.subheader("Per-Quarter Detail")

    # Per-quarter tabs with key metrics
    tabs = st.tabs(quarter_order)
    for tab, q in zip(tabs, quarter_order):
        with tab:
            adf = multi_quarters[q].get("analyzed_df")
            if adf is None:
                st.warning(f"No data for {q}")
                continue

            q_feat = quarter_features.get(q, {})
            q_pred = quarter_predictions.get(q)

            # Show prediction if available
            if q_pred:
                direction = q_pred["direction_label"]
                confidence = q_pred["confidence"]
                prob_up = q_pred["prob_up"]
                dir_color = "#34d399" if direction == "UP" else "#ef4444"
                st.markdown(
                    f'<div style="background:#181825;border:1px solid {dir_color}40;'
                    f'border-radius:12px;padding:16px 24px;text-align:center;margin-bottom:16px">'
                    f'<span style="color:{dir_color};font-size:28px;font-weight:800">{direction}</span>'
                    f'<span style="color:#94a3b8;font-size:14px;margin-left:12px">'
                    f'Confidence: {confidence:.1%} | P(Up): {prob_up:.1%}</span></div>',
                    unsafe_allow_html=True)

            # Key sentiment metrics
            uploaded_lm_q = adf["lm_net_score"].mean()
            uploaded_vader_q = adf["vader_compound"].mean()
            risk_count_q = int(adf["risk_count"].sum())

            c1, c2, c3 = st.columns(3)
            c1.metric("LM Net Score", f"{uploaded_lm_q:.4f}")
            c2.metric("VADER Compound", f"{uploaded_vader_q:.4f}")
            c3.metric("Risk Signals", risk_count_q)

            # Momentum if available
            mom = q_feat.get("mean_lm_net_score_momentum")
            if mom is not None and not (isinstance(mom, float) and np.isnan(mom)):
                st.caption(f"Sentiment momentum vs prior quarter: **{mom:+.4f}**")

    # Restore latest quarter state
    latest_q = quarter_order[-1]
    st.session_state.upload_analyzed_df = multi_quarters[latest_q].get("analyzed_df")
    st.session_state["_analyzed_quarter"] = latest_q
    st.stop()

analyzed_df = st.session_state.upload_analyzed_df
ticker = st.session_state.get("_analyzed_ticker", "")
quarter = st.session_state.get("_analyzed_quarter", "")

st.caption(f"Comparing {ticker} {quarter} against the historical dataset")

# ── Load all data upfront ──
hist_sentiment = load_sentiment()
if hist_sentiment.empty:
    st.warning("No historical data available for benchmarking. Run the full pipeline first.")
    st.stop()

hist_quarterly = hist_sentiment.groupby(["ticker", "quarter"]).agg(
    lm_net=("lm_net_score", "mean"),
    vader=("vader_compound", "mean"),
).reset_index()

has_finbert = "finbert_positive" in analyzed_df.columns
uploaded_lm = analyzed_df["lm_net_score"].mean()
uploaded_vader = analyzed_df["vader_compound"].mean()
lm_pct = (hist_quarterly["lm_net"] < uploaded_lm).mean() * 100

features = load_features()
fin_results = load_finance_results()

# Pre-compute bucket info
bucket = None
bucket_data = None
similar = None
q_low = q_high = None
if not features.empty and "abnormal_ret_1d" in features.columns:
    similar = features.dropna(subset=["mean_lm_net_score", "abnormal_ret_1d"])
    if not similar.empty:
        q_low = similar["mean_lm_net_score"].quantile(0.33)
        q_high = similar["mean_lm_net_score"].quantile(0.67)
        if uploaded_lm < q_low:
            bucket = "Low"
            bucket_data = similar[similar["mean_lm_net_score"] < q_low]
        elif uploaded_lm > q_high:
            bucket = "High"
            bucket_data = similar[similar["mean_lm_net_score"] > q_high]
        else:
            bucket = "Mid"
            bucket_data = similar[(similar["mean_lm_net_score"] >= q_low) &
                                   (similar["mean_lm_net_score"] <= q_high)]


# ══════════════════════════════════════════════════
# SECTION 1: ML PREDICTION (what traders care about)
# ══════════════════════════════════════════════════

# Extract features including momentum (using historical data for prior quarter lookup)
uploaded_features = extract_uploaded_features(analyzed_df, features, ticker, quarter)

# Binary classification prediction
ml_prediction = None
model_state = _load_direction_model()
if model_state is not None:
    ml_prediction = predict_direction(uploaded_features, model_state)

if ml_prediction is not None:
    direction = ml_prediction["direction_label"]
    confidence = ml_prediction["confidence"]
    prob_up = ml_prediction["prob_up"]

    if direction == "UP":
        dir_color = "#34d399"
        dir_icon = "+"
    else:
        dir_color = "#ef4444"
        dir_icon = "-"

    # Hero card — direction prediction
    st.markdown(
        f'<div style="background:#181825;border:1px solid {dir_color}40;'
        f'border-radius:16px;padding:24px 32px;margin:8px 0 24px 0;text-align:center">'
        f'<div style="color:#94a3b8;font-size:14px;margin-bottom:8px">'
        f'ML Classification — trained on {ml_prediction["n_train"]:,} historical earnings calls '
        f'({ml_prediction["n_features"]} NLP features)</div>'
        f'<div style="color:{dir_color};font-size:48px;font-weight:800;letter-spacing:-1px">'
        f'{direction}</div>'
        f'<div style="color:#c6cdd5;font-size:18px;margin-top:4px">'
        f'Predicted 1-day post-earnings direction for <b>{ticker} {quarter}</b></div>'
        f'<div style="color:#94a3b8;font-size:14px;margin-top:8px">'
        f'Confidence: {confidence:.1%} &nbsp;|&nbsp; P(Up): {prob_up:.1%} &nbsp;|&nbsp; P(Down): {1-prob_up:.1%}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Plain-English interpretation
    if direction == "UP":
        interp = (f"Based on the language patterns in this earnings call, our models predict "
                  f"**{ticker}** is **more likely to rise** in the day after earnings "
                  f"({confidence:.0%} confidence). This is based on {ml_prediction['n_features']} "
                  f"NLP features extracted from the transcript.")
    else:
        interp = (f"Based on the language patterns in this earnings call, our models predict "
                  f"**{ticker}** is **more likely to decline** in the day after earnings "
                  f"({confidence:.0%} confidence). Negative language cues in the call suggest "
                  f"the market may react unfavorably.")
    st.markdown(f'<div style="color:#94a3b8;font-size:13px;text-align:center;margin:-12px 0 16px 0">'
                f'{interp}</div>', unsafe_allow_html=True)

    # Individual model predictions
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ensemble", direction, delta=f"{confidence:.1%} confidence")
    for col_widget, (name, pred) in zip([col2, col3, col4], ml_prediction["individual"].items()):
        display_name = name.replace("_", " ").title()
        model_dir = "UP" if pred == 1 else "DOWN"
        model_prob = ml_prediction["probabilities"][name]["up"]
        col_widget.metric(display_name, model_dir, delta=f"P(Up): {model_prob:.1%}")

    # Momentum features indicator
    has_momentum = any(k.endswith("_momentum") or k == "risk_delta"
                      for k in uploaded_features if uploaded_features.get(k) is not None
                      and not (isinstance(uploaded_features.get(k), float)
                               and np.isnan(uploaded_features.get(k))))
    if has_momentum:
        st.caption(f"Sentiment momentum available (vs prior quarter {_get_prior_quarter(quarter)})")
    else:
        st.caption(f"No prior quarter data for {ticker} — momentum features imputed")

elif bucket_data is not None:
    avg_ret_1d = bucket_data["abnormal_ret_1d"].mean()
    ret_color = "#34d399" if avg_ret_1d >= 0 else "#ef4444"
    st.markdown(
        f'<div style="background:#181825;border:1px solid {ret_color}40;'
        f'border-radius:16px;padding:24px 32px;margin:8px 0 24px 0;text-align:center">'
        f'<div style="color:#94a3b8;font-size:14px;margin-bottom:8px">'
        f'Based on {len(bucket_data)} historically similar earnings calls</div>'
        f'<div style="color:{ret_color};font-size:42px;font-weight:800;letter-spacing:-1px">'
        f'{avg_ret_1d*100:+.2f}%</div>'
        f'<div style="color:#c6cdd5;font-size:16px;margin-top:4px">'
        f'Average 1-day abnormal return for <b>{bucket} Sentiment</b> calls</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

if bucket_data is not None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sentiment Bucket", f"{bucket}")
    col2.metric("LM Net Score", f"{uploaded_lm:.4f}", delta=f"Percentile: {lm_pct:.0f}%")
    if "abnormal_ret_3d" in bucket_data.columns:
        avg_ret_3d = bucket_data["abnormal_ret_3d"].mean()
        col3.metric("Avg 3-Day Return", f"{avg_ret_3d*100:+.2f}%")
    else:
        col3.metric("Std Dev (1-Day)", f"{bucket_data['abnormal_ret_1d'].std()*100:.2f}%")
    col4.metric("Sample Size", f"{len(bucket_data):,} calls")

    # Plain-English bucket interpretation
    bucket_interp = {
        "Low": "This call's tone is in the bottom third of all historical earnings calls — negative language dominates.",
        "Mid": "This call's tone is in the middle of the pack — neither strongly positive nor negative compared to peers.",
        "High": "This call's tone is in the top third — management used significantly more positive language than average.",
    }
    st.caption(bucket_interp.get(bucket, ""))

    # Scatter: sentiment vs return (dynamic count)
    n_events = len(similar)
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=similar["mean_lm_net_score"], y=similar["abnormal_ret_1d"] * 100,
        mode="markers", marker=dict(size=5, color="#4f7df5", opacity=0.4),
        name="Historical Calls",
        hovertemplate="LM: %{x:.4f}<br>Return: %{y:.2f}%<extra></extra>",
    ))
    fig_scatter.add_vline(x=uploaded_lm, line_dash="dash", line_color="#ef4444",
                          annotation_text=f"{ticker}", annotation_font_size=14)
    fig_scatter.add_hline(y=0, line_color="rgba(255,255,255,0.15)")
    fig_scatter.add_vrect(x0=similar["mean_lm_net_score"].min(), x1=q_low,
                          fillcolor="rgba(239,68,68,0.06)", line_width=0)
    fig_scatter.add_vrect(x0=q_high, x1=similar["mean_lm_net_score"].max(),
                          fillcolor="rgba(52,211,153,0.06)", line_width=0)
    fig_scatter.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Sentiment vs 1-Day Abnormal Return ({n_events:,} earnings events)",
        xaxis_title="Mean LM Net Score",
        yaxis_title="1-Day Abnormal Return (%)",
    )
    st.plotly_chart(styled_plotly(fig_scatter), use_container_width=True)

    # Portfolio sort bars
    if fin_results and "portfolio_sorts_mean_lm_net_score" in fin_results:
        sorts = fin_results["portfolio_sorts_mean_lm_net_score"]
        labels = [s["bucket_label"] for s in sorts]
        returns = [s["mean_return"] * 100 for s in sorts]
        counts = [s["count"] for s in sorts]
        colors = ["#ef4444" if r < 0 else "#34d399" for r in returns]

        fig_port = go.Figure()
        fig_port.add_trace(go.Bar(
            x=labels, y=returns,
            marker_color=colors,
            text=[f"{r:+.3f}%<br>(n={c:,})" for r, c in zip(returns, counts)],
            textposition="outside",
            textfont=dict(size=13),
            width=0.4,
        ))
        bucket_idx = 0 if uploaded_lm < q_low else (2 if uploaded_lm > q_high else 1)
        fig_port.add_annotation(
            x=labels[bucket_idx], y=returns[bucket_idx],
            text=f"{ticker}",
            showarrow=True, arrowhead=2, arrowcolor="#ef4444",
            font=dict(color="#ef4444", size=14, family="Inter"),
        )
        fig_port.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Avg 1-Day Abnormal Return by Sentiment Tercile",
            xaxis_title="Sentiment Bucket (LM Net Score)",
            yaxis_title="Abnormal Return (%)",
            yaxis=dict(tickformat=".3f"),
        )
        st.plotly_chart(styled_plotly(fig_port), use_container_width=True)

        # Portfolio sort interpretation
        spread = returns[-1] - returns[0]
        st.caption(f"Historically, the most positive earnings calls outperform the most negative by "
                   f"**{spread:+.3f}%** on average in the first day after the call. "
                   f"This long-short spread shows that earnings call sentiment carries real predictive signal.")


# ══════════════════════════════════════════════════
# SECTION 2: SENTIMENT PERCENTILE & RISK BENCHMARK
# ══════════════════════════════════════════════════
st.markdown("---")
st.subheader("Sentiment Percentile")

vader_pct = (hist_quarterly["vader"] < uploaded_vader).mean() * 100

if has_finbert:
    label_counts = analyzed_df["finbert_label"].value_counts()
    pos_pct = label_counts.get("positive", 0) / len(analyzed_df) * 100
    neg_pct = label_counts.get("negative", 0) / len(analyzed_df) * 100
    col1, col2, col3 = st.columns(3)
    col3.metric("FinBERT Positive", f"{pos_pct:.0f}%",
                delta=f"Negative: {neg_pct:.0f}%", delta_color="inverse")
else:
    col1, col2 = st.columns(2)

col1.metric("LM Net Score", f"{uploaded_lm:.4f}", delta=f"Percentile: {lm_pct:.0f}%")
col2.metric("VADER Compound", f"{uploaded_vader:.4f}", delta=f"Percentile: {vader_pct:.0f}%")

# Sector comparison
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

# Risk benchmark
hist_risk = load_risk()
if not hist_risk.empty:
    uploaded_risk_count = analyzed_df["risk_count"].sum()
    hist_risk_quarterly = hist_risk.groupby(["ticker", "quarter"])["risk_count"].sum().reset_index()
    risk_pct = (hist_risk_quarterly["risk_count"] < uploaded_risk_count).mean() * 100
    st.metric("Total Risk Signals", int(uploaded_risk_count),
              delta=f"Percentile: {risk_pct:.0f}%")

# Distribution plots behind expander
with st.expander("Sentiment Distribution vs Historical"):
    if has_finbert:
        col_lm, col_vader, col_fb = st.columns(3)
    else:
        col_lm, col_vader = st.columns(2)

    with col_lm:
        fig_bench_lm = go.Figure()
        fig_bench_lm.add_trace(go.Histogram(x=hist_quarterly["lm_net"], name="Historical",
                                             marker_color="#cccccc", opacity=0.7, nbinsx=40))
        fig_bench_lm.add_vline(x=uploaded_lm, line_dash="dash", line_color="red",
                               annotation_text=f"{ticker}")
        fig_bench_lm.update_layout(template=PLOTLY_TEMPLATE, title="LM Net vs Historical",
                                   xaxis_title="LM Net Score", yaxis_title="Count")
        st.plotly_chart(styled_plotly(fig_bench_lm), use_container_width=True)

    with col_vader:
        fig_bench_v = go.Figure()
        fig_bench_v.add_trace(go.Histogram(x=hist_quarterly["vader"], name="Historical",
                                            marker_color="#cccccc", opacity=0.7, nbinsx=40))
        fig_bench_v.add_vline(x=uploaded_vader, line_dash="dash", line_color="red",
                              annotation_text=f"{ticker}")
        fig_bench_v.update_layout(template=PLOTLY_TEMPLATE, title="VADER vs Historical",
                                  xaxis_title="VADER Compound", yaxis_title="Count")
        st.plotly_chart(styled_plotly(fig_bench_v), use_container_width=True)

    if has_finbert:
        with col_fb:
            labels = ["Positive", "Negative", "Neutral"]
            values = [label_counts.get("positive", 0),
                      label_counts.get("negative", 0),
                      label_counts.get("neutral", 0)]
            fig_fb_pie = go.Figure(go.Pie(
                labels=labels, values=values,
                marker=dict(colors=["#34d399", "#ef4444", "#94a3b8"]),
                hole=0.4,
            ))
            fig_fb_pie.update_layout(template=PLOTLY_TEMPLATE, title="FinBERT Labels",
                                     showlegend=True)
            st.plotly_chart(styled_plotly(fig_fb_pie), use_container_width=True)


# ══════════════════════════════════════════════════
# SECTION 3: MODEL DETAILS (behind expanders)
# ══════════════════════════════════════════════════
if fin_results:
    st.markdown("---")
    st.subheader("Model Details")
    st.caption("Expand sections below for methodology and model performance")

    # Event Study
    if "event_study" in fin_results:
        with st.expander("Event Study — Cumulative Abnormal Returns"):
            n_events = fin_results['event_study'].get('car_1d', {}).get('n', 0)
            st.markdown(f"Based on **{n_events:,}** historical earnings events. "
                        "CAR measures average stock price reaction around earnings calls, "
                        "adjusted for market (S&P 500) movement.")
            es = fin_results["event_study"]
            col1, col2, col3 = st.columns(3)
            for col_w, window, key in [(col1, "1-Day", "car_1d"), (col2, "3-Day", "car_3d"), (col3, "5-Day", "car_5d")]:
                if key in es:
                    data = es[key]
                    col_w.metric(
                        f"CAR {window}",
                        f"{data['mean']*100:+.3f}%",
                        delta=f"t={data['t_stat']:.2f}, p={data['p_value']:.3f}",
                        delta_color="off",
                    )

    # Portfolio Sort (VADER)
    if "portfolio_sorts_mean_vader_compound" in fin_results:
        with st.expander("Portfolio Sort — VADER Sentiment"):
            sorts_v = fin_results["portfolio_sorts_mean_vader_compound"]
            cols = st.columns(len(sorts_v))
            for col_w, s in zip(cols, sorts_v):
                col_w.metric(
                    f"VADER {s['bucket_label']}",
                    f"{s['mean_return']*100:+.3f}%",
                    delta=f"n={s['count']:,}",
                    delta_color="off",
                )

    # ML Model Comparison
    if "ml_models" in fin_results:
        with st.expander("ML Model Performance & Feature Importance"):
            models = fin_results["ml_models"]
            reg = fin_results.get("regression", {})

            st.markdown("Cross-validated R² on predicting 1-day abnormal returns from NLP features. "
                        "Negative R² means the model performs worse than simply predicting the mean return.")

            model_names = ["OLS"]
            r2_scores = [reg.get("r_squared", 0)]
            for name, key in [("Lasso", "lasso"), ("Ridge", "ridge"),
                              ("Random Forest", "random_forest"), ("Gradient Boosting", "gradient_boosting")]:
                if key in models:
                    model_names.append(name)
                    r2_scores.append(models[key]["cv_r2_mean"])

            fig_models = go.Figure()
            bar_colors = ["#4f7df5" if r >= 0 else "#ef4444" for r in r2_scores]
            fig_models.add_trace(go.Bar(
                x=model_names, y=r2_scores,
                marker_color=bar_colors,
                text=[f"{r:.4f}" for r in r2_scores],
                textposition="outside",
                width=0.5,
            ))
            fig_models.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_dash="dot")
            fig_models.update_layout(
                template=PLOTLY_TEMPLATE,
                title="Model R² Comparison (Regression)",
                xaxis_title="Model", yaxis_title="R²",
            )
            st.plotly_chart(styled_plotly(fig_models), use_container_width=True)

            # Feature importance from classification models
            if ml_prediction and ml_prediction.get("feature_importances"):
                st.markdown("#### Feature Importance (Classification Models)")
                imp_keys = [k for k in ml_prediction["feature_importances"]
                           if k in ("gradient_boosting", "random_forest")]
                if imp_keys:
                    imp_cols = st.columns(len(imp_keys))
                    for col_w, key in zip(imp_cols, imp_keys):
                        imp = ml_prediction["feature_importances"][key]
                        sorted_imp = sorted(imp.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
                        feat_names = [x[0].replace("mean_", "").replace("_score", "")
                                     .replace("_", " ").title() for x in sorted_imp]
                        feat_vals = [abs(x[1]) * 100 for x in sorted_imp]

                        fig_imp = go.Figure(go.Bar(
                            x=feat_vals, y=feat_names, orientation="h",
                            marker_color="#8b5cf6",
                        ))
                        title = key.replace("_", " ").title()
                        fig_imp.update_layout(
                            template=PLOTLY_TEMPLATE, title=f"{title} (Top 15)",
                            xaxis_title="Importance (%)",
                            yaxis=dict(autorange="reversed"), height=450,
                        )
                        with col_w:
                            st.plotly_chart(styled_plotly(fig_imp), use_container_width=True)

            st.markdown("*Note: NLP features alone have limited predictive power for stock returns. "
                        "Binary classification (UP/DOWN) achieves ~54% accuracy vs ~51% baseline — "
                        "a small but meaningful edge consistent with academic findings.*")

    # OLS Regression
    if fin_results.get("regression"):
        with st.expander("OLS Regression Coefficients"):
            reg = fin_results["regression"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R²", f"{reg['r_squared']:.4f}")
            col2.metric("Adj R²", f"{reg['adj_r_squared']:.4f}")
            col3.metric("F-statistic", f"{reg['f_statistic']:.3f}",
                        delta=f"p={reg['f_pvalue']:.3f}", delta_color="off")
            col4.metric("Observations", f"{reg['n_obs']:,}")

            coefs = reg.get("coefficients", {})
            if coefs:
                sorted_coefs = sorted(
                    [(k, v) for k, v in coefs.items() if k != "const"],
                    key=lambda x: abs(x[1]["coef"]),
                    reverse=True,
                )

                feat_names = []
                coef_vals = []
                bar_colors = []
                for name, vals in sorted_coefs:
                    clean_name = (name.replace("mean_", "").replace("_score", "")
                                  .replace("_", " ").title())
                    sig = vals["p_value"] < 0.05
                    if sig:
                        clean_name += " *"
                    feat_names.append(clean_name)
                    coef_vals.append(vals["coef"] * 100)
                    bar_colors.append("#34d399" if vals["coef"] >= 0 else "#ef4444")

                fig_coef = go.Figure(go.Bar(
                    x=coef_vals, y=feat_names, orientation="h",
                    marker_color=bar_colors,
                    text=[f"{v:+.4f}" for v in coef_vals],
                    textposition="outside",
                    textfont=dict(size=11),
                ))
                fig_coef.add_vline(x=0, line_color="rgba(255,255,255,0.3)")
                fig_coef.update_layout(
                    template=PLOTLY_TEMPLATE,
                    title="OLS Coefficients (x100 for readability)",
                    xaxis_title="Coefficient (x100)",
                    yaxis=dict(autorange="reversed"),
                    height=max(300, len(sorted_coefs) * 35 + 80),
                )
                st.plotly_chart(styled_plotly(fig_coef), use_container_width=True)
                st.caption("Green = positive effect on returns, Red = negative. "
                           "Features marked with **\\*** are statistically significant (p < 0.05).")
