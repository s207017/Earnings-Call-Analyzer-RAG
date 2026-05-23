"""Sentiment Analysis — detailed sentiment breakdown of the uploaded transcript."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import apply_theme, PLOTLY_TEMPLATE, styled_plotly, highlight_sentiment_words, require_upload, is_multi_mode

apply_theme()

st.header("Sentiment Analysis")

require_upload("sentiment analysis")

# ── Glossary / Legend ──
with st.expander("What do these models mean?", expanded=False):
    st.markdown(
        """
| Model | What it measures | How it works | Best for |
|-------|-----------------|--------------|----------|
| **Loughran-McDonald (LM)** | Finance-specific tone | Counts positive/negative words from a dictionary built for 10-K/10-Q filings | Detecting formal financial language (e.g. *impairment*, *restructuring*) |
| **VADER** | General sentiment | Rule-based scorer tuned for social-media-style text; handles punctuation, caps, slang | Catching everyday positive/negative language that LM misses |
| **FinBERT** | Deep contextual sentiment | Transformer neural network fine-tuned on financial news | Understanding context (e.g. "not bad" = positive) |

**Key signals shown on this page:**
- **Management vs Analyst gap** — when management is much more positive than analysts, research shows this *overoptimism* often precedes negative stock movement (p < 0.001).
- **Prepared vs Q&A shift** — if sentiment drops when executives face live questions, it suggests the prepared remarks were overly polished (p = 0.002).
- **Sentiment momentum** (multi-quarter) — quarter-over-quarter *change* in sentiment is a stronger return predictor than the raw level.
""")


MGMT_ROLES = {"CEO", "CFO", "COO", "CTO", "VP", "IR", "Management"}


def _compute_signals(analyzed_df):
    """Extract the key sentiment signals from analyzed data."""
    signals = {}

    # Overall
    signals["lm_net"] = analyzed_df["lm_net_score"].mean()
    signals["vader"] = analyzed_df["vader_compound"].mean()
    has_finbert = "finbert_label" in analyzed_df.columns
    signals["has_finbert"] = has_finbert
    if has_finbert:
        label_counts = analyzed_df["finbert_label"].value_counts()
        total = len(analyzed_df)
        signals["fb_positive_pct"] = label_counts.get("positive", 0) / total
        signals["fb_negative_pct"] = label_counts.get("negative", 0) / total
        signals["fb_neutral_pct"] = label_counts.get("neutral", 0) / total

    # Management vs Analyst
    mgmt = analyzed_df[analyzed_df["role"].isin(MGMT_ROLES)]
    analysts = analyzed_df[analyzed_df["role"] == "Analyst"]
    signals["mgmt_sent"] = mgmt["lm_net_score"].mean() if not mgmt.empty else np.nan
    signals["analyst_sent"] = analysts["lm_net_score"].mean() if not analysts.empty else np.nan
    if pd.notna(signals["mgmt_sent"]) and pd.notna(signals["analyst_sent"]):
        signals["mgmt_analyst_gap"] = signals["mgmt_sent"] - signals["analyst_sent"]
    else:
        signals["mgmt_analyst_gap"] = np.nan

    # Prepared vs Q&A
    prepared = analyzed_df[analyzed_df["section"] == "prepared_remarks"]
    qa = analyzed_df[analyzed_df["section"] == "qa"]
    signals["prepared_sent"] = prepared["lm_net_score"].mean() if not prepared.empty else np.nan
    signals["qa_sent"] = qa["lm_net_score"].mean() if not qa.empty else np.nan
    if pd.notna(signals["prepared_sent"]) and pd.notna(signals["qa_sent"]):
        signals["pq_divergence"] = signals["prepared_sent"] - signals["qa_sent"]
    else:
        signals["pq_divergence"] = np.nan

    return signals


def _tone_label(score):
    """Map a sentiment score to a tone label and color."""
    if score > 0.02:
        return "Positive", "#34d399"
    elif score < -0.02:
        return "Negative", "#ef4444"
    else:
        return "Neutral", "#94a3b8"


def _render_summary_box(signals, ticker, quarter):
    """Auto-generated key takeaway box at the top."""
    findings = []

    # Overall tone
    tone_label, tone_color = _tone_label(signals["lm_net"])
    findings.append(f"Overall tone is <b style='color:{tone_color}'>{tone_label.lower()}</b> "
                    f"(LM net: {signals['lm_net']:.4f})")

    # Management vs Analyst gap
    gap = signals.get("mgmt_analyst_gap", np.nan)
    if pd.notna(gap):
        if gap > 0.01:
            findings.append(f"Management is <b style='color:#fbbf24'>significantly more positive</b> "
                            f"than analysts (gap: {gap:+.4f}) — historically correlates with "
                            f"negative post-earnings drift")
        elif gap < -0.01:
            findings.append(f"Analysts are <b style='color:#34d399'>more positive</b> than management "
                            f"(gap: {gap:+.4f}) — suggests external confidence")
        else:
            findings.append(f"Management and analyst sentiment are <b>aligned</b> (gap: {gap:+.4f})")

    # Prepared vs Q&A divergence
    div = signals.get("pq_divergence", np.nan)
    if pd.notna(div):
        if div > 0.01:
            findings.append(f"Sentiment <b style='color:#ef4444'>drops in Q&A</b> vs prepared remarks "
                            f"(shift: {div:+.4f}) — tone weakens under analyst questioning")
        elif div < -0.01:
            findings.append(f"Sentiment <b style='color:#34d399'>improves in Q&A</b> vs prepared remarks "
                            f"(shift: {div:+.4f}) — management holds up well under pressure")

    # FinBERT
    if signals.get("has_finbert"):
        pos_pct = signals["fb_positive_pct"]
        neg_pct = signals["fb_negative_pct"]
        if pos_pct > 0.4:
            findings.append(f"FinBERT classifies {pos_pct:.0%} of chunks as positive")
        elif neg_pct > 0.3:
            findings.append(f"FinBERT flags {neg_pct:.0%} of chunks as negative")

    bullets = "".join(f"<li style='margin:4px 0'>{f}</li>" for f in findings)
    st.markdown(
        f'<div style="background:#181825;border:1px solid #4285f440;border-radius:12px;'
        f'padding:20px 24px;margin:8px 0 20px 0">'
        f'<div style="color:#f1f5f9;font-size:16px;font-weight:700;margin-bottom:8px">'
        f'Key Findings — {ticker} {quarter}</div>'
        f'<ul style="color:#c6cdd5;font-size:14px;margin:0;padding-left:20px">{bullets}</ul>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_scorecard(signals):
    """Compact sentiment scorecard replacing histograms."""
    def _card(label, score, subtitle=""):
        tone, color = _tone_label(score)
        sub_html = f'<div style="color:#64748b;font-size:11px;margin-top:2px">{subtitle}</div>' if subtitle else ""
        return (
            f'<div style="background:#181825;border-radius:10px;padding:16px 20px;text-align:center;'
            f'border:1px solid {color}30">'
            f'<div style="color:#94a3b8;font-size:12px;letter-spacing:0.5px">{label}</div>'
            f'<div style="color:{color};font-size:28px;font-weight:800;margin:4px 0">{tone}</div>'
            f'<div style="color:#c6cdd5;font-size:14px">{score:.4f}</div>'
            f'{sub_html}'
            f'</div>'
        )

    cols = st.columns(4)

    # Overall (LM)
    cols[0].markdown(_card("Overall Tone", signals["lm_net"], "Loughran-McDonald"), unsafe_allow_html=True)

    # VADER
    cols[1].markdown(_card("General Sentiment", signals["vader"], "VADER"), unsafe_allow_html=True)

    # FinBERT
    if signals.get("has_finbert"):
        pos = signals["fb_positive_pct"]
        neg = signals["fb_negative_pct"]
        fb_score = pos - neg  # net positive ratio
        fb_sub = f"{pos:.0%} pos / {neg:.0%} neg"
        cols[2].markdown(_card("Deep Sentiment", fb_score, f"FinBERT — {fb_sub}"), unsafe_allow_html=True)
    else:
        cols[2].markdown(_card("Deep Sentiment", 0, "FinBERT unavailable"), unsafe_allow_html=True)

    # Uncertainty
    if "lm_uncertainty_score" in signals:
        cols[3].markdown(_card("Uncertainty", signals["lm_uncertainty_score"], "LM Uncertainty Words"),
                         unsafe_allow_html=True)
    else:
        # Use the raw column if available
        cols[3].markdown(
            f'<div style="background:#181825;border-radius:10px;padding:16px 20px;text-align:center;'
            f'border:1px solid #94a3b830">'
            f'<div style="color:#94a3b8;font-size:12px;letter-spacing:0.5px">Chunks Analyzed</div>'
            f'<div style="color:#f1f5f9;font-size:28px;font-weight:800;margin:4px 0">{signals["n_chunks"]}</div>'
            f'<div style="color:#64748b;font-size:11px;margin-top:2px">{signals["n_speakers"]} speakers</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_mgmt_vs_analyst(signals, analyzed_df, kp=""):
    """Management vs Analyst sentiment gap visualization."""
    mgmt_s = signals["mgmt_sent"]
    analyst_s = signals["analyst_sent"]
    gap = signals["mgmt_analyst_gap"]

    if pd.isna(mgmt_s) or pd.isna(analyst_s):
        return

    st.markdown("#### Management vs Analyst Sentiment")
    st.caption("The #1 predictor of post-earnings stock movement (p < 0.001)")

    # Bullet chart: two horizontal bars side by side
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=["Analysts", "Management"],
        x=[analyst_s, mgmt_s],
        orientation="h",
        marker=dict(
            color=["#ff9900", "#4285f4"],
            line=dict(width=0),
        ),
        text=[f"{analyst_s:.4f}", f"{mgmt_s:.4f}"],
        textposition="outside",
        textfont=dict(size=14, color="#c6cdd5"),
        width=0.5,
    ))

    # Add zero line and gap annotation
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_width=1)

    gap_color = "#fbbf24" if gap > 0.01 else "#34d399" if gap < -0.01 else "#94a3b8"
    gap_label = "OVEROPTIMISM" if gap > 0.01 else "ANALYST CONFIDENCE" if gap < -0.01 else "ALIGNED"

    fig.add_annotation(
        x=max(mgmt_s, analyst_s) + 0.005,
        y=0.5,
        text=f"<b>Gap: {gap:+.4f}</b><br><span style='font-size:11px'>{gap_label}</span>",
        showarrow=False,
        font=dict(color=gap_color, size=13),
        xanchor="left",
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=180,
        margin=dict(l=100, r=150, t=10, b=10),
        xaxis=dict(title="LM Net Score", gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0)"),
    )
    st.plotly_chart(styled_plotly(fig), use_container_width=True, key=f"mgmt_analyst_{kp}")


def _render_credibility_waterfall(signals, kp=""):
    """Prepared Remarks → Q&A sentiment waterfall showing credibility shift."""
    prep = signals["prepared_sent"]
    qa = signals["qa_sent"]
    div = signals["pq_divergence"]

    if pd.isna(prep) or pd.isna(qa):
        return

    st.markdown("#### Prepared Remarks vs Q&A Credibility")
    st.caption("Sentiment shift under analyst questioning (p = 0.002)")

    shift_color = "#ef4444" if div > 0.005 else "#34d399" if div < -0.005 else "#94a3b8"
    shift_label = "drops" if div > 0.005 else "improves" if div < -0.005 else "holds steady"

    fig = go.Figure(go.Waterfall(
        name="Sentiment",
        orientation="v",
        x=["Prepared Remarks", "Shift under Q&A", "Q&A Section"],
        y=[prep, -div, 0],
        measure=["absolute", "relative", "total"],
        text=[f"{prep:.4f}", f"{-div:+.4f}", f"{qa:.4f}"],
        textposition="outside",
        textfont=dict(size=13, color="#c6cdd5"),
        connector=dict(line=dict(color="rgba(255,255,255,0.15)", width=1)),
        increasing=dict(marker=dict(color="#34d399")),
        decreasing=dict(marker=dict(color="#ef4444")),
        totals=dict(marker=dict(color="#4285f4")),
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=320,
        margin=dict(t=10, b=40),
        yaxis=dict(title="LM Net Score"),
        showlegend=False,
    )

    st.plotly_chart(styled_plotly(fig), use_container_width=True, key=f"waterfall_{kp}")

    # Interpretation
    st.markdown(
        f'<div style="color:#94a3b8;font-size:13px;text-align:center;margin-top:-8px">'
        f'Tone <b style="color:{shift_color}">{shift_label}</b> from prepared remarks to Q&A '
        f'(shift: {-div:+.4f})</div>',
        unsafe_allow_html=True,
    )


def _show_single_quarter_sentiment(analyzed_df, ticker, quarter, key_prefix=None):
    """Render single-quarter sentiment details."""
    kp = key_prefix or quarter  # unique prefix for chart keys

    signals = _compute_signals(analyzed_df)
    signals["n_chunks"] = len(analyzed_df)
    signals["n_speakers"] = analyzed_df["speaker"].nunique()

    # Check for uncertainty column
    if "lm_uncertainty_score" in analyzed_df.columns:
        signals["lm_uncertainty_score"] = analyzed_df["lm_uncertainty_score"].mean()

    # ── Summary box ──
    _render_summary_box(signals, ticker, quarter)

    # ── Scorecard ──
    _render_scorecard(signals)

    st.markdown("")  # spacing

    # ── Management vs Analyst ──
    # ── Credibility Waterfall ──
    col1, col2 = st.columns(2)
    with col1:
        _render_mgmt_vs_analyst(signals, analyzed_df, kp=kp)
    with col2:
        _render_credibility_waterfall(signals, kp=kp)

    # ── Speaker breakdown ──
    st.divider()
    st.markdown("#### Sentiment by Speaker")

    speaker_data = analyzed_df.groupby(["speaker", "role"])["lm_net_score"].mean().reset_index()
    speaker_data = speaker_data.sort_values("lm_net_score", ascending=True)

    if len(speaker_data) > 1:
        ROLE_COLORS = {
            "CEO": "#ef4444", "CFO": "#22d3ee", "COO": "#a78bfa",
            "CTO": "#34d399", "VP": "#fbbf24", "IR": "#8b5cf6",
            "Operator": "#6ee7b7", "Analyst": "#ff9900", "Management": "#4285f4",
            "Other": "#64748b", "Unknown": "#64748b",
        }

        bar_colors = [ROLE_COLORS.get(r, "#94a3b8") for r in speaker_data["role"]]

        fig_speaker = go.Figure(go.Bar(
            x=speaker_data["lm_net_score"].values,
            y=[f"{r['speaker']} ({r['role']})" for _, r in speaker_data.iterrows()],
            orientation="h",
            marker=dict(color=bar_colors),
            text=[f"{v:.4f}" for v in speaker_data["lm_net_score"]],
            textposition="outside",
            textfont=dict(size=11, color="#c6cdd5"),
        ))
        fig_speaker.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_width=1)
        fig_speaker.update_layout(
            template=PLOTLY_TEMPLATE,
            height=max(250, len(speaker_data) * 32),
            margin=dict(l=180, r=80, t=10, b=30),
            xaxis=dict(title="LM Net Score"),
        )
        st.plotly_chart(styled_plotly(fig_speaker), use_container_width=True, key=f"speaker_{kp}")

    # ── Most positive / negative chunks ──
    st.divider()
    st.markdown('<span style="color:#34d399">green = positive</span> &nbsp; '
                '<span style="color:#ef4444">red = negative</span> &nbsp; '
                '<span style="color:#fbbf24">yellow = uncertainty</span> &nbsp; '
                '(Loughran-McDonald lexicon)',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Most Positive Chunks**")
        for _, row in analyzed_df.nlargest(3, "lm_net_score").iterrows():
            st.markdown(f"Score: **{row['lm_net_score']:.4f}** | {row['speaker']} ({row['role']})")
            highlighted = highlight_sentiment_words(row["text"])
            st.markdown(
                f'<div style="border-left:3px solid rgba(52,211,153,0.4);'
                f'background:#1e1e2e;border-radius:0 8px 8px 0;'
                f'padding:12px 16px;color:#c6cdd5;margin:8px 0;line-height:1.7">{highlighted}</div>',
                unsafe_allow_html=True)
    with c2:
        st.markdown("**Most Negative Chunks**")
        for _, row in analyzed_df.nsmallest(3, "lm_net_score").iterrows():
            st.markdown(f"Score: **{row['lm_net_score']:.4f}** | {row['speaker']} ({row['role']})")
            highlighted = highlight_sentiment_words(row["text"])
            st.markdown(
                f'<div style="border-left:3px solid rgba(239,68,68,0.4);'
                f'background:#1e1e2e;border-radius:0 8px 8px 0;'
                f'padding:12px 16px;color:#c6cdd5;margin:8px 0;line-height:1.7">{highlighted}</div>',
                unsafe_allow_html=True)

    # ── Raw distributions (collapsed) ──
    with st.expander("Raw Score Distributions", expanded=False):
        has_finbert = signals["has_finbert"]
        if has_finbert:
            col_lm, col_vader, col_finbert = st.columns(3)
        else:
            col_lm, col_vader = st.columns(2)

        with col_lm:
            fig_lm = go.Figure()
            fig_lm.add_trace(go.Histogram(x=analyzed_df["lm_net_score"], marker_color="#4285f4", nbinsx=20))
            fig_lm.update_layout(template=PLOTLY_TEMPLATE, title="LM Net Score",
                                 xaxis_title="Score", yaxis_title="Count", height=280)
            st.plotly_chart(styled_plotly(fig_lm), use_container_width=True, key=f"hist_lm_{kp}")

        with col_vader:
            fig_vader = go.Figure()
            fig_vader.add_trace(go.Histogram(x=analyzed_df["vader_compound"], marker_color="#ff9900", nbinsx=20))
            fig_vader.update_layout(template=PLOTLY_TEMPLATE, title="VADER Compound",
                                    xaxis_title="Score", yaxis_title="Count", height=280)
            st.plotly_chart(styled_plotly(fig_vader), use_container_width=True, key=f"hist_vader_{kp}")

        if has_finbert:
            with col_finbert:
                label_counts = analyzed_df["finbert_label"].value_counts()
                colors = {"positive": "#34d399", "negative": "#ef4444", "neutral": "#94a3b8", "error": "#64748b"}
                fig_fb = go.Figure(go.Bar(
                    x=label_counts.index,
                    y=label_counts.values,
                    marker_color=[colors.get(l, "#94a3b8") for l in label_counts.index],
                ))
                fig_fb.update_layout(template=PLOTLY_TEMPLATE, title="FinBERT Labels",
                                     xaxis_title="Label", yaxis_title="Count", height=280)
                st.plotly_chart(styled_plotly(fig_fb), use_container_width=True, key=f"hist_fb_{kp}")


# ═══════════════════════════════════════════════════
# MULTI-QUARTER MODE
# ═══════════════════════════════════════════════════
if is_multi_mode():
    ticker = st.session_state.get("multi_ticker", "")
    quarter_order = st.session_state.get("multi_quarter_order", [])
    multi_quarters = st.session_state.get("multi_quarters", {})

    st.caption(f"{ticker} | {len(quarter_order)} quarters: {', '.join(quarter_order)}")

    # Compute signals per quarter
    all_signals = {}
    for q in quarter_order:
        adf = multi_quarters[q].get("analyzed_df")
        if adf is None:
            continue
        all_signals[q] = _compute_signals(adf)

    if not all_signals:
        st.error("No analyzed data available.")
        st.stop()

    # ── Multi-quarter summary box ──
    latest_q = quarter_order[-1]
    latest_sig = all_signals[latest_q]

    findings = []
    tone_label, tone_color = _tone_label(latest_sig["lm_net"])
    findings.append(f"Latest quarter ({latest_q}) tone is "
                    f"<b style='color:{tone_color}'>{tone_label.lower()}</b> "
                    f"(LM: {latest_sig['lm_net']:.4f})")

    # Sentiment momentum
    if len(all_signals) >= 2:
        prev_q = quarter_order[-2]
        prev_sig = all_signals[prev_q]
        momentum = latest_sig["lm_net"] - prev_sig["lm_net"]
        mom_color = "#34d399" if momentum > 0 else "#ef4444"
        findings.append(f"Sentiment <b style='color:{mom_color}'>{'improved' if momentum > 0 else 'declined'}</b> "
                        f"vs {prev_q} (change: {momentum:+.4f}) — sentiment momentum is the #2 return predictor")

    # Mgmt-analyst gap trend
    gaps = [(q, s["mgmt_analyst_gap"]) for q, s in all_signals.items() if pd.notna(s.get("mgmt_analyst_gap"))]
    if len(gaps) >= 2:
        latest_gap = gaps[-1][1]
        if latest_gap > 0.01:
            findings.append(f"Management-analyst gap is widening ({latest_gap:+.4f}) — "
                            f"<b style='color:#fbbf24'>potential overoptimism</b>")
        elif latest_gap < -0.01:
            findings.append(f"Analysts more positive than management ({latest_gap:+.4f}) — "
                            f"<b style='color:#34d399'>external confidence signal</b>")

    bullets = "".join(f"<li style='margin:4px 0'>{f}</li>" for f in findings)
    st.markdown(
        f'<div style="background:#181825;border:1px solid #4285f440;border-radius:12px;'
        f'padding:20px 24px;margin:8px 0 20px 0">'
        f'<div style="color:#f1f5f9;font-size:16px;font-weight:700;margin-bottom:8px">'
        f'Cross-Quarter Insights — {ticker}</div>'
        f'<ul style="color:#c6cdd5;font-size:14px;margin:0;padding-left:20px">{bullets}</ul>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Trend charts ──
    q_list = list(all_signals.keys())
    col1, col2 = st.columns(2)

    with col1:
        # Mgmt vs Analyst gap trend
        gap_data = [(q, s["mgmt_sent"], s["analyst_sent"]) for q, s in all_signals.items()
                    if pd.notna(s.get("mgmt_sent")) and pd.notna(s.get("analyst_sent"))]
        if gap_data:
            gq, gm, ga = zip(*gap_data)
            fig_gap = go.Figure()
            fig_gap.add_trace(go.Scatter(x=list(gq), y=list(gm), mode="lines+markers",
                                         name="Management", marker=dict(size=10, color="#4285f4"),
                                         line=dict(color="#4285f4", width=2.5)))
            fig_gap.add_trace(go.Scatter(x=list(gq), y=list(ga), mode="lines+markers",
                                         name="Analysts", marker=dict(size=10, color="#ff9900"),
                                         line=dict(color="#ff9900", width=2.5)))
            fig_gap.update_layout(template=PLOTLY_TEMPLATE, title="Management vs Analyst Tone",
                                  yaxis_title="LM Net Score")
            st.plotly_chart(styled_plotly(fig_gap), use_container_width=True)

    with col2:
        # Prepared vs Q&A divergence trend
        div_data = [(q, s["prepared_sent"], s["qa_sent"]) for q, s in all_signals.items()
                    if pd.notna(s.get("prepared_sent")) and pd.notna(s.get("qa_sent"))]
        if div_data:
            dq, dp, dqa = zip(*div_data)
            fig_div = go.Figure()
            fig_div.add_trace(go.Scatter(x=list(dq), y=list(dp), mode="lines+markers",
                                         name="Prepared Remarks", marker=dict(size=10, color="#34d399"),
                                         line=dict(color="#34d399", width=2.5)))
            fig_div.add_trace(go.Scatter(x=list(dq), y=list(dqa), mode="lines+markers",
                                         name="Q&A Section", marker=dict(size=10, color="#ef4444"),
                                         line=dict(color="#ef4444", width=2.5)))
            fig_div.update_layout(template=PLOTLY_TEMPLATE, title="Prepared vs Q&A Credibility",
                                  yaxis_title="LM Net Score")
            st.plotly_chart(styled_plotly(fig_div), use_container_width=True)

    # Momentum delta bars
    if len(all_signals) >= 2:
        st.markdown("#### Sentiment Momentum (Quarter-over-Quarter Change)")
        lm_deltas = []
        for i in range(1, len(q_list)):
            delta = all_signals[q_list[i]]["lm_net"] - all_signals[q_list[i - 1]]["lm_net"]
            lm_deltas.append({"Quarter": q_list[i], "Delta": delta})
        delta_df = pd.DataFrame(lm_deltas)

        fig_mom = go.Figure(go.Bar(
            x=delta_df["Quarter"],
            y=delta_df["Delta"],
            marker_color=["#34d399" if d >= 0 else "#ef4444" for d in delta_df["Delta"]],
            text=[f"{d:+.4f}" for d in delta_df["Delta"]],
            textposition="outside",
            textfont=dict(size=13, color="#c6cdd5"),
            width=0.4,
        ))
        fig_mom.add_hline(y=0, line_color="rgba(255,255,255,0.2)")
        fig_mom.update_layout(template=PLOTLY_TEMPLATE, yaxis_title="Change in LM Net Score",
                              height=300)
        st.plotly_chart(styled_plotly(fig_mom), use_container_width=True)

    # Per-quarter tabs
    st.divider()
    st.subheader("Per-Quarter Detail")
    tabs = st.tabs(quarter_order)
    for tab, q in zip(tabs, quarter_order):
        with tab:
            adf = multi_quarters[q].get("analyzed_df")
            if adf is None:
                st.warning(f"No data for {q}")
                continue
            _show_single_quarter_sentiment(adf, ticker, q)

    st.stop()


# ═══════════════════════════════════════════════════
# SINGLE-QUARTER MODE
# ═══════════════════════════════════════════════════
analyzed_df = st.session_state.upload_analyzed_df
ticker = st.session_state.get("_analyzed_ticker", "")
quarter = st.session_state.get("_analyzed_quarter", "")

_show_single_quarter_sentiment(analyzed_df, ticker, quarter)
