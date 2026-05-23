"""Risk Detection — risk signal breakdown of the uploaded transcript."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import (apply_theme, PLOTLY_TEMPLATE, styled_plotly,
                   fmt_risk_category, fmt_risk_categories, highlight_risk_words, require_upload,
                   is_multi_mode)

apply_theme()

st.header("Risk Signal Detection")

require_upload("risk analysis")

MGMT_ROLES = {"CEO", "CFO", "COO", "CTO", "VP", "IR", "Management"}


# ── Helper: extract risk stats from a DataFrame ──
def _extract_risk_stats(analyzed_df):
    """Extract comprehensive risk statistics from analyzed data."""
    stats = {}

    # All categories
    all_cats = []
    for cats in analyzed_df["risk_categories"]:
        all_cats.extend(cats)
    stats["all_cats"] = all_cats
    stats["cat_counts"] = pd.Series(all_cats).value_counts() if all_cats else pd.Series(dtype=int)

    # Severity + detection method breakdown
    all_severities = []
    keyword_count = 0
    semantic_count = 0
    semantic_new_count = 0
    semantic_confirm_count = 0
    keyword_details = []
    semantic_details = []
    for details in analyzed_df["risk_details"]:
        for d in details:
            if d.get("detection_method") == "semantic":
                semantic_count += 1
                semantic_details.append(d)
                if d.get("is_confirmation"):
                    semantic_confirm_count += 1
                else:
                    semantic_new_count += 1
                    all_severities.append(d.get("severity", "medium"))
            else:
                keyword_count += 1
                keyword_details.append(d)
                all_severities.append(d.get("severity", "medium"))

    stats["all_severities"] = all_severities
    stats["sev_counts"] = pd.Series(all_severities).value_counts() if all_severities else pd.Series(dtype=int)
    stats["keyword_count"] = keyword_count
    stats["semantic_count"] = semantic_count
    stats["semantic_new_count"] = semantic_new_count
    stats["semantic_confirm_count"] = semantic_confirm_count
    stats["keyword_details"] = keyword_details
    stats["semantic_details"] = semantic_details

    # Semantic-only categories (found by RAG but not keywords)
    kw_cats = set(d.get("category") for d in keyword_details)
    sem_new_cats = set(d.get("category") for d in semantic_details if not d.get("is_confirmation"))
    sem_confirm_cats = set(d.get("category") for d in semantic_details if d.get("is_confirmation"))
    stats["semantic_only_cats"] = sem_new_cats - kw_cats
    stats["semantic_confirmed_cats"] = sem_confirm_cats

    # Management vs Analyst risk breakdown
    mgmt_df = analyzed_df[analyzed_df["role"].isin(MGMT_ROLES)]
    analyst_df = analyzed_df[analyzed_df["role"] == "Analyst"]

    mgmt_cats = []
    for cats in mgmt_df["risk_categories"]:
        mgmt_cats.extend(cats)
    analyst_cats = []
    for cats in analyst_df["risk_categories"]:
        analyst_cats.extend(cats)

    stats["mgmt_risk_count"] = int(mgmt_df["risk_count"].sum()) if not mgmt_df.empty else 0
    stats["analyst_risk_count"] = int(analyst_df["risk_count"].sum()) if not analyst_df.empty else 0
    stats["mgmt_cats"] = pd.Series(mgmt_cats).value_counts() if mgmt_cats else pd.Series(dtype=int)
    stats["analyst_cats"] = pd.Series(analyst_cats).value_counts() if analyst_cats else pd.Series(dtype=int)

    # Prepared vs Q&A risk breakdown
    prep_df = analyzed_df[analyzed_df["section"] == "prepared_remarks"]
    qa_df = analyzed_df[analyzed_df["section"] == "qa"]
    stats["prepared_risk_count"] = int(prep_df["risk_count"].sum()) if not prep_df.empty else 0
    stats["qa_risk_count"] = int(qa_df["risk_count"].sum()) if not qa_df.empty else 0

    return stats


# ── Key Findings summary box ──
def _render_risk_summary(stats, ticker, quarter):
    """Auto-generated key takeaway box."""
    findings = []
    total = stats["keyword_count"] + stats["semantic_count"]

    if total == 0:
        st.success("No risk signals detected.")
        return

    # Top risk categories
    top_cats = stats["cat_counts"].head(3)
    top_labels = [fmt_risk_category(c) for c in top_cats.index]
    findings.append(
        f"Top risk areas: <b style='color:#ef4444'>{', '.join(top_labels)}</b> "
        f"({total} total signals across {len(stats['cat_counts'])} categories)")

    # High severity count
    high_count = int(stats["sev_counts"].get("high", 0))
    if high_count > 0:
        findings.append(
            f"<b style='color:#ef4444'>{high_count} high-severity</b> signal{'s' if high_count != 1 else ''} "
            f"detected — flagged by intensifying language (e.g. \"significant\", \"major\", \"critical\")")

    # Management vs Analyst
    mgmt_r = stats["mgmt_risk_count"]
    analyst_r = stats["analyst_risk_count"]
    if mgmt_r > 0 or analyst_r > 0:
        if analyst_r > mgmt_r and analyst_r > 0:
            findings.append(
                f"Analysts raised <b style='color:#fbbf24'>more risk signals</b> ({analyst_r}) "
                f"than management ({mgmt_r}) — suggests external concerns management may be downplaying")
        elif mgmt_r > analyst_r * 1.5 and mgmt_r > 0:
            findings.append(
                f"Management flagged <b style='color:#34d399'>more risks</b> ({mgmt_r}) "
                f"than analysts ({analyst_r}) — proactive risk disclosure can signal transparency")

    # Prepared vs Q&A
    prep_r = stats["prepared_risk_count"]
    qa_r = stats["qa_risk_count"]
    if qa_r > prep_r and qa_r > 0:
        findings.append(
            f"More risks surfaced in <b style='color:#fbbf24'>Q&A</b> ({qa_r}) "
            f"than prepared remarks ({prep_r}) — analyst questioning uncovered additional concerns")

    # Semantic detection summary
    sem_only = stats["semantic_only_cats"]
    sem_confirm = stats["semantic_confirm_count"]
    if sem_only:
        sem_labels = [fmt_risk_category(c) for c in sorted(sem_only)]
        findings.append(
            f"RAG semantic detection found <b style='color:#8b5cf6'>{len(sem_only)} additional "
            f"risk categor{'y' if len(sem_only) == 1 else 'ies'}</b> missed by keywords: "
            f"{', '.join(sem_labels)}")
    if sem_confirm > 0:
        confirm_labels = [fmt_risk_category(c) for c in sorted(stats.get("semantic_confirmed_cats", set()))]
        findings.append(
            f"RAG <b style='color:#8b5cf6'>independently confirmed</b> {sem_confirm} keyword "
            f"detection{'s' if sem_confirm != 1 else ''} via embeddings: {', '.join(confirm_labels)}")

    bullets = "".join(f"<li style='margin:4px 0'>{f}</li>" for f in findings)
    st.markdown(
        f'<div style="background:#181825;border:1px solid #ef444440;border-radius:12px;'
        f'padding:20px 24px;margin:8px 0 20px 0">'
        f'<div style="color:#f1f5f9;font-size:16px;font-weight:700;margin-bottom:8px">'
        f'Key Findings — {ticker} {quarter}</div>'
        f'<ul style="color:#c6cdd5;font-size:14px;margin:0;padding-left:20px">{bullets}</ul>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Detection Method Breakdown ──
def _render_detection_breakdown(stats, kp=""):
    """Show keyword vs semantic (RAG) detection comparison."""
    kw = stats["keyword_count"]
    sem = stats["semantic_count"]
    total = kw + sem

    if total == 0:
        return

    st.markdown("#### Detection Method Breakdown")
    st.caption("Keyword matching uses the Loughran-McDonald financial risk lexicon; "
               "semantic detection uses RAG embeddings to catch risks expressed in novel language")

    col1, col2 = st.columns(2)

    with col1:
        # Stacked bar showing keyword vs semantic (new) vs semantic (confirmed) per category
        all_cats_set = set()
        kw_cat_counts = {}
        sem_new_cat_counts = {}
        sem_conf_cat_counts = {}
        for d in stats["keyword_details"]:
            cat = d.get("category", "unknown")
            all_cats_set.add(cat)
            kw_cat_counts[cat] = kw_cat_counts.get(cat, 0) + 1
        for d in stats["semantic_details"]:
            cat = d.get("category", "unknown")
            all_cats_set.add(cat)
            if d.get("is_confirmation"):
                sem_conf_cat_counts[cat] = sem_conf_cat_counts.get(cat, 0) + 1
            else:
                sem_new_cat_counts[cat] = sem_new_cat_counts.get(cat, 0) + 1

        sorted_cats = sorted(all_cats_set,
                             key=lambda c: (kw_cat_counts.get(c, 0) +
                                            sem_new_cat_counts.get(c, 0) +
                                            sem_conf_cat_counts.get(c, 0)),
                             reverse=True)
        cat_labels = [fmt_risk_category(c) for c in sorted_cats]
        kw_vals = [kw_cat_counts.get(c, 0) for c in sorted_cats]
        sem_new_vals = [sem_new_cat_counts.get(c, 0) for c in sorted_cats]
        sem_conf_vals = [sem_conf_cat_counts.get(c, 0) for c in sorted_cats]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=cat_labels, x=kw_vals, orientation="h",
            name="Keyword", marker_color="#4285f4",
        ))
        fig.add_trace(go.Bar(
            y=cat_labels, x=sem_new_vals, orientation="h",
            name="Semantic (new)", marker_color="#34d399",
        ))
        fig.add_trace(go.Bar(
            y=cat_labels, x=sem_conf_vals, orientation="h",
            name="Semantic (confirmed)", marker_color="#8b5cf6",
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, barmode="stack",
            title="Detections by Category & Method",
            xaxis_title="Count", height=max(280, len(sorted_cats) * 32),
            margin=dict(l=140, r=20, t=40, b=30),
            legend=dict(orientation="h", y=1.12),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(styled_plotly(fig), use_container_width=True, key=f"det_stack_{kp}")

    with col2:
        # Summary metrics
        sem_only = stats["semantic_only_cats"]
        sem_new = stats["semantic_new_count"]
        sem_confirm = stats["semantic_confirm_count"]
        sem_confirmed_cats = stats.get("semantic_confirmed_cats", set())

        st.markdown(
            f'<div style="background:#181825;border-radius:10px;padding:20px;'
            f'border:1px solid #8b5cf630;margin-top:8px">'
            f'<div style="color:#f1f5f9;font-size:14px;font-weight:700;margin-bottom:12px">'
            f'RAG Semantic Detection</div>'
            f'<div style="color:#c6cdd5;font-size:32px;font-weight:800;color:#8b5cf6">'
            f'{sem} <span style="font-size:16px;color:#94a3b8">detections</span></div>'
            f'<div style="color:#94a3b8;font-size:13px;margin-top:8px">'
            f'<span style="color:#34d399">{sem_new} new</span> categories keywords missed<br>'
            f'<span style="color:#8b5cf6">{sem_confirm} confirmations</span> of keyword findings</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if sem_only:
            sem_labels = [fmt_risk_category(c) for c in sorted(sem_only)]
            st.markdown(
                f'<div style="background:#181825;border-radius:10px;padding:16px;'
                f'border:1px solid #34d39930;margin-top:12px">'
                f'<div style="color:#34d399;font-size:13px;font-weight:600;margin-bottom:6px">'
                f'New risks found ONLY by RAG</div>'
                f'<div style="color:#c6cdd5;font-size:13px">{", ".join(sem_labels)}</div>'
                f'<div style="color:#64748b;font-size:11px;margin-top:4px">'
                f'These risks were expressed in language not in the keyword lexicon — '
                f'only semantic similarity detected them.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if sem_confirmed_cats:
            confirm_labels = [fmt_risk_category(c) for c in sorted(sem_confirmed_cats)]
            st.markdown(
                f'<div style="background:#181825;border-radius:10px;padding:16px;'
                f'border:1px solid #8b5cf620;margin-top:12px">'
                f'<div style="color:#8b5cf6;font-size:13px;font-weight:600;margin-bottom:6px">'
                f'Confirmed by RAG</div>'
                f'<div style="color:#c6cdd5;font-size:13px">{", ".join(confirm_labels)}</div>'
                f'<div style="color:#64748b;font-size:11px;margin-top:4px">'
                f'Keyword detections independently validated by semantic embeddings — '
                f'higher confidence these are genuine risk signals.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if sem == 0:
            st.markdown(
                f'<div style="background:#181825;border-radius:10px;padding:16px;'
                f'border:1px solid #64748b20;margin-top:12px">'
                f'<div style="color:#94a3b8;font-size:13px;font-weight:600;margin-bottom:6px">'
                f'No semantic detections</div>'
                f'<div style="color:#64748b;font-size:12px">'
                f'The semantic model did not flag additional risks beyond keyword matches. '
                f'This can happen when risk language is explicit and well-covered by the lexicon.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Management vs Analyst Risk Exposure ──
def _render_mgmt_vs_analyst_risk(stats, kp=""):
    """Show who is raising risk signals — management or analysts."""
    mgmt_r = stats["mgmt_risk_count"]
    analyst_r = stats["analyst_risk_count"]

    if mgmt_r == 0 and analyst_r == 0:
        return

    st.markdown("#### Management vs Analyst Risk Exposure")
    st.caption("When analysts raise more risks than management, it may signal concerns "
               "that leadership is not adequately addressing")

    col1, col2 = st.columns(2)

    with col1:
        # Combined category comparison
        all_cats_set = set(stats["mgmt_cats"].index) | set(stats["analyst_cats"].index)
        sorted_cats = sorted(all_cats_set,
                             key=lambda c: (stats["mgmt_cats"].get(c, 0) +
                                            stats["analyst_cats"].get(c, 0)),
                             reverse=True)
        cat_labels = [fmt_risk_category(c) for c in sorted_cats]
        mgmt_vals = [int(stats["mgmt_cats"].get(c, 0)) for c in sorted_cats]
        analyst_vals = [int(stats["analyst_cats"].get(c, 0)) for c in sorted_cats]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=cat_labels, x=mgmt_vals, orientation="h",
            name="Management", marker_color="#4285f4",
        ))
        fig.add_trace(go.Bar(
            y=cat_labels, x=analyst_vals, orientation="h",
            name="Analysts", marker_color="#ff9900",
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, barmode="group",
            title="Risk Signals by Source & Category",
            xaxis_title="Count", height=max(280, len(sorted_cats) * 40),
            margin=dict(l=140, r=20, t=40, b=30),
            legend=dict(orientation="h", y=1.12),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(styled_plotly(fig), use_container_width=True, key=f"mgmt_analyst_risk_{kp}")

    with col2:
        # Summary comparison
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=["Management", "Analysts"],
            y=[mgmt_r, analyst_r],
            marker_color=["#4285f4", "#ff9900"],
            text=[str(mgmt_r), str(analyst_r)],
            textposition="outside",
            textfont=dict(size=16, color="#c6cdd5"),
            width=0.4,
        ))
        fig_comp.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Total Risk Signals by Source",
            yaxis_title="Count", yaxis=dict(dtick=1),
            height=280,
        )
        st.plotly_chart(styled_plotly(fig_comp), use_container_width=True, key=f"mgmt_analyst_total_{kp}")

        # Analyst-only categories
        analyst_only_cats = set(stats["analyst_cats"].index) - set(stats["mgmt_cats"].index)
        if analyst_only_cats:
            labels = [fmt_risk_category(c) for c in sorted(analyst_only_cats)]
            st.markdown(
                f'<div style="background:#181825;border-radius:8px;padding:12px 16px;'
                f'border:1px solid #ff990030">'
                f'<div style="color:#ff9900;font-size:12px;font-weight:600;margin-bottom:4px">'
                f'Risks raised ONLY by analysts</div>'
                f'<div style="color:#c6cdd5;font-size:13px">{", ".join(labels)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _show_single_quarter_risk(analyzed_df, ticker, quarter, key_prefix=None):
    """Render single-quarter risk details."""
    kp = key_prefix or quarter
    st.caption(f"{ticker} {quarter} | {int(analyzed_df['risk_count'].sum())} risk signals detected")

    stats = _extract_risk_stats(analyzed_df)

    if not stats["all_cats"]:
        st.success("No risk signals detected.")
        return

    # ── Summary box ──
    _render_risk_summary(stats, ticker, quarter)

    # ── Risk categories bar chart ──
    cat_counts = stats["cat_counts"]
    cat_labels = [fmt_risk_category(c) for c in cat_counts.index]
    fig_risk = go.Figure(go.Bar(
        x=cat_labels, y=cat_counts.values,
        marker=dict(color=cat_counts.values, colorscale="YlOrRd"), width=0.4,
    ))
    fig_risk.update_layout(template=PLOTLY_TEMPLATE, title="Risk Categories Detected",
                           xaxis_title="Category", yaxis_title="Count", yaxis=dict(dtick=1))
    st.plotly_chart(styled_plotly(fig_risk), use_container_width=True, key=f"risk_cats_{kp}")

    # ── Severity metrics ──
    if stats["all_severities"]:
        sev_counts = stats["sev_counts"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("High Severity", int(sev_counts.get("high", 0)))
        c2.metric("Medium Severity", int(sev_counts.get("medium", 0)))
        c3.metric("Low Severity", int(sev_counts.get("low", 0)))
        c4.metric("Keyword", stats["keyword_count"])
        sem_delta = (f"{stats['semantic_new_count']} new, "
                     f"{stats['semantic_confirm_count']} confirmed")
        c5.metric("Semantic (RAG)", stats["semantic_count"], delta=sem_delta, delta_color="off")

    st.divider()

    # ── Detection Method Breakdown ──
    _render_detection_breakdown(stats, kp=kp)

    st.divider()

    # ── Management vs Analyst Risk ──
    _render_mgmt_vs_analyst_risk(stats, kp=kp)

    st.divider()

    # ── Top risk excerpts ──
    # Include chunks that have any risk_details (keyword, semantic new, or semantic confirmation)
    has_risk = analyzed_df[analyzed_df["risk_details"].apply(lambda d: len(d) > 0)]
    if has_risk.empty:
        return

    st.markdown("**Top Risk Excerpts**")
    risk_chunks = has_risk.nlargest(5, "risk_intensity")
    for _, row in risk_chunks.iterrows():
        details = row["risk_details"]
        kw_dets = [d for d in details if d.get("detection_method") != "semantic"]
        sem_new_dets = [d for d in details if d.get("detection_method") == "semantic" and not d.get("is_confirmation")]
        sem_conf_dets = [d for d in details if d.get("detection_method") == "semantic" and d.get("is_confirmation")]

        # Build method badges
        badges = []
        if kw_dets:
            badges.append(f'<span style="background:#4285f430;color:#4285f4;padding:2px 8px;'
                          f'border-radius:4px;font-size:11px">Keyword ({len(kw_dets)})</span>')
        if sem_new_dets:
            badges.append(f'<span style="background:#34d39930;color:#34d399;padding:2px 8px;'
                          f'border-radius:4px;font-size:11px">RAG new ({len(sem_new_dets)})</span>')
        if sem_conf_dets:
            badges.append(f'<span style="background:#8b5cf630;color:#8b5cf6;padding:2px 8px;'
                          f'border-radius:4px;font-size:11px">RAG confirmed ({len(sem_conf_dets)})</span>')
        badge_html = " ".join(badges)

        st.markdown(f"**Categories:** {fmt_risk_categories(row['risk_categories'])} | "
                    f"Intensity: {row['risk_intensity']}")
        st.markdown(f"_{row['speaker']} ({row['role']})_ &nbsp; {badge_html}",
                    unsafe_allow_html=True)

        # Highlight keyword matches in the text
        highlighted = highlight_risk_words(row["text"], row["risk_details"])

        # Build semantic detection annotations with WHY explanation
        sem_annotations = []
        for d in details:
            if d.get("detection_method") == "semantic":
                cat_label = fmt_risk_category(d["category"])
                anchor = d.get("matched_anchor", "")
                sim = d.get("risk_similarity", 0)
                is_conf = d.get("is_confirmation", False)

                if is_conf:
                    label_color = "#8b5cf6"
                    label_prefix = "RAG confirmed"
                    bg_color = "#8b5cf620"
                else:
                    label_color = "#34d399"
                    label_prefix = "RAG detected"
                    bg_color = "#34d39920"

                anchor_html = ""
                if anchor:
                    anchor_html = (
                        f'<div style="color:#64748b;font-size:11px;margin-top:4px;font-style:italic">'
                        f'Matched pattern: "{anchor}" '
                        f'(similarity: {sim:.2f})</div>')

                sem_annotations.append(
                    f'<div style="background:{bg_color};border-radius:6px;'
                    f'padding:8px 12px;margin:4px 0">'
                    f'<span style="color:{label_color};font-size:12px;font-weight:600">'
                    f'{label_prefix}: {cat_label}</span>'
                    f'{anchor_html}</div>')

        # Use purple left border if semantic-only, red if keyword, gradient if both
        has_kw = bool(kw_dets)
        has_sem_new = bool(sem_new_dets)
        if has_sem_new and not has_kw:
            border_color = "rgba(139,92,246,0.6)"  # purple for semantic-only
        elif has_sem_new and has_kw:
            border_color = "rgba(239,68,68,0.4)"   # red (keyword primary)
        else:
            border_color = "rgba(239,68,68,0.4)"   # red default

        st.markdown(
            f'<div style="border-left:3px solid {border_color};'
            f'background:#1e1e2e;border-radius:0 8px 8px 0;'
            f'padding:12px 16px;color:#c6cdd5;margin:8px 0;line-height:1.7">{highlighted}</div>',
            unsafe_allow_html=True)

        if sem_annotations:
            st.markdown(
                f'<div style="margin:-4px 0 8px 16px">{"".join(sem_annotations)}</div>',
                unsafe_allow_html=True)

        st.divider()


# ── Multi-quarter mode ──
if is_multi_mode():
    ticker = st.session_state.get("multi_ticker", "")
    quarter_order = st.session_state.get("multi_quarter_order", [])
    multi_quarters = st.session_state.get("multi_quarters", {})

    st.caption(f"{ticker} | {len(quarter_order)} quarters: {', '.join(quarter_order)}")

    # Risk category heatmap across quarters
    all_risk_cats = set()
    q_cat_counts = {}
    for q in quarter_order:
        adf = multi_quarters[q].get("analyzed_df")
        if adf is None:
            continue
        cats = []
        for c_list in adf["risk_categories"]:
            cats.extend(c_list)
        cat_series = pd.Series(cats)
        q_cat_counts[q] = cat_series.value_counts().to_dict() if len(cat_series) > 0 else {}
        all_risk_cats.update(q_cat_counts[q].keys())

    if all_risk_cats:
        sorted_cats = sorted(all_risk_cats)
        heatmap_data = []
        for cat in sorted_cats:
            row = [q_cat_counts.get(q, {}).get(cat, 0) for q in quarter_order]
            heatmap_data.append(row)

        fig_hm = go.Figure(go.Heatmap(
            z=heatmap_data,
            x=quarter_order,
            y=[fmt_risk_category(c) for c in sorted_cats],
            colorscale="YlOrRd",
            text=[[str(v) if v > 0 else "" for v in row] for row in heatmap_data],
            texttemplate="%{text}",
            hovertemplate="Quarter: %{x}<br>Category: %{y}<br>Count: %{z}<extra></extra>",
        ))
        fig_hm.update_layout(template=PLOTLY_TEMPLATE, title="Risk Category Heatmap Across Quarters",
                             xaxis_title="Quarter", yaxis_title="Risk Category",
                             height=max(400, len(sorted_cats) * 35))
        st.plotly_chart(styled_plotly(fig_hm), use_container_width=True)

    # Total risk trend
    risk_trend = []
    for q in quarter_order:
        adf = multi_quarters[q].get("analyzed_df")
        if adf is None:
            continue
        risk_trend.append({
            "Quarter": q,
            "Total Signals": int(adf["risk_count"].sum()),
            "Total Intensity": int(adf["risk_intensity"].sum()),
            "Chunks with Risk": int((adf["risk_count"] > 0).sum()),
        })

    if risk_trend:
        trend_df = pd.DataFrame(risk_trend)
        col1, col2 = st.columns(2)
        with col1:
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(x=trend_df["Quarter"], y=trend_df["Total Signals"],
                                       mode="lines+markers", name="Signals",
                                       marker=dict(size=10, color="#ef4444"),
                                       line=dict(color="#ef4444", width=2.5)))
            fig_t.update_layout(template=PLOTLY_TEMPLATE, title="Risk Signal Trend",
                                yaxis_title="Count")
            st.plotly_chart(styled_plotly(fig_t), use_container_width=True)
        with col2:
            fig_i = go.Figure()
            fig_i.add_trace(go.Scatter(x=trend_df["Quarter"], y=trend_df["Total Intensity"],
                                       mode="lines+markers", name="Intensity",
                                       marker=dict(size=10, color="#f97316"),
                                       line=dict(color="#f97316", width=2.5)))
            fig_i.update_layout(template=PLOTLY_TEMPLATE, title="Risk Intensity Trend",
                                yaxis_title="Intensity")
            st.plotly_chart(styled_plotly(fig_i), use_container_width=True)

        # Delta metrics
        if len(trend_df) >= 2:
            latest = trend_df.iloc[-1]
            prev = trend_df.iloc[-2]
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Signals ({latest['Quarter']})", latest["Total Signals"],
                      delta=f"{latest['Total Signals'] - prev['Total Signals']:+d} vs {prev['Quarter']}")
            c2.metric(f"Intensity ({latest['Quarter']})", latest["Total Intensity"],
                      delta=f"{latest['Total Intensity'] - prev['Total Intensity']:+d} vs {prev['Quarter']}")
            c3.metric(f"Risky Chunks ({latest['Quarter']})", latest["Chunks with Risk"],
                      delta=f"{latest['Chunks with Risk'] - prev['Chunks with Risk']:+d} vs {prev['Quarter']}")

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
            _show_single_quarter_risk(adf, ticker, q)
    st.stop()


# ── Single-quarter mode (original) ──
analyzed_df = st.session_state.upload_analyzed_df
ticker = st.session_state.get("_analyzed_ticker", "")
quarter = st.session_state.get("_analyzed_quarter", "")

_show_single_quarter_risk(analyzed_df, ticker, quarter)
