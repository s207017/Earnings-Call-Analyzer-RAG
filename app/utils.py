"""Shared dashboard utilities."""

import json
import re
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent

COMPANIES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "META": "Meta", "NVDA": "NVIDIA",
    "TSLA": "Tesla", "NFLX": "Netflix", "CRM": "Salesforce", "ORCL": "Oracle",
}

COLORS = {
    "AAPL": "#94a3b8", "MSFT": "#4f7df5", "GOOGL": "#22d3ee",
    "AMZN": "#f97316", "META": "#8b5cf6", "NVDA": "#34d399",
    "TSLA": "#ef4444", "NFLX": "#ec4899", "CRM": "#22d3ee", "ORCL": "#f97316",
}

PLOTLY_TEMPLATE = "plotly_dark"

# ── Plotly dark layout matching the presentation theme ──
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#94a3b8"),
    title_font=dict(color="#f1f5f9", size=18),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
    colorway=["#4f7df5", "#8b5cf6", "#22d3ee", "#34d399", "#f97316", "#ef4444", "#ec4899",
              "#a78bfa", "#67e8f9", "#6ee7b7", "#fbbf24", "#fb923c"],
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=40, r=20, t=50, b=40),
)


def apply_theme():
    """Inject custom CSS from external file to match the presentation dark theme."""
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        css = css_path.read_text()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def styled_plotly(fig):
    """Apply the dark theme layout to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


@st.cache_data
def load_chunks():
    path = PROJECT_ROOT / "data" / "processed" / "all_chunks.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_sentiment():
    path = PROJECT_ROOT / "data" / "processed" / "chunks_with_sentiment.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_risk():
    path = PROJECT_ROOT / "data" / "processed" / "chunks_with_risk.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "risk_categories" in df.columns and df["risk_categories"].dtype == object:
            df["risk_categories"] = df["risk_categories"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        if "risk_details" in df.columns and df["risk_details"].dtype == object:
            df["risk_details"] = df["risk_details"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        return df
    return pd.DataFrame()


@st.cache_data
def load_market():
    path = PROJECT_ROOT / "data" / "market" / "market_reactions.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_features():
    path = PROJECT_ROOT / "outputs" / "feature_matrix.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_finance_results():
    path = PROJECT_ROOT / "outputs" / "finance_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ── Quarter parsing and gap detection helpers ──

def parse_quarter(q: str) -> Optional[Tuple[int, int]]:
    """Parse '2024Q3' into (2024, 3). Returns None if invalid."""
    m = re.match(r"(\d{4})Q([1-4])$", q.strip().upper())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def quarter_to_str(year: int, q: int) -> str:
    """Convert (2024, 3) to '2024Q3'."""
    return f"{year}Q{q}"


def next_quarter(year: int, q: int) -> Tuple[int, int]:
    """Return the next quarter after (year, q)."""
    if q == 4:
        return year + 1, 1
    return year, q + 1


def prev_quarter(year: int, q: int) -> Tuple[int, int]:
    """Return the quarter before (year, q)."""
    if q == 1:
        return year - 1, 4
    return year, q - 1


def find_missing_quarters(quarters: List[str]) -> List[str]:
    """Given a list of quarter strings like ['2024Q1', '2024Q3'], return missing ones in between.

    Returns list of missing quarter strings sorted chronologically.
    """
    parsed = []
    for q in quarters:
        p = parse_quarter(q)
        if p:
            parsed.append(p)
    if len(parsed) < 2:
        return []

    parsed.sort()
    missing = []
    for i in range(len(parsed) - 1):
        current = parsed[i]
        nxt = next_quarter(*current)
        while nxt < parsed[i + 1]:
            missing.append(quarter_to_str(*nxt))
            nxt = next_quarter(*nxt)
    return missing


def sort_quarters(quarters: List[str]) -> List[str]:
    """Sort quarter strings chronologically."""
    return sorted(quarters, key=lambda q: parse_quarter(q) or (0, 0))


def is_multi_mode() -> bool:
    """Check if the session is in multi-quarter mode."""
    return st.session_state.get("multi_mode", False)


def require_upload(page_name: str = "this page"):
    """Gate for sub-pages: show loading if analysis is in progress, or upload prompt if no data."""
    # Multi-mode: check that at least one quarter is analyzed
    if st.session_state.get("multi_mode"):
        mq = st.session_state.get("multi_quarters", {})
        if any(q.get("analyzed_df") is not None for q in mq.values()):
            return  # at least one quarter ready
    if st.session_state.get("upload_analyzed_df") is None:
        if st.session_state.get("upload_raw_text") is not None:
            st.info("Analysis is still in progress. Please return to the **Home** page and wait for it to finish.")
            st.spinner("Waiting for analysis to complete...")
        else:
            st.info(f"Upload an earnings call transcript on the **Home** page to view {page_name}.")
        st.stop()


def get_company_list():
    return list(COMPANIES.keys())


def get_quarter_list():
    return ["2023Q1", "2023Q2", "2023Q3", "2023Q4",
            "2024Q1", "2024Q2", "2024Q3", "2024Q4"]


def company_quarter_filter(sidebar=True):
    """Reusable company + quarter filter component."""
    container = st.sidebar if sidebar else st
    tickers = container.multiselect(
        "Companies", get_company_list(),
        default=get_company_list(),
        format_func=lambda x: f"{x} - {COMPANIES.get(x, x)}"
    )
    quarters = container.multiselect(
        "Quarters", get_quarter_list(), default=get_quarter_list()
    )
    return tickers, quarters


def fmt_pct(val, decimals=2):
    """Format as percentage string."""
    if pd.isna(val):
        return "N/A"
    return f"{val*100:.{decimals}f}%"


def highlight_sentiment_words(text: str, max_chars: int = 300) -> str:
    """Return HTML with LM positive/negative/uncertainty words color-highlighted."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "configs"))
    try:
        from loughran_mcdonald import POSITIVE_WORDS, NEGATIVE_WORDS, UNCERTAINTY_WORDS
    except ImportError:
        return text[:max_chars] + "..."

    display = text[:max_chars]
    words = display.split()
    result = []
    for word in words:
        clean = word.strip(".,;:!?()[]\"'").lower()
        if clean in NEGATIVE_WORDS:
            result.append(f'<span style="color:#ef4444;font-weight:600">{word}</span>')
        elif clean in POSITIVE_WORDS:
            result.append(f'<span style="color:#34d399;font-weight:600">{word}</span>')
        elif clean in UNCERTAINTY_WORDS:
            result.append(f'<span style="color:#fbbf24;font-weight:600">{word}</span>')
        else:
            result.append(word)
    return " ".join(result) + "..."


def highlight_risk_words(text: str, risk_details: list, max_chars: int = 400) -> str:
    """Return HTML with risk-triggering keywords highlighted, showing context around keywords."""
    import re

    if not risk_details:
        return text[:max_chars] + "..."

    SEVERITY_COLORS = {
        "high": "#ef4444",
        "medium": "#f97316",
        "low": "#fbbf24",
    }

    # Find the earliest keyword position to center the excerpt around it
    text_lower = text.lower()
    earliest_pos = len(text)
    for d in risk_details:
        kw = d["keyword_matched"].lower()
        idx = text_lower.find(kw)
        if idx != -1 and idx < earliest_pos:
            earliest_pos = idx

    # Build excerpt window centered on keywords
    if earliest_pos > max_chars:
        start = max(0, earliest_pos - 80)
        display = "..." + text[start:start + max_chars]
    else:
        display = text[:max_chars]

    # Highlight each keyword (longest first to avoid partial matches)
    seen = set()
    for d in sorted(risk_details, key=lambda x: -len(x["keyword_matched"])):
        kw = d["keyword_matched"]
        if kw.lower() in seen:
            continue
        seen.add(kw.lower())
        color = SEVERITY_COLORS.get(d.get("severity", "medium"), "#f97316")
        pattern = re.compile(
            r'(?<![a-zA-Z])(' + re.escape(kw) + r'(?:s|es|ed|ing)?)(?![a-zA-Z])',
            re.IGNORECASE,
        )
        display = pattern.sub(
            rf'<span style="color:{color};font-weight:700;text-decoration:underline">\1</span>',
            display,
        )
    return display + "..."


def highlight_transcript_chunk(text: str, risk_details: list = None,
                                finbert_sentences: list = None) -> str:
    """Highlight text with FinBERT sentence-level background + risk keyword background.

    - Positive sentences: green background
    - Negative sentences: red background
    - Risk keywords: orange/red underline + bold
    """
    import re

    clean = " ".join(text.split())

    # Step 1: FinBERT sentence-level background highlighting
    if finbert_sentences:
        for sent_info in finbert_sentences:
            sent_text = sent_info["sentence"].strip()
            label = sent_info["label"]
            score = sent_info["score"]

            if len(sent_text) < 15:
                continue

            # Only highlight sentences with strong sentiment
            if label == "positive" and score > 0.7:
                bg = "rgba(52,211,153,0.25)"
            elif label == "negative" and score > 0.7:
                bg = "rgba(239,68,68,0.25)"
            else:
                continue

            # Use plain string search instead of regex to avoid escape issues
            # Normalize the sentence text the same way as clean
            needle = " ".join(sent_text.split())
            idx = clean.lower().find(needle.lower()[:60])
            if idx == -1:
                continue

            # Find end of sentence from match point
            end_match = re.search(r'[.!?]', clean[idx + 60:])
            if end_match:
                end = idx + 60 + end_match.end()
            else:
                end = idx + len(needle)

            sentence_text = clean[idx:end]
            # Skip if this segment already contains HTML tags (already highlighted)
            if "<span" in sentence_text:
                continue
            highlighted = (f'<span style="background:{bg};padding:2px 4px;'
                         f'border-radius:4px">{sentence_text}</span>')
            clean = clean[:idx] + highlighted + clean[end:]

    # Step 2: Risk keyword highlighting (underline + bold, not background to avoid conflict)
    risk_placeholders = {}
    if risk_details:
        SEVERITY_COLORS = {
            "high": "#ef4444",
            "medium": "#f97316",
            "low": "#fbbf24",
        }
        seen_kw = set()
        for d in sorted(risk_details, key=lambda x: -len(x["keyword_matched"])):
            kw = d["keyword_matched"]
            if kw.lower() in seen_kw:
                continue
            seen_kw.add(kw.lower())
            color = SEVERITY_COLORS.get(d.get("severity", "medium"), "#f97316")
            pattern = re.compile(
                r'(?<![a-zA-Z])(' + re.escape(kw) + r'(?:s|es|ed|ing)?)(?![a-zA-Z])',
                re.IGNORECASE,
            )
            def make_replacer(col):
                def replacer(m):
                    token = f"__RISK_{len(risk_placeholders)}__"
                    risk_placeholders[token] = (
                        f'<span style="color:{col};font-weight:700;'
                        f'text-decoration:underline">{m.group(1)}</span>'
                    )
                    return token
                return replacer
            clean = pattern.sub(make_replacer(color), clean)

    # Step 3: Replace risk placeholders with actual HTML
    for token, html in risk_placeholders.items():
        clean = clean.replace(token, html)

    return clean


def fmt_risk_category(cat: str) -> str:
    """Format risk category for display: 'labor_risk' -> 'Labor Risk'."""
    return cat.replace("_", " ").title()


def fmt_risk_categories(cats: list) -> str:
    """Format a list of risk categories for display."""
    return ", ".join(fmt_risk_category(c) for c in cats)
