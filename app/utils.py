"""Shared dashboard utilities."""

import json
from pathlib import Path

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


def fmt_risk_category(cat: str) -> str:
    """Format risk category for display: 'labor_risk' -> 'Labor Risk'."""
    return cat.replace("_", " ").title()


def fmt_risk_categories(cats: list) -> str:
    """Format a list of risk categories for display."""
    return ", ".join(fmt_risk_category(c) for c in cats)
