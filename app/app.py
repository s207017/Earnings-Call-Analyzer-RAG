"""Earnings Call Analyzer — Upload & Analyze (Entry Point)."""

import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Earnings Call Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import apply_theme

apply_theme()


# ── Auto-detect ticker and quarter from text ──
def detect_ticker_and_quarter(text: str, filename: str) -> tuple:
    detected_ticker = ""
    detected_quarter = ""
    text_upper = text[:3000].upper()
    filename_upper = filename.upper()

    name_to_ticker = {
        "APPLE": "AAPL", "MICROSOFT": "MSFT", "ALPHABET": "GOOGL", "GOOGLE": "GOOGL",
        "AMAZON": "AMZN", "META PLATFORMS": "META", "FACEBOOK": "META",
        "NVIDIA": "NVDA", "TESLA": "TSLA", "NETFLIX": "NFLX",
        "SALESFORCE": "CRM", "ORACLE": "ORCL", "AMD": "AMD",
        "ADVANCED MICRO DEVICES": "AMD", "CISCO": "CSCO",
        "IBM": "IBM", "ADOBE": "ADBE", "ACCENTURE": "ACN",
        "PAYPAL": "PYPL", "MASTERCARD": "MA", "AMERICAN EXPRESS": "AXP",
        "JPMORGAN": "JPM", "BANK OF AMERICA": "BAC", "CITIGROUP": "C",
        "DEUTSCHE BANK": "DB", "COSTCO": "COST", "WALMART": "WMT",
        "NIKE": "NKE", "LULULEMON": "LULU", "WALT DISNEY": "DIS", "DISNEY": "DIS",
        "MARRIOTT": "MAR", "BOOKING": "BKNG", "FORD": "F",
        "GENERAL MOTORS": "GM", "ALLSTATE": "ALL",
        "CARDINAL HEALTH": "CAH", "ELEVANCE": "ELV", "UNITEDHEALTH": "UNH",
        "BP": "BP", "SHELL": "SHEL", "BROOKFIELD": "BAM",
        "VOLVO": "VOLVY", "BMW": "BMW.DE", "SIEMENS": "SIE.DE",
        "BBVA": "BBVA", "AIRBUS": "AIR.PA", "AMADEUS": "AMS.PA",
        "CAPGEMINI": "CAP.PA", "L'OREAL": "OR.PA", "LOREAL": "OR.PA",
        "LVMH": "MC.PA", "SCHNEIDER ELECTRIC": "SCHNEIDER.PA",
        "ENGIE": "ENGI.PA", "EDF": "EDF.PA",
    }

    for name, ticker in sorted(name_to_ticker.items(), key=lambda x: -len(x[0])):
        if name in text_upper:
            detected_ticker = ticker
            break

    ticker_pattern = re.search(r"(?:NASDAQ|NYSE|TICKER)[:\s]+([A-Z]{1,5})", text_upper)
    if ticker_pattern:
        detected_ticker = ticker_pattern.group(1)

    fn_ticker = re.search(r"^([A-Z]{1,5})[-_ ]", filename_upper)
    if fn_ticker and not detected_ticker:
        detected_ticker = fn_ticker.group(1)
    for name, ticker in name_to_ticker.items():
        if name.replace(" ", "").replace("'", "") in filename_upper.replace(" ", "").replace("-", "").replace("_", ""):
            detected_ticker = ticker
            break

    ordinal_map = {
        "FIRST": "1", "SECOND": "2", "THIRD": "3", "FOURTH": "4",
        "1ST": "1", "2ND": "2", "3RD": "3", "4TH": "4",
    }

    q_match = re.search(r"Q(\d)\s*[-/]?\s*(20\d{2})", text_upper)
    if q_match:
        detected_quarter = f"{q_match.group(2)}Q{q_match.group(1)}"

    if not detected_quarter:
        q_match = re.search(r"(20\d{2})\s*[-/]?\s*Q(\d)", text_upper)
        if q_match:
            detected_quarter = f"{q_match.group(1)}Q{q_match.group(2)}"

    if not detected_quarter:
        for word, num in ordinal_map.items():
            pattern = re.search(
                rf"{word}\s+QUARTER\s+(?:OF\s+)?(?:FISCAL\s+(?:YEAR\s+)?)?(20\d{{2}})",
                text_upper,
            )
            if pattern:
                detected_quarter = f"{pattern.group(1)}Q{num}"
                break

    if not detected_quarter:
        fn_q = re.search(r"(20\d{2})\s*[-_]?\s*Q(\d)", filename_upper)
        if fn_q:
            detected_quarter = f"{fn_q.group(1)}Q{fn_q.group(2)}"
        else:
            fn_q = re.search(r"Q(\d)\s*[-_]?\s*(20\d{2})", filename_upper)
            if fn_q:
                detected_quarter = f"{fn_q.group(2)}Q{fn_q.group(1)}"

    return detected_ticker, detected_quarter


def _extract_text(file_bytes, file_name):
    if file_name.lower().endswith(".pdf"):
        try:
            import pdfplumber
            import io
            pages = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
            return "\n\n".join(pages)
        except ImportError:
            st.error("pdfplumber is required for PDF upload. Install with: pip install pdfplumber")
            return None
    else:
        return file_bytes.decode("utf-8", errors="replace")


_PARSER_VERSION = 5  # bump to invalidate cache after parser changes

@st.cache_data
def parse_and_analyze(text, ticker, quarter, _version=_PARSER_VERSION):
    """Parse transcript into chunks and run sentiment + risk analysis."""
    from src.agents.transcript_ingestion import TranscriptParser
    from src.agents.sentiment_analysis import SentimentAnalyzer
    from src.agents.risk_detection import RiskDetector

    parser = TranscriptParser()
    chunks = parser.parse_transcript(text, ticker, quarter)
    if not chunks:
        return pd.DataFrame(), pd.DataFrame()

    chunks_df = pd.DataFrame(chunks)

    analyzer = SentimentAnalyzer()
    texts = chunks_df["text"].tolist()

    lm_results = [analyzer.loughran_mcdonald_sentiment(t) for t in texts]
    lm_df = pd.DataFrame(lm_results)

    vader_results = [analyzer.vader_sentiment(t) for t in texts]
    vader_df = pd.DataFrame(vader_results)

    sent_df = pd.concat([chunks_df.reset_index(drop=True), lm_df, vader_df], axis=1)

    detector = RiskDetector()
    all_categories, all_counts, all_intensities, all_details = [], [], [], []

    for _, row in sent_df.iterrows():
        detections = detector.detect_risks(row["text"])
        active = [d for d in detections if not d["negated"]]
        categories = list(set(d["category"] for d in active))
        count = len(active)
        severity_weights = {"high": 3, "medium": 2, "low": 1}
        intensity = sum(severity_weights.get(d["severity"], 1) for d in active)
        all_categories.append(categories)
        all_counts.append(count)
        all_intensities.append(intensity)
        all_details.append(active)

    sent_df["risk_categories"] = all_categories
    sent_df["risk_count"] = all_counts
    sent_df["risk_intensity"] = all_intensities
    sent_df["risk_details"] = all_details

    return chunks_df, sent_df


# ── Session state initialization ──
for key in ["upload_raw_text", "upload_filename", "upload_ticker", "upload_quarter",
            "upload_chunks_df", "upload_analyzed_df", "upload_qa_messages", "upload_rag_engine"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ══════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════
st.title("Earnings Call Analyzer")
st.caption("Upload a transcript to get started with sentiment analysis, risk detection, and RAG-based Q&A")

st.divider()

# ── Upload ──
uploaded_file = st.file_uploader(
    "Upload a PDF or TXT earnings call transcript",
    type=["pdf", "txt"],
    help="Supports Motley Fool, Seeking Alpha, and company IR PDF formats",
)

if uploaded_file is not None:
    if (st.session_state.upload_filename != uploaded_file.name
            or st.session_state.upload_raw_text is None):
        raw_text = _extract_text(uploaded_file.getvalue(), uploaded_file.name)
        if raw_text and len(raw_text) >= 100:
            st.session_state.upload_raw_text = raw_text
            st.session_state.upload_filename = uploaded_file.name
            auto_ticker, auto_quarter = detect_ticker_and_quarter(raw_text, uploaded_file.name)
            st.session_state.upload_ticker = auto_ticker
            st.session_state.upload_quarter = auto_quarter
            st.session_state.upload_chunks_df = None
            st.session_state.upload_analyzed_df = None
            st.session_state.upload_qa_messages = []
            st.session_state.upload_rag_engine = None
        else:
            st.error("Could not extract sufficient text from the uploaded file.")
            st.stop()


# ── Nothing uploaded yet ──
if st.session_state.upload_raw_text is None:
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader("Sentiment")
        st.markdown("Loughran-McDonald, VADER, and FinBERT scoring per chunk")
    with col2:
        st.subheader("Risk Detection")
        st.markdown("255 keywords across 10 financial risk categories")
    with col3:
        st.subheader("RAG Q&A")
        st.markdown("Ask questions answered by the transcript with source citations")
    with col4:
        st.subheader("Benchmark")
        st.markdown("Compare against 1,100+ historical earnings calls")
    st.stop()


# ══════════════════════════════════════════════════
# TRANSCRIPT LOADED
# ══════════════════════════════════════════════════
raw_text = st.session_state.upload_raw_text

st.success(f"Loaded: **{st.session_state.upload_filename}** ({len(raw_text):,} characters)")

auto_ticker = st.session_state.upload_ticker or ""
auto_quarter = st.session_state.upload_quarter or ""

if auto_ticker or auto_quarter:
    st.info(f"Auto-detected: **{auto_ticker or '?'}** | **{auto_quarter or '?'}** — confirm or edit below.")

col1, col2, col3 = st.columns([2, 2, 1])
ticker_input = col1.text_input("Ticker Symbol", value=auto_ticker,
                                placeholder="e.g. AAPL").strip().upper()
quarter_input = col2.text_input("Quarter", value=auto_quarter,
                                 placeholder="e.g. 2024Q3").strip().upper()

if col3.button("Clear Transcript"):
    for key in ["upload_raw_text", "upload_filename", "upload_ticker", "upload_quarter",
                "upload_chunks_df", "upload_analyzed_df", "upload_qa_messages", "upload_rag_engine"]:
        st.session_state[key] = None
    st.rerun()

if not ticker_input or not quarter_input:
    st.warning("Please confirm or enter the ticker symbol and quarter to proceed.")
    st.stop()


# ── Parse & Analyze ──
if (st.session_state.upload_analyzed_df is not None
        and st.session_state.upload_chunks_df is not None
        and st.session_state.get("_analyzed_ticker") == ticker_input
        and st.session_state.get("_analyzed_quarter") == quarter_input):
    chunks_df = st.session_state.upload_chunks_df
    analyzed_df = st.session_state.upload_analyzed_df
else:
    with st.spinner("Parsing transcript and running analysis..."):
        chunks_df, analyzed_df = parse_and_analyze(raw_text, ticker_input, quarter_input)
    st.session_state.upload_chunks_df = chunks_df
    st.session_state.upload_analyzed_df = analyzed_df
    st.session_state["_analyzed_ticker"] = ticker_input
    st.session_state["_analyzed_quarter"] = quarter_input

if analyzed_df.empty:
    st.error("Could not parse any chunks from the transcript. The format may not be recognized.")
    st.stop()

# ══════════════════════════════════════════════════
# SUMMARY DASHBOARD
# ══════════════════════════════════════════════════
st.divider()
st.subheader(f"Analysis Summary — {ticker_input} {quarter_input}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Chunks", len(analyzed_df))
col2.metric("Speakers", analyzed_df["speaker"].nunique())
col3.metric("Avg Sentiment (LM)", f"{analyzed_df['lm_net_score'].mean():.4f}")
col4.metric("Risk Signals", int(analyzed_df["risk_count"].sum()))

# Section breakdown
col1, col2 = st.columns(2)
with col1:
    section_counts = analyzed_df["section"].value_counts()
    st.markdown("**Sections**")
    for section, count in section_counts.items():
        label = "Prepared Remarks" if section == "prepared_remarks" else "Q&A"
        st.markdown(f"- {label}: {count} chunks")

with col2:
    st.markdown("**Speakers**")
    speaker_counts = analyzed_df.groupby(["speaker", "role"]).size().reset_index(name="chunks")
    for _, row in speaker_counts.iterrows():
        st.markdown(f"- {row['speaker']} ({row['role']}): {row['chunks']} chunks")

st.markdown("---")
st.markdown("Use the **sidebar** to navigate to detailed Sentiment, Risk, Benchmark, Transcript, and Q&A pages.")
