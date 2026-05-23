"""Earnings Call Analyzer — Upload & Analyze (Entry Point)."""

import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Earnings Call Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import apply_theme, parse_quarter, find_missing_quarters, sort_quarters, quarter_to_str

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


_PARSER_VERSION = 10  # bump to invalidate cache after parser changes

@st.cache_data(show_spinner="Analyzing transcript — sentiment, risk detection, and NLP scoring...")
def analyze_transcript(text, ticker, quarter, _version=_PARSER_VERSION):
    """Parse transcript into chunks and run sentiment + risk analysis."""
    from src.agents.transcript_ingestion import TranscriptParser
    from src.agents.sentiment_analysis import SentimentAnalyzer
    from src.agents.risk_detection import RiskDetector, SemanticRiskDetector

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

    # FinBERT (transformer-based)
    try:
        finbert_results = analyzer.finbert_sentiment(texts)
        finbert_df = pd.DataFrame(finbert_results)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"FinBERT unavailable: {e}")
        finbert_df = pd.DataFrame()

    dfs = [chunks_df.reset_index(drop=True), lm_df, vader_df]
    if not finbert_df.empty:
        dfs.append(finbert_df)
    sent_df = pd.concat(dfs, axis=1)

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

    # Semantic risk detection (embedding-based) — merge with keyword results
    try:
        sem_detector = SemanticRiskDetector()
        sem_results = sem_detector.detect_semantic_risks(texts)
        for i, sem_dets in enumerate(sem_results):
            kw_cats = set(all_categories[i])
            for sd in sem_dets:
                if sd["category"] not in kw_cats:
                    # New category only found by semantic — add to counts
                    all_categories[i].append(sd["category"])
                    all_counts[i] += 1
                    all_intensities[i] += 2  # medium weight
                    all_details[i].append({
                        "category": sd["category"],
                        "keyword_matched": f"[semantic: {sd['differential']:.3f}]",
                        "severity": "medium",
                        "negated": False,
                        "context_snippet": texts[i][:300],
                        "detection_method": "semantic",
                        "matched_anchor": sd.get("matched_anchor", ""),
                        "risk_similarity": sd.get("risk_similarity", 0),
                    })
                else:
                    # Category already found by keyword — record as semantic confirmation
                    # (does NOT increase risk_count/intensity, but shows RAG also detected it)
                    all_details[i].append({
                        "category": sd["category"],
                        "keyword_matched": f"[semantic confirmation: {sd['differential']:.3f}]",
                        "severity": "medium",
                        "negated": False,
                        "context_snippet": texts[i][:300],
                        "detection_method": "semantic",
                        "is_confirmation": True,
                        "matched_anchor": sd.get("matched_anchor", ""),
                        "risk_similarity": sd.get("risk_similarity", 0),
                    })
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Semantic risk detection unavailable: {e}")

    sent_df["risk_categories"] = all_categories
    sent_df["risk_count"] = all_counts
    sent_df["risk_intensity"] = all_intensities
    sent_df["risk_details"] = all_details

    # Sentence-level FinBERT for highlighting
    try:
        sentence_results = analyzer.finbert_sentence_sentiment(texts)
        sent_df["finbert_sentences"] = sentence_results
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"FinBERT sentence scoring unavailable: {e}")
        sent_df["finbert_sentences"] = [[] for _ in range(len(sent_df))]

    return chunks_df, sent_df


# ── Session state initialization ──
for key in ["upload_raw_text", "upload_filename", "upload_ticker", "upload_quarter",
            "upload_chunks_df", "upload_analyzed_df", "upload_qa_messages", "upload_rag_engine",
            "_analyzing"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Multi-quarter state
if "multi_mode" not in st.session_state:
    st.session_state.multi_mode = False
if "multi_quarters" not in st.session_state:
    st.session_state.multi_quarters = {}  # {quarter_str: {raw_text, filename, chunks_df, analyzed_df}}
if "multi_quarter_order" not in st.session_state:
    st.session_state.multi_quarter_order = []  # sorted list of quarter strings
if "multi_ticker" not in st.session_state:
    st.session_state.multi_ticker = ""
if "multi_missing_quarters" not in st.session_state:
    st.session_state.multi_missing_quarters = []


# ══════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════
st.title("Earnings Call Analyzer")
st.caption("Upload a transcript to get started with sentiment analysis, risk detection, and RAG-based Q&A")

st.divider()

# ── Upload (supports single or multiple files) ──
uploaded_files = st.file_uploader(
    "Upload earnings call transcripts (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="Upload one transcript for single-quarter analysis, or multiple for cross-quarter trend analysis.",
)


def _clear_all_state():
    """Reset all upload and multi-quarter state."""
    for key in ["upload_raw_text", "upload_filename", "upload_ticker", "upload_quarter",
                "upload_chunks_df", "upload_analyzed_df", "upload_qa_messages", "upload_rag_engine"]:
        st.session_state[key] = None
    st.session_state.multi_mode = False
    st.session_state.multi_quarters = {}
    st.session_state.multi_quarter_order = []
    st.session_state.multi_ticker = ""
    st.session_state.multi_missing_quarters = []


def _process_uploaded_files(files):
    """Extract text and detect ticker/quarter for each uploaded file.

    Returns list of dicts: [{filename, raw_text, ticker, quarter}, ...]
    """
    results = []
    for f in files:
        raw_text = _extract_text(f.getvalue(), f.name)
        if not raw_text or len(raw_text) < 100:
            st.warning(f"Could not extract text from **{f.name}** — skipping.")
            continue
        ticker, quarter = detect_ticker_and_quarter(raw_text, f.name)
        results.append({
            "filename": f.name,
            "raw_text": raw_text,
            "ticker": ticker,
            "quarter": quarter,
        })
    return results


# ── Determine if new files were uploaded ──
if uploaded_files:
    current_names = sorted([f.name for f in uploaded_files])
    prev_names = st.session_state.get("_prev_upload_names", [])

    if current_names != prev_names:
        # New upload detected — process files
        st.session_state["_prev_upload_names"] = current_names
        with st.spinner("Reading and analyzing uploaded transcripts..."):
            file_infos = _process_uploaded_files(uploaded_files)

        if not file_infos:
            st.error("No valid transcripts found in uploaded files.")
            st.stop()

        if len(file_infos) == 1:
            # ── Single file mode ──
            info = file_infos[0]
            st.session_state.multi_mode = False
            st.session_state.multi_quarters = {}
            st.session_state.multi_quarter_order = []
            st.session_state.multi_ticker = ""
            st.session_state.multi_missing_quarters = []
            st.session_state.upload_raw_text = info["raw_text"]
            st.session_state.upload_filename = info["filename"]
            st.session_state.upload_ticker = info["ticker"]
            st.session_state.upload_quarter = info["quarter"]
            st.session_state.upload_chunks_df = None
            st.session_state.upload_analyzed_df = None
            st.session_state.upload_qa_messages = []
            st.session_state.upload_rag_engine = None
        else:
            # ── Multi-file mode ──
            # Validate: all files must have detectable tickers and quarters
            tickers_found = set()
            quarters_found = {}
            issues = []

            for info in file_infos:
                if not info["ticker"]:
                    issues.append(f"**{info['filename']}**: Could not detect ticker.")
                if not info["quarter"]:
                    issues.append(f"**{info['filename']}**: Could not detect quarter.")
                elif not parse_quarter(info["quarter"]):
                    issues.append(f"**{info['filename']}**: Invalid quarter format '{info['quarter']}'.")
                else:
                    if info["quarter"] in quarters_found:
                        issues.append(f"**{info['filename']}**: Duplicate quarter {info['quarter']} "
                                      f"(also in {quarters_found[info['quarter']]['filename']}).")
                    else:
                        quarters_found[info["quarter"]] = info
                if info["ticker"]:
                    tickers_found.add(info["ticker"])

            # Validate same company
            if len(tickers_found) > 1:
                issues.append(f"Multiple tickers detected: {', '.join(sorted(tickers_found))}. "
                              f"All transcripts must be from the same company.")

            if issues:
                st.error("Issues detected with uploaded files:")
                for issue in issues:
                    st.markdown(f"- {issue}")
                st.stop()

            # All valid — set multi-quarter state
            common_ticker = tickers_found.pop()
            quarter_list = sort_quarters(list(quarters_found.keys()))
            missing = find_missing_quarters(quarter_list)

            st.session_state.multi_mode = True
            st.session_state.multi_ticker = common_ticker
            st.session_state.multi_quarter_order = quarter_list
            st.session_state.multi_missing_quarters = missing
            st.session_state.multi_quarters = {}
            for q_str, info in quarters_found.items():
                st.session_state.multi_quarters[q_str] = {
                    "filename": info["filename"],
                    "raw_text": info["raw_text"],
                    "chunks_df": None,
                    "analyzed_df": None,
                }

            # Also set single-file state to latest quarter for backward compat with pages
            latest = quarter_list[-1]
            st.session_state.upload_raw_text = quarters_found[latest]["raw_text"]
            st.session_state.upload_filename = quarters_found[latest]["filename"]
            st.session_state.upload_ticker = common_ticker
            st.session_state.upload_quarter = latest
            st.session_state.upload_chunks_df = None
            st.session_state.upload_analyzed_df = None
            st.session_state.upload_qa_messages = []
            st.session_state.upload_rag_engine = None

        # Single file needs rerun to show ticker/quarter input fields
        # Multi-file flows directly into the analysis section below (no blank flash)
        if not st.session_state.multi_mode:
            st.rerun()
# (Don't clear state when uploader is empty — files don't persist across page navigation.
#  State is only cleared via the explicit "Clear" buttons.)


# ── Nothing uploaded yet ──
if st.session_state.upload_raw_text is None and not st.session_state.multi_mode:
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div style="background:#181825;border:1px solid rgba(66,133,244,.2);border-radius:12px;padding:24px">'
            '<div style="font-size:24px;margin-bottom:8px">📊</div>'
            '<div style="color:#f1f5f9;font-size:16px;font-weight:700;margin-bottom:8px">Sentiment</div>'
            '<div style="color:#94a3b8;font-size:13px;line-height:1.6">'
            'Three NLP models (LM, VADER, FinBERT) with management vs analyst gap, '
            'prepared vs Q&A credibility shift, and cross-quarter momentum tracking'
            '</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(
            '<div style="background:#181825;border:1px solid rgba(239,68,68,.2);border-radius:12px;padding:24px">'
            '<div style="font-size:24px;margin-bottom:8px">🛡️</div>'
            '<div style="color:#f1f5f9;font-size:16px;font-weight:700;margin-bottom:8px">Risk Detection</div>'
            '<div style="color:#94a3b8;font-size:13px;line-height:1.6">'
            '255 keywords + semantic RAG detection across 10 risk categories, '
            'with severity scoring, management vs analyst exposure, and explainable anchor matching'
            '</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(
            '<div style="background:#181825;border:1px solid rgba(139,92,246,.2);border-radius:12px;padding:24px">'
            '<div style="font-size:24px;margin-bottom:8px">💬</div>'
            '<div style="color:#f1f5f9;font-size:16px;font-weight:700;margin-bottom:8px">RAG Q&A</div>'
            '<div style="color:#94a3b8;font-size:13px;line-height:1.6">'
            'Hybrid FAISS + BM25 search with cross-encoder re-ranking. '
            'Ask questions and get cited answers powered by Llama 3.2 (local)'
            '</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(
            '<div style="background:#181825;border:1px solid rgba(52,211,153,.2);border-radius:12px;padding:24px">'
            '<div style="font-size:24px;margin-bottom:8px">📈</div>'
            '<div style="color:#f1f5f9;font-size:16px;font-weight:700;margin-bottom:8px">Benchmark</div>'
            '<div style="color:#94a3b8;font-size:13px;line-height:1.6">'
            'ML direction prediction (UP/DOWN) trained on 13,000+ earnings events. '
            'Sentiment percentile vs 1,100+ historical calls with portfolio backtest'
            '</div></div>', unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════
# MULTI-QUARTER MODE
# ══════════════════════════════════════════════════
if st.session_state.multi_mode:
    ticker = st.session_state.multi_ticker
    quarter_order = st.session_state.multi_quarter_order
    missing = st.session_state.multi_missing_quarters

    st.success(f"Loaded **{len(quarter_order)} transcripts** for **{ticker}**: "
               f"{', '.join(quarter_order)}")

    # Show file details
    with st.expander("Uploaded files"):
        for q in quarter_order:
            info = st.session_state.multi_quarters[q]
            st.markdown(f"- **{q}**: {info['filename']} ({len(info['raw_text']):,} chars)")

    # Gap warning
    if missing:
        missing_str = ", ".join(missing)
        st.warning(f"Non-consecutive quarters detected. Missing: **{missing_str}**. "
                   f"Cross-quarter trend analysis may have gaps.")

    # Clear button
    if st.button("Clear All Transcripts"):
        _clear_all_state()
        st.session_state["_prev_upload_names"] = []
        st.rerun()

    # ── Parse & Analyze all quarters ──
    needs_analysis = any(
        st.session_state.multi_quarters[q]["analyzed_df"] is None for q in quarter_order
    )
    all_analyzed = True

    if needs_analysis:
        status_container = st.empty()
        progress_bar = st.progress(0, text="Analyzing transcripts...")

        for i, q in enumerate(quarter_order):
            qdata = st.session_state.multi_quarters[q]
            if qdata["analyzed_df"] is not None:
                progress_bar.progress((i + 1) / len(quarter_order),
                                      text=f"[{i+1}/{len(quarter_order)}] {q} — cached")
                continue

            progress_bar.progress(i / len(quarter_order),
                                  text=f"[{i+1}/{len(quarter_order)}] Analyzing {q}... (parsing, sentiment, risk detection)")
            chunks_df, analyzed_df = analyze_transcript(qdata["raw_text"], ticker, q)

            if analyzed_df.empty:
                st.error(f"Could not parse transcript for {q}. Format not recognized.")
                all_analyzed = False
                continue

            st.session_state.multi_quarters[q]["chunks_df"] = chunks_df
            st.session_state.multi_quarters[q]["analyzed_df"] = analyzed_df
            progress_bar.progress((i + 1) / len(quarter_order),
                                  text=f"[{i+1}/{len(quarter_order)}] {q} — done")

        progress_bar.empty()
        status_container.empty()

    if not all_analyzed:
        st.stop()

    # Set single-file state to latest quarter for page compatibility
    latest_q = quarter_order[-1]
    latest_data = st.session_state.multi_quarters[latest_q]
    st.session_state.upload_chunks_df = latest_data["chunks_df"]
    st.session_state.upload_analyzed_df = latest_data["analyzed_df"]
    st.session_state["_analyzed_ticker"] = ticker
    st.session_state["_analyzed_quarter"] = latest_q

    # ══════════════════════════════════════════════════
    # MULTI-QUARTER SUMMARY DASHBOARD
    # ══════════════════════════════════════════════════
    st.divider()
    st.subheader(f"Cross-Quarter Overview — {ticker}")

    # Summary metrics per quarter
    summary_rows = []
    for q in quarter_order:
        adf = st.session_state.multi_quarters[q]["analyzed_df"]
        summary_rows.append({
            "Quarter": q,
            "Chunks": len(adf),
            "Speakers": adf["speaker"].nunique(),
            "Avg LM Sentiment": adf["lm_net_score"].mean(),
            "Avg VADER": adf["vader_compound"].mean(),
            "Risk Signals": int(adf["risk_count"].sum()),
            "Risk Intensity": int(adf["risk_intensity"].sum()),
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(
        summary_df.style.format({
            "Avg LM Sentiment": "{:.4f}",
            "Avg VADER": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Trend sparklines
    import plotly.graph_objects as go
    from utils import PLOTLY_TEMPLATE, styled_plotly

    col1, col2 = st.columns(2)
    with col1:
        fig_sent = go.Figure()
        fig_sent.add_trace(go.Scatter(
            x=summary_df["Quarter"], y=summary_df["Avg LM Sentiment"],
            mode="lines+markers", name="LM Net", marker=dict(size=8, color="#4285f4"),
            line=dict(color="#4285f4", width=2),
        ))
        fig_sent.add_trace(go.Scatter(
            x=summary_df["Quarter"], y=summary_df["Avg VADER"],
            mode="lines+markers", name="VADER", marker=dict(size=8, color="#ff9900"),
            line=dict(color="#ff9900", width=2),
        ))
        fig_sent.update_layout(template=PLOTLY_TEMPLATE, title="Sentiment Trend",
                               yaxis_title="Score")
        st.plotly_chart(styled_plotly(fig_sent), use_container_width=True)

    with col2:
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Bar(
            x=summary_df["Quarter"], y=summary_df["Risk Signals"],
            marker_color="#ef4444", name="Risk Signals",
        ))
        fig_risk.update_layout(template=PLOTLY_TEMPLATE, title="Risk Signals per Quarter",
                               yaxis_title="Count")
        st.plotly_chart(styled_plotly(fig_risk), use_container_width=True)

    st.markdown("---")
    st.markdown("Use the **sidebar** to navigate to detailed analysis pages. "
                "Each page will show cross-quarter trends automatically.")
    st.stop()


# ══════════════════════════════════════════════════
# SINGLE-QUARTER MODE (original flow)
# ══════════════════════════════════════════════════
raw_text = st.session_state.upload_raw_text

st.success(f"Loaded: **{st.session_state.upload_filename}** ({len(raw_text):,} characters)")

auto_ticker = st.session_state.upload_ticker or ""
auto_quarter = st.session_state.upload_quarter or ""

if auto_ticker or auto_quarter:
    st.info(f"Auto-detected: **{auto_ticker or '?'}** | **{auto_quarter or '?'}** — confirm or edit below.")

col1, col2, col3 = st.columns([2, 2, 1])
ticker_input = col1.text_input("Ticker Symbol", value=auto_ticker,
                                placeholder="e.g. AAPL",
                                key="ticker_input_field").strip().upper()
quarter_input = col2.text_input("Quarter", value=auto_quarter,
                                 placeholder="e.g. 2024Q3",
                                 key="quarter_input_field").strip().upper()

# Persist edits so they survive page navigation
if ticker_input:
    st.session_state.upload_ticker = ticker_input
if quarter_input:
    st.session_state.upload_quarter = quarter_input

if col3.button("Clear Transcript"):
    _clear_all_state()
    st.session_state["_prev_upload_names"] = []
    st.rerun()

status_placeholder = st.empty()

if not ticker_input or not quarter_input:
    status_placeholder.warning("Please confirm or enter the ticker symbol and quarter to proceed.")
    st.stop()


# ── Parse & Analyze ──
if (st.session_state.upload_analyzed_df is not None
        and st.session_state.upload_chunks_df is not None
        and st.session_state.get("_analyzed_ticker") == ticker_input
        and st.session_state.get("_analyzed_quarter") == quarter_input):
    chunks_df = st.session_state.upload_chunks_df
    analyzed_df = st.session_state.upload_analyzed_df
else:
    status_placeholder.empty()
    with st.spinner("Parsing transcript, running sentiment (LM + VADER + FinBERT) and risk analysis... This may take 10-20 seconds."):
        chunks_df, analyzed_df = analyze_transcript(raw_text, ticker_input, quarter_input)
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
