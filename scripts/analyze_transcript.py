"""
Standalone script to parse a PDF/TXT earnings call transcript
and run the full analysis pipeline (same as the Streamlit dashboard).

Usage:
    python scripts/analyze_transcript.py <path_to_pdf_or_txt> [--ticker GOOGL] [--quarter 2025Q4]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "app"))


# ── 1. Extract text from PDF or TXT ──
def extract_text(file_path: Path) -> str:
    if file_path.suffix.lower() == ".pdf":
        import pdfplumber
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)
    else:
        return file_path.read_text(encoding="utf-8", errors="replace")


# ── 2. Auto-detect ticker and quarter ──
def detect_ticker_and_quarter(text: str, filename: str) -> tuple[str, str]:
    import re
    text_upper = text[:3000].upper()
    filename_upper = filename.upper()
    detected_ticker = ""
    detected_quarter = ""

    name_to_ticker = {
        "APPLE": "AAPL", "MICROSOFT": "MSFT", "ALPHABET": "GOOGL", "GOOGLE": "GOOGL",
        "AMAZON": "AMZN", "META PLATFORMS": "META", "NVIDIA": "NVDA",
        "TESLA": "TSLA", "NETFLIX": "NFLX", "SALESFORCE": "CRM", "ORACLE": "ORCL",
        "JPMORGAN": "JPM", "BANK OF AMERICA": "BAC", "WALMART": "WMT",
    }

    for name, ticker in sorted(name_to_ticker.items(), key=lambda x: -len(x[0])):
        if name in text_upper:
            detected_ticker = ticker
            break

    ticker_pattern = re.search(r"(?:NASDAQ|NYSE|TICKER)[:\s]+([A-Z]{1,5})", text_upper)
    if ticker_pattern:
        detected_ticker = ticker_pattern.group(1)

    q_match = re.search(r"Q(\d)\s*[-/]?\s*(20\d{2})", text_upper)
    if q_match:
        detected_quarter = f"{q_match.group(2)}Q{q_match.group(1)}"
    if not detected_quarter:
        q_match = re.search(r"(20\d{2})\s*[-/]?\s*Q(\d)", text_upper)
        if q_match:
            detected_quarter = f"{q_match.group(1)}Q{q_match.group(2)}"

    ordinal_map = {"FIRST": "1", "SECOND": "2", "THIRD": "3", "FOURTH": "4",
                   "1ST": "1", "2ND": "2", "3RD": "3", "4TH": "4"}
    if not detected_quarter:
        for word, num in ordinal_map.items():
            pattern = re.search(
                rf"{word}\s+QUARTER\s+(?:OF\s+)?(?:FISCAL\s+(?:YEAR\s+)?)?(20\d{{2}})", text_upper)
            if pattern:
                detected_quarter = f"{pattern.group(1)}Q{num}"
                break

    return detected_ticker, detected_quarter


# ── 3. Parse, chunk, and analyze ──
def analyze(text: str, ticker: str, quarter: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.agents.transcript_ingestion import TranscriptParser
    from src.agents.sentiment_analysis import SentimentAnalyzer
    from src.agents.risk_detection import RiskDetector

    # Parse into chunks
    parser = TranscriptParser()
    chunks = parser.parse_transcript(text, ticker, quarter)
    if not chunks:
        print("ERROR: Could not parse any chunks from the transcript.")
        sys.exit(1)

    chunks_df = pd.DataFrame(chunks)
    print(f"Parsed {len(chunks_df)} chunks, {chunks_df['speaker'].nunique()} speakers")

    # Sentiment analysis (LM + VADER)
    analyzer = SentimentAnalyzer()
    texts = chunks_df["text"].tolist()

    lm_results = [analyzer.loughran_mcdonald_sentiment(t) for t in texts]
    lm_df = pd.DataFrame(lm_results)

    vader_results = [analyzer.vader_sentiment(t) for t in texts]
    vader_df = pd.DataFrame(vader_results)

    analyzed_df = pd.concat([chunks_df.reset_index(drop=True), lm_df, vader_df], axis=1)

    # Risk detection
    detector = RiskDetector()
    all_categories, all_counts, all_intensities, all_details = [], [], [], []

    for _, row in analyzed_df.iterrows():
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

    analyzed_df["risk_categories"] = all_categories
    analyzed_df["risk_count"] = all_counts
    analyzed_df["risk_intensity"] = all_intensities
    analyzed_df["risk_details"] = all_details

    return chunks_df, analyzed_df


# ── 4. Print results ──
def print_results(analyzed_df: pd.DataFrame, ticker: str, quarter: str):
    sep = "=" * 70

    # Summary
    print(f"\n{sep}")
    print(f"  ANALYSIS RESULTS — {ticker} {quarter}")
    print(sep)
    print(f"  Chunks:           {len(analyzed_df)}")
    print(f"  Speakers:         {analyzed_df['speaker'].nunique()}")
    print(f"  Avg LM Net Score: {analyzed_df['lm_net_score'].mean():.4f}")
    print(f"  Avg VADER Score:  {analyzed_df['vader_compound'].mean():.4f}")
    print(f"  Total Risk Sigs:  {int(analyzed_df['risk_count'].sum())}")
    print(sep)

    # Sentiment by section
    print("\n── SENTIMENT BY SECTION ──")
    section_avg = analyzed_df.groupby("section")[["lm_net_score", "vader_compound"]].mean()
    print(section_avg.to_string(float_format=lambda x: f"{x:.4f}"))

    # Sentiment by role
    print("\n── SENTIMENT BY SPEAKER ROLE ──")
    role_avg = analyzed_df.groupby("role")[["lm_net_score", "vader_compound"]].mean()
    print(role_avg.to_string(float_format=lambda x: f"{x:.4f}"))

    # Sentiment by speaker
    print("\n── SENTIMENT BY SPEAKER ──")
    speaker_sent = analyzed_df.groupby("speaker")["lm_net_score"].mean().sort_values(ascending=False)
    for speaker, score in speaker_sent.items():
        bar = "+" * int(max(0, score) * 200) or "-" * int(max(0, -score) * 200) or "·"
        print(f"  {score:+.4f}  {bar:20s}  {speaker}")

    # Most positive chunks
    print("\n── MOST POSITIVE CHUNKS ──")
    for _, row in analyzed_df.nlargest(3, "lm_net_score").iterrows():
        print(f"  [{row['lm_net_score']:+.4f}] {row['speaker']} ({row['role']})")
        print(f"    \"{row['text'][:150]}...\"")

    # Most negative chunks
    print("\n── MOST NEGATIVE CHUNKS ──")
    for _, row in analyzed_df.nsmallest(3, "lm_net_score").iterrows():
        print(f"  [{row['lm_net_score']:+.4f}] {row['speaker']} ({row['role']})")
        print(f"    \"{row['text'][:150]}...\"")

    # Risk categories
    print("\n── RISK CATEGORIES ──")
    all_cats = []
    for cats in analyzed_df["risk_categories"]:
        all_cats.extend(cats)
    if all_cats:
        cat_counts = pd.Series(all_cats).value_counts()
        for cat, count in cat_counts.items():
            print(f"  {count:4d}  {cat}")
    else:
        print("  No risk signals detected.")

    # Risk severity
    all_sev = []
    for details in analyzed_df["risk_details"]:
        for d in details:
            all_sev.append(d.get("severity", "medium"))
    if all_sev:
        sev_counts = pd.Series(all_sev).value_counts()
        print(f"\n  Severity: High={sev_counts.get('high', 0)}  "
              f"Medium={sev_counts.get('medium', 0)}  "
              f"Low={sev_counts.get('low', 0)}")

    # Top risk excerpts
    print("\n── TOP RISK EXCERPTS ──")
    risk_chunks = analyzed_df[analyzed_df["risk_count"] > 0].nlargest(5, "risk_intensity")
    for _, row in risk_chunks.iterrows():
        cats = ", ".join(row["risk_categories"])
        print(f"  [{cats}] intensity={row['risk_intensity']}  — {row['speaker']} ({row['role']})")
        print(f"    \"{row['text'][:200]}...\"")
        print()

    # Benchmark against historical data
    hist_path = PROJECT_ROOT / "data" / "processed" / "chunks_with_sentiment.parquet"
    if hist_path.exists():
        try:
            print(f"\n{sep}")
            print("  BENCHMARK VS HISTORICAL DATASET")
            print(sep)
            hist = pd.read_parquet(hist_path)
            hist_quarterly = hist.groupby(["ticker", "quarter"]).agg(
                lm_net=("lm_net_score", "mean"),
                vader=("vader_compound", "mean"),
            ).reset_index()

            uploaded_lm = analyzed_df["lm_net_score"].mean()
            uploaded_vader = analyzed_df["vader_compound"].mean()

            lm_pct = (hist_quarterly["lm_net"] < uploaded_lm).mean() * 100
            vader_pct = (hist_quarterly["vader"] < uploaded_vader).mean() * 100

            print(f"  LM Net Score:  {uploaded_lm:.4f}  (percentile: {lm_pct:.0f}%)")
            print(f"  VADER Score:   {uploaded_vader:.4f}  (percentile: {vader_pct:.0f}%)")
            print(f"  Compared against {len(hist_quarterly)} historical earnings calls")

            hist_risk_path = PROJECT_ROOT / "data" / "processed" / "chunks_with_risk.parquet"
            if hist_risk_path.exists():
                hist_risk = pd.read_parquet(hist_risk_path)
                hist_risk_q = hist_risk.groupby(["ticker", "quarter"])["risk_count"].sum().reset_index()
                uploaded_risk = analyzed_df["risk_count"].sum()
                risk_pct = (hist_risk_q["risk_count"] < uploaded_risk).mean() * 100
                print(f"  Risk Signals:  {int(uploaded_risk)}  (percentile: {risk_pct:.0f}%)")
        except ImportError:
            print("\n  (Skipping benchmark — pyarrow not installed)")

    print(f"\n{sep}")
    print("  DONE")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Analyze an earnings call transcript")
    parser.add_argument("file", type=str, help="Path to PDF or TXT transcript file")
    parser.add_argument("--ticker", type=str, default="", help="Ticker symbol (auto-detected if omitted)")
    parser.add_argument("--quarter", type=str, default="", help="Quarter e.g. 2025Q4 (auto-detected if omitted)")
    parser.add_argument("--output", type=str, default="", help="Save results to CSV")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    print(f"Reading {file_path.name}...")
    text = extract_text(file_path)
    print(f"Extracted {len(text):,} characters")

    # Detect or use provided ticker/quarter
    ticker = args.ticker.upper()
    quarter = args.quarter.upper()
    if not ticker or not quarter:
        auto_ticker, auto_quarter = detect_ticker_and_quarter(text, file_path.name)
        ticker = ticker or auto_ticker
        quarter = quarter or auto_quarter
        print(f"Auto-detected: {ticker} {quarter}")

    if not ticker or not quarter:
        print("ERROR: Could not detect ticker/quarter. Provide --ticker and --quarter.")
        sys.exit(1)

    # Run analysis
    print(f"Analyzing {ticker} {quarter}...")
    chunks_df, analyzed_df = analyze(text, ticker, quarter)

    # Print results
    print_results(analyzed_df, ticker, quarter)

    # Optionally save
    if args.output:
        out_path = Path(args.output)
        analyzed_df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
