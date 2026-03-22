"""Example usage script for sentiment analysis pipeline."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.agents.sentiment_analysis import SentimentAnalyzer

def main():
    chunks_path = project_root / "data" / "processed" / "all_chunks.parquet"
    if not chunks_path.exists():
        print("No chunks found. Run data pipeline first:")
        print("  python src/data_collection.py")
        print("  python -c \"from src.agents.transcript_ingestion import TranscriptParser; TranscriptParser().process_all()\"")
        return

    # Load chunks
    df = pd.read_parquet(chunks_path)
    print(f"Loaded {len(df)} chunks")
    print(f"Companies: {df['ticker'].nunique()}, Quarters: {df['quarter'].nunique()}")

    # Initialize analyzer
    analyzer = SentimentAnalyzer(str(project_root))

    # Run LM + VADER (skip FinBERT for quick test)
    result = analyzer.analyze_all(df, methods=["lm", "vader"])
    print(f"\nAnalyzed {len(result)} chunks")

    # Aggregate by company
    print("\n=== Average Sentiment by Company ===")
    company_avg = result.groupby("ticker")[["lm_net_score", "vader_compound"]].mean()
    print(company_avg.sort_values("lm_net_score", ascending=False).to_string())

    # Aggregate by section
    print("\n=== Average Sentiment by Section ===")
    section_avg = result.groupby("section")[["lm_net_score", "vader_compound"]].mean()
    print(section_avg.to_string())

    # Sentiment divergence for first company
    ticker = df["ticker"].iloc[0]
    quarter = df["quarter"].iloc[0]
    div = analyzer.sentiment_divergence(result, ticker, quarter)
    print(f"\n=== Sentiment Divergence ({ticker} {quarter}) ===")
    for k, v in div.items():
        print(f"  {k}: {v:.4f}")

    # Trends for first company
    trends = analyzer.sentiment_trends(result, ticker)
    print(f"\n=== Sentiment Trends ({ticker}) ===")
    print(trends.to_string())


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
