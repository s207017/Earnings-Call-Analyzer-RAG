"""
Preprocess Motley Fool dataset into the pipeline's expected format.

Reads data/kaggle/motley-fool-data.pkl and writes transcripts to data/raw/{TICKER}/{QUARTER}.txt
Only processes tickers listed in configs/companies.json.
Skips transcripts that already exist in data/raw/.

Usage:
    python3.11 scripts/preprocess_motley_fool.py                # pipeline tickers only
    python3.11 scripts/preprocess_motley_fool.py --all          # all tickers
    python3.11 scripts/preprocess_motley_fool.py --market-only  # tickers with market data
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
MARKET_PATH = PROJECT_ROOT / "data" / "market" / "market_reactions.parquet"


def load_pipeline_tickers() -> set:
    """Load tickers tracked by the pipeline."""
    companies_path = PROJECT_ROOT / "configs" / "companies.json"
    with open(companies_path) as f:
        companies = json.load(f)
    return {c["ticker"] for c in companies}


def load_market_tickers() -> set:
    """Load tickers that have market data (abnormal returns)."""
    if not MARKET_PATH.exists():
        print(f"WARNING: {MARKET_PATH} not found")
        return set()
    market_df = pd.read_parquet(MARKET_PATH, columns=["ticker", "quarter"])
    return set(zip(market_df["ticker"], market_df["quarter"]))


def normalize_quarter(q: str) -> str:
    """Convert '2020-Q2' → '2020Q2'."""
    m = re.match(r"(\d{4})-?Q(\d)", q)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    return q


def load_manifest() -> list:
    """Load existing manifest entries."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return []


def save_manifest(entries: list):
    """Save manifest."""
    with open(MANIFEST_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Preprocess Motley Fool dataset")
    parser.add_argument("--all", action="store_true",
                        help="Include all tickers, not just pipeline tickers")
    parser.add_argument("--market-only", action="store_true",
                        help="Only include tickers/quarters that have market data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing files")
    args = parser.parse_args()

    pkl_path = PROJECT_ROOT / "data" / "kaggle" / "motley-fool-data.pkl"
    if not pkl_path.exists():
        print(f"ERROR: {pkl_path} not found")
        sys.exit(1)

    print("Loading Motley Fool dataset...")
    with open(pkl_path, "rb") as f:
        mf = pickle.load(f)
    print(f"Loaded {len(mf)} transcripts, {mf['ticker'].nunique()} tickers")

    # Normalize quarters early for market matching
    mf["quarter_norm"] = mf["q"].apply(normalize_quarter)

    # Filter based on mode
    if args.market_only:
        market_pairs = load_market_tickers()
        market_tickers = {t for t, _ in market_pairs}
        mf = mf[mf["ticker"].isin(market_tickers)]
        print(f"Filtered to {len(mf)} transcripts for {mf['ticker'].nunique()} tickers with market data")
        # Further filter to only ticker-quarter pairs with market data
        mf = mf[mf.apply(lambda r: (r["ticker"], r["quarter_norm"]) in market_pairs, axis=1)]
        print(f"After quarter-level filter: {len(mf)} transcripts")
    elif not args.all:
        pipeline_tickers = load_pipeline_tickers()
        mf = mf[mf["ticker"].isin(pipeline_tickers)]
        print(f"Filtered to {len(mf)} transcripts for {mf['ticker'].nunique()} pipeline tickers")

    # Load existing manifest to check for duplicates
    manifest = load_manifest()
    existing = {(e["ticker"], e["quarter"]) for e in manifest}

    written = 0
    skipped_exists = 0
    skipped_short = 0
    new_manifest_entries = []

    for _, row in mf.iterrows():
        ticker = row["ticker"]
        quarter = row.get("quarter_norm") or normalize_quarter(row["q"])
        transcript = row["transcript"]

        if not quarter or not re.match(r"\d{4}Q\d", quarter):
            continue

        # Skip if already in pipeline
        if (ticker, quarter) in existing:
            skipped_exists += 1
            continue

        # Skip very short transcripts
        if not transcript or len(transcript) < 500:
            skipped_short += 1
            continue

        # Write transcript file
        ticker_dir = RAW_DIR / ticker
        out_path = ticker_dir / f"{quarter}.txt"

        if out_path.exists():
            skipped_exists += 1
            existing.add((ticker, quarter))
            continue

        if args.dry_run:
            print(f"  [DRY RUN] Would write {out_path} ({len(transcript):,} chars)")
            written += 1
            continue

        ticker_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(transcript, encoding="utf-8")

        new_manifest_entries.append({
            "file_path": str(out_path.relative_to(PROJECT_ROOT)),
            "ticker": ticker,
            "quarter": quarter,
            "date": row.get("date", ""),
            "source": "motley_fool",
        })
        existing.add((ticker, quarter))
        written += 1

    # Update manifest
    if new_manifest_entries and not args.dry_run:
        manifest.extend(new_manifest_entries)
        save_manifest(manifest)

    print(f"\nResults:")
    print(f"  Written:  {written} new transcripts")
    print(f"  Skipped:  {skipped_exists} (already exist) + {skipped_short} (too short)")
    print(f"  Manifest: {len(manifest)} total entries")


if __name__ == "__main__":
    main()
