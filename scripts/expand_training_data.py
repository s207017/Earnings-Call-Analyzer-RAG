"""
Expand training data by incorporating all Motley Fool transcripts that have market data.

This script:
  1. Identifies MF transcripts with matching market data (abnormal returns)
  2. Writes new transcripts to data/raw/{TICKER}/{QUARTER}.txt
  3. Runs the NLP pipeline (parse → sentiment → risk → feature matrix)

Usage:
    python3.11 scripts/expand_training_data.py [--dry-run] [--skip-pipeline]
"""

import argparse
import json
import logging
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
MARKET_PATH = PROJECT_ROOT / "data" / "market" / "market_reactions.parquet"


def normalize_quarter(q: str) -> str:
    m = re.match(r"(\d{4})-?Q(\d)", str(q))
    return f"{m.group(1)}Q{m.group(2)}" if m else str(q)


def main():
    parser = argparse.ArgumentParser(description="Expand training data with MF transcripts")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Only write transcripts, skip NLP pipeline")
    parser.add_argument("--steps", type=str, default="1,2,3,4",
                        help="Pipeline steps to run (default: 1,2,3,4)")
    args = parser.parse_args()

    # ── 1. Load market data to know which ticker-quarters have returns ──
    if not MARKET_PATH.exists():
        logger.error(f"Market data not found at {MARKET_PATH}. Run fetch_market_data.py first.")
        sys.exit(1)

    market_df = pd.read_parquet(MARKET_PATH)
    market_pairs = set(zip(market_df["ticker"], market_df["quarter"]))
    market_tickers = {t for t, _ in market_pairs}
    valid_returns = market_df["abnormal_ret_1d"].notna().sum()
    logger.info(f"Market data: {len(market_df)} events, {len(market_tickers)} tickers, "
                f"{valid_returns} with valid abnormal returns")

    # ── 2. Load Motley Fool dataset ──
    pkl_path = PROJECT_ROOT / "data" / "kaggle" / "motley-fool-data.pkl"
    logger.info("Loading Motley Fool dataset...")
    with open(pkl_path, "rb") as f:
        mf = pickle.load(f)
    mf["quarter_norm"] = mf["q"].apply(normalize_quarter)
    logger.info(f"Loaded {len(mf)} transcripts, {mf['ticker'].nunique()} tickers")

    # ── 3. Filter to transcripts with matching market data ──
    mf["has_market"] = mf.apply(lambda r: (r["ticker"], r["quarter_norm"]) in market_pairs, axis=1)
    mf_with_market = mf[mf["has_market"]].copy()
    logger.info(f"Transcripts with market data: {len(mf_with_market)} "
                f"({mf_with_market['ticker'].nunique()} tickers)")

    # ── 4. Check what's already in manifest ──
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    else:
        manifest = []
    existing = {(e["ticker"], e["quarter"]) for e in manifest}
    logger.info(f"Current manifest: {len(manifest)} entries")

    # ── 5. Write new transcripts ──
    written = 0
    skipped = 0
    new_entries = []

    for _, row in mf_with_market.iterrows():
        ticker = row["ticker"]
        quarter = row["quarter_norm"]

        if not re.match(r"\d{4}Q\d", quarter):
            continue

        if (ticker, quarter) in existing:
            skipped += 1
            continue

        transcript = row["transcript"]
        if not transcript or len(transcript) < 500:
            continue

        out_path = RAW_DIR / ticker / f"{quarter}.txt"
        if out_path.exists():
            # File exists but not in manifest — add to manifest
            existing.add((ticker, quarter))
            new_entries.append({
                "file_path": str(out_path.relative_to(PROJECT_ROOT)),
                "ticker": ticker,
                "quarter": quarter,
                "source": "motley_fool",
            })
            skipped += 1
            continue

        if args.dry_run:
            written += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(transcript, encoding="utf-8")

        new_entries.append({
            "file_path": str(out_path.relative_to(PROJECT_ROOT)),
            "ticker": ticker,
            "quarter": quarter,
            "source": "motley_fool",
        })
        existing.add((ticker, quarter))
        written += 1

    logger.info(f"Written: {written} new transcripts, Skipped: {skipped} (already exist)")
    logger.info(f"New manifest entries: {len(new_entries)}")

    if args.dry_run:
        logger.info("[DRY RUN] No files written.")
        return

    # Update manifest
    if new_entries:
        manifest.extend(new_entries)
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest updated: {len(manifest)} total entries")

    # ── 6. Run pipeline ──
    if args.skip_pipeline:
        logger.info("Skipping pipeline (--skip-pipeline)")
        return

    if written == 0 and len(new_entries) == 0:
        logger.info("No new transcripts to process")
        return

    logger.info(f"Running NLP pipeline (steps {args.steps})...")
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
           "--steps", args.steps]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        logger.error(f"Pipeline failed with exit code {result.returncode}")
        sys.exit(1)

    logger.info("Expansion complete!")

    # Summary
    from src.agents.finance_analysis import FinanceAnalyzer
    try:
        feat_path = PROJECT_ROOT / "data" / "outputs" / "feature_matrix.parquet"
        if feat_path.exists():
            fm = pd.read_parquet(feat_path)
            has_target = fm["abnormal_ret_1d"].notna().sum() if "abnormal_ret_1d" in fm.columns else 0
            logger.info(f"Feature matrix: {len(fm)} rows, {has_target} with target variable")
    except Exception:
        pass


if __name__ == "__main__":
    main()
