"""
Run the full NLP + finance pipeline on all transcripts in data/raw/.

Steps:
  1. Parse transcripts into chunks (TranscriptParser)
  2. Run sentiment analysis (LM + VADER) on all chunks
  3. Run risk detection on all chunks
  4. Save processed parquets
  5. Rebuild feature matrix and finance results

Incremental: only processes transcripts not already in the processed data.

Usage:
    python3.11 scripts/run_pipeline.py [--full]     # --full to reprocess everything
    python3.11 scripts/run_pipeline.py --steps 1,2   # run only specific steps
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"

SENTIMENT_PATH = PROCESSED_DIR / "chunks_with_sentiment.parquet"
RISK_PATH = PROCESSED_DIR / "chunks_with_risk.parquet"


def load_manifest() -> list:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return []


def get_existing_pairs(parquet_path: Path) -> set:
    """Get (ticker, quarter) pairs already processed."""
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path, columns=["ticker", "quarter"])
        return set(zip(df["ticker"], df["quarter"]))
    return set()


def _checkpoint_save(existing: pd.DataFrame, new_dfs: list, path: Path,
                     label: str, serialize_lists: bool = False):
    """Save intermediate checkpoint during batch processing."""
    import json as _json
    new_df = pd.concat(new_dfs, ignore_index=True)
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ticker", "quarter", "chunk_index"], keep="last")
    else:
        combined = new_df
    if serialize_lists:
        for col in ["risk_categories", "risk_details"]:
            if col in combined.columns:
                combined[col] = combined[col].apply(
                    lambda x: _json.dumps(x) if isinstance(x, list) else x
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    logger.info(f"  Checkpoint saved ({label}): {len(combined)} total chunks")


# ── Step 1: Parse transcripts into chunks ──
def step1_parse(manifest: list, full: bool = False) -> pd.DataFrame:
    """Parse all raw transcripts into chunks."""
    from src.agents.transcript_ingestion import TranscriptParser

    existing_pairs = set() if full else get_existing_pairs(SENTIMENT_PATH)
    parser = TranscriptParser()

    all_chunks = []
    skipped = 0
    errors = 0

    for entry in manifest:
        ticker = entry["ticker"]
        quarter = entry["quarter"]

        if (ticker, quarter) in existing_pairs:
            skipped += 1
            continue

        file_path = PROJECT_ROOT / entry["file_path"]
        if not file_path.exists():
            continue

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            if len(text) < 200:
                continue

            chunks = parser.parse_transcript(text, ticker, quarter)
            if chunks:
                all_chunks.extend(chunks)
        except Exception as e:
            logger.warning(f"Error parsing {ticker} {quarter}: {e}")
            errors += 1

    logger.info(f"Step 1: Parsed {len(all_chunks)} new chunks "
                f"(skipped {skipped} existing, {errors} errors)")
    return pd.DataFrame(all_chunks) if all_chunks else pd.DataFrame()


# ── Step 2: Sentiment analysis ──
def step2_sentiment(new_chunks: pd.DataFrame, full: bool = False) -> pd.DataFrame:
    """Run LM + VADER sentiment on chunks, with batch checkpointing."""
    from src.agents.sentiment_analysis import SentimentAnalyzer

    # Load existing data
    if not full and SENTIMENT_PATH.exists():
        existing = pd.read_parquet(SENTIMENT_PATH)
    else:
        existing = pd.DataFrame()

    if new_chunks.empty:
        logger.info("Step 2: No new chunks to analyze")
        return existing

    analyzer = SentimentAnalyzer()
    total = len(new_chunks)
    BATCH_SIZE = 2000  # checkpoint every N chunks

    logger.info(f"Step 2: Running sentiment on {total} chunks (batch size {BATCH_SIZE})...")
    t0 = time.time()

    all_sent = []
    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = new_chunks.iloc[batch_start:batch_end]
        texts = batch["text"].tolist()

        lm_results = [analyzer.loughran_mcdonald_sentiment(t) for t in texts]
        lm_df = pd.DataFrame(lm_results)

        vader_results = [analyzer.vader_sentiment(t) for t in texts]
        vader_df = pd.DataFrame(vader_results)

        sent_df = pd.concat([batch.reset_index(drop=True), lm_df, vader_df], axis=1)
        all_sent.append(sent_df)

        elapsed = time.time() - t0
        rate = batch_end / elapsed if elapsed > 0 else 0
        logger.info(f"  Step 2: {batch_end}/{total} chunks ({rate:.0f}/s)")

        # Checkpoint save every batch
        if batch_end < total and batch_end % (BATCH_SIZE * 3) == 0:
            _checkpoint_save(existing, all_sent, SENTIMENT_PATH, "sentiment")

    new_sent = pd.concat(all_sent, ignore_index=True) if all_sent else pd.DataFrame()

    # Merge with existing
    if not existing.empty and not new_sent.empty:
        combined = pd.concat([existing, new_sent], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ticker", "quarter", "chunk_index"], keep="last")
    elif not new_sent.empty:
        combined = new_sent
    else:
        combined = existing

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(SENTIMENT_PATH, index=False)
    logger.info(f"Step 2: Done in {time.time()-t0:.1f}s. "
                f"Total: {len(combined)} chunks saved to {SENTIMENT_PATH.name}")
    return combined


# ── Step 3: Risk detection ──
def step3_risk(new_chunks: pd.DataFrame, full: bool = False) -> pd.DataFrame:
    """Run risk detection on chunks, with batch checkpointing."""
    from src.agents.risk_detection import RiskDetector

    if not full and RISK_PATH.exists():
        existing = pd.read_parquet(RISK_PATH)
    else:
        existing = pd.DataFrame()

    if new_chunks.empty:
        logger.info("Step 3: No new chunks to analyze")
        return existing

    detector = RiskDetector()
    total = len(new_chunks)
    BATCH_SIZE = 2000
    logger.info(f"Step 3: Running risk detection on {total} chunks (batch size {BATCH_SIZE})...")
    t0 = time.time()

    all_risk_dfs = []
    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = new_chunks.iloc[batch_start:batch_end]

        all_categories, all_counts, all_intensities, all_details = [], [], [], []
        for _, row in batch.iterrows():
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

        risk_df = batch[["chunk_id", "ticker", "quarter", "speaker", "role",
                         "section", "text", "chunk_index"]].copy()
        risk_df["risk_categories"] = all_categories
        risk_df["risk_count"] = all_counts
        risk_df["risk_intensity"] = all_intensities
        risk_df["risk_details"] = all_details
        all_risk_dfs.append(risk_df)

        elapsed = time.time() - t0
        rate = batch_end / elapsed if elapsed > 0 else 0
        logger.info(f"  Step 3: {batch_end}/{total} chunks ({rate:.0f}/s)")

        # Checkpoint save
        if batch_end < total and batch_end % (BATCH_SIZE * 3) == 0:
            _checkpoint_save(existing, all_risk_dfs, RISK_PATH, "risk",
                             serialize_lists=True)

    new_risk = pd.concat(all_risk_dfs, ignore_index=True) if all_risk_dfs else pd.DataFrame()

    if not existing.empty and not new_risk.empty:
        combined = pd.concat([existing, new_risk], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ticker", "quarter", "chunk_index"], keep="last")
    elif not new_risk.empty:
        combined = new_risk
    else:
        combined = existing

    # Ensure list columns are serialized as JSON strings for parquet compatibility
    import json as _json
    for col in ["risk_categories", "risk_details"]:
        if col in combined.columns:
            combined[col] = combined[col].apply(
                lambda x: _json.dumps(x) if isinstance(x, list) else x
            )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(RISK_PATH, index=False)
    logger.info(f"Step 3: Done in {time.time()-t0:.1f}s. "
                f"Total: {len(combined)} chunks saved to {RISK_PATH.name}")
    return combined


# ── Step 4: Feature matrix + finance analysis ──
def step4_finance():
    """Rebuild feature matrix and run finance analysis."""
    from src.agents.finance_analysis import FinanceAnalyzer

    logger.info("Step 4: Rebuilding feature matrix and running finance analysis...")
    t0 = time.time()

    fa = FinanceAnalyzer()
    results = fa.run_all()

    logger.info(f"Step 4: Done in {time.time()-t0:.1f}s. "
                f"Feature matrix: {fa.feature_matrix.shape}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run full NLP + finance pipeline")
    parser.add_argument("--full", action="store_true",
                        help="Reprocess everything from scratch")
    parser.add_argument("--steps", type=str, default="1,2,3,4",
                        help="Comma-separated step numbers to run (default: 1,2,3,4)")
    args = parser.parse_args()

    steps = {int(s) for s in args.steps.split(",")}
    manifest = load_manifest()
    logger.info(f"Manifest: {len(manifest)} transcripts")

    new_chunks = pd.DataFrame()

    if 1 in steps:
        new_chunks = step1_parse(manifest, full=args.full)

    if 2 in steps:
        step2_sentiment(new_chunks, full=args.full)

    if 3 in steps:
        step3_risk(new_chunks, full=args.full)

    if 4 in steps:
        step4_finance()

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
