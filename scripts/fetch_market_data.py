"""
Fetch market data (stock returns, abnormal returns) for Motley Fool transcripts
that are missing this data.

Uses yfinance to get stock prices and computes:
- 1-day, 3-day, 5-day returns
- Abnormal returns (stock return - S&P 500 return)
- Abnormal volume

Appends results to data/market/market_reactions.parquet.

Usage:
    python3.11 scripts/fetch_market_data.py [--max-tickers 100] [--dry-run]
"""

import argparse
import logging
import pickle
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MARKET_PATH = PROJECT_ROOT / "data" / "market" / "market_reactions.parquet"
CACHE_DIR = PROJECT_ROOT / "data" / "market" / "cache"
BENCHMARK = "^GSPC"  # S&P 500


def parse_mf_date(date_str: str) -> str:
    """Parse Motley Fool date format to YYYY-MM-DD.

    Examples:
        'Aug 27, 2020, 9:00 p.m. ET' → '2020-08-27'
        'Jul 30, 2020, 4:30 p.m. ET' → '2020-07-30'
    """
    if date_str is None:
        return None
    # Some entries are lists — take the last element which usually has the date
    if isinstance(date_str, list):
        date_str = date_str[-1] if date_str else None
    if not isinstance(date_str, str) or not date_str.strip():
        return None
    try:
        # Remove timezone suffix
        clean = re.sub(r',?\s*\d{1,2}:\d{2}\s*[ap]\.?m\.?\s*ET\s*$', '', date_str, flags=re.IGNORECASE)
        clean = clean.strip().rstrip(',')
        dt = datetime.strptime(clean, "%b %d, %Y")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        # Try alternate format: "Jan. 31, 2019 11:00 a.m. ET"
        try:
            clean = re.sub(r'\s*\d{1,2}:\d{2}\s*[ap]\.?m\.?\s*ET\s*$', '', date_str, flags=re.IGNORECASE)
            clean = clean.strip().rstrip(',')
            for fmt in ["%b. %d, %Y", "%b %d, %Y", "%B %d, %Y"]:
                try:
                    dt = datetime.strptime(clean, fmt)
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue
        except Exception:
            pass
        return None


def normalize_quarter(q: str) -> str:
    """Convert '2020-Q2' → '2020Q2'."""
    m = re.match(r"(\d{4})-?Q(\d)", str(q))
    return f"{m.group(1)}Q{m.group(2)}" if m else str(q)


def fetch_prices(ticker: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame:
    """Fetch price data with caching."""
    import yfinance as yf

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{ticker}_{start}_{end}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    for attempt in range(max_retries):
        try:
            t = yf.Ticker(ticker)
            data = t.history(start=start, end=end, auto_adjust=True)
            if data.empty:
                return pd.DataFrame()
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            data.to_parquet(cache_file)
            return data
        except Exception as e:
            wait = 2 ** (attempt + 1)
            logger.debug(f"Attempt {attempt+1} for {ticker}: {e}, waiting {wait}s")
            time.sleep(wait)

    return pd.DataFrame()


def compute_returns(prices: pd.DataFrame, earnings_date: str, windows=[1, 3, 5]) -> dict:
    """Compute post-earnings returns."""
    if prices.empty or "Close" not in prices.columns:
        return {f"ret_{w}d": np.nan for w in windows}

    close = prices["Close"]
    ed = pd.Timestamp(earnings_date)

    # Find trading day on or before earnings
    past = close.index[close.index <= ed]
    if len(past) == 0:
        return {f"ret_{w}d": np.nan for w in windows}
    base_date = past[-1]
    base_price = close.loc[base_date]

    results = {}
    for w in windows:
        future = close.index[close.index > base_date]
        if len(future) >= w:
            end_price = close.iloc[close.index.get_loc(future[w - 1])]
            results[f"ret_{w}d"] = float((end_price - base_price) / base_price)
        else:
            results[f"ret_{w}d"] = np.nan
    return results


def compute_abnormal_volume(prices: pd.DataFrame, earnings_date: str, lookback: int = 20) -> float:
    """Compute abnormal volume ratio."""
    if prices.empty or "Volume" not in prices.columns:
        return np.nan

    vol = prices["Volume"]
    ed = pd.Timestamp(earnings_date)
    future = vol.index[vol.index >= ed]
    if len(future) == 0:
        return np.nan

    ed_idx = future[0]
    prior = vol[vol.index < ed_idx].tail(lookback)
    if len(prior) < 5:
        return np.nan

    avg_vol = prior.mean()
    if avg_vol == 0:
        return np.nan

    return float(vol.loc[ed_idx] / avg_vol)


def process_ticker_events(ticker: str, events: list) -> list:
    """Process all earnings events for one ticker."""
    if not events:
        return []

    # Determine date range needed
    dates = [pd.Timestamp(e["date"]) for e in events]
    start = (min(dates) - timedelta(days=40)).strftime("%Y-%m-%d")
    end = (max(dates) + timedelta(days=10)).strftime("%Y-%m-%d")

    # Fetch stock prices
    prices = fetch_prices(ticker, start, end)
    if prices.empty:
        return []

    # Fetch benchmark
    bench = fetch_prices(BENCHMARK, start, end)

    records = []
    for event in events:
        earnings_date = event["date"]
        quarter = event["quarter"]

        stock_rets = compute_returns(prices, earnings_date)
        bench_rets = compute_returns(bench, earnings_date) if not bench.empty else {}

        abnormal = {}
        for w in [1, 3, 5]:
            sr = stock_rets.get(f"ret_{w}d", np.nan)
            br = bench_rets.get(f"ret_{w}d", np.nan)
            if np.isnan(sr) or np.isnan(br):
                abnormal[f"abnormal_ret_{w}d"] = np.nan
            else:
                abnormal[f"abnormal_ret_{w}d"] = sr - br

        abn_vol = compute_abnormal_volume(prices, earnings_date)

        records.append({
            "ticker": ticker,
            "quarter": quarter,
            "earnings_date": earnings_date,
            **stock_rets,
            **abnormal,
            "abnormal_volume": abn_vol,
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="Fetch market data for Motley Fool transcripts")
    parser.add_argument("--max-tickers", type=int, default=0,
                        help="Max tickers to process (0 = all)")
    parser.add_argument("--min-transcripts", type=int, default=3,
                        help="Minimum transcripts per ticker to process")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load Motley Fool data
    pkl_path = PROJECT_ROOT / "data" / "kaggle" / "motley-fool-data.pkl"
    logger.info("Loading Motley Fool dataset...")
    with open(pkl_path, "rb") as f:
        mf = pd.DataFrame(pickle.load(f))

    # Parse dates and quarters
    mf["earnings_date"] = mf["date"].apply(parse_mf_date)
    mf["quarter_norm"] = mf["q"].apply(normalize_quarter)

    # Drop rows with unparseable dates
    mf = mf.dropna(subset=["earnings_date"])
    logger.info(f"Loaded {len(mf)} transcripts with valid dates")

    # Load existing market data
    if MARKET_PATH.exists():
        existing = pd.read_parquet(MARKET_PATH)
        existing_pairs = set(zip(existing["ticker"], existing["quarter"]))
        logger.info(f"Existing market data: {len(existing_pairs)} events")
    else:
        existing = pd.DataFrame()
        existing_pairs = set()

    # Find missing events
    mf["has_data"] = mf.apply(lambda r: (r["ticker"], r["quarter_norm"]) in existing_pairs, axis=1)
    missing = mf[~mf["has_data"]].copy()
    logger.info(f"Missing market data: {len(missing)} transcripts")

    # Filter by min transcripts
    ticker_counts = missing["ticker"].value_counts()
    valid_tickers = ticker_counts[ticker_counts >= args.min_transcripts].index.tolist()
    missing = missing[missing["ticker"].isin(valid_tickers)]
    logger.info(f"After min-transcripts filter ({args.min_transcripts}+): "
                f"{len(valid_tickers)} tickers, {len(missing)} transcripts")

    if args.max_tickers > 0:
        valid_tickers = valid_tickers[:args.max_tickers]
        missing = missing[missing["ticker"].isin(valid_tickers)]
        logger.info(f"Limited to {args.max_tickers} tickers: {len(missing)} transcripts")

    if args.dry_run:
        logger.info("[DRY RUN] Would process:")
        for ticker in valid_tickers[:20]:
            n = len(missing[missing["ticker"] == ticker])
            logger.info(f"  {ticker}: {n} events")
        if len(valid_tickers) > 20:
            logger.info(f"  ... and {len(valid_tickers) - 20} more tickers")
        return

    # Process ticker by ticker
    all_new_records = []
    failed_tickers = []

    for i, ticker in enumerate(valid_tickers):
        ticker_events = missing[missing["ticker"] == ticker]
        events = [{"date": r["earnings_date"], "quarter": r["quarter_norm"]}
                  for _, r in ticker_events.iterrows()]

        logger.info(f"[{i+1}/{len(valid_tickers)}] {ticker}: {len(events)} events")

        try:
            records = process_ticker_events(ticker, events)
            all_new_records.extend(records)
            logger.info(f"  → {len(records)} records (with data)")
        except Exception as e:
            logger.warning(f"  → FAILED: {e}")
            failed_tickers.append(ticker)

        # Rate limit: yfinance throttles aggressively
        # ~1 request per ticker (cached benchmark), so 1s delay is safe
        time.sleep(1.0)
        if (i + 1) % 50 == 0:
            logger.info(f"  → Rate limit pause (5s)...")
            time.sleep(5)

        # Periodic save every 50 tickers
        if (i + 1) % 50 == 0 and all_new_records:
            _save(existing, all_new_records)
            logger.info(f"  → Checkpoint saved ({len(all_new_records)} new records so far)")

    # Final save
    if all_new_records:
        total = _save(existing, all_new_records)
        logger.info(f"Done! Total market data: {total} events")
    else:
        logger.info("No new records to save")

    if failed_tickers:
        logger.warning(f"Failed tickers ({len(failed_tickers)}): {failed_tickers[:20]}")


def _save(existing: pd.DataFrame, new_records: list) -> int:
    """Merge new records with existing and save."""
    new_df = pd.DataFrame(new_records)
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ticker", "quarter"], keep="last")
    else:
        combined = new_df

    MARKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(MARKET_PATH, index=False)
    return len(combined)


if __name__ == "__main__":
    main()
