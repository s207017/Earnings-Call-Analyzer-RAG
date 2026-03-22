"""Market data collection and post-earnings return computation."""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MarketDataCollector:
    """Fetches stock prices and computes post-earnings returns."""

    def __init__(self, project_root: str = None):
        if project_root is None:
            project_root = str(Path(__file__).parent.parent.parent)
        self.project_root = Path(project_root)

        config_path = self.project_root / "configs" / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        calendar_path = self.project_root / "configs" / "earnings_calendar.json"
        with open(calendar_path) as f:
            self.calendar = json.load(f)

        self.cache_dir = self.project_root / self.config["paths"]["market_cache"]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark = self.config["market"]["benchmark"]
        self.windows = self.config["market"]["return_windows"]
        self.volume_lookback = self.config["market"]["volume_lookback"]

    def fetch_prices(self, ticker: str, start: str, end: str,
                     max_retries: int = 3) -> pd.DataFrame:
        """Fetch price data with caching and retry logic."""
        import yfinance as yf

        cache_file = self.cache_dir / f"{ticker}_{start}_{end}.parquet"
        if cache_file.exists():
            logger.debug(f"Cache hit for {ticker}")
            return pd.read_parquet(cache_file)

        for attempt in range(max_retries):
            try:
                t = yf.Ticker(ticker)
                data = t.history(start=start, end=end, auto_adjust=True)
                if data.empty:
                    logger.warning(f"No data for {ticker} ({start} to {end})")
                    return pd.DataFrame()
                # Ensure timezone-naive index for parquet compatibility
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                data.to_parquet(cache_file)
                return data
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {ticker}: {e}")
                time.sleep(2 ** attempt)

        logger.error(f"Failed to fetch {ticker} after {max_retries} attempts")
        return pd.DataFrame()

    def _find_trading_day(self, prices: pd.DataFrame, target_date: str,
                          direction: int = 1) -> Optional[pd.Timestamp]:
        """Find nearest trading day in given direction (1=forward, -1=backward)."""
        target = pd.Timestamp(target_date)
        dates = prices.index.sort_values()

        if direction >= 0:
            future = dates[dates >= target]
            return future[0] if len(future) > 0 else None
        else:
            past = dates[dates <= target]
            return past[-1] if len(past) > 0 else None

    def compute_returns(self, ticker: str, earnings_date: str,
                        windows: List[int] = None) -> Dict:
        """Compute post-earnings returns over multiple windows."""
        if windows is None:
            windows = self.windows

        # Fetch wide window of data
        ed = pd.Timestamp(earnings_date)
        start = (ed - pd.Timedelta(days=self.volume_lookback + 10)).strftime("%Y-%m-%d")
        end = (ed + pd.Timedelta(days=max(windows) + 5)).strftime("%Y-%m-%d")

        prices = self.fetch_prices(ticker, start, end)
        if prices.empty or "Close" not in prices.columns:
            return {f"ret_{w}d": np.nan for w in windows}

        close = prices["Close"]

        # Find the trading day on or just before earnings
        base_date = self._find_trading_day(prices, earnings_date, direction=-1)
        if base_date is None:
            return {f"ret_{w}d": np.nan for w in windows}

        base_price = close.loc[base_date]
        results = {}

        for w in windows:
            future_dates = close.index[close.index > base_date]
            if len(future_dates) >= w:
                end_price = close.iloc[close.index.get_indexer([future_dates[w-1]], method="nearest")[0]]
                results[f"ret_{w}d"] = float((end_price - base_price) / base_price)
            else:
                results[f"ret_{w}d"] = np.nan

        return results

    def compute_abnormal_returns(self, ticker: str, earnings_date: str,
                                  windows: List[int] = None) -> Dict:
        """Compute abnormal returns relative to benchmark."""
        if windows is None:
            windows = self.windows

        stock_returns = self.compute_returns(ticker, earnings_date, windows)
        bench_returns = self.compute_returns(self.benchmark, earnings_date, windows)

        results = {}
        for w in windows:
            sr = stock_returns.get(f"ret_{w}d", np.nan)
            br = bench_returns.get(f"ret_{w}d", np.nan)
            if np.isnan(sr) or np.isnan(br):
                results[f"abnormal_ret_{w}d"] = np.nan
            else:
                results[f"abnormal_ret_{w}d"] = sr - br

        return results

    def compute_abnormal_volume(self, ticker: str, earnings_date: str) -> float:
        """Compute abnormal volume ratio around earnings vs trailing average."""
        ed = pd.Timestamp(earnings_date)
        start = (ed - pd.Timedelta(days=self.volume_lookback + 10)).strftime("%Y-%m-%d")
        end = (ed + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

        prices = self.fetch_prices(ticker, start, end)
        if prices.empty or "Volume" not in prices.columns:
            return np.nan

        volume = prices["Volume"]
        ed_idx = self._find_trading_day(prices, earnings_date, direction=1)
        if ed_idx is None:
            return np.nan

        # Average volume in lookback window
        prior = volume[volume.index < ed_idx].tail(self.volume_lookback)
        if len(prior) < 5:
            return np.nan

        avg_vol = prior.mean()
        if avg_vol == 0:
            return np.nan

        earnings_vol = volume.loc[ed_idx] if ed_idx in volume.index else np.nan
        if np.isnan(earnings_vol):
            return np.nan

        return float(earnings_vol / avg_vol)

    def collect_all(self) -> pd.DataFrame:
        """Fetch all market data for all company-quarter combos."""
        records = []
        for event in tqdm(self.calendar, desc="Market Data"):
            ticker = event["ticker"]
            quarter = event["quarter"]
            earnings_date = event["earnings_date"]

            logger.info(f"Processing {ticker} {quarter} ({earnings_date})")

            returns = self.compute_returns(ticker, earnings_date)
            abnormal = self.compute_abnormal_returns(ticker, earnings_date)
            abn_vol = self.compute_abnormal_volume(ticker, earnings_date)

            record = {
                "ticker": ticker,
                "quarter": quarter,
                "earnings_date": earnings_date,
                **returns,
                **abnormal,
                "abnormal_volume": abn_vol,
            }
            records.append(record)
            time.sleep(0.5)  # Rate limiting

        df = pd.DataFrame(records)
        out_path = self.project_root / self.config["paths"]["market"] / "market_reactions.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved market data to {out_path}")
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    collector = MarketDataCollector()
    df = collector.collect_all()
    print(f"Collected market data for {len(df)} events")
    print(df.head())
