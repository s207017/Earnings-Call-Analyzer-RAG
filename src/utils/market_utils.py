"""Market data utility functions."""

import numpy as np
import pandas as pd


def next_trading_day(date: pd.Timestamp, market_dates: pd.DatetimeIndex = None) -> pd.Timestamp:
    """Find next trading day (skip weekends)."""
    d = pd.Timestamp(date)
    while d.weekday() >= 5:  # Saturday=5, Sunday=6
        d += pd.Timedelta(days=1)
    return d


def prev_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    """Find previous trading day (skip weekends)."""
    d = pd.Timestamp(date)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def align_to_trading_day(date: str) -> pd.Timestamp:
    """Snap date to nearest trading day."""
    d = pd.Timestamp(date)
    if d.weekday() >= 5:
        return next_trading_day(d)
    return d


def compute_simple_return(prices: pd.Series, start_idx: int, end_idx: int) -> float:
    """Compute simple return between two indices."""
    if start_idx >= len(prices) or end_idx >= len(prices):
        return np.nan
    return (prices.iloc[end_idx] - prices.iloc[start_idx]) / prices.iloc[start_idx]


def compute_log_return(prices: pd.Series, start_idx: int, end_idx: int) -> float:
    """Compute log return between two indices."""
    if start_idx >= len(prices) or end_idx >= len(prices):
        return np.nan
    return np.log(prices.iloc[end_idx] / prices.iloc[start_idx])


def compute_cumulative_return(returns_series: pd.Series) -> pd.Series:
    """Compute cumulative returns from a series of simple returns."""
    return (1 + returns_series).cumprod() - 1
