"""Finance analysis backtesting and evaluation."""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


def walk_forward_cv(model, X, y, n_splits: int = 3) -> Dict:
    """Time-series aware walk-forward cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(y_train) < 5:
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = out_of_sample_r2(y_test, y_pred)
        scores.append(r2)

    return {
        "cv_scores": scores,
        "mean_r2": float(np.mean(scores)) if scores else np.nan,
        "std_r2": float(np.std(scores)) if scores else np.nan,
        "n_splits": len(scores),
    }


def out_of_sample_r2(y_true, y_pred) -> float:
    """Compute out-of-sample R-squared."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def model_comparison_table(results: Dict) -> pd.DataFrame:
    """Create comparison table across models."""
    rows = []
    for name, metrics in results.items():
        if isinstance(metrics, dict) and "cv_r2_mean" in metrics:
            rows.append({
                "model": name,
                "cv_r2_mean": metrics["cv_r2_mean"],
                "cv_r2_std": metrics["cv_r2_std"],
            })
    return pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False)


def portfolio_backtest(df: pd.DataFrame, feature_col: str,
                       return_col: str, n_periods: int = None) -> Dict:
    """Simulate portfolio returns over time based on NLP feature sorting."""
    df = df.dropna(subset=[feature_col, return_col]).sort_values("quarter")
    quarters = sorted(df["quarter"].unique())

    if n_periods is None:
        n_periods = len(quarters)

    long_returns = []
    short_returns = []

    for q in quarters[-n_periods:]:
        q_df = df[df["quarter"] == q]
        if len(q_df) < 4:
            continue

        median_feat = q_df[feature_col].median()
        long = q_df[q_df[feature_col] >= median_feat][return_col].mean()
        short = q_df[q_df[feature_col] < median_feat][return_col].mean()
        long_returns.append(long)
        short_returns.append(short)

    long_cum = np.cumprod([1 + r for r in long_returns]) - 1 if long_returns else []
    short_cum = np.cumprod([1 + r for r in short_returns]) - 1 if short_returns else []

    return {
        "quarters": quarters[-len(long_returns):] if long_returns else [],
        "long_returns": [float(r) for r in long_returns],
        "short_returns": [float(r) for r in short_returns],
        "long_cumulative": [float(r) for r in long_cum],
        "short_cumulative": [float(r) for r in short_cum],
        "spread_mean": float(np.mean([l - s for l, s in zip(long_returns, short_returns)])) if long_returns else np.nan,
    }
