"""Sentiment analysis utility functions."""

import re
import numpy as np
import pandas as pd


def preprocess_for_sentiment(text: str) -> list:
    """Lowercase, remove punctuation, tokenize."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def normalize_sentiment_score(score: float, method: str) -> float:
    """Normalize sentiment score to [-1, 1] range."""
    if method == "vader":
        return np.clip(score, -1, 1)
    elif method == "lm":
        return np.clip(score * 10, -1, 1)  # Scale up small LM scores
    elif method == "finbert":
        return np.clip(score, -1, 1)
    return score


def aggregate_sentiment(df: pd.DataFrame, group_by: list,
                        score_cols: list = None) -> pd.DataFrame:
    """Aggregate sentiment scores by grouping columns."""
    if score_cols is None:
        score_cols = [c for c in df.columns
                     if c.startswith(("lm_", "vader_", "finbert_"))
                     and df[c].dtype in ["float64", "float32"]]

    agg_dict = {col: ["mean", "std", "count"] for col in score_cols}
    result = df.groupby(group_by).agg(agg_dict)
    result.columns = ["_".join(col).strip("_") for col in result.columns]
    return result.reset_index()


def prepare_sentiment_chart_data(df: pd.DataFrame, ticker: str,
                                  score_col: str = "lm_net_score") -> dict:
    """Format sentiment data for plotly charts."""
    company_df = df[df["ticker"] == ticker].copy()
    if company_df.empty:
        return {"x": [], "y": [], "ticker": ticker}

    quarterly = company_df.groupby("quarter")[score_col].mean().reset_index()
    quarterly = quarterly.sort_values("quarter")

    return {
        "x": quarterly["quarter"].tolist(),
        "y": quarterly[score_col].tolist(),
        "ticker": ticker,
        "score_col": score_col,
    }
