"""Finance analysis utility functions."""

import numpy as np
import pandas as pd
from scipy import stats


def winsorize(series: pd.Series, limits=(0.05, 0.95)) -> pd.Series:
    """Winsorize a series at given percentile limits."""
    lower = series.quantile(limits[0])
    upper = series.quantile(limits[1])
    return series.clip(lower=lower, upper=upper)


def bootstrap_ci(data, stat_func=np.mean, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval."""
    data = np.array(data)
    data = data[~np.isnan(data)]
    if len(data) < 3:
        return {"estimate": float(stat_func(data)), "ci_lower": np.nan, "ci_upper": np.nan}

    boot_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(stat_func(sample))

    alpha = (1 - ci) / 2
    return {
        "estimate": float(stat_func(data)),
        "ci_lower": float(np.percentile(boot_stats, alpha * 100)),
        "ci_upper": float(np.percentile(boot_stats, (1 - alpha) * 100)),
    }


def newey_west_se(model, max_lags=None):
    """Compute Newey-West HAC standard errors for OLS model."""
    import statsmodels.api as sm
    if max_lags is None:
        max_lags = int(np.floor(4 * (model.nobs / 100) ** (2 / 9)))
    robust = model.get_robustcov_results(cov_type="HAC", maxlags=max_lags)
    return robust.bse


def compute_car(returns: pd.Series, window: int) -> float:
    """Compute Cumulative Abnormal Return over a window."""
    if len(returns) < window:
        return np.nan
    return float(returns.iloc[:window].sum())


def portfolio_sort_table(df: pd.DataFrame, feature_col: str,
                         return_col: str, n_buckets: int = 3) -> pd.DataFrame:
    """Create portfolio sort results table."""
    clean = df.dropna(subset=[feature_col, return_col])
    if len(clean) < n_buckets * 2:
        return pd.DataFrame()

    clean["bucket"] = pd.qcut(clean[feature_col], n_buckets, labels=False, duplicates="drop")
    result = clean.groupby("bucket").agg(
        n=(return_col, "count"),
        mean_return=(return_col, "mean"),
        std_return=(return_col, "std"),
        mean_feature=(feature_col, "mean"),
    ).reset_index()

    # T-test for long-short spread
    if len(result) >= 2:
        top = clean[clean["bucket"] == result["bucket"].max()][return_col]
        bottom = clean[clean["bucket"] == result["bucket"].min()][return_col]
        if len(top) > 1 and len(bottom) > 1:
            t_stat, p_val = stats.ttest_ind(top, bottom)
            result.attrs["long_short_t"] = float(t_stat)
            result.attrs["long_short_p"] = float(p_val)

    return result
