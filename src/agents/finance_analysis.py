"""Finance analysis pipeline linking NLP features to post-earnings stock returns."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class FinanceAnalyzer:
    """Links NLP features (sentiment, risk) to market returns."""

    def __init__(self, project_root: str = None):
        if project_root is None:
            project_root = str(Path(__file__).parent.parent.parent)
        self.project_root = Path(project_root)

        config_path = self.project_root / "configs" / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.feature_matrix = None

    def _load_data(self):
        """Load sentiment, risk, and market data."""
        proc = self.project_root / self.config["paths"]["processed"]
        market = self.project_root / self.config["paths"]["market"]

        sent_path = proc / "chunks_with_sentiment.parquet"
        risk_path = proc / "chunks_with_risk.parquet"
        market_path = market / "market_reactions.parquet"

        dfs = {}
        for name, path in [("sentiment", sent_path), ("risk", risk_path), ("market", market_path)]:
            if path.exists():
                dfs[name] = pd.read_parquet(path)
                logger.info(f"Loaded {name}: {len(dfs[name])} rows")
            else:
                logger.warning(f"{name} data not found at {path}")
                dfs[name] = pd.DataFrame()

        return dfs

    def build_feature_matrix(self) -> pd.DataFrame:
        """Join NLP + market data into feature matrix per company-quarter."""
        dfs = self._load_data()
        sent_df = dfs["sentiment"]
        risk_df = dfs["risk"]
        market_df = dfs["market"]

        features = []
        # Use sentiment data as base (or risk if sentiment missing)
        base_df = sent_df if not sent_df.empty else risk_df
        if base_df.empty:
            logger.error("No NLP data available")
            return pd.DataFrame()

        for (ticker, quarter), group in base_df.groupby(["ticker", "quarter"]):
            feat = {"ticker": ticker, "quarter": quarter}

            # Sentiment features
            for col in ["lm_net_score", "lm_positive_score", "lm_negative_score",
                        "lm_uncertainty_score", "vader_compound",
                        "finbert_positive", "finbert_negative", "finbert_neutral"]:
                if col in group.columns:
                    feat[f"mean_{col}"] = group[col].mean()

            # Management vs analyst sentiment
            if "role" in group.columns and "lm_net_score" in group.columns:
                mgmt = group[group["role"].isin(["CEO", "CFO", "COO", "CTO"])]
                analysts = group[group["role"] == "Analyst"]
                feat["mgmt_sentiment"] = mgmt["lm_net_score"].mean() if not mgmt.empty else np.nan
                feat["analyst_sentiment"] = analysts["lm_net_score"].mean() if not analysts.empty else np.nan

            # Prepared vs Q&A sentiment
            if "section" in group.columns and "lm_net_score" in group.columns:
                prepared = group[group["section"] == "prepared_remarks"]
                qa = group[group["section"] == "qa"]
                feat["prepared_sentiment"] = prepared["lm_net_score"].mean() if not prepared.empty else np.nan
                feat["qa_sentiment"] = qa["lm_net_score"].mean() if not qa.empty else np.nan
                if not np.isnan(feat.get("prepared_sentiment", np.nan)) and not np.isnan(feat.get("qa_sentiment", np.nan)):
                    feat["sentiment_divergence"] = feat["prepared_sentiment"] - feat["qa_sentiment"]

            # Risk features (from risk_df if available)
            if not risk_df.empty:
                risk_group = risk_df[(risk_df["ticker"] == ticker) & (risk_df["quarter"] == quarter)]
                if not risk_group.empty:
                    feat["total_risk_count"] = risk_group["risk_count"].sum() if "risk_count" in risk_group.columns else 0
                    feat["avg_risk_intensity"] = risk_group["risk_intensity"].mean() if "risk_intensity" in risk_group.columns else 0
                    # Count unique risk categories
                    if "risk_categories" in risk_group.columns:
                        all_cats = []
                        for cats in risk_group["risk_categories"]:
                            if isinstance(cats, str):
                                try:
                                    all_cats.extend(json.loads(cats))
                                except:
                                    pass
                            elif isinstance(cats, list):
                                all_cats.extend(cats)
                        feat["num_risk_categories"] = len(set(all_cats))

            features.append(feat)

        feat_df = pd.DataFrame(features)

        # Add sentiment momentum (change vs prior quarter)
        feat_df = feat_df.sort_values(["ticker", "quarter"])
        for col in ["mean_lm_net_score", "mean_vader_compound"]:
            if col in feat_df.columns:
                feat_df[f"{col}_momentum"] = feat_df.groupby("ticker")[col].diff()

        # Risk delta
        if "total_risk_count" in feat_df.columns:
            feat_df["risk_delta"] = feat_df.groupby("ticker")["total_risk_count"].diff()

        # Merge with market data
        if not market_df.empty:
            feat_df = feat_df.merge(market_df, on=["ticker", "quarter"], how="left")

        self.feature_matrix = feat_df

        # Save
        out_path = self.project_root / self.config["paths"]["outputs"] / "feature_matrix.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        feat_df.to_parquet(out_path, index=False)
        logger.info(f"Feature matrix saved: {len(feat_df)} rows, {len(feat_df.columns)} columns")
        return feat_df

    def event_study(self, windows: List[int] = None) -> Dict:
        """Compute CARs with t-test significance."""
        from scipy import stats

        if self.feature_matrix is None:
            self.build_feature_matrix()

        if windows is None:
            windows = [1, 3, 5]

        results = {}
        for w in windows:
            col = f"abnormal_ret_{w}d"
            if col not in self.feature_matrix.columns:
                continue
            returns = self.feature_matrix[col].dropna()
            if len(returns) < 3:
                continue

            t_stat, p_value = stats.ttest_1samp(returns, 0)
            results[f"car_{w}d"] = {
                "mean": float(returns.mean()),
                "std": float(returns.std()),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "n": int(len(returns)),
                "significant_5pct": bool(p_value < 0.05),
            }
        return results

    def portfolio_sorts(self, feature: str = "mean_lm_net_score",
                        target: str = "abnormal_ret_1d",
                        n_buckets: int = 3) -> pd.DataFrame:
        """Sort companies into buckets by NLP feature, compute avg returns."""
        if self.feature_matrix is None:
            self.build_feature_matrix()

        df = self.feature_matrix.dropna(subset=[feature, target])
        if len(df) < n_buckets * 2:
            logger.warning(f"Not enough observations for {n_buckets} buckets")
            return pd.DataFrame()

        df["bucket"] = pd.qcut(df[feature], n_buckets, labels=False, duplicates="drop")
        result = df.groupby("bucket").agg(
            count=(target, "count"),
            mean_return=(target, "mean"),
            std_return=(target, "std"),
            mean_feature=(feature, "mean"),
        ).reset_index()
        result["bucket_label"] = ["Low", "Mid", "High"][:len(result)]
        return result

    def run_regression(self, target: str = "abnormal_ret_1d",
                       features: List[str] = None) -> Dict:
        """Run OLS regression of NLP features on returns."""
        import statsmodels.api as sm

        if self.feature_matrix is None:
            self.build_feature_matrix()

        if features is None:
            # Only use features with >80% non-null coverage
            candidates = [c for c in self.feature_matrix.columns
                         if c.startswith(("mean_", "mgmt_", "analyst_", "prepared_", "qa_",
                                         "sentiment_", "total_risk", "avg_risk", "num_risk"))]
            threshold = len(self.feature_matrix) * 0.8
            features = [c for c in candidates if self.feature_matrix[c].notna().sum() >= threshold]

        df = self.feature_matrix.dropna(subset=[target] + features)
        if len(df) < len(features) + 2:
            logger.warning(f"Not enough observations for regression ({len(df)} rows, {len(features)} features)")
            return {}

        X = sm.add_constant(df[features].astype(float))
        y = df[target].astype(float)

        model = sm.OLS(y, X).fit()

        return {
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_statistic": float(model.fvalue),
            "f_pvalue": float(model.f_pvalue),
            "n_obs": int(model.nobs),
            "coefficients": {
                name: {
                    "coef": float(model.params[name]),
                    "std_err": float(model.bse[name]),
                    "t_stat": float(model.tvalues[name]),
                    "p_value": float(model.pvalues[name]),
                }
                for name in model.params.index
            },
            "summary": str(model.summary()),
        }

    def run_ml_models(self, target: str = "abnormal_ret_1d",
                      features: List[str] = None) -> Dict:
        """Run ML models (Lasso, Ridge, RF, GBM) with cross-validation."""
        from sklearn.linear_model import Lasso, Ridge
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        if self.feature_matrix is None:
            self.build_feature_matrix()

        if features is None:
            candidates = [c for c in self.feature_matrix.columns
                         if c.startswith(("mean_", "mgmt_", "analyst_", "prepared_", "qa_",
                                         "sentiment_", "total_risk", "avg_risk", "num_risk"))]
            threshold = len(self.feature_matrix) * 0.8
            features = [c for c in candidates if self.feature_matrix[c].notna().sum() >= threshold]

        df = self.feature_matrix.dropna(subset=[target] + features)
        if len(df) < 10:
            logger.warning(f"Not enough data for ML models ({len(df)} rows)")
            return {}

        X = df[features].astype(float).values
        y = df[target].astype(float).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_cv = min(5, len(df) // 3)
        if n_cv < 2:
            n_cv = 2

        models = {
            "lasso": Lasso(alpha=0.01, max_iter=10000),
            "ridge": Ridge(alpha=1.0),
            "random_forest": RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=42),
        }

        results = {}
        for name, model in models.items():
            try:
                cv_scores = cross_val_score(model, X_scaled, y, cv=n_cv, scoring="r2")
                model.fit(X_scaled, y)

                result = {
                    "cv_r2_mean": float(cv_scores.mean()),
                    "cv_r2_std": float(cv_scores.std()),
                    "cv_scores": [float(s) for s in cv_scores],
                }

                # Feature importance for tree models
                if hasattr(model, "feature_importances_"):
                    importance = dict(zip(features, [float(x) for x in model.feature_importances_]))
                    result["feature_importance"] = dict(sorted(importance.items(), key=lambda x: -x[1]))
                elif hasattr(model, "coef_"):
                    coefs = dict(zip(features, [float(x) for x in model.coef_]))
                    result["coefficients"] = coefs

                results[name] = result
                logger.info(f"{name}: CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            except Exception as e:
                logger.error(f"{name} failed: {e}")
                results[name] = {"error": str(e)}

        return results

    def run_all(self) -> Dict:
        """Run complete finance analysis pipeline."""
        logger.info("Building feature matrix...")
        self.build_feature_matrix()

        results = {}
        logger.info("Running event study...")
        results["event_study"] = self.event_study()

        logger.info("Running portfolio sorts...")
        for feat in ["mean_lm_net_score", "mean_vader_compound"]:
            if feat in self.feature_matrix.columns:
                sorts = self.portfolio_sorts(feature=feat)
                if not sorts.empty:
                    results[f"portfolio_sorts_{feat}"] = sorts.to_dict(orient="records")

        logger.info("Running regression...")
        results["regression"] = self.run_regression()

        logger.info("Running ML models...")
        results["ml_models"] = self.run_ml_models()

        # Save results
        out_path = self.project_root / self.config["paths"]["outputs"] / "finance_results.json"

        def default_serializer(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=default_serializer)
        logger.info(f"Results saved to {out_path}")
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    analyzer = FinanceAnalyzer()
    results = analyzer.run_all()
    print("Finance analysis complete.")
    for key in results:
        print(f"  {key}: {type(results[key])}")
