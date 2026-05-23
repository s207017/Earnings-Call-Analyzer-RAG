"""
Train binary classification models for post-earnings stock direction prediction.
Saves trained models to models/direction_model.pkl for use by the Streamlit app.

Usage:
    python3.11 scripts/train_direction_model.py
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_MATRIX_PATH = PROJECT_ROOT / "outputs" / "feature_matrix.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "direction_model.pkl"


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "qa_sentiment" in df.columns and "prepared_sentiment" in df.columns:
        df["qa_weighted_sentiment"] = df["qa_sentiment"].fillna(0) * 0.65 + df["prepared_sentiment"].fillna(0) * 0.35
    if "mean_lm_negative_score" in df.columns:
        df["negative_emphasis"] = df["mean_lm_negative_score"] * 1.5
    if "mean_lm_negative_score" in df.columns and "mean_lm_positive_score" in df.columns:
        df["neg_pos_ratio"] = df["mean_lm_negative_score"] / (df["mean_lm_positive_score"] + 1e-8)
    if "mean_lm_net_score" in df.columns and "mean_lm_uncertainty_score" in df.columns:
        df["uncertainty_adj_sentiment"] = df["mean_lm_net_score"] - (df["mean_lm_uncertainty_score"] * 0.5)
    if "mgmt_sentiment" in df.columns and "analyst_sentiment" in df.columns:
        df["mgmt_analyst_divergence"] = df["mgmt_sentiment"] - df["analyst_sentiment"]
    if "mean_lm_net_score" in df.columns and "total_risk_count" in df.columns:
        df["risk_sentiment_interaction"] = df["mean_lm_net_score"] * df["total_risk_count"]
    return df


def main():
    logger.info("Loading feature matrix...")
    fm = pd.read_parquet(FEATURE_MATRIX_PATH)
    logger.info(f"Feature matrix: {len(fm)} rows, {fm.shape[1]} columns")

    enhanced = _engineer_features(fm)
    target = "abnormal_ret_1d"

    # Select feature columns with >70% coverage
    candidate_cols = [c for c in enhanced.columns
                      if c.startswith(("mean_", "mgmt_", "analyst_", "prepared_", "qa_",
                                       "sentiment_", "total_risk", "avg_risk", "num_risk",
                                       "negative_emphasis", "neg_pos", "uncertainty_adj",
                                       "mgmt_analyst", "risk_sentiment", "qa_weighted",
                                       "call_", "risk_delta"))]
    threshold = len(enhanced) * 0.7
    feature_cols = [c for c in candidate_cols if enhanced[c].notna().sum() >= threshold]
    logger.info(f"Features ({len(feature_cols)}): {feature_cols}")

    # Prepare data
    train_df = enhanced.dropna(subset=[target]).copy()
    train_df["direction"] = (train_df[target] > 0).astype(int)
    logger.info(f"Training samples: {len(train_df)}, positive rate: {train_df['direction'].mean():.4f}")

    X = train_df[feature_cols].astype(float).values
    y = train_df["direction"].values

    # Imputation
    medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        nan_mask = np.isnan(X[:, j])
        X[nan_mask, j] = medians[j]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Models
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
    }

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    baseline = max(y.mean(), 1 - y.mean())
    logger.info(f"Baseline accuracy: {baseline:.4f}")

    cv_results = {}
    feature_importances = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
        cv_results[name] = {"mean": float(scores.mean()), "std": float(scores.std())}
        logger.info(f"  {name}: {scores.mean():.4f} +/- {scores.std():.4f}")

        # Train on all data
        model.fit(X_scaled, y)
        if hasattr(model, "feature_importances_"):
            feature_importances[name] = dict(zip(feature_cols,
                                                  [float(x) for x in model.feature_importances_]))
        elif hasattr(model, "coef_"):
            feature_importances[name] = dict(zip(feature_cols,
                                                  [float(x) for x in model.coef_[0]]))

    # Save everything
    model_bundle = {
        "models": models,
        "scaler": scaler,
        "medians": medians,
        "feature_cols": feature_cols,
        "n_train": len(train_df),
        "baseline_accuracy": baseline,
        "cv_results": cv_results,
        "feature_importances": feature_importances,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_bundle, f)
    logger.info(f"Model saved to {MODEL_PATH}")

    # Also save a readable summary
    summary = {
        "n_train": len(train_df),
        "n_features": len(feature_cols),
        "baseline_accuracy": float(baseline),
        "cv_results": cv_results,
        "feature_cols": feature_cols,
    }
    summary_path = MODEL_PATH.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
