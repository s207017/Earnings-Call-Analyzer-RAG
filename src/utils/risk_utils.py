"""Risk detection utility functions."""

import pandas as pd


def prepare_risk_heatmap(risk_matrix: pd.DataFrame) -> dict:
    """Format risk matrix for plotly heatmap."""
    return {
        "z": risk_matrix.values.tolist(),
        "x": risk_matrix.columns.tolist(),
        "y": risk_matrix.index.tolist(),
    }


def prepare_risk_trends(trends_data: pd.DataFrame) -> dict:
    """Format risk trends for plotly line chart."""
    risk_cols = [c for c in trends_data.columns
                 if c not in ("quarter", "total_risk_count", "avg_risk_intensity")]
    return {
        "quarters": trends_data["quarter"].tolist(),
        "series": {
            col: trends_data[col].tolist() for col in risk_cols
        },
    }


def summarize_risks(risk_detections: list) -> dict:
    """Aggregate risk detections into summary."""
    cat_counts = {}
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    for d in risk_detections:
        if d.get("negated"):
            continue
        cat = d["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        sev = d.get("severity", "medium")
        if sev in severity_counts:
            severity_counts[sev] += 1

    return {
        "category_counts": dict(sorted(cat_counts.items(), key=lambda x: -x[1])),
        "severity_distribution": severity_counts,
        "total": sum(cat_counts.values()),
    }


def risk_severity_color_map() -> dict:
    """Color mapping for risk severity levels."""
    return {
        "high": "#e74c3c",
        "medium": "#f39c12",
        "low": "#27ae60",
        "negated": "#95a5a6",
    }
