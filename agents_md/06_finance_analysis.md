# Agent 6: Finance Analysis Agent

## Objective
Build the finance analysis pipeline that links NLP features (sentiment, risk) from earnings calls to post-earnings stock returns. This is a CORE deliverable — not optional.

## Tasks

### 1. Rewrite `src/agents/finance_analysis.py`
The current scaffold has basic correlation and regression. Build a comprehensive analysis:

#### Feature Engineering
- **Sentiment features**: Mean sentiment per call, management vs analyst sentiment, prepared remarks vs Q&A sentiment
- **Sentiment momentum**: Change in sentiment vs prior quarter
- **Risk features**: Total risk intensity, count of risk categories flagged, dominant risk category
- **Risk delta**: Change in risk signals vs prior quarter
- **Divergence features**: Gap between Q&A tone and prepared remarks tone (management confidence proxy)

#### Event Study
- **CAR (Cumulative Abnormal Returns)**: Compute CARs over [-1, +1], [-1, +3], [-1, +5] windows
- **Abnormal returns** relative to market (SPY) benchmark
- **Statistical significance** of CARs (t-test)

#### Portfolio Analysis
- **Portfolio sorts**: Sort companies into quintiles/terciles by sentiment score, compute average returns per bucket
- **Long-short portfolio**: Long top-sentiment, short bottom-sentiment, track returns

#### Regression & ML
- **OLS regression**: NLP features → returns, with proper statistical output (coefficients, p-values, R², F-stat)
- **Lasso/Ridge**: For feature selection when many features
- **Random Forest & Gradient Boosting**: Non-linear models with feature importance
- **Cross-validation**: Walk-forward or k-fold CV with proper time-series handling
- **Bootstrap confidence intervals** for key estimates

#### Output
- Feature importance rankings
- Regression summary tables
- Portfolio return tables
- All results saved to `outputs/`

### 2. Create `src/utils/finance_utils.py`
Helper functions:
- Return calculations (simple, log, cumulative, abnormal)
- Statistical tests (t-test, bootstrap CI, Newey-West standard errors)
- Feature engineering helpers (momentum, delta, divergence)
- Portfolio sort helpers
- Winsorization for outlier handling

### 3. Create `src/evaluation/finance_eval.py`
Backtesting framework:
- Walk-forward validation (train on quarters 1-N, test on N+1)
- Out-of-sample R² and accuracy metrics
- Portfolio backtest (simulated returns over time)
- Comparison table across models

## Files to Create/Modify
- `src/agents/finance_analysis.py` (rewrite existing)
- `src/utils/finance_utils.py` (new)
- `src/evaluation/finance_eval.py` (new)

## Dependencies
- `statsmodels` (OLS, statistical tests)
- `scikit-learn` (RF, GBM, Lasso, Ridge, cross-validation)
- `scipy` (statistical tests)
- `pandas`, `numpy`

## Quality Requirements
- Must handle small sample sizes gracefully (80 observations max)
- Statistical tests must be properly implemented (not just correlations)
- Feature matrix must handle missing data (dropna, imputation)
- Walk-forward validation must respect time ordering (no look-ahead bias)
- All results should be serializable to JSON for the dashboard
- Code should be importable and usable by the Orchestrator agent
