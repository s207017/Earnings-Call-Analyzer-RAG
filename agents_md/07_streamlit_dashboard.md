# Agent 7: Streamlit Dashboard Agent

## Objective
Build a Streamlit multi-page dashboard for the Earnings Call Analyzer with 6 pages covering all system outputs.

## Tasks

### 1. Create `app/app.py` — Main Entry Point
- Streamlit multipage app setup
- Sidebar with navigation, company/quarter filters
- Page title, project description
- Load shared data once and cache with `@st.cache_data`

### 2. Create `app/pages/01_company_overview.py`
- Company selector dropdown (10 companies)
- Basic stats: number of transcripts, total chunks, date range
- Summary table of all earnings calls for selected company
- Key metrics at a glance (average sentiment, top risks)

### 3. Create `app/pages/02_sentiment_trends.py`
- Line charts: sentiment score over quarters by company (plotly)
- Comparison: prepared remarks vs Q&A sentiment
- Heatmap: companies × quarters with sentiment color coding
- Speaker-level breakdown (CEO vs CFO vs Analysts)
- Filters: company, quarter range, speaker role, sentiment method (LM/FinBERT/VADER)

### 4. Create `app/pages/03_risk_monitoring.py`
- Risk heatmap: companies × 10 risk categories (plotly heatmap)
- Risk trends over time per company (line chart)
- Top risks per earnings call (bar chart)
- Risk severity distribution
- Drill-down: click a risk category to see the matching text excerpts

### 5. Create `app/pages/04_transcript_explorer.py`
- Browse transcripts by company and quarter
- View speaker turns with role labels
- Search within transcripts (keyword search)
- Highlight sentiment (color code positive/negative chunks)
- Section toggle (prepared remarks vs Q&A)

### 6. Create `app/pages/05_rag_qa.py`
- Chat-style interface for asking questions about earnings calls
- Filters: ticker, quarter (optional, or search across all)
- Display retrieved source chunks with citations
- Show relevance scores for each source
- Conversation history within the session

### 7. Create `app/pages/06_market_reaction.py`
- Post-earnings return charts (1d, 3d, 5d bars per company per quarter)
- Scatter plots: sentiment vs returns, risk vs returns
- Regression results display (coefficients table, R², p-values)
- Feature importance bar chart from ML models
- Portfolio sort results (return by sentiment quintile)
- CAR (Cumulative Abnormal Return) charts

### 8. Create `app/utils.py`
Shared dashboard helpers:
- Data loading functions (load parquet files from data/processed/ and outputs/)
- Chart styling (consistent color palette, plotly theme)
- Filter components (reusable company/quarter/role selectors)
- Formatting helpers (percentages, sentiment scores, dates)

## Data Sources
The dashboard reads from:
- `data/processed/all_chunks.parquet` — raw chunks
- `data/processed/chunks_with_sentiment.parquet` — chunks + sentiment scores
- `data/processed/chunks_with_risk.parquet` — chunks + risk labels
- `data/market/market_reactions.parquet` — market return data
- `outputs/feature_matrix.parquet` — joined NLP + market features
- `outputs/finance_results.json` — regression/ML results

## Files to Create
- `app/app.py` (new)
- `app/utils.py` (new)
- `app/pages/01_company_overview.py` (new)
- `app/pages/02_sentiment_trends.py` (new)
- `app/pages/03_risk_monitoring.py` (new)
- `app/pages/04_transcript_explorer.py` (new)
- `app/pages/05_rag_qa.py` (new)
- `app/pages/06_market_reaction.py` (new)

## Dependencies
- `streamlit`
- `plotly`
- `pandas`

## Quality Requirements
- All pages must handle missing data gracefully (show "No data available" messages, not crash)
- Charts should be interactive (plotly, not matplotlib)
- Use consistent color palette across all pages
- Cache data loading with `@st.cache_data` for performance
- Dashboard should be runnable with `streamlit run app/app.py`
- Responsive layout using `st.columns` where appropriate
