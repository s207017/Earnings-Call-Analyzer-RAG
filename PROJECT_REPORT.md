# Earnings Call Analyzer

A finance-focused NLP and Retrieval-Augmented Generation (RAG) system that analyzes earnings call transcripts from 49 publicly traded companies across 28 quarters (2018–2024). The system extracts sentiment signals, detects financial risk factors, links NLP features to post-earnings stock returns, and provides an interactive Q&A interface powered by a local LLM — all through a multi-page Streamlit dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Data Pipeline](#data-pipeline)
4. [NLP Analysis](#nlp-analysis)
   - [Sentiment Analysis](#sentiment-analysis)
   - [Risk Signal Detection](#risk-signal-detection)
5. [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
6. [Finance Analysis](#finance-analysis)
7. [Interactive Dashboard](#interactive-dashboard)
8. [PDF Upload Feature](#pdf-upload-feature)
9. [Project Structure](#project-structure)
10. [Setup & Running](#setup--running)

---

## Project Overview

Earnings calls are quarterly events where company executives discuss financial results and take questions from analysts. These calls contain rich qualitative information — management tone, forward guidance, risk disclosures — that complements quantitative financial data.

This project builds an end-to-end analytics pipeline that:

- **Ingests** 1,108 earnings call transcripts (30,235 text chunks) from two Kaggle datasets
- **Analyzes sentiment** using three complementary methods (Loughran-McDonald, VADER, FinBERT)
- **Detects risk signals** across 10 financial risk categories with negation-awareness and severity scoring
- **Links NLP features to market reactions** via regression, machine learning models, and event studies
- **Enables natural language Q&A** over any transcript using hybrid retrieval (FAISS + BM25) and a local LLM (Llama 3.2 via Ollama)
- **Provides a live upload feature** where users can upload a new earnings call PDF and receive instant analytics with benchmarking against the historical dataset

### Scale

| Metric | Count |
|--------|-------|
| Companies | 49 |
| Quarters | 28 (2018Q1–2024Q4) |
| Transcripts | 1,108 |
| Text Chunks | 30,235 |
| Market Events | 1,372 |
| Feature Matrix Rows | 1,108 |
| Feature Columns | 26 |
| Risk Categories | 10 |
| Risk Keywords | 255 |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard                      │
│  ┌──────────┐ ┌──────────┐ ┌──────┐ ┌──────┐ ┌───────────┐ │
│  │ Sentiment│ │   Risk   │ │ RAG  │ │Market│ │  Upload   │ │
│  │  Trends  │ │ Monitor  │ │ Q&A  │ │React.│ │ & Analyze │ │
│  └────┬─────┘ └────┬─────┘ └──┬───┘ └──┬───┘ └─────┬─────┘ │
└───────┼────────────┼──────────┼────────┼───────────┼────────┘
        │            │          │        │           │
┌───────▼────────────▼──────────▼────────▼───────────▼────────┐
│                      Processing Layer                        │
│                                                              │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │  Sentiment   │  │ Risk Detection│  │    Retrieval     │  │
│  │  Analyzer    │  │   (Taxonomy)  │  │  (FAISS + BM25)  │  │
│  │ LM/VADER/FB  │  │  10 categories│  │  Hybrid Search   │  │
│  └──────┬───────┘  └──────┬────────┘  └────────┬─────────┘  │
│         │                 │                     │            │
│  ┌──────▼─────────────────▼─────────────────────▼─────────┐ │
│  │              Feature Matrix (1,108 rows)               │ │
│  │    NLP features + Market returns per company-quarter    │ │
│  └────────────────────────┬───────────────────────────────┘ │
│                           │                                  │
│  ┌────────────────────────▼───────────────────────────────┐ │
│  │              Finance Analysis                          │ │
│  │  Event Study | Regression | Portfolio Sorts | ML Models│ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────────┐
│                        Data Layer                             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ Kaggle Data  │  │ yfinance API │  │  Ollama (Llama 3.2) │ │
│  │ Motley Fool  │  │ Stock Prices │  │  Local LLM for Q&A  │ │
│  │ AlphaSpread  │  │ SPY Benchmark│  │  Free, no API key   │ │
│  └──────────────┘  └──────────────┘  └─────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline

### Data Sources

1. **Motley Fool Dataset** (`data/kaggle/motley-fool-data.pkl`)
   - Pickle file containing earnings call transcripts
   - Columns: `ticker`, `q` (quarter), `date`, `transcript`
   - Primary source for US-listed companies

2. **AlphaSpread Dataset** (`data/kaggle/archive (2)/`)
   - Two subdirectories: `cleaned_ECTs_dataset/` and `NLP_Dataset/`
   - Organized as folders per company, with `.txt` files per quarter
   - Filename pattern: `2023_Q3_aapl_processed.txt`

### Data Collection (`src/data_collection.py`)

The `DataCollector` class:

1. **Loads** transcripts from both Kaggle datasets
2. **Normalizes** ticker symbols (e.g., `GOOG` → `GOOGL`, `FB` → `META`, `ANTM` → `ELV`)
3. **Normalizes** quarter formats (e.g., `Q1 2023`, `2023-Q1`, `FY2023Q1` → `2023Q1`)
4. **Maps** AlphaSpread folder names to tickers (e.g., `"Schneider Electric"` → `SCHNEIDER.PA`)
5. **Deduplicates** by keeping the longer transcript when both sources cover the same company-quarter
6. **Saves** individual files to `data/raw/{TICKER}/{QUARTER}.txt`
7. **Generates** `data/manifest.json` listing all available transcripts

### Transcript Parsing (`src/agents/transcript_ingestion.py`)

The `TranscriptParser` converts raw transcripts into structured chunks:

1. **Boilerplate removal** — strips safe harbor disclaimers, copyright notices, Motley Fool attribution
2. **Speaker detection** — uses 6 regex patterns to identify speaker turns (e.g., `Name -- Title`, `Name (Title)`, `Name:`)
3. **Role classification** — maps titles to roles: CEO, CFO, COO, CTO, VP, IR, Analyst, Operator, Other
4. **Section detection** — identifies prepared remarks vs. Q&A using keyword cues (e.g., "question-and-answer", "open the call to questions")
5. **Sentence-aware chunking** — splits text into ~400-token chunks with 50-token overlap, respecting sentence boundaries

**Output:** `data/processed/all_chunks.parquet` with columns:
- `chunk_id`: unique identifier (e.g., `AAPL_2024Q3_0012`)
- `ticker`, `quarter`: company and period
- `speaker`, `role`: who said it and their role
- `section`: `prepared_remarks` or `qa`
- `text`: the chunk content
- `chunk_index`: position within the transcript

### Companies Covered

The system covers 49 companies across 9 sectors:

| Sector | Companies |
|--------|-----------|
| Technology | AAPL, MSFT, GOOGL, AMZN, META, NVDA, CRM, ORCL, AMD, CSCO, IBM, ADBE, ACN, AMS.PA, CAP.PA |
| Financials | PYPL, MA, AXP, JPM, BAC, C, DB, ALL, BAM, BBVA |
| Consumer Discretionary | TSLA, NKE, LULU, MAR, BKNG, F, GM, BMW.DE, MC.PA |
| Communication Services | NFLX, DIS |
| Consumer Staples | COST, WMT, OR.PA |
| Healthcare | CAH, ELV, UNH |
| Energy | BP, SHEL |
| Industrials | VOLVY, SIE.DE, AIR.PA, SCHNEIDER.PA |
| Utilities | ENGI.PA, EDF.PA |

---

## NLP Analysis

### Sentiment Analysis

Three complementary sentiment methods are applied to every chunk (`src/agents/sentiment_analysis.py`):

#### 1. Loughran-McDonald (LM) Dictionary

A finance-specific lexicon designed for 10-K filings and earnings calls. Unlike general-purpose sentiment dictionaries, LM correctly handles financial language (e.g., "liability" is negative in general English but neutral in finance).

- **341 positive words** (e.g., "achieve", "benefit", "growth", "improve", "profitable")
- **951 negative words** (e.g., "adverse", "decline", "impairment", "restructuring", "weakness")
- **219 uncertainty words** (e.g., "approximately", "contingent", "uncertain", "volatile")
- **100+ litigious words** (e.g., "lawsuit", "litigation", "plaintiff", "tribunal")
- **100+ constraining words** (e.g., "commit", "obligation", "restrict", "binding")

**Scoring:** For each chunk, counts matching words and normalizes by total word count:
```
lm_net_score = (positive_count - negative_count) / total_words
```

#### 2. VADER (Valence Aware Dictionary and sEntiment Reasoner)

A rule-based model that handles:
- Punctuation emphasis ("Great!!!" > "Great")
- Capitalization ("GREAT" > "great")
- Degree modifiers ("extremely good" > "good")
- Conjunctions ("good but not great")
- Negation ("not good")

**Output:** Compound score from -1 (most negative) to +1 (most positive).

#### 3. FinBERT (Optional)

A pre-trained BERT model fine-tuned on financial text (ProsusAI/finbert). Classifies each chunk as positive, negative, or neutral with a confidence score. Runs with batching (batch_size=16, max_length=512).

#### Derived Sentiment Features

Beyond raw scores, the system computes:

- **Management vs. Analyst sentiment** — compares sentiment of CEO/CFO statements against analyst questions
- **Prepared remarks vs. Q&A sentiment** — compares the scripted portion against spontaneous responses
- **Sentiment divergence** — the gap between prepared and Q&A sentiment (large divergence may signal management overly positive in scripted remarks)
- **Sentiment momentum** — quarter-over-quarter change in sentiment for the same company

### Risk Signal Detection

The risk detection module (`src/agents/risk_detection.py`) identifies financial risk signals using a taxonomy-based approach with context awareness.

#### Risk Taxonomy

10 risk categories with 255 total keywords (`configs/risk_taxonomy.json`):

| Category | Example Keywords | Description |
|----------|-----------------|-------------|
| `demand_risk` | "declining demand", "customer churn", "lower bookings" | Signals of weakening demand |
| `margin_risk` | "margin pressure", "cost inflation", "pricing pressure" | Threats to profitability |
| `supply_chain_risk` | "supply constraints", "chip shortage", "logistics challenges" | Supply disruptions |
| `regulatory_risk` | "regulatory scrutiny", "antitrust", "compliance costs" | Legal/regulatory threats |
| `competition_risk` | "competitive pressure", "market share loss", "price war" | Competitive dynamics |
| `fx_macro_risk` | "currency headwinds", "forex impact", "inflation" | Macro/FX exposure |
| `liquidity_risk` | "cash burn", "debt maturity", "liquidity concerns" | Balance sheet stress |
| `tech_execution_risk` | "product delays", "integration challenges", "technical debt" | Technology/execution risk |
| `labor_risk` | "talent shortage", "attrition", "labor costs" | Workforce challenges |
| `geopolitical_risk` | "tariff", "trade war", "sanctions", "geopolitical" | Political/trade risks |

#### Context-Aware Detection

The detector goes beyond simple keyword matching:

1. **Negation detection** — checks a 3-word window before each keyword for negation words (e.g., "no", "not", "without", "never"). A negated risk signal (e.g., "no margin pressure") is flagged but excluded from active risk counts.

2. **Severity scoring** — examines a 5-word window around each keyword for:
   - **Intensifiers** (→ high severity): "significant", "severe", "major", "unprecedented", "critical"
   - **Diminishers** (→ low severity): "slight", "minor", "modest", "temporary", "manageable"
   - Default: medium severity

3. **Risk intensity** — weighted score: high=3, medium=2, low=1, summed across all detections per chunk

**Output columns added to each chunk:**
- `risk_categories`: list of detected categories (e.g., `["margin_risk", "supply_chain_risk"]`)
- `risk_count`: number of active (non-negated) detections
- `risk_intensity`: severity-weighted score
- `risk_details`: full detection details including matched keyword, severity, negation status, and context snippet

---

## Retrieval-Augmented Generation (RAG)

The RAG system enables natural language Q&A over the earnings call corpus.

### Retrieval Engine (`src/agents/retrieval.py`)

#### Hybrid Search: Dense + Sparse

The system combines two complementary retrieval methods:

1. **Dense retrieval (FAISS)**
   - Embeds all 30,235 chunks using `all-MiniLM-L6-v2` (384-dimensional vectors)
   - Indexes with FAISS `IndexFlatIP` (inner product = cosine similarity on normalized vectors)
   - Captures semantic meaning: "revenue growth" matches "top-line expansion"

2. **Sparse retrieval (BM25)**
   - BM25Okapi over tokenized chunks
   - Captures exact keyword matches: searching for "EBITDA" finds chunks with that exact term
   - Important for financial terminology and acronyms

#### Reciprocal Rank Fusion (RRF)

Results from both methods are merged using RRF:

```
score(doc) = dense_weight / (k + rank_dense) + sparse_weight / (k + rank_sparse)
```

Where `k=60` (standard RRF constant), `dense_weight=0.6`, `sparse_weight=0.4`.

This ensures documents ranked highly by either method appear in the final results, while documents ranked highly by both methods are boosted further.

#### Metadata Filtering

Results can be filtered by:
- `ticker`: specific company
- `quarter`: specific time period
- `role`: speaker role (CEO, CFO, Analyst, etc.)
- `section`: prepared remarks or Q&A

### Answer Generation (`src/agents/rag_qa.py`)

The Q&A pipeline:

1. **Retrieve** — hybrid search returns top-K relevant chunks (default K=10, truncated to fit context window of 3,000 tokens)
2. **Format** — chunks are numbered and formatted with metadata headers:
   ```
   [1] AAPL 2024Q3 | Tim Cook (CEO) | prepared_remarks
   <chunk text>
   ```
3. **Generate** — a local LLM (Llama 3.2 3B via Ollama) generates an answer using a financial analyst system prompt
4. **Cite** — the system extracts `[1]`, `[2]` citations from the answer and maps them back to source chunks

#### LLM Configuration

The system uses **Ollama** with **Llama 3.2 (3B)** running locally — completely free with no API keys required. The system also supports OpenAI and Anthropic as alternative providers if configured.

#### System Prompt

The LLM is instructed to:
- Only use information from the provided context (no hallucination)
- Cite sources using numbered references
- Distinguish between management statements and analyst Q&A
- Note the speaker and their role
- Be specific with numbers, dates, and quotes

---

## Finance Analysis

The finance analysis module (`src/agents/finance_analysis.py`) links NLP features to post-earnings stock market reactions.

### Market Data Collection (`src/agents/market_data.py`)

For each of the 1,372 company-quarter events:

1. **Fetch stock prices** via `yfinance` (with caching and retry logic)
2. **Compute raw returns** over 1-day, 3-day, and 5-day windows after the earnings date
3. **Compute abnormal returns** by subtracting SPY (S&P 500 ETF) benchmark returns:
   ```
   abnormal_return = stock_return - benchmark_return
   ```
4. **Compute abnormal volume** — ratio of earnings-day volume to trailing 20-day average

### Feature Matrix

The system builds a feature matrix with 1,108 rows (one per company-quarter) and 26 columns:

**NLP Features (aggregated from chunk-level to call-level):**
- `mean_lm_net_score`, `mean_lm_positive_score`, `mean_lm_negative_score`, `mean_lm_uncertainty_score`
- `mean_vader_compound`
- `mgmt_sentiment` — average sentiment of CEO/CFO/COO/CTO statements
- `analyst_sentiment` — average sentiment of analyst questions
- `prepared_sentiment`, `qa_sentiment` — sentiment by section
- `sentiment_divergence` — prepared minus Q&A sentiment
- `total_risk_count`, `avg_risk_intensity`, `num_risk_categories`
- `mean_lm_net_score_momentum` — quarter-over-quarter sentiment change
- `mean_vader_compound_momentum`
- `risk_delta` — quarter-over-quarter risk change

**Market Features (target variables):**
- `ret_1d`, `ret_3d`, `ret_5d` — raw post-earnings returns
- `abnormal_ret_1d`, `abnormal_ret_3d`, `abnormal_ret_5d` — abnormal returns vs SPY
- `abnormal_volume` — volume ratio

### Statistical Analysis

#### 1. Event Study

Tests whether cumulative abnormal returns (CARs) are significantly different from zero using a one-sample t-test:

| Window | Mean CAR | t-stat | p-value | n |
|--------|---------|--------|---------|---|
| 1-day | 0.05% | 0.86 | 0.39 | 1,045 |
| 3-day | 0.08% | 0.93 | 0.35 | 1,045 |
| 5-day | -0.05% | -0.39 | 0.70 | 1,004 |

**Interpretation:** CARs are not significantly different from zero — consistent with semi-strong efficient market hypothesis. Earnings call sentiment alone does not systematically predict abnormal returns.

#### 2. OLS Regression

Regresses 1-day abnormal returns on NLP features (features must have >80% non-null coverage):

- **R² = 0.0107** (1.07% of return variance explained)
- **Adjusted R² = 0.0017**
- **F-statistic = 1.19, p = 0.30** (not significant)
- **n = 1,000 observations**

Features used: `mean_lm_net_score`, `mean_lm_positive_score`, `mean_lm_negative_score`, `mean_lm_uncertainty_score`, `mean_vader_compound`, `total_risk_count`, `avg_risk_intensity`, `num_risk_categories`, `mean_lm_net_score_momentum`, `mean_vader_compound_momentum`

#### 3. Portfolio Sorts

Companies are sorted into 3 buckets (Low/Mid/High) by sentiment score, and average returns are computed per bucket. This non-parametric approach doesn't assume linearity.

#### 4. Machine Learning Models

Four models with cross-validation:

| Model | CV R² (mean) | CV R² (std) |
|-------|-------------|-------------|
| Lasso | -0.006 | 0.005 |
| Ridge | -0.015 | 0.025 |
| Random Forest | -0.027 | 0.009 |
| Gradient Boosting | -0.125 | 0.045 |

**Interpretation:** Negative CV R² means the models perform worse than simply predicting the mean — confirming that earnings call NLP features alone have limited standalone predictive power for short-term stock returns. This is a valid and expected academic finding given market efficiency.

---

## Interactive Dashboard

The Streamlit dashboard (`app/`) provides 7 pages:

### Page 1: Company Overview
- Select any of the 49 companies
- View: total chunks, quarters available, unique speakers
- Earnings call summary table (chunks per quarter, speakers, sections)
- Speaker breakdown with role classification
- Key sentiment metrics

### Page 2: Sentiment Trends
- Interactive line chart: sentiment over time per company
- Heatmap: companies x quarters color-coded by sentiment (red-yellow-green)
- Prepared remarks vs. Q&A sentiment comparison (grouped bar chart)
- Sentiment breakdown by speaker role

### Page 3: Risk Monitoring
- Risk heatmap: companies x 10 risk categories
- Risk signal trends over time
- Drill-down: select a company + quarter to see specific risk categories and text excerpts

### Page 4: Transcript Explorer
- Browse full transcripts chunk by chunk
- Search within transcripts
- Filter by section (prepared remarks / Q&A)
- Each chunk color-coded by sentiment (green = positive, red = negative)

### Page 5: RAG Q&A
- Chat interface for natural language questions
- Filter by company and/or quarter
- Answers generated by Llama 3.2 (local, free) with source citations
- Expandable source panel showing retrieved chunks with relevance scores

### Page 6: Market Reaction Analysis
- Post-earnings return charts (1d, 3d, 5d)
- Abnormal return charts vs. SPY benchmark
- Scatter plot: any sentiment feature vs. any return metric
- Regression results with coefficient table
- ML model comparison table
- Feature importance from tree-based models
- Portfolio sort charts (Low/Mid/High sentiment buckets)

### Page 7: Upload & Analyze
- Upload a new earnings call PDF or TXT
- Auto-detects ticker and quarter from the transcript text and filename
- Instant sentiment analysis (LM + VADER)
- Instant risk detection (10 categories)
- Benchmarking against the 1,108 historical calls (percentile rank, sector comparison)
- Expected market reaction based on historical sentiment buckets
- Full transcript browser with sentiment highlighting
- Q&A chat over the uploaded transcript

---

## PDF Upload Feature

The upload feature (`app/pages/07_upload_transcript.py`) allows users to analyze any new earnings call transcript:

### Auto-Detection

When a PDF/TXT is uploaded, the system automatically detects:

- **Ticker symbol** — by scanning for company names (e.g., "Apple Inc." → AAPL), exchange tags (e.g., "(NASDAQ: AAPL)"), and filename patterns
- **Quarter** — by matching patterns like "Q3 2024", "third quarter 2024", "2024Q3" in both text and filename

The detected values are pre-filled and the user is asked to confirm or edit.

### Analytics Pipeline

For an uploaded transcript, the system runs:

1. **Text extraction** — pdfplumber for PDFs, UTF-8 decode for TXT
2. **Parsing** — speaker detection, role classification, section detection, chunking
3. **Sentiment analysis** — Loughran-McDonald + VADER (fast, no GPU needed)
4. **Risk detection** — all 10 categories with negation/severity
5. **Benchmarking** — percentile rank vs. historical dataset, sector comparison
6. **Q&A** — builds a mini FAISS+BM25 index on the fly for the uploaded document

### Session Persistence

The uploaded transcript and analysis results are stored in Streamlit session state, so navigating to other pages and returning does not lose the upload or require re-computation.

---

## Project Structure

```
indiv_assignment_RAG/
├── app/
│   ├── app.py                      # Dashboard home page
│   ├── utils.py                    # Shared utilities, data loaders
│   └── pages/
│       ├── 01_company_overview.py   # Company details
│       ├── 02_sentiment_trends.py   # Sentiment visualizations
│       ├── 03_risk_monitoring.py    # Risk heatmaps and trends
│       ├── 04_transcript_explorer.py # Browse transcripts
│       ├── 05_rag_qa.py             # Q&A chat interface
│       ├── 06_market_reaction.py    # Finance analysis results
│       └── 07_upload_transcript.py  # PDF upload & analyze
├── configs/
│   ├── config.yaml                 # Central pipeline configuration
│   ├── companies.json              # 49 target companies
│   ├── earnings_calendar.json      # 1,372 earnings dates
│   ├── risk_taxonomy.json          # 10 risk categories, 255 keywords
│   └── loughran_mcdonald.py        # LM sentiment word lists
├── src/
│   ├── data_collection.py          # Kaggle data ingestion
│   ├── agents/
│   │   ├── transcript_ingestion.py  # Parsing and chunking
│   │   ├── sentiment_analysis.py    # LM, VADER, FinBERT
│   │   ├── risk_detection.py        # Taxonomy-based risk detection
│   │   ├── retrieval.py             # FAISS + BM25 hybrid search
│   │   ├── rag_qa.py                # LLM answer generation
│   │   ├── market_data.py           # yfinance price fetching
│   │   └── finance_analysis.py      # Regression, ML, event study
│   ├── utils/                      # Helper functions
│   └── evaluation/                 # Evaluation scripts
├── data/
│   ├── kaggle/                     # Raw Kaggle datasets
│   ├── raw/                        # Processed transcripts (per ticker)
│   ├── processed/                  # Parquet files (chunks, sentiment, risk)
│   ├── embeddings/                 # FAISS index, BM25 index
│   └── market/                     # Market data and cache
├── outputs/
│   ├── feature_matrix.parquet      # NLP + market feature matrix
│   └── finance_results.json        # Regression and ML results
├── requirements.txt
└── PROJECT_REPORT.md               # This file
```

---

## Setup & Running

### Prerequisites

- Python 3.10+
- Ollama (for local LLM Q&A)

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt
pip install pdfplumber  # For PDF upload feature

# Install and start Ollama
brew install ollama
brew services start ollama
ollama pull llama3.2:3b
```

### Running the Pipeline

```bash
# 1. Data collection (requires Kaggle datasets in data/kaggle/)
python src/data_collection.py

# 2. Transcript parsing
python -c "from src.agents.transcript_ingestion import TranscriptParser; TranscriptParser().process_all()"

# 3. Sentiment analysis
python -c "
from src.agents.sentiment_analysis import SentimentAnalyzer
import pandas as pd
analyzer = SentimentAnalyzer()
chunks = pd.read_parquet('data/processed/all_chunks.parquet')
analyzer.analyze_all(chunks, methods=['lm', 'vader'])
"

# 4. Risk detection
python -c "
from src.agents.risk_detection import RiskDetector
import pandas as pd
detector = RiskDetector()
chunks = pd.read_parquet('data/processed/all_chunks.parquet')
detector.analyze_chunks(chunks)
"

# 5. Build RAG index
python -c "
from src.agents.retrieval import RetrievalEngine
import pandas as pd
engine = RetrievalEngine()
chunks = pd.read_parquet('data/processed/all_chunks.parquet')
engine.build_index(chunks)
engine.save_index()
"

# 6. Market data (takes ~15 minutes)
python src/agents/market_data.py

# 7. Finance analysis
python src/agents/finance_analysis.py
```

### Launching the Dashboard

```bash
streamlit run app/app.py --server.port 8501
```

Open http://localhost:8501 in your browser.

---

## Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Embedding | sentence-transformers (all-MiniLM-L6-v2) | 384-dim dense vectors for semantic search |
| Dense Index | FAISS (IndexFlatIP) | Fast similarity search over 30K vectors |
| Sparse Index | BM25Okapi (rank_bm25) | Keyword-based retrieval |
| Sentiment | Loughran-McDonald lexicon | Finance-specific dictionary |
| Sentiment | vaderSentiment | Rule-based with modifier handling |
| Sentiment | ProsusAI/finbert | Pre-trained transformer for financial text |
| LLM | Llama 3.2 3B (via Ollama) | Local, free answer generation |
| Market Data | yfinance | Stock prices and benchmark returns |
| Statistics | statsmodels, scipy | OLS regression, t-tests |
| ML | scikit-learn | Lasso, Ridge, Random Forest, Gradient Boosting |
| Dashboard | Streamlit + Plotly | Interactive multi-page web app |
| PDF Parsing | pdfplumber | Text extraction from PDF uploads |
| Data Format | Apache Parquet | Efficient columnar storage |
