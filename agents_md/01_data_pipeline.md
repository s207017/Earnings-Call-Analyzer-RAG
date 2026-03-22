# Agent 1: Data Pipeline Agent

## Objective
Build the transcript data collection and ingestion pipeline for the Earnings Call Analyzer.

## Scope
- 10 Big Tech companies: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, NFLX, CRM, ORCL
- 8 quarters: 2023Q1 through 2024Q4

## Tasks

### 1. Create `configs/companies.json`
- Ticker, company name, sector, industry
- Fiscal year end, IR page URL
- All 10 companies

### 2. Create `configs/config.yaml`
- All pipeline configuration in one place
- Paths (raw, processed, embeddings, market, outputs)
- Chunking params (chunk_size: 400, chunk_overlap: 50)
- Quarters list, company list reference
- Model settings (embedding model, sentiment model, LLM provider)

### 3. Create `src/data_collection.py`
A `DataCollector` class that loads transcripts from TWO Kaggle datasets:

#### Dataset 1: Motley Fool Scraped Earnings Call Transcripts
- URL: https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts
- Format: `.pkl` (pandas pickle), ~281 MB
- Columns: Date, Exchange, Quarter, Ticker, Transcript
- 18,755 transcripts, up to ~mid-2023

#### Dataset 2: AlphaSpread Earnings Call Transcripts
- URL: https://www.kaggle.com/datasets/ramssvimala/earning-call-transcripts
- Format: ZIP, ~63 MB
- Top 50 companies, updated Oct 2024 (likely covers 2023-2024)
- CC0 Public Domain license

#### Data Collection Logic
- Expect both datasets to be downloaded and placed in `data/kaggle/` directory
  - `data/kaggle/motley_fool/` — extracted Motley Fool pickle file
  - `data/kaggle/alphaspread/` — extracted AlphaSpread files
- Load both datasets, normalize to a common schema: ticker, quarter, date, transcript_text, source
- Filter for the 10 target companies and 8 target quarters (2023Q1-2024Q4)
- Prefer AlphaSpread for recent quarters (2023-2024), use Motley Fool as fallback/supplement
- Deduplicate: if both datasets have the same ticker+quarter, keep the longer/more complete transcript
- Save individual transcript files to `data/raw/{ticker}/{quarter}.txt`
- Create `data/manifest.json` listing all transcripts with: file path, ticker, quarter, earnings date, source dataset
- Print a coverage report showing which company-quarters are available vs missing
- Includes a CLI entry point to run collection

### 4. Create `src/agents/transcript_ingestion.py`
Build a robust transcript parser to handle real-world transcripts from both datasets:
- Multiple speaker formats (e.g., "Name -- Title", "Name (Title)", "Name, Title:", "[Name]")
- Boilerplate removal (legal disclaimers, copyright notices, "Safe Harbor" statements)
- Sentence-aware chunking (don't break mid-sentence)
- Fallback when no speakers detected (treat whole text as single speaker)
- Section detection for prepared remarks vs Q&A (multiple cue phrases)
- Proper logging with Python `logging` module
- Edge case handling (empty files, encoding issues)
- Handle format differences between Motley Fool and AlphaSpread transcripts

## Files to Create/Modify
- `configs/companies.json` (new)
- `configs/config.yaml` (new)
- `src/data_collection.py` (new)
- `src/agents/transcript_ingestion.py` (new)

## Dependencies
- `pyyaml` for config loading
- `pandas` for data handling
- `kagglehub` or manual download instructions

## Quality Requirements
- All functions should have docstrings
- Use Python `logging` module (not print statements in production code)
- Handle errors gracefully with informative messages
- Code should be importable and usable by the Orchestrator agent
