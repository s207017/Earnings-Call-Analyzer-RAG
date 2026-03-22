# Agent 5: Market Data Agent

## Objective
Build the market data pipeline that fetches stock prices around earnings dates, computes post-earnings returns, abnormal returns, and volatility for linking with NLP features.

## Tasks

### 1. Improve `src/agents/market_data.py`
The current scaffold uses basic yfinance calls. Upgrade to:
- **Rate limiting & retry**: Respect yfinance rate limits, exponential backoff on failure
- **Abnormal returns**: Compute returns relative to S&P 500 (SPY) benchmark
- **Volume analysis**: Compute abnormal trading volume around earnings (vs 20-day average)
- **Sector-relative returns**: Compare company returns to sector ETF performance
- **Caching**: Cache downloaded price data to `data/market/cache/` to avoid refetching
- **Robust error handling**: Handle missing data, delisted tickers, weekends/holidays
- **Earnings date auto-detection**: Try to pull earnings dates from yfinance if available

### 2. Create `configs/earnings_calendar.json`
Earnings calendar for all 10 companies across 8 quarters (2023Q1-2024Q4):
- Use approximate real earnings dates
- Format: list of `{"ticker": "AAPL", "quarter": "2023Q1", "earnings_date": "2023-02-02"}`
- Include all 80 events (10 companies × 8 quarters)

### 3. Create `src/utils/market_utils.py`
Helper functions:
- Trading day calculations (next/previous trading day)
- Date alignment (align earnings date to nearest trading day)
- Return computation helpers (simple, log, cumulative)
- Benchmark-relative return calculation
- Abnormal volume computation

## Files to Create/Modify
- `src/agents/market_data.py` (modify existing)
- `configs/earnings_calendar.json` (new)
- `src/utils/market_utils.py` (new)

## Dependencies
- `yfinance`
- `pandas`, `numpy`
- `requests` (fallback data fetching)

## Quality Requirements
- Must handle real-world data issues (missing prices, stock splits, after-hours earnings)
- Caching should prevent unnecessary API calls on reruns
- All return calculations should clearly document whether they use close-to-close or open-to-close
- Earnings calendar dates should be reasonably accurate (within 1-2 days of actual dates)
- Code should be importable and usable by the Orchestrator agent
