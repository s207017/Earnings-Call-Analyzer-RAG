# Agent 2: NLP/Sentiment Analysis Agent

## Objective
Build a production-quality sentiment analysis pipeline for earnings call transcript chunks using multiple approaches.

## Tasks

### 1. Improve `src/agents/sentiment_analysis.py`
The current scaffold has a hardcoded subset of Loughran-McDonald words. Upgrade to:
- **Loughran-McDonald**: Load full dictionary from `configs/loughran_mcdonald.py` (positive, negative, uncertainty, litigious, constraining word lists)
- **FinBERT**: Proper batching for efficiency, GPU support, truncation handling
- **VADER**: Add as a third baseline (general-purpose sentiment)
- Sentiment divergence: method to compare prepared remarks vs Q&A sentiment
- Sentiment trends: method to track sentiment change across quarters for a company
- Proper error handling, progress logging with `logging` module
- Batch processing with configurable batch size

### 2. Create `configs/loughran_mcdonald.py`
- Full Loughran-McDonald financial word lists
- Categories: positive, negative, uncertainty, litigious, constraining, superfluous
- These should be comprehensive lists (the real LM dictionary has ~350 positive, ~2300 negative words — include as many as practical)

### 3. Create `src/utils/sentiment_utils.py`
Helper functions:
- Text preprocessing for sentiment (lowercase, remove punctuation, tokenize)
- Sentiment score normalization
- Visualization data preparation (format for plotly charts)
- Comparison helpers (cross-quarter, cross-speaker)

### 4. Create `notebooks/02_sentiment_analysis.py`
Example usage script showing:
- Loading processed chunks
- Running all three sentiment methods
- Aggregating by speaker, section, quarter
- Printing summary statistics

## Files to Create/Modify
- `src/agents/sentiment_analysis.py` (modify existing)
- `configs/loughran_mcdonald.py` (new)
- `src/utils/__init__.py` (new, empty)
- `src/utils/sentiment_utils.py` (new)
- `notebooks/02_sentiment_analysis.py` (new)

## Dependencies
- `transformers` (FinBERT)
- `vaderSentiment`
- `pandas`, `numpy`
- `torch`

## Quality Requirements
- FinBERT batching should handle large datasets without OOM
- Lexicon matching should use efficient lookups (sets, not list iteration)
- All methods should return consistent output formats
- Code should be importable and usable by the Orchestrator agent
