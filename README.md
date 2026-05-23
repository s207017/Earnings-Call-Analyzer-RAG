# Earnings Call Analyzer (RAG)

An NLP-powered tool that analyzes earnings call transcripts using sentiment analysis, risk detection, and retrieval-augmented generation (RAG) for interactive Q&A.

## Features

- **Sentiment Analysis** — Three NLP models (Loughran-McDonald, VADER, FinBERT) with management vs. analyst comparison, prepared remarks vs. Q&A credibility shift, and cross-quarter momentum tracking
- **Risk Detection** — 255 keyword patterns + semantic embedding-based detection across 10 risk categories, with severity scoring and explainable anchor matching
- **RAG-based Q&A** — Hybrid FAISS + BM25 retrieval with cross-encoder re-ranking, powered by Llama 3.2 (local via Ollama)
- **Benchmark & Prediction** — ML direction prediction (UP/DOWN) trained on 13,000+ earnings events, sentiment percentile ranking vs. 1,100+ historical calls, and portfolio backtest simulation
- **Multi-Quarter Analysis** — Upload multiple transcripts to track sentiment trends and risk evolution across quarters

## Architecture

```
src/agents/
├── transcript_ingestion.py   # Transcript parsing and chunking
├── sentiment_analysis.py     # LM, VADER, and FinBERT sentiment
├── risk_detection.py         # Keyword + semantic risk detection
├── retrieval.py              # Hybrid FAISS/BM25 retrieval engine
├── rag_qa.py                 # RAG question-answering agent
├── finance_analysis.py       # Market data and benchmark analysis
├── market_data.py            # Stock price and volume fetching
├── role_classifier.py        # Speaker role classification
├── text_features.py          # NLP text feature extraction
└── qa_features.py            # Q&A-specific feature engineering

app/
├── app.py                    # Streamlit entry point (upload & analyze)
├── utils.py                  # Shared UI utilities
├── style.css                 # Custom dark theme
└── pages/                    # Streamlit multi-page navigation
    ├── 01_Sentiment.py
    ├── 02_Risk.py
    ├── 03_Benchmark.py
    ├── 04_Transcript.py
    └── 05_Q&A.py

scripts/
├── run_pipeline.py           # End-to-end analysis pipeline
├── run_evaluation.py         # Model evaluation
├── train_direction_model.py  # Train post-earnings direction classifier
├── fetch_market_data.py      # Download historical market data
├── preprocess_motley_fool.py # Preprocess Motley Fool transcripts
├── preprocess_nlp_dataset.py # Preprocess NLP training data
└── expand_training_data.py   # Augment training dataset
```

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Sentiment | FinBERT, Loughran-McDonald, VADER |
| Vector Store | FAISS (dense) + BM25 (sparse) |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Llama 3.2 3B (via Ollama) |
| Frontend | Streamlit |
| Visualization | Plotly |

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) with `llama3.2:3b` model pulled

### Installation

```bash
pip install -r requirements.txt
ollama pull llama3.2:3b
```

### Running the App

```bash
streamlit run app/app.py
```

Upload an earnings call transcript (PDF or TXT) and the system will automatically detect the company ticker and quarter, then run the full analysis pipeline.

## Usage

1. **Upload** — Drop one or more earnings call transcripts on the home page
2. **Sentiment** — View sentiment breakdown by speaker, section, and model
3. **Risk** — Explore detected risk signals with severity and category details
4. **Benchmark** — Compare sentiment against historical percentiles and see ML-based direction predictions
5. **Transcript** — Browse the parsed transcript with inline sentiment highlighting
6. **Q&A** — Ask natural language questions about the transcript with cited answers
