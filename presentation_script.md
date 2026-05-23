# Presentation Script — Earnings Call Analyzer
**Target: 5 minutes total | 2 min slides + 3 min demo | Recorded video**

---

# PART 1 — SLIDES (2 minutes)

---

## Slide 1 — Title (10s)

Hi, my name is Seunghwan and this is my individual assignment for BC3409. I built an Earnings Call Analyzer — a fully local NLP system that extracts sentiment, risk signals, and answers questions from earnings call transcripts using RAG. It covers over 1,700 companies with three sentiment models, 10 risk categories, and an ML model trained on over 13,000 earnings events.

---

## Slide 2 — The Problem (15s)

Earnings calls are 60-plus pages of dialogue, and buried in that text are language patterns that predict stock movement. Research shows raw sentiment is insignificant — what matters is sentiment *change* over time, the gap between management and analyst tone, and whether sentiment drops during Q&A. Those are the signals my system extracts.

---

## Slide 3 — Architecture (15s)

Upload a PDF — the system parses speakers and roles, runs three sentiment models, and applies a two-layer risk detector. Chunks are embedded into a FAISS index for RAG Q&A. Over 30,000 chunks, 39 NLP features, everything runs locally — no API keys needed.

---

## Slide 4 — Sentiment Analysis (12s)

Three models: Loughran-McDonald for finance-specific terms, VADER for general language, and FinBERT for contextual understanding. The dashboard shows management-analyst gaps, credibility waterfalls from prepared remarks to Q&A, and quarter-over-quarter momentum.

---

## Slide 5 — Risk Detection (15s)

Two layers: keyword detection with 255 terms, negation handling, and severity scoring. Semantic detection uses RAG embeddings to catch risks in non-standard language — like "our pipeline is weaker than expected" matching demand risk. Every semantic detection shows which anchor sentence it matched, so it's fully explainable.

---

## Slide 6 — RAG Q&A (12s)

Hybrid retrieval combining FAISS semantic search and BM25 keyword matching, re-ranked by a cross-encoder, then answered by Llama 3.2 running locally. Every answer includes source citations, and users can filter by speaker, role, or section.

---

## Slide 7 — Evaluation Results (15s)

Retrieval: Recall@3 is 87%, MRR is 100% — the top result is always relevant. ML direction accuracy is 53%, only slightly above baseline — but consistent with academic literature. The portfolio backtest shows a 0.72% spread per trade on a long-short sentiment strategy, which compounds meaningfully over hundreds of events.

---

## Slide 8 — Dashboard Features (10s)

Multi-quarter analysis with auto-detected tickers, plain-English key findings on every page, keyword-vs-RAG detection breakdowns on the risk page, and suggested questions on the Q&A page — designed for both technical and non-technical users.

---

## Slide 9 — Tech Stack (8s)

Fully local: FinBERT and sentence-transformers on-device, Llama 3.2 via Ollama. Zero API costs, full data privacy — suitable for confidential pre-release earnings.

---

## Slide 10 — Transition to Demo (8s)

That's the system in theory. Now let me show you how it works in practice using real Google earnings transcripts from 2025.

---

# PART 2 — DEMO (3 minutes)

> **Pre-record**: Have the Streamlit app running. Have all 4 Google PDFs (Q1–Q4 2025) ready. Screen-record the demo walkthrough.

---

## Step 1 — Upload & Analyse (30s)

*[On the Home page, upload all 4 Google transcripts at once]*

"I'm uploading four Google earnings transcripts — Q1 through Q4 2025. The system automatically detects the ticker as GOOG and identifies each quarter. You can see the analysis kicks off — it's running sentiment models, risk detection, and NLP feature extraction on each transcript."

*[Wait for processing to complete — point out the spinner message]*

---

## Step 2 — Sentiment Page (40s)

*[Navigate to the Sentiment page]*

"Starting with sentiment. The key findings box at the top gives a plain-English summary — you can immediately see whether management tone is more positive or negative than analysts, and how that shifted across quarters."

*[Scroll to the management-vs-analyst gap chart]*

"This chart shows the management-analyst sentiment gap. When management is significantly more positive than analysts, that's a potential credibility concern."

*[Click on the credibility waterfall]*

"The waterfall shows how sentiment changes from prepared remarks to the Q&A section — a big drop here means management's optimism doesn't hold up under analyst questioning."

*[Point to the glossary expander]*

"There's also a glossary explaining what each model measures, so non-technical users can follow along."

---

## Step 3 — Risk Page (40s)

*[Navigate to the Risk page]*

"The risk page opens with a key findings summary and a detection method breakdown. You can see exactly how many risks were caught by keywords versus RAG semantic detection."

*[Point to the detection breakdown chart]*

"The stacked bar shows keyword detections in blue and semantic detections in purple. The RAG Value-Add card tells you what percentage of new risks were only caught by embeddings — risks with no explicit keyword match."

*[Scroll to Top Risk Excerpts]*

"Each risk excerpt shows a badge — Keyword, RAG New, or RAG Confirmed. For semantic detections, you can see the matched anchor sentence — this is *why* the system flagged it. For example, this chunk about competitive pressure matched the 'competition risk' anchor about market share loss. That's full explainability."

*[Point to management vs analyst risk chart if visible]*

"You can also compare which risks management raised versus what analysts pressed on."

---

## Step 4 — Benchmark Page (30s)

*[Navigate to the Benchmark page]*

"The benchmark page shows the ML model's prediction for each quarter. It predicts direction — whether the stock moves up or down post-earnings — with a plain-English interpretation."

*[Point to the per-quarter tabs]*

"Each tab shows the key NLP features driving that quarter's prediction — Loughran-McDonald negativity, VADER compound score, risk count — and the sentiment momentum across quarters."

"I'll be transparent — the ML accuracy is 53%, marginal above baseline. But the portfolio backtest shows the signal has economic value when applied systematically."

---

## Step 5 — Q&A Page (40s)

*[Navigate to the Q&A page]*

"Finally, the Q&A system. You can see suggested questions to get started."

*[Click: "What were the main risk factors discussed?"]*

"I'll ask about risk factors. The system searches across all four quarters using hybrid retrieval — FAISS plus BM25 — re-ranks with a cross-encoder, and generates an answer with Llama 3.2 running locally."

*[Wait for response, then point to citations]*

"Every answer includes source citations linking back to specific transcript chunks — you can verify exactly where the information came from."

*[Type: "How did Google's AI strategy evolve across quarters?"]*

"Now a cross-quarter question. Because we uploaded all four transcripts, it searches across all of them simultaneously and synthesises the answer. This is where multi-quarter RAG really shines — you get a longitudinal view that would take hours to compile manually."

*[Point out the source citations spanning multiple quarters]*

---

## Wrap-up (10s)

"So that's the full pipeline — from raw PDF to sentiment analysis, explainable risk detection, ML-based benchmarking, and RAG-powered Q&A — all running locally with zero API costs. Thank you for watching."
