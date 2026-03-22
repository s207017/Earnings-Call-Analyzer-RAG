# Agent 4: RAG System Agent

## Objective
Build the Retrieval-Augmented Generation system for Q&A over earnings call transcripts, including embedding, indexing, retrieval, reranking, and LLM answer generation with citations.

## Tasks

### 1. Improve `src/agents/retrieval.py`
The current scaffold uses basic FAISS flat index. Upgrade to:
- **Hybrid retrieval**: Combine dense embeddings (sentence-transformers) with sparse/BM25 retrieval using rank fusion
- **Compound metadata filtering**: Filter by multiple fields simultaneously (ticker AND quarter AND role)
- **Evaluation methods**: Recall@K, MRR (Mean Reciprocal Rank) computation given ground truth
- **Batch embedding**: Progress tracking with tqdm, configurable batch size
- **Index management**: Save/load with versioning, rebuild incrementally

### 2. Improve `src/agents/rag_qa.py`
The current scaffold uses OpenAI only. Upgrade to:
- **Multiple LLM providers**: Support both OpenAI and Anthropic Claude (via their respective SDKs)
- **Answer evaluation**: Built-in checks for groundedness (is the answer supported by context?), faithfulness, completeness
- **Conversation history**: Support follow-up questions with context from prior Q&A turns
- **Structured citations**: Parse and return citations in a structured format (source index, ticker, quarter, speaker)
- **Financial domain prompts**: Optimized system prompt for financial analysis Q&A
- **Streaming support**: Optional streaming for long answers

### 3. Create `src/utils/rag_utils.py`
Helper functions:
- Chunk formatting for display
- Citation extraction from LLM output (parse [1], [2] references)
- Answer quality scoring helpers
- Context window management (truncate if too long for LLM)

### 4. Create `src/evaluation/rag_eval.py`
RAG evaluation pipeline:
- Define test questions with expected answers
- Score retrieval quality (Recall@K, MRR)
- Score answer quality (groundedness, faithfulness, completeness using LLM-as-judge)
- Generate evaluation report

## Files to Create/Modify
- `src/agents/retrieval.py` (modify existing)
- `src/agents/rag_qa.py` (modify existing)
- `src/utils/rag_utils.py` (new)
- `src/evaluation/__init__.py` (new, empty)
- `src/evaluation/rag_eval.py` (new)

## Dependencies
- `sentence-transformers`
- `faiss-cpu`
- `rank_bm25` (for BM25 sparse retrieval)
- `openai`
- `anthropic`
- `tqdm`

## Quality Requirements
- Hybrid retrieval should outperform dense-only on diverse query types
- Citations must be traceable back to specific chunks
- Answer evaluation should catch hallucinations
- Code should be importable and usable by the Orchestrator agent
