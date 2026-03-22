"""RAG evaluation pipeline."""

import logging
from typing import List, Dict

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_QUESTIONS = [
    {"question": "What was Apple's revenue in the most recent quarter?", "expected_tickers": ["AAPL"]},
    {"question": "How did Microsoft's cloud business perform?", "expected_tickers": ["MSFT"]},
    {"question": "What guidance did NVIDIA provide for next quarter?", "expected_tickers": ["NVDA"]},
    {"question": "What are Tesla's gross margins?", "expected_tickers": ["TSLA"]},
    {"question": "How is Meta's advertising revenue trending?", "expected_tickers": ["META"]},
    {"question": "What risks did Amazon mention in their earnings call?", "expected_tickers": ["AMZN"]},
    {"question": "What is Netflix's subscriber growth?", "expected_tickers": ["NFLX"]},
    {"question": "How did Salesforce's AI products perform?", "expected_tickers": ["CRM"]},
    {"question": "What was Oracle's cloud infrastructure revenue?", "expected_tickers": ["ORCL"]},
    {"question": "Compare Google and Microsoft's AI strategies", "expected_tickers": ["GOOGL", "MSFT"]},
]


def score_retrieval(retrieval_engine, queries: List[str],
                    ground_truth: List[List[str]],
                    k_values: List[int] = None) -> Dict:
    """Score retrieval quality with Recall@K and MRR."""
    if k_values is None:
        k_values = [3, 5, 10]
    return retrieval_engine.evaluate(queries, ground_truth, k_values)


def score_answer_quality(question: str, answer: str, context: str,
                          llm_client=None) -> Dict:
    """Score answer quality using LLM-as-judge for groundedness."""
    if llm_client is None:
        # Simple heuristic check
        answer_sentences = [s.strip() for s in answer.split(".") if s.strip()]
        has_citations = bool(len([s for s in answer_sentences if "[" in s]))
        return {
            "has_citations": has_citations,
            "answer_length": len(answer.split()),
            "groundedness": "check_manually",
        }

    # LLM-as-judge evaluation
    eval_prompt = f"""Evaluate this RAG answer for:
1. Groundedness (0-1): Is the answer supported by the context?
2. Faithfulness (0-1): Does it avoid adding information not in the context?
3. Completeness (0-1): Does it address all parts of the question?

Context: {context[:2000]}
Question: {question}
Answer: {answer}

Return scores as: groundedness=X, faithfulness=Y, completeness=Z"""

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0,
            max_tokens=200,
        )
        return {"eval_text": response.choices[0].message.content}
    except Exception as e:
        logger.error(f"LLM eval failed: {e}")
        return {"error": str(e)}


def generate_eval_report(retrieval_metrics: Dict, answer_scores: List[Dict] = None) -> str:
    """Generate evaluation summary report."""
    lines = ["=== RAG Evaluation Report ===\n"]
    lines.append("Retrieval Metrics:")
    for k, v in retrieval_metrics.items():
        lines.append(f"  {k}: {v:.4f}")

    if answer_scores:
        lines.append("\nAnswer Quality:")
        for i, score in enumerate(answer_scores):
            lines.append(f"  Q{i+1}: {score}")

    return "\n".join(lines)
