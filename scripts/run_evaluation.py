"""
Run full evaluation pipeline — RAG retrieval metrics + answer quality + finance backtesting.
Saves results to outputs/evaluation_results.json.

Usage:
    python3.11 scripts/run_evaluation.py
"""

import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Evaluation queries with ground truth ──
# Each query has: question, expected tickers/sections to validate retrieval,
# and keywords that should appear in retrieved chunks.
RETRIEVAL_EVAL = [
    {
        "question": "What was Apple's revenue in the most recent quarter?",
        "expected_tickers": ["AAPL"],
        "expected_keywords": ["revenue", "billion"],
    },
    {
        "question": "How did Microsoft's cloud business perform?",
        "expected_tickers": ["MSFT"],
        "expected_keywords": ["cloud", "azure"],
    },
    {
        "question": "What guidance did NVIDIA provide for next quarter?",
        "expected_tickers": ["NVDA"],
        "expected_keywords": ["guidance", "expect", "quarter"],
    },
    {
        "question": "What are Tesla's gross margins?",
        "expected_tickers": ["TSLA"],
        "expected_keywords": ["margin", "gross"],
    },
    {
        "question": "How is Meta's advertising revenue trending?",
        "expected_tickers": ["META"],
        "expected_keywords": ["advertising", "ad", "revenue"],
    },
    {
        "question": "What risks did Amazon mention in their earnings call?",
        "expected_tickers": ["AMZN"],
        "expected_keywords": ["risk", "challenge", "headwind", "pressure"],
    },
    {
        "question": "What is Netflix's subscriber growth?",
        "expected_tickers": ["NFLX"],
        "expected_keywords": ["subscriber", "member", "growth"],
    },
    {
        "question": "How did Salesforce's AI products perform?",
        "expected_tickers": ["CRM"],
        "expected_keywords": ["ai", "artificial", "intelligence", "einstein"],
    },
    {
        "question": "What was Oracle's cloud infrastructure revenue?",
        "expected_tickers": ["ORCL"],
        "expected_keywords": ["cloud", "infrastructure", "oci", "revenue"],
    },
    {
        "question": "What did the CFO say about operating expenses?",
        "expected_tickers": [],  # any ticker
        "expected_keywords": ["operating", "expense", "cost"],
        "expected_roles": ["CFO"],
    },
    {
        "question": "What concerns did analysts raise about competition?",
        "expected_tickers": [],
        "expected_keywords": ["competition", "competitive", "market share"],
        "expected_sections": ["qa"],
    },
    {
        "question": "What is the company's strategy for growth?",
        "expected_tickers": [],
        "expected_keywords": ["strategy", "growth", "invest", "plan", "initiative"],
    },
]


def evaluate_retrieval(engine, queries, k_values=None):
    """Evaluate retrieval quality with multiple metrics."""
    if k_values is None:
        k_values = [3, 5, 10]

    results = {f"recall@{k}": [] for k in k_values}
    results["mrr"] = []
    results["ticker_accuracy"] = []
    results["keyword_hit_rate"] = []
    results["section_accuracy"] = []
    results["role_accuracy"] = []
    per_query = []

    for q_info in queries:
        question = q_info["question"]
        expected_tickers = q_info.get("expected_tickers", [])
        expected_keywords = q_info.get("expected_keywords", [])
        expected_sections = q_info.get("expected_sections", [])
        expected_roles = q_info.get("expected_roles", [])

        retrieved = engine.hybrid_search(question, top_k=max(k_values), rerank=True)
        retrieved_texts = [r["text"].lower() for r in retrieved]
        retrieved_tickers = [r["ticker"] for r in retrieved]

        # Recall@K: what fraction of expected keywords appear in top-k results
        for k in k_values:
            top_k_text = " ".join(retrieved_texts[:k])
            if expected_keywords:
                hits = sum(1 for kw in expected_keywords if kw.lower() in top_k_text)
                recall = hits / len(expected_keywords)
            else:
                recall = 1.0  # no keywords to check
            results[f"recall@{k}"].append(recall)

        # MRR: rank of first relevant result (contains any expected keyword)
        rr = 0.0
        if expected_keywords:
            for i, text in enumerate(retrieved_texts):
                if any(kw.lower() in text for kw in expected_keywords):
                    rr = 1.0 / (i + 1)
                    break
        else:
            rr = 1.0
        results["mrr"].append(rr)

        # Ticker accuracy: do retrieved chunks match expected company
        if expected_tickers:
            ticker_hits = sum(1 for t in retrieved_tickers[:5] if t in expected_tickers)
            results["ticker_accuracy"].append(ticker_hits / min(5, len(retrieved)))
        else:
            results["ticker_accuracy"].append(1.0)

        # Keyword hit rate across all retrieved
        all_text = " ".join(retrieved_texts)
        if expected_keywords:
            kw_hits = sum(1 for kw in expected_keywords if kw.lower() in all_text)
            results["keyword_hit_rate"].append(kw_hits / len(expected_keywords))
        else:
            results["keyword_hit_rate"].append(1.0)

        # Section accuracy
        if expected_sections:
            sec_hits = sum(1 for r in retrieved[:5] if r.get("section") in expected_sections)
            results["section_accuracy"].append(sec_hits / min(5, len(retrieved)))

        # Role accuracy
        if expected_roles:
            role_hits = sum(1 for r in retrieved[:5] if r.get("role") in expected_roles)
            results["role_accuracy"].append(role_hits / min(5, len(retrieved)))

        per_query.append({
            "question": question,
            "n_retrieved": len(retrieved),
            "top_ticker": retrieved_tickers[0] if retrieved_tickers else None,
            "keyword_recall": results[f"recall@{k_values[-1]}"][-1],
            "mrr": rr,
        })

    # Aggregate
    metrics = {}
    for key in [f"recall@{k}" for k in k_values] + ["mrr", "ticker_accuracy", "keyword_hit_rate"]:
        vals = results[key]
        if vals:
            metrics[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    if results["section_accuracy"]:
        metrics["section_accuracy"] = {"mean": float(np.mean(results["section_accuracy"]))}
    if results["role_accuracy"]:
        metrics["role_accuracy"] = {"mean": float(np.mean(results["role_accuracy"]))}

    return metrics, per_query


def evaluate_answer_quality(rag_engine, queries, top_k=5):
    """Evaluate RAG answer quality with heuristic metrics (no LLM judge needed)."""
    scores = []

    for q_info in queries:
        question = q_info["question"]
        expected_keywords = q_info.get("expected_keywords", [])

        try:
            result = rag_engine.answer(question, top_k=top_k)
            answer = result["answer"]
            sources = result.get("sources", [])
            queries_used = result.get("queries_used", [])

            # Heuristic quality metrics
            citation_count = len(re.findall(r"\[\d+\]", answer))
            answer_words = len(answer.split())
            has_hedge = any(phrase in answer.lower() for phrase in
                           ["context doesn't", "not enough information", "no relevant",
                            "cannot determine", "not mentioned"])

            # Keyword coverage in answer
            answer_lower = answer.lower()
            if expected_keywords:
                kw_in_answer = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
                keyword_coverage = kw_in_answer / len(expected_keywords)
            else:
                keyword_coverage = 1.0

            # Source diversity
            unique_speakers = len(set(s.get("speaker", "") for s in sources))
            unique_sections = len(set(s.get("section", "") for s in sources))

            scores.append({
                "question": question,
                "answer_length_words": answer_words,
                "citation_count": citation_count,
                "has_citations": citation_count > 0,
                "hedged_answer": has_hedge,
                "keyword_coverage": keyword_coverage,
                "n_sources": len(sources),
                "unique_speakers": unique_speakers,
                "unique_sections": unique_sections,
                "query_expanded": len(queries_used) > 1,
                "queries_used": queries_used,
            })
        except Exception as e:
            logger.warning(f"Answer generation failed for '{question}': {e}")
            scores.append({
                "question": question,
                "error": str(e),
            })

    # Aggregate
    valid = [s for s in scores if "error" not in s]
    summary = {}
    if valid:
        summary = {
            "n_questions": len(scores),
            "n_successful": len(valid),
            "avg_answer_length": float(np.mean([s["answer_length_words"] for s in valid])),
            "avg_citations": float(np.mean([s["citation_count"] for s in valid])),
            "citation_rate": float(np.mean([s["has_citations"] for s in valid])),
            "avg_keyword_coverage": float(np.mean([s["keyword_coverage"] for s in valid])),
            "hedge_rate": float(np.mean([s["hedged_answer"] for s in valid])),
            "query_expansion_rate": float(np.mean([s["query_expanded"] for s in valid])),
            "avg_sources": float(np.mean([s["n_sources"] for s in valid])),
            "avg_unique_speakers": float(np.mean([s["unique_speakers"] for s in valid])),
        }

    return summary, scores


def evaluate_finance(feature_matrix_path, finance_results_path):
    """Run finance evaluation: walk-forward CV + portfolio backtest."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from src.evaluation.finance_eval import walk_forward_cv, portfolio_backtest

    fm = pd.read_parquet(feature_matrix_path)
    logger.info(f"Feature matrix: {len(fm)} rows")

    target = "abnormal_ret_1d"
    if target not in fm.columns:
        logger.warning(f"No {target} column in feature matrix")
        return {}

    # Feature selection
    candidate_cols = [c for c in fm.columns
                      if c.startswith(("mean_", "mgmt_", "analyst_", "prepared_", "qa_",
                                       "sentiment_", "total_risk", "avg_risk", "num_risk",
                                       "call_", "risk_delta"))]
    threshold = len(fm) * 0.7
    feature_cols = [c for c in candidate_cols if fm[c].notna().sum() >= threshold]

    train_df = fm.dropna(subset=[target]).copy()
    logger.info(f"Training samples: {len(train_df)}, features: {len(feature_cols)}")

    if len(train_df) < 50:
        logger.warning("Too few samples for evaluation")
        return {}

    # Sort by quarter for time-series CV
    train_df = train_df.sort_values("quarter")

    X = train_df[feature_cols].astype(float).values
    y = train_df[target].values

    # Impute
    medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        nan_mask = np.isnan(X[:, j])
        X[nan_mask, j] = medians[j]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Walk-forward CV for regression models
    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    }

    wf_results = {}
    for name, model in models.items():
        logger.info(f"  Walk-forward CV: {name}")
        wf = walk_forward_cv(model, X_scaled, y, n_splits=5)
        wf_results[name] = wf
        logger.info(f"    R²: {wf['mean_r2']:.4f} +/- {wf['std_r2']:.4f}")

    # Binary classification evaluation
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    y_dir = (y > 0).astype(int)
    baseline_acc = max(y_dir.mean(), 1 - y_dir.mean())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest_clf": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
        "gradient_boosting_clf": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
    }

    clf_results = {}
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_scaled, y_dir, cv=cv, scoring="accuracy")
        clf_results[name] = {
            "accuracy_mean": float(scores.mean()),
            "accuracy_std": float(scores.std()),
        }
        logger.info(f"  Classification {name}: {scores.mean():.4f} +/- {scores.std():.4f}")

    # Portfolio backtest
    backtest_results = {}
    for feat in ["mean_lm_net_score", "mean_vader_compound"]:
        if feat in train_df.columns:
            bt = portfolio_backtest(train_df, feat, target)
            backtest_results[feat] = bt
            logger.info(f"  Backtest {feat}: spread = {bt['spread_mean']*100:.3f}%")

    return {
        "n_samples": len(train_df),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "baseline_accuracy": float(baseline_acc),
        "walk_forward_cv": wf_results,
        "classification": clf_results,
        "portfolio_backtest": backtest_results,
    }


def main():
    start_time = time.time()
    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    chunks_path = PROJECT_ROOT / "data" / "processed" / "all_chunks.parquet"
    feature_matrix_path = PROJECT_ROOT / "outputs" / "feature_matrix.parquet"
    finance_results_path = PROJECT_ROOT / "outputs" / "finance_results.json"
    output_path = PROJECT_ROOT / "outputs" / "evaluation_results.json"

    # ═══════════════════════════════════════
    # 1. RAG Retrieval Evaluation
    # ═══════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 1: RAG Retrieval Evaluation")
    logger.info("=" * 60)

    if chunks_path.exists():
        from src.agents.retrieval import RetrievalEngine

        engine = RetrievalEngine()

        # Try loading pre-built index first
        embeddings_path = PROJECT_ROOT / "data" / "embeddings"
        if (embeddings_path / "faiss.index").exists():
            logger.info("Loading pre-built index...")
            engine.load_index()
        else:
            logger.info("Building index from chunks...")
            df = pd.read_parquet(chunks_path)
            engine.build_index(df, use_speaker_turns=True)
            engine.save_index()

        logger.info(f"Index ready: {engine.index.ntotal} vectors")

        retrieval_metrics, per_query_retrieval = evaluate_retrieval(engine, RETRIEVAL_EVAL)
        results["retrieval"] = {
            "metrics": retrieval_metrics,
            "per_query": per_query_retrieval,
            "index_size": engine.index.ntotal,
        }

        logger.info("\nRetrieval Results:")
        for metric, vals in retrieval_metrics.items():
            if isinstance(vals, dict):
                logger.info(f"  {metric}: {vals['mean']:.4f}" +
                            (f" +/- {vals['std']:.4f}" if "std" in vals else ""))

        # ═══════════════════════════════════════
        # 2. RAG Answer Quality Evaluation
        # ═══════════════════════════════════════
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: RAG Answer Quality Evaluation")
        logger.info("=" * 60)

        try:
            from src.agents.rag_qa import RAGEngine
            rag = RAGEngine(engine)

            answer_summary, answer_details = evaluate_answer_quality(rag, RETRIEVAL_EVAL[:6])
            results["answer_quality"] = {
                "summary": answer_summary,
                "per_query": answer_details,
            }

            logger.info("\nAnswer Quality:")
            for k, v in answer_summary.items():
                logger.info(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

        except Exception as e:
            logger.warning(f"Answer quality eval skipped (LLM unavailable): {e}")
            results["answer_quality"] = {"error": str(e)}
    else:
        logger.warning(f"No chunks found at {chunks_path}")
        results["retrieval"] = {"error": "No chunks data"}

    # ═══════════════════════════════════════
    # 3. Finance Model Evaluation
    # ═══════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Finance Model Evaluation")
    logger.info("=" * 60)

    if feature_matrix_path.exists():
        finance_results = evaluate_finance(feature_matrix_path, finance_results_path)
        results["finance"] = finance_results
    else:
        logger.warning(f"No feature matrix at {feature_matrix_path}")
        results["finance"] = {"error": "No feature matrix"}

    # ═══════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = round(elapsed, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info(f"EVALUATION COMPLETE — {elapsed:.1f}s")
    logger.info(f"Results saved to {output_path}")
    logger.info("=" * 60)

    # Print summary table
    print("\n╔══════════════════════════════════════════════════╗")
    print("║           EVALUATION RESULTS SUMMARY             ║")
    print("╠══════════════════════════════════════════════════╣")

    if "retrieval" in results and "metrics" in results["retrieval"]:
        rm = results["retrieval"]["metrics"]
        print(f"║  Retrieval                                       ║")
        print(f"║    Recall@3:          {rm.get('recall@3', {}).get('mean', 0):.3f}                      ║")
        print(f"║    Recall@5:          {rm.get('recall@5', {}).get('mean', 0):.3f}                      ║")
        print(f"║    Recall@10:         {rm.get('recall@10', {}).get('mean', 0):.3f}                      ║")
        print(f"║    MRR:               {rm.get('mrr', {}).get('mean', 0):.3f}                      ║")
        print(f"║    Ticker Accuracy:   {rm.get('ticker_accuracy', {}).get('mean', 0):.3f}                      ║")
        print(f"║    Keyword Hit Rate:  {rm.get('keyword_hit_rate', {}).get('mean', 0):.3f}                      ║")

    if "answer_quality" in results and "summary" in results["answer_quality"]:
        aq = results["answer_quality"]["summary"]
        print(f"║  Answer Quality                                  ║")
        print(f"║    Citation Rate:     {aq.get('citation_rate', 0):.3f}                      ║")
        print(f"║    Keyword Coverage:  {aq.get('avg_keyword_coverage', 0):.3f}                      ║")
        print(f"║    Avg Answer Length: {aq.get('avg_answer_length', 0):.0f} words                 ║")

    if "finance" in results and "classification" in results["finance"]:
        fc = results["finance"]
        print(f"║  Finance                                         ║")
        print(f"║    Baseline Accuracy: {fc.get('baseline_accuracy', 0):.3f}                      ║")
        for name, m in fc.get("classification", {}).items():
            short = name.replace("_clf", "").replace("_", " ").title()[:20]
            print(f"║    {short:20s} {m['accuracy_mean']:.3f} +/- {m['accuracy_std']:.3f}     ║")

    print(f"╚══════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
