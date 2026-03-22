"""Sentiment analysis pipeline for earnings call transcript chunks."""

import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Multi-method sentiment analyzer for financial text."""

    def __init__(self, project_root: str = None):
        if project_root is None:
            project_root = str(Path(__file__).parent.parent.parent)
        self.project_root = Path(project_root)

        config_path = self.project_root / "configs" / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self._finbert_pipeline = None
        self._vader_analyzer = None
        self._lm_words = None

    def _load_lm_words(self):
        """Lazy-load Loughran-McDonald word lists."""
        if self._lm_words is None:
            import sys
            sys.path.insert(0, str(self.project_root / "configs"))
            from loughran_mcdonald import (
                POSITIVE_WORDS, NEGATIVE_WORDS, UNCERTAINTY_WORDS,
                LITIGIOUS_WORDS, CONSTRAINING_WORDS
            )
            self._lm_words = {
                "positive": POSITIVE_WORDS,
                "negative": NEGATIVE_WORDS,
                "uncertainty": UNCERTAINTY_WORDS,
                "litigious": LITIGIOUS_WORDS,
                "constraining": CONSTRAINING_WORDS,
            }
        return self._lm_words

    def _load_finbert(self):
        """Lazy-load FinBERT pipeline."""
        if self._finbert_pipeline is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            import torch

            model_name = self.config["models"]["sentiment_model"]
            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Loading FinBERT model: {model_name} (device={device})")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._finbert_pipeline = pipeline(
                "sentiment-analysis", model=model, tokenizer=tokenizer,
                device=device, truncation=True,
                max_length=self.config["sentiment"]["finbert_max_length"],
            )
        return self._finbert_pipeline

    def _load_vader(self):
        """Lazy-load VADER analyzer."""
        if self._vader_analyzer is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader_analyzer = SentimentIntensityAnalyzer()
        return self._vader_analyzer

    def loughran_mcdonald_sentiment(self, text: str) -> Dict[str, float]:
        """Compute Loughran-McDonald sentiment scores."""
        words = self._load_lm_words()
        tokens = set(text.lower().split())
        total = max(len(tokens), 1)

        pos_count = len(tokens & words["positive"])
        neg_count = len(tokens & words["negative"])
        unc_count = len(tokens & words["uncertainty"])

        return {
            "lm_positive_score": pos_count / total,
            "lm_negative_score": neg_count / total,
            "lm_net_score": (pos_count - neg_count) / total,
            "lm_uncertainty_score": unc_count / total,
        }

    def finbert_sentiment(self, texts: List[str], batch_size: int = None) -> List[Dict]:
        """Run FinBERT sentiment on a list of texts with batching."""
        if batch_size is None:
            batch_size = self.config["sentiment"]["finbert_batch_size"]

        pipe = self._load_finbert()
        results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT"):
            batch = texts[i:i + batch_size]
            # Truncate long texts
            batch = [t[:2000] for t in batch]
            try:
                outputs = pipe(batch, batch_size=batch_size)
                for out in outputs:
                    label = out["label"].lower()
                    score = out["score"]
                    results.append({
                        "finbert_label": label,
                        "finbert_score": score,
                        "finbert_positive": score if label == "positive" else 0.0,
                        "finbert_negative": score if label == "negative" else 0.0,
                        "finbert_neutral": score if label == "neutral" else 0.0,
                    })
            except Exception as e:
                logger.error(f"FinBERT batch error: {e}")
                for _ in batch:
                    results.append({
                        "finbert_label": "error", "finbert_score": 0.0,
                        "finbert_positive": 0.0, "finbert_negative": 0.0, "finbert_neutral": 0.0,
                    })

        return results

    def vader_sentiment(self, text: str) -> Dict[str, float]:
        """Compute VADER sentiment scores."""
        analyzer = self._load_vader()
        scores = analyzer.polarity_scores(text)
        return {
            "vader_compound": scores["compound"],
            "vader_pos": scores["pos"],
            "vader_neg": scores["neg"],
            "vader_neu": scores["neu"],
        }

    def analyze_all(self, chunks_df: pd.DataFrame,
                    methods: List[str] = None) -> pd.DataFrame:
        """Run all sentiment methods on chunks dataframe."""
        if methods is None:
            methods = ["lm", "finbert", "vader"]

        df = chunks_df.copy()
        texts = df["text"].tolist()

        if "lm" in methods:
            logger.info("Running Loughran-McDonald sentiment...")
            lm_results = [self.loughran_mcdonald_sentiment(t) for t in tqdm(texts, desc="LM")]
            lm_df = pd.DataFrame(lm_results)
            df = pd.concat([df.reset_index(drop=True), lm_df], axis=1)

        if "finbert" in methods:
            logger.info("Running FinBERT sentiment...")
            fb_results = self.finbert_sentiment(texts)
            fb_df = pd.DataFrame(fb_results)
            df = pd.concat([df.reset_index(drop=True), fb_df], axis=1)

        if "vader" in methods:
            logger.info("Running VADER sentiment...")
            vader_results = [self.vader_sentiment(t) for t in tqdm(texts, desc="VADER")]
            vader_df = pd.DataFrame(vader_results)
            df = pd.concat([df.reset_index(drop=True), vader_df], axis=1)

        # Save
        out_path = self.project_root / self.config["paths"]["processed"] / "chunks_with_sentiment.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved sentiment results to {out_path}")
        return df

    def sentiment_divergence(self, chunks_df: pd.DataFrame,
                             ticker: str, quarter: str) -> Dict[str, float]:
        """Compare prepared remarks vs Q&A sentiment for a specific call."""
        mask = (chunks_df["ticker"] == ticker) & (chunks_df["quarter"] == quarter)
        call_df = chunks_df[mask]

        prepared = call_df[call_df["section"] == "prepared_remarks"]
        qa = call_df[call_df["section"] == "qa"]

        result = {}
        for col in ["lm_net_score", "vader_compound"]:
            if col in call_df.columns:
                prep_mean = prepared[col].mean() if not prepared.empty else 0
                qa_mean = qa[col].mean() if not qa.empty else 0
                result[f"{col}_prepared"] = prep_mean
                result[f"{col}_qa"] = qa_mean
                result[f"{col}_divergence"] = prep_mean - qa_mean

        return result

    def sentiment_trends(self, chunks_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Track sentiment across quarters for a company."""
        mask = chunks_df["ticker"] == ticker
        company_df = chunks_df[mask]

        score_cols = [c for c in company_df.columns if c.startswith(("lm_", "vader_", "finbert_"))
                      and company_df[c].dtype in ["float64", "float32"]]

        trends = company_df.groupby("quarter")[score_cols].mean().reset_index()
        trends = trends.sort_values("quarter")
        return trends


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    analyzer = SentimentAnalyzer()
    chunks_path = Path(analyzer.project_root) / "data" / "processed" / "all_chunks.parquet"
    if chunks_path.exists():
        df = pd.read_parquet(chunks_path)
        result = analyzer.analyze_all(df, methods=["lm", "vader"])
        print(f"Analyzed {len(result)} chunks")
    else:
        print("No chunks found. Run data pipeline first.")
