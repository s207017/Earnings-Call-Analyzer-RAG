"""Risk signal detection for earnings call transcript chunks."""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


class RiskDetector:
    """Taxonomy-based risk detection with context-awareness and severity scoring."""

    def __init__(self, project_root: str = None, negation_window: int = 3):
        if project_root is None:
            project_root = str(Path(__file__).parent.parent.parent)
        self.project_root = Path(project_root)

        config_path = self.project_root / "configs" / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        taxonomy_path = self.project_root / "configs" / "risk_taxonomy.json"
        with open(taxonomy_path) as f:
            self.taxonomy = json.load(f)

        self.negation_window = negation_window
        self.negation_words = set(self.taxonomy.get("negation_words", []))
        self.intensifiers = set(self.taxonomy.get("severity_modifiers", {}).get("intensifiers", []))
        self.diminishers = set(self.taxonomy.get("severity_modifiers", {}).get("diminishers", []))
        self.categories = {k: v for k, v in self.taxonomy.get("categories", {}).items()}

    def _check_negation(self, words: List[str], keyword_pos: int) -> bool:
        """Check if keyword is negated within window."""
        start = max(0, keyword_pos - self.negation_window)
        window = words[start:keyword_pos]
        return bool(set(w.lower() for w in window) & self.negation_words)

    def _score_severity(self, words: List[str], keyword_pos: int) -> str:
        """Score severity based on nearby intensifiers/diminishers."""
        window_start = max(0, keyword_pos - 5)
        window_end = min(len(words), keyword_pos + 5)
        window = set(w.lower() for w in words[window_start:window_end])

        if window & self.intensifiers:
            return "high"
        elif window & self.diminishers:
            return "low"
        return "medium"

    def detect_risks(self, text: str) -> List[Dict]:
        """Detect risk signals in text with context awareness."""
        detections = []
        text_lower = text.lower()
        words = text_lower.split()

        for category, keywords in self.categories.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Use word-boundary matching to avoid substring false positives
                # (e.g., "war" matching "software", "towards")
                # Use lookaround for non-alphanumeric boundaries to handle
                # special chars like & in "R&D" where \b fails
                escaped = re.escape(keyword_lower)
                # Allow optional trailing 's'/'es'/'ed'/'ing' for plural/verb forms
                pattern = re.compile(r'(?<![a-zA-Z0-9])' + escaped + r'(?:s|es|ed|ing)?(?![a-zA-Z0-9])')
                for match in pattern.finditer(text_lower):
                    idx = match.start()
                    # Find word position
                    word_pos = len(text_lower[:idx].split()) - 1
                    word_pos = max(0, word_pos)

                    negated = self._check_negation(words, word_pos)
                    severity = self._score_severity(words, word_pos)

                    # Extract context snippet
                    start = max(0, idx - 50)
                    end = min(len(text), idx + len(keyword_lower) + 50)
                    snippet = text[start:end].strip()

                    detections.append({
                        "category": category,
                        "keyword_matched": keyword,
                        "severity": severity if not negated else "negated",
                        "negated": negated,
                        "context_snippet": snippet,
                    })

        return detections

    def analyze_chunks(self, chunks_df: pd.DataFrame) -> pd.DataFrame:
        """Process all chunks and add risk columns."""
        df = chunks_df.copy()
        all_categories = []
        all_counts = []
        all_intensities = []
        all_details = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Risk Detection"):
            detections = self.detect_risks(row["text"])
            # Filter out negated
            active = [d for d in detections if not d["negated"]]

            categories = list(set(d["category"] for d in active))
            count = len(active)
            # Intensity: weighted count (high=3, medium=2, low=1)
            severity_weights = {"high": 3, "medium": 2, "low": 1}
            intensity = sum(severity_weights.get(d["severity"], 1) for d in active)

            all_categories.append(categories)
            all_counts.append(count)
            all_intensities.append(intensity)
            all_details.append(active)

        df["risk_categories"] = all_categories
        df["risk_count"] = all_counts
        df["risk_intensity"] = all_intensities
        df["risk_details"] = all_details

        # Save
        out_path = self.project_root / self.config["paths"]["processed"] / "chunks_with_risk.parquet"
        # Convert lists to strings for parquet compatibility
        save_df = df.copy()
        save_df["risk_categories"] = save_df["risk_categories"].apply(json.dumps)
        save_df["risk_details"] = save_df["risk_details"].apply(json.dumps)
        save_df.to_parquet(out_path, index=False)
        logger.info(f"Saved risk results to {out_path}")
        return df

    def risk_heatmap_data(self, chunks_df: pd.DataFrame) -> pd.DataFrame:
        """Create companies × risk categories matrix."""
        rows = []
        for ticker in chunks_df["ticker"].unique():
            mask = chunks_df["ticker"] == ticker
            company_df = chunks_df[mask]
            cat_counts = {}
            for cats in company_df["risk_categories"]:
                cat_list = cats if isinstance(cats, list) else json.loads(cats)
                for c in cat_list:
                    cat_counts[c] = cat_counts.get(c, 0) + 1
            cat_counts["ticker"] = ticker
            rows.append(cat_counts)

        heatmap_df = pd.DataFrame(rows).fillna(0)
        if "ticker" in heatmap_df.columns:
            heatmap_df = heatmap_df.set_index("ticker")
        return heatmap_df

    def risk_trends(self, chunks_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Track risk signals across quarters for a company."""
        mask = chunks_df["ticker"] == ticker
        company_df = chunks_df[mask]

        rows = []
        for quarter in sorted(company_df["quarter"].unique()):
            q_df = company_df[company_df["quarter"] == quarter]
            cat_counts = {}
            for cats in q_df["risk_categories"]:
                cat_list = cats if isinstance(cats, list) else json.loads(cats)
                for c in cat_list:
                    cat_counts[c] = cat_counts.get(c, 0) + 1
            cat_counts["quarter"] = quarter
            cat_counts["total_risk_count"] = q_df["risk_count"].sum()
            cat_counts["avg_risk_intensity"] = q_df["risk_intensity"].mean()
            rows.append(cat_counts)

        return pd.DataFrame(rows).fillna(0)

    def risk_summary(self, chunks_df: pd.DataFrame, ticker: str, quarter: str) -> Dict:
        """Get top risks for a specific earnings call."""
        mask = (chunks_df["ticker"] == ticker) & (chunks_df["quarter"] == quarter)
        call_df = chunks_df[mask]

        cat_counts = {}
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        examples = {}

        for _, row in call_df.iterrows():
            details = row["risk_details"]
            if isinstance(details, str):
                details = json.loads(details)
            for d in details:
                cat = d["category"]
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                sev = d.get("severity", "medium")
                if sev in severity_counts:
                    severity_counts[sev] += 1
                if cat not in examples:
                    examples[cat] = d.get("context_snippet", "")

        top_risks = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
        return {
            "ticker": ticker,
            "quarter": quarter,
            "top_risks": top_risks[:5],
            "severity_distribution": severity_counts,
            "total_detections": sum(cat_counts.values()),
            "examples": examples,
        }


class SupervisedRiskClassifier:
    """Skeleton for supervised multi-label risk classification using transformers."""

    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 10):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None

    def prepare_data(self, texts: List[str], labels: np.ndarray):
        """Prepare dataset for training (multi-label)."""
        from torch.utils.data import Dataset

        class RiskDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len=512):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.texts[idx], truncation=True, padding="max_length",
                    max_length=self.max_len, return_tensors="pt"
                )
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": self.labels[idx],
                }

        return RiskDataset

    def train(self, train_texts, train_labels, val_texts=None, val_labels=None,
              epochs=3, batch_size=16, lr=2e-5):
        """Training loop skeleton — requires labeled data to run."""
        logger.info("SupervisedRiskClassifier.train() called")
        logger.info("This is a skeleton — provide labeled data to train.")
        logger.info(f"Would train on {len(train_texts)} examples for {epochs} epochs")
        # Full implementation would use:
        # - AutoModelForSequenceClassification with problem_type="multi_label_classification"
        # - BCEWithLogitsLoss
        # - AdamW optimizer with linear schedule
        raise NotImplementedError("Provide labeled risk data to enable training")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    detector = RiskDetector()
    chunks_path = Path(detector.project_root) / "data" / "processed" / "all_chunks.parquet"
    if chunks_path.exists():
        df = pd.read_parquet(chunks_path)
        result = detector.analyze_chunks(df)
        print(f"Analyzed {len(result)} chunks for risk")
    else:
        print("No chunks found. Run data pipeline first.")
