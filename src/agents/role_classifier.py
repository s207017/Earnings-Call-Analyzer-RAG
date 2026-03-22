"""ML-based speaker role classifier for earnings call transcripts.

Trains on labeled chunks from the dataset and predicts roles
for speakers where regex-based detection fails.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

# Map fine-grained roles to broader classes for training
ROLE_MAP = {
    "CEO": "Management",
    "CFO": "Management",
    "COO": "Management",
    "CTO": "Management",
    "VP": "Management",
    "IR": "IR",
    "Analyst": "Analyst",
    "Operator": "Operator",
}

MODEL_FILENAME = "role_classifier.pkl"


class RoleClassifier:
    """TF-IDF + Logistic Regression classifier for speaker roles."""

    def __init__(self, project_root: str = None):
        if project_root is None:
            project_root = str(Path(__file__).parent.parent.parent)
        self.project_root = Path(project_root)
        self.model_path = self.project_root / "models" / MODEL_FILENAME
        self.pipeline: Optional[Pipeline] = None

    def _load_training_data(self) -> tuple:
        """Load labeled chunks from the processed dataset."""
        chunks_path = self.project_root / "data" / "processed" / "all_chunks.parquet"
        df = pd.read_parquet(chunks_path)

        # Keep only chunks with known roles
        df = df[~df["role"].isin(["Unknown", "Other"])].copy()
        df["label"] = df["role"].map(ROLE_MAP)
        df = df.dropna(subset=["label"])

        # Minimum text length filter
        df = df[df["text"].str.len() >= 50]

        logger.info(f"Training data: {len(df)} labeled chunks")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")

        return df["text"].tolist(), df["label"].tolist()

    def train(self) -> dict:
        """Train the role classifier and save the model."""
        texts, labels = self._load_training_data()

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=1.0,
                solver="lbfgs",
                multi_class="multinomial",
            )),
        ])

        # Cross-validate
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self.pipeline, texts, labels, cv=cv, scoring="f1_macro")
        logger.info(f"CV F1 (macro): {scores.mean():.3f} +/- {scores.std():.3f}")

        # Train on full data
        self.pipeline.fit(texts, labels)

        # Classification report on training data (for reference)
        y_pred = self.pipeline.predict(texts)
        report = classification_report(labels, y_pred, output_dict=True)

        # Save model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.pipeline, f)
        logger.info(f"Model saved to {self.model_path}")

        return {
            "cv_f1_mean": float(scores.mean()),
            "cv_f1_std": float(scores.std()),
            "train_samples": len(texts),
            "report": report,
        }

    def load(self) -> bool:
        """Load a previously trained model."""
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                self.pipeline = pickle.load(f)
            return True
        return False

    def predict(self, text: str) -> str:
        """Predict role for a single text chunk."""
        if self.pipeline is None:
            if not self.load():
                return "Unknown"
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        max_idx = np.argmax(proba)
        confidence = proba[max_idx]

        # Only predict if confident enough
        if confidence < 0.4:
            return "Unknown"
        return classes[max_idx]

    def predict_batch(self, texts: list) -> list:
        """Predict roles for multiple text chunks."""
        if self.pipeline is None:
            if not self.load():
                return ["Unknown"] * len(texts)
        probas = self.pipeline.predict_proba(texts)
        classes = self.pipeline.classes_
        results = []
        for proba in probas:
            max_idx = np.argmax(proba)
            if proba[max_idx] < 0.4:
                results.append("Unknown")
            else:
                results.append(classes[max_idx])
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    clf = RoleClassifier()
    results = clf.train()
    print(f"\nCV F1 (macro): {results['cv_f1_mean']:.3f} +/- {results['cv_f1_std']:.3f}")
    print(f"Training samples: {results['train_samples']}")
    print("\nClassification Report (train):")
    for label, metrics in results["report"].items():
        if isinstance(metrics, dict):
            print(f"  {label:12s}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1-score']:.3f}  n={metrics['support']}")
