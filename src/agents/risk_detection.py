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


class SemanticRiskDetector:
    """Embedding-based semantic risk detection using sentence-transformers.

    Complements keyword detection by finding risk signals that keywords miss.
    Uses differential scoring: risk_similarity - safe_similarity to filter
    out chunks that merely *mention* a topic vs ones that express actual risk.
    """

    RISK_ANCHORS = {
        "demand_risk": {
            "risk": [
                "Customer demand is declining and orders are being cancelled.",
                "Revenue is falling because fewer customers are buying.",
                "Sales volume dropped significantly, the market is shrinking.",
                "We are losing subscribers and churn is increasing.",
                "Our pipeline is weak with fewer deals closing.",
                "Bookings declined and backlog is shrinking.",
            ],
            "safe": [
                "Demand is strong and growing across all segments.",
                "Revenue grew double digits driven by robust customer demand.",
                "We saw record bookings and a healthy pipeline.",
                "Customer acquisition accelerated this quarter.",
            ],
        },
        "margin_risk": {
            "risk": [
                "Profit margins are being squeezed by rising costs.",
                "Gross margin declined due to cost inflation and unfavorable mix.",
                "Operating expenses grew faster than revenue, hurting profitability.",
                "Pricing pressure from competitors is eroding our margins.",
                "Higher input costs and wage inflation are compressing margins.",
                "We expect margin headwinds from elevated costs next quarter.",
            ],
            "safe": [
                "Gross margin expanded by 3 points driven by favorable mix.",
                "Operating margin improved due to disciplined cost management.",
                "Profitability increased as we achieved operating leverage.",
                "Our margins are healthy and trending upward.",
            ],
        },
        "supply_chain_risk": {
            "risk": [
                "Our supply chain has been disrupted by component shortages.",
                "Shipping delays and logistics issues are affecting deliveries.",
                "We have excess inventory that may require write-downs.",
                "Dependency on a single supplier creates vulnerability.",
                "Manufacturing bottlenecks are limiting our production capacity.",
                "Lead times for critical materials have extended significantly.",
            ],
            "safe": [
                "Supply chain operations are running smoothly.",
                "Inventory levels are healthy and well-managed.",
                "We diversified our supplier base and reduced concentration risk.",
                "Logistics and fulfillment are operating efficiently.",
            ],
        },
        "regulatory_risk": {
            "risk": [
                "We face potential fines from regulatory investigations.",
                "New regulations will increase our compliance costs.",
                "Antitrust scrutiny could force changes to our business model.",
                "Ongoing litigation may result in material financial settlements.",
                "Data privacy enforcement actions pose operational risk.",
                "Tax authorities are challenging our transfer pricing arrangements.",
            ],
            "safe": [
                "We are in full compliance with all regulatory requirements.",
                "The regulatory environment is stable and predictable.",
                "We resolved all outstanding legal matters favorably.",
                "Our compliance program is robust and well-resourced.",
            ],
        },
        "competition_risk": {
            "risk": [
                "Competitors are taking market share with aggressive pricing.",
                "New entrants are disrupting our core business.",
                "Win rates have declined as competition intensifies.",
                "Our products are being commoditized, reducing differentiation.",
                "We are losing deals to competitors offering better alternatives.",
                "Market fragmentation is making it harder to maintain share.",
            ],
            "safe": [
                "We gained market share across all key segments.",
                "Our competitive position is strong and differentiated.",
                "We consistently win against competitors in head-to-head deals.",
                "Our moat continues to widen with product innovation.",
            ],
        },
        "fx_macro_risk": {
            "risk": [
                "The strong dollar is creating significant revenue headwinds.",
                "Macroeconomic uncertainty is causing customers to delay purchases.",
                "Rising interest rates are increasing our cost of capital.",
                "Inflation is reducing consumer spending in key markets.",
                "Economic recession is dampening enterprise spending.",
                "Currency fluctuations negatively impacted international revenue.",
            ],
            "safe": [
                "The macroeconomic environment is favorable for our business.",
                "FX was a tailwind this quarter, boosting reported revenue.",
                "Customer spending remains resilient despite macro concerns.",
                "Interest rates are stabilizing, reducing uncertainty.",
            ],
        },
        "liquidity_risk": {
            "risk": [
                "Our cash burn rate is unsustainable without new funding.",
                "Debt maturities coming due create refinancing pressure.",
                "Free cash flow turned negative this quarter.",
                "Our credit rating is at risk of downgrade.",
                "Working capital needs increased substantially.",
                "We suspended dividends and buybacks to preserve cash.",
            ],
            "safe": [
                "We generated strong free cash flow this quarter.",
                "Our balance sheet is healthy with ample liquidity.",
                "We returned significant capital to shareholders.",
                "Cash position is strong with low leverage.",
            ],
        },
        "tech_execution_risk": {
            "risk": [
                "Our product launch has been delayed due to technical issues.",
                "We experienced a cybersecurity breach compromising data.",
                "Cloud migration is taking longer and costing more than planned.",
                "Legacy system integration is causing engineering challenges.",
                "Quality issues forced us to patch our latest release.",
                "Technical debt is slowing down our development velocity.",
            ],
            "safe": [
                "Product launches are on track and ahead of schedule.",
                "Our platform is stable, scalable, and performing well.",
                "The migration completed successfully on time and budget.",
                "Our engineering team is executing at a high level.",
            ],
        },
        "labor_risk": {
            "risk": [
                "We are struggling to hire and retain engineering talent.",
                "Employee attrition has increased to concerning levels.",
                "Layoffs are affecting morale and productivity.",
                "Rising wage costs are pressuring operating margins.",
                "Skills gaps in key areas are slowing our roadmap.",
                "Union negotiations may result in higher labor costs.",
            ],
            "safe": [
                "We attracted top talent and retention rates improved.",
                "Employee satisfaction scores are at all-time highs.",
                "Our workforce is stable and highly engaged.",
                "We are fully staffed in all critical positions.",
            ],
        },
        "geopolitical_risk": {
            "risk": [
                "Trade tensions are disrupting our international operations.",
                "New tariffs will increase our import costs significantly.",
                "Political instability in key markets is hurting our business.",
                "Export restrictions are limiting our ability to serve customers.",
                "Regional conflicts have disrupted our local operations.",
                "Decoupling pressures are forcing costly supply chain changes.",
            ],
            "safe": [
                "International operations are running smoothly.",
                "Trade relations are stable in our key markets.",
                "We have minimal exposure to geopolitically sensitive regions.",
                "Our global diversification provides resilience.",
            ],
        },
    }

    # Per-category thresholds tuned on 49-company evaluation
    # v5: relaxed min_diff to improve recall on real transcripts where risk language
    # is more subtle than anchor sentences; min_sim kept to avoid false positives
    CATEGORY_THRESHOLDS = {
        "fx_macro_risk":       {"min_sim": 0.36, "min_diff": 0.02},
        "geopolitical_risk":   {"min_sim": 0.35, "min_diff": 0.05},
        "margin_risk":         {"min_sim": 0.40, "min_diff": 0.05},
        "regulatory_risk":     {"min_sim": 0.33, "min_diff": 0.07},
        "tech_execution_risk": {"min_sim": 0.38, "min_diff": 0.07},
        "supply_chain_risk":   {"min_sim": 0.37, "min_diff": 0.08},
        "liquidity_risk":      {"min_sim": 0.40, "min_diff": 0.05},
        "demand_risk":         {"min_sim": 0.40, "min_diff": 0.08},
        "competition_risk":    {"min_sim": 0.40, "min_diff": 0.10},
        "labor_risk":          {"min_sim": 0.38, "min_diff": 0.10},
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = None
        self._model_name = model_name
        self._risk_matrix = None
        self._safe_matrix = None
        self._cat_names = None

    def _load(self):
        """Lazy-load model and precompute anchor embeddings."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401

        logger.info(f"Loading semantic risk model: {self._model_name}")
        self._model = SentenceTransformer(self._model_name)

        self._cat_names = list(self.RISK_ANCHORS.keys())
        risk_centroids = []
        safe_centroids = []
        for cat in self._cat_names:
            anchors = self.RISK_ANCHORS[cat]
            risk_centroids.append(
                np.mean(self._model.encode(anchors["risk"], show_progress_bar=False), axis=0)
            )
            safe_centroids.append(
                np.mean(self._model.encode(anchors["safe"], show_progress_bar=False), axis=0)
            )
        self._risk_matrix = np.stack(risk_centroids)
        self._safe_matrix = np.stack(safe_centroids)

    def detect_semantic_risks(self, texts: List[str]) -> List[List[Dict]]:
        """Detect risk categories semantically for a list of texts.

        Returns a list (one per text) of detected risk dicts with category,
        similarity score, differential score, and the closest matching risk
        anchor sentence (explains *why* the chunk was flagged).
        """
        from sklearn.metrics.pairwise import cosine_similarity

        self._load()

        embeddings = self._model.encode(texts, show_progress_bar=False, batch_size=32)
        risk_sims = cosine_similarity(embeddings, self._risk_matrix)
        safe_sims = cosine_similarity(embeddings, self._safe_matrix)
        diff_scores = risk_sims - safe_sims

        # Pre-encode all individual anchor sentences for closest-match lookup
        if not hasattr(self, "_anchor_embeddings"):
            self._anchor_embeddings = {}
            for cat in self._cat_names:
                anchors = self.RISK_ANCHORS[cat]["risk"]
                self._anchor_embeddings[cat] = self._model.encode(
                    anchors, show_progress_bar=False)

        results = []
        for i in range(len(texts)):
            detections = []
            for j, cat in enumerate(self._cat_names):
                t = self.CATEGORY_THRESHOLDS[cat]
                if risk_sims[i][j] > t["min_sim"] and diff_scores[i][j] > t["min_diff"]:
                    # Find closest matching anchor sentence
                    anchor_sims = cosine_similarity(
                        embeddings[i:i+1], self._anchor_embeddings[cat])[0]
                    best_anchor_idx = int(anchor_sims.argmax())
                    best_anchor = self.RISK_ANCHORS[cat]["risk"][best_anchor_idx]

                    detections.append({
                        "category": cat,
                        "risk_similarity": float(risk_sims[i][j]),
                        "differential": float(diff_scores[i][j]),
                        "severity": "medium",
                        "detection_method": "semantic",
                        "matched_anchor": best_anchor,
                    })
            results.append(detections)

        return results


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
