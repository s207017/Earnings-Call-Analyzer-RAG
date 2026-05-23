"""Test semantic risk detection v2: embedding similarity + sentiment filter.

Only flags chunks that are both topically relevant to a risk category
AND carry negative sentiment (not just mentioning the topic positively).
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).parent.parent

# ── 1. Load model ──
print("Loading sentence-transformer model...")
t0 = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Model loaded in {time.time() - t0:.1f}s\n")

# ── 2. Risk descriptions: negative framing only ──
# Each category has RISK descriptions (negative) and SAFE counter-examples (positive)
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

# ── 3. Embed risk and safe anchors, compute directional vectors ──
print("Embedding risk anchors (risk + safe pairs)...")
category_risk_centroids = {}
category_safe_centroids = {}
category_directions = {}  # risk_centroid - safe_centroid = direction toward risk

for cat, anchors in RISK_ANCHORS.items():
    risk_embs = model.encode(anchors["risk"], show_progress_bar=False)
    safe_embs = model.encode(anchors["safe"], show_progress_bar=False)
    risk_centroid = np.mean(risk_embs, axis=0)
    safe_centroid = np.mean(safe_embs, axis=0)
    category_risk_centroids[cat] = risk_centroid
    category_safe_centroids[cat] = safe_centroid
    # Direction vector: from safe toward risk
    direction = risk_centroid - safe_centroid
    category_directions[cat] = direction / np.linalg.norm(direction)

print(f"Embedded {len(category_risk_centroids)} categories with risk/safe pairs\n")

# ── 4. Load data ──
chunks_path = PROJECT_ROOT / "data" / "processed" / "chunks_with_risk.parquet"
df = pd.read_parquet(chunks_path)
if "risk_categories" in df.columns and df["risk_categories"].dtype == object:
    df["risk_categories"] = df["risk_categories"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

msft = df[df["ticker"] == "MSFT"].copy()
print(f"MSFT: {len(msft)} chunks, {(msft['risk_count'] > 0).sum()} with keyword risks\n")

# ── 5. Embed all chunks ──
print("Embedding all MSFT chunks...")
t0 = time.time()
all_embeddings = model.encode(msft["text"].tolist(), show_progress_bar=True, batch_size=32)
print(f"Embedded in {time.time() - t0:.1f}s\n")

# ── 6. Scoring method: risk similarity minus safe similarity ──
# A chunk is risky if it's much closer to the risk anchors than the safe anchors

cat_names = list(category_risk_centroids.keys())
risk_matrix = np.stack([category_risk_centroids[c] for c in cat_names])
safe_matrix = np.stack([category_safe_centroids[c] for c in cat_names])

risk_sims = cosine_similarity(all_embeddings, risk_matrix)
safe_sims = cosine_similarity(all_embeddings, safe_matrix)

# Differential score: how much more similar to risk than safe
diff_scores = risk_sims - safe_sims

print("=" * 90)
print("METHOD: Differential scoring (risk_similarity - safe_similarity)")
print("=" * 90)

for threshold in [0.02, 0.04, 0.06, 0.08, 0.10]:
    keyword_total = 0
    semantic_total = 0
    overlap = 0
    semantic_only = 0
    keyword_only = 0

    for i, (_, row) in enumerate(msft.iterrows()):
        kw_cats = set(row["risk_categories"]) if isinstance(row["risk_categories"], list) else set()
        # Require: risk_sim > 0.35 AND diff > threshold
        sem_cats = set()
        for j, cat in enumerate(cat_names):
            if risk_sims[i][j] > 0.35 and diff_scores[i][j] > threshold:
                sem_cats.add(cat)

        keyword_total += len(kw_cats)
        semantic_total += len(sem_cats)
        overlap += len(kw_cats & sem_cats)
        semantic_only += len(sem_cats - kw_cats)
        keyword_only += len(kw_cats - sem_cats)

    print(f"\nDiff threshold: {threshold} (+ min risk_sim > 0.35)")
    print(f"  Keyword detections:  {keyword_total}")
    print(f"  Semantic detections: {semantic_total}")
    print(f"  Overlap (both):      {overlap}")
    print(f"  Semantic only (new): {semantic_only}")
    print(f"  Keyword only:        {keyword_only}")

# ── 7. Show sample detections at best threshold ──
BEST_THRESHOLD = 0.06
MIN_RISK_SIM = 0.35

print("\n" + "=" * 90)
print(f"SAMPLE DETECTIONS (diff > {BEST_THRESHOLD}, risk_sim > {MIN_RISK_SIM})")
print("=" * 90)

# Show chunks where semantic found risks that keywords missed
semantic_only_finds = []
for i, (_, row) in enumerate(msft.iterrows()):
    kw_cats = set(row["risk_categories"]) if isinstance(row["risk_categories"], list) else set()
    for j, cat in enumerate(cat_names):
        if (risk_sims[i][j] > MIN_RISK_SIM
                and diff_scores[i][j] > BEST_THRESHOLD
                and cat not in kw_cats):
            semantic_only_finds.append({
                "text": row["text"][:250],
                "speaker": row["speaker"],
                "category": cat,
                "risk_sim": risk_sims[i][j],
                "safe_sim": safe_sims[i][j],
                "diff": diff_scores[i][j],
            })

semantic_only_finds.sort(key=lambda x: -x["diff"])

print(f"\nTop 20 SEMANTIC-ONLY detections (keywords missed):\n")
for item in semantic_only_finds[:20]:
    print(f"  Category: {item['category']}")
    print(f"  Risk sim: {item['risk_sim']:.3f} | Safe sim: {item['safe_sim']:.3f} | Diff: {item['diff']:.3f}")
    print(f"  Speaker: {item['speaker']}")
    print(f"  Text: {item['text']}...")
    print()

# ── 8. Show false positive check: chunks that keywords flagged but semantic didn't ──
print("=" * 90)
print("KEYWORD-ONLY DETECTIONS (semantic filtered these out — are they false positives?)")
print("=" * 90)

keyword_only_finds = []
for i, (_, row) in enumerate(msft.iterrows()):
    kw_cats = set(row["risk_categories"]) if isinstance(row["risk_categories"], list) else set()
    for cat in kw_cats:
        j = cat_names.index(cat)
        if not (risk_sims[i][j] > MIN_RISK_SIM and diff_scores[i][j] > BEST_THRESHOLD):
            keyword_only_finds.append({
                "text": row["text"][:250],
                "speaker": row["speaker"],
                "category": cat,
                "risk_sim": risk_sims[i][j],
                "safe_sim": safe_sims[i][j],
                "diff": diff_scores[i][j],
            })

print(f"\nTotal keyword-only: {len(keyword_only_finds)}")
print(f"Showing 10 random samples:\n")
import random
random.seed(42)
for item in random.sample(keyword_only_finds, min(10, len(keyword_only_finds))):
    print(f"  Category: {item['category']}")
    print(f"  Risk sim: {item['risk_sim']:.3f} | Safe sim: {item['safe_sim']:.3f} | Diff: {item['diff']:.3f}")
    print(f"  Speaker: {item['speaker']}")
    print(f"  Text: {item['text']}...")
    print()

# ── 9. Hybrid approach: combine keyword + semantic ──
print("=" * 90)
print("HYBRID: keyword OR (semantic with high confidence)")
print("=" * 90)

for sem_thresh in [0.06, 0.08, 0.10]:
    hybrid_total = 0
    keyword_total = 0

    for i, (_, row) in enumerate(msft.iterrows()):
        kw_cats = set(row["risk_categories"]) if isinstance(row["risk_categories"], list) else set()
        sem_cats = set()
        for j, cat in enumerate(cat_names):
            if risk_sims[i][j] > MIN_RISK_SIM and diff_scores[i][j] > sem_thresh:
                sem_cats.add(cat)

        hybrid = kw_cats | sem_cats
        keyword_total += len(kw_cats)
        hybrid_total += len(hybrid)

    extra = hybrid_total - keyword_total
    print(f"\n  Semantic diff threshold: {sem_thresh}")
    print(f"  Keyword-only total: {keyword_total}")
    print(f"  Hybrid total:       {hybrid_total} (+{extra} new, {extra/max(keyword_total,1)*100:.0f}% more)")
