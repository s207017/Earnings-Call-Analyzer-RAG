"""Test semantic risk detection v3: multi-company evaluation.

Tests across all 10 companies in the dataset to validate the hybrid approach.
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

# ── 2. Risk/Safe anchors ──
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

# ── 3. Embed anchors ──
print("Embedding risk anchors...")
cat_names = list(RISK_ANCHORS.keys())
risk_centroids = {}
safe_centroids = {}
for cat, anchors in RISK_ANCHORS.items():
    risk_centroids[cat] = np.mean(model.encode(anchors["risk"], show_progress_bar=False), axis=0)
    safe_centroids[cat] = np.mean(model.encode(anchors["safe"], show_progress_bar=False), axis=0)

risk_matrix = np.stack([risk_centroids[c] for c in cat_names])
safe_matrix = np.stack([safe_centroids[c] for c in cat_names])

# ── 4. Load full dataset ──
chunks_path = PROJECT_ROOT / "data" / "processed" / "chunks_with_risk.parquet"
df = pd.read_parquet(chunks_path)
if "risk_categories" in df.columns and df["risk_categories"].dtype == object:
    df["risk_categories"] = df["risk_categories"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

tickers = sorted(df["ticker"].unique())
print(f"Loaded {len(df)} chunks across {len(tickers)} companies: {', '.join(tickers)}\n")

# ── 5. Per-company evaluation ──
MIN_RISK_SIM = 0.35
THRESHOLDS = [0.06, 0.08, 0.10]

print("=" * 100)
print(f"{'COMPANY':<8} {'CHUNKS':>7} {'KW_DET':>7} | ", end="")
for t in THRESHOLDS:
    print(f"t={t}: {'SEM':>5} {'OLAP':>5} {'S_ONLY':>6} {'K_ONLY':>6} | ", end="")
print()
print("=" * 100)

all_results = {}

for ticker in tickers:
    company_df = df[df["ticker"] == ticker]
    n_chunks = len(company_df)

    # Embed
    embeddings = model.encode(company_df["text"].tolist(), show_progress_bar=False, batch_size=32)
    risk_sims = cosine_similarity(embeddings, risk_matrix)
    safe_sims = cosine_similarity(embeddings, safe_matrix)
    diff_scores = risk_sims - safe_sims

    kw_total = sum(
        len(row["risk_categories"]) if isinstance(row["risk_categories"], list) else 0
        for _, row in company_df.iterrows()
    )

    print(f"{ticker:<8} {n_chunks:>7} {kw_total:>7} | ", end="")

    for threshold in THRESHOLDS:
        sem_total = 0
        overlap = 0
        sem_only = 0
        kw_only = 0

        for i, (_, row) in enumerate(company_df.iterrows()):
            kw_cats = set(row["risk_categories"]) if isinstance(row["risk_categories"], list) else set()
            sem_cats = set()
            for j, cat in enumerate(cat_names):
                if risk_sims[i][j] > MIN_RISK_SIM and diff_scores[i][j] > threshold:
                    sem_cats.add(cat)

            sem_total += len(sem_cats)
            overlap += len(kw_cats & sem_cats)
            sem_only += len(sem_cats - kw_cats)
            kw_only += len(kw_cats - sem_cats)

        print(f"     {sem_total:>5} {overlap:>5} {sem_only:>6} {kw_only:>6} | ", end="")

        if threshold == 0.08:
            all_results[ticker] = {
                "chunks": n_chunks, "kw_det": kw_total,
                "sem_det": sem_total, "overlap": overlap,
                "sem_only": sem_only, "kw_only": kw_only,
                "embeddings": embeddings, "risk_sims": risk_sims,
                "safe_sims": safe_sims, "diff_scores": diff_scores,
                "df": company_df,
            }

    print()

# ── 6. Aggregate stats at threshold 0.08 ──
print("\n" + "=" * 100)
print("AGGREGATE STATS (threshold = 0.08)")
print("=" * 100)

total_kw = sum(r["kw_det"] for r in all_results.values())
total_sem = sum(r["sem_det"] for r in all_results.values())
total_overlap = sum(r["overlap"] for r in all_results.values())
total_sem_only = sum(r["sem_only"] for r in all_results.values())
total_kw_only = sum(r["kw_only"] for r in all_results.values())

print(f"Total keyword detections:     {total_kw}")
print(f"Total semantic detections:    {total_sem}")
print(f"Overlap (both agree):         {total_overlap}")
print(f"Semantic only (new finds):    {total_sem_only}")
print(f"Keyword only (sem filtered):  {total_kw_only}")
print(f"Hybrid total:                 {total_kw + total_sem_only}")
print(f"New finds as % of keywords:   {total_sem_only / max(total_kw, 1) * 100:.1f}%")

# ── 7. Per-category breakdown across all companies ──
print("\n" + "=" * 100)
print("PER-CATEGORY BREAKDOWN (threshold = 0.08)")
print("=" * 100)

cat_stats = {cat: {"kw": 0, "sem": 0, "overlap": 0, "sem_only": 0, "kw_only": 0} for cat in cat_names}

for ticker, res in all_results.items():
    company_df = res["df"]
    risk_sims = res["risk_sims"]
    diff_scores = res["diff_scores"]

    for i, (_, row) in enumerate(company_df.iterrows()):
        kw_cats = set(row["risk_categories"]) if isinstance(row["risk_categories"], list) else set()
        for j, cat in enumerate(cat_names):
            is_sem = risk_sims[i][j] > MIN_RISK_SIM and diff_scores[i][j] > 0.08
            is_kw = cat in kw_cats

            if is_kw:
                cat_stats[cat]["kw"] += 1
            if is_sem:
                cat_stats[cat]["sem"] += 1
            if is_kw and is_sem:
                cat_stats[cat]["overlap"] += 1
            if is_sem and not is_kw:
                cat_stats[cat]["sem_only"] += 1
            if is_kw and not is_sem:
                cat_stats[cat]["kw_only"] += 1

print(f"\n{'CATEGORY':<22} {'KW':>6} {'SEM':>6} {'OLAP':>6} {'S_ONLY':>7} {'K_ONLY':>7} {'NOISE_RATIO':>12}")
print("-" * 75)
for cat in cat_names:
    s = cat_stats[cat]
    noise = s["sem_only"] / max(s["sem"], 1)
    print(f"{cat:<22} {s['kw']:>6} {s['sem']:>6} {s['overlap']:>6} {s['sem_only']:>7} {s['kw_only']:>7} {noise:>11.1%}")

# ── 8. Quality check: top semantic-only finds per category ──
print("\n" + "=" * 100)
print("TOP SEMANTIC-ONLY FINDS PER CATEGORY (quality check)")
print("=" * 100)

for cat_idx, cat in enumerate(cat_names):
    finds = []
    for ticker, res in all_results.items():
        company_df = res["df"]
        risk_sims = res["risk_sims"]
        diff_scores = res["diff_scores"]

        for i, (_, row) in enumerate(company_df.iterrows()):
            kw_cats = set(row["risk_categories"]) if isinstance(row["risk_categories"], list) else set()
            if (risk_sims[i][cat_idx] > MIN_RISK_SIM
                    and diff_scores[i][cat_idx] > 0.08
                    and cat not in kw_cats):
                finds.append({
                    "ticker": ticker,
                    "text": row["text"][:200],
                    "speaker": row["speaker"],
                    "diff": diff_scores[i][cat_idx],
                    "risk_sim": risk_sims[i][cat_idx],
                })

    finds.sort(key=lambda x: -x["diff"])

    print(f"\n── {cat} ({len(finds)} semantic-only finds) ──")
    for item in finds[:3]:
        print(f"  [{item['ticker']}] diff={item['diff']:.3f} risk_sim={item['risk_sim']:.3f}")
        print(f"  {item['text']}...")
        print()

# ── 9. Quality check: keyword-only finds per category (what semantic misses) ──
print("\n" + "=" * 100)
print("KEYWORD-ONLY SAMPLES PER CATEGORY (what semantic misses)")
print("=" * 100)

import random
random.seed(42)

for cat_idx, cat in enumerate(cat_names):
    finds = []
    for ticker, res in all_results.items():
        company_df = res["df"]
        risk_sims = res["risk_sims"]
        diff_scores = res["diff_scores"]

        for i, (_, row) in enumerate(company_df.iterrows()):
            kw_cats = set(row["risk_categories"]) if isinstance(row["risk_categories"], list) else set()
            if (cat in kw_cats
                    and not (risk_sims[i][cat_idx] > MIN_RISK_SIM
                             and diff_scores[i][cat_idx] > 0.08)):
                finds.append({
                    "ticker": ticker,
                    "text": row["text"][:200],
                    "diff": diff_scores[i][cat_idx],
                    "risk_sim": risk_sims[i][cat_idx],
                })

    print(f"\n── {cat} ({len(finds)} keyword-only) ──")
    for item in random.sample(finds, min(2, len(finds))):
        print(f"  [{item['ticker']}] diff={item['diff']:.3f} risk_sim={item['risk_sim']:.3f}")
        print(f"  {item['text']}...")
        print()
