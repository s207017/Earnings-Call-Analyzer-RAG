"""Test semantic risk detection v4: per-category thresholds.

Uses aggressive thresholds for noisy categories and relaxed thresholds
for categories where semantic detection performs well.
"""

import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).parent.parent
random.seed(42)

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

# Per-category thresholds based on v3 noise analysis
# Low noise → lower threshold (more sensitive)
# High noise → higher threshold (more conservative) or disabled
CATEGORY_THRESHOLDS = {
    "fx_macro_risk":      {"min_sim": 0.40, "min_diff": 0.08},   # 45% noise — best performer
    "geopolitical_risk":  {"min_sim": 0.40, "min_diff": 0.10},   # 52% noise — decent
    "margin_risk":        {"min_sim": 0.45, "min_diff": 0.12},   # 89% noise — tighten
    "regulatory_risk":    {"min_sim": 0.42, "min_diff": 0.14},   # 80% noise — tighten
    "tech_execution_risk":{"min_sim": 0.42, "min_diff": 0.13},   # 87% noise — tighten
    "supply_chain_risk":  {"min_sim": 0.42, "min_diff": 0.14},   # 91% noise — tighten hard
    "liquidity_risk":     {"min_sim": 0.45, "min_diff": 0.15},   # 96% noise — very tight
    "demand_risk":        {"min_sim": 0.45, "min_diff": 0.15},   # 96% noise — very tight
    "competition_risk":   {"min_sim": 0.50, "min_diff": 0.18},   # 99% noise — near-disabled
    "labor_risk":         {"min_sim": 0.45, "min_diff": 0.18},   # 96% noise — near-disabled
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
print(f"Loaded {len(df)} chunks across {len(tickers)} companies\n")


def semantic_detect(risk_sims_row, diff_scores_row):
    """Return set of categories detected by semantic model for one chunk."""
    cats = set()
    for j, cat in enumerate(cat_names):
        t = CATEGORY_THRESHOLDS[cat]
        if risk_sims_row[j] > t["min_sim"] and diff_scores_row[j] > t["min_diff"]:
            cats.add(cat)
    return cats


# ── 5. Per-company evaluation ──
print("=" * 110)
print(f"{'COMPANY':<10} {'CHUNKS':>6} {'KW':>6} {'SEM':>6} {'OLAP':>5} {'S_ONLY':>6} "
      f"{'K_ONLY':>6} {'HYBRID':>6} {'NEW%':>6}")
print("=" * 110)

all_company_results = {}

for ticker in tickers:
    company_df = df[df["ticker"] == ticker].copy()
    n = len(company_df)

    embeddings = model.encode(company_df["text"].tolist(), show_progress_bar=False, batch_size=32)
    r_sims = cosine_similarity(embeddings, risk_matrix)
    s_sims = cosine_similarity(embeddings, safe_matrix)
    diffs = r_sims - s_sims

    kw_total = 0
    sem_total = 0
    overlap = 0
    sem_only = 0
    kw_only = 0

    rows_data = []
    for i, (_, row) in enumerate(company_df.iterrows()):
        kw_cats = set(row["risk_categories"]) if isinstance(row["risk_categories"], list) else set()
        sem_cats = semantic_detect(r_sims[i], diffs[i])

        kw_total += len(kw_cats)
        sem_total += len(sem_cats)
        overlap += len(kw_cats & sem_cats)
        sem_only += len(sem_cats - kw_cats)
        kw_only += len(kw_cats - sem_cats)

        rows_data.append({
            "kw_cats": kw_cats, "sem_cats": sem_cats,
            "text": row["text"], "speaker": row["speaker"],
            "risk_sims": r_sims[i], "diffs": diffs[i],
        })

    hybrid = kw_total + sem_only
    new_pct = sem_only / max(kw_total, 1) * 100

    print(f"{ticker:<10} {n:>6} {kw_total:>6} {sem_total:>6} {overlap:>5} {sem_only:>6} "
          f"{kw_only:>6} {hybrid:>6} {new_pct:>5.1f}%")

    all_company_results[ticker] = {
        "kw": kw_total, "sem": sem_total, "overlap": overlap,
        "sem_only": sem_only, "kw_only": kw_only, "rows": rows_data,
    }

# ── 6. Aggregates ──
total_kw = sum(r["kw"] for r in all_company_results.values())
total_sem = sum(r["sem"] for r in all_company_results.values())
total_olap = sum(r["overlap"] for r in all_company_results.values())
total_sem_only = sum(r["sem_only"] for r in all_company_results.values())
total_kw_only = sum(r["kw_only"] for r in all_company_results.values())

print("=" * 110)
print(f"{'TOTAL':<10} {'':>6} {total_kw:>6} {total_sem:>6} {total_olap:>5} {total_sem_only:>6} "
      f"{total_kw_only:>6} {total_kw + total_sem_only:>6} "
      f"{total_sem_only / max(total_kw, 1) * 100:>5.1f}%")

# ── 7. Per-category stats ──
print("\n" + "=" * 110)
print("PER-CATEGORY BREAKDOWN (tuned thresholds)")
print("=" * 110)

cat_stats = {cat: {"kw": 0, "sem": 0, "overlap": 0, "sem_only": 0, "kw_only": 0} for cat in cat_names}

for res in all_company_results.values():
    for rd in res["rows"]:
        for cat in cat_names:
            is_kw = cat in rd["kw_cats"]
            is_sem = cat in rd["sem_cats"]
            if is_kw: cat_stats[cat]["kw"] += 1
            if is_sem: cat_stats[cat]["sem"] += 1
            if is_kw and is_sem: cat_stats[cat]["overlap"] += 1
            if is_sem and not is_kw: cat_stats[cat]["sem_only"] += 1
            if is_kw and not is_sem: cat_stats[cat]["kw_only"] += 1

t = CATEGORY_THRESHOLDS
print(f"\n{'CATEGORY':<22} {'THRESH':>14} {'KW':>6} {'SEM':>6} {'OLAP':>6} {'S_ONLY':>7} "
      f"{'K_ONLY':>7} {'NOISE%':>7}")
print("-" * 85)
for cat in cat_names:
    s = cat_stats[cat]
    noise = s["sem_only"] / max(s["sem"], 1) * 100
    thresh_str = f"s>{t[cat]['min_sim']:.2f} d>{t[cat]['min_diff']:.2f}"
    print(f"{cat:<22} {thresh_str:>14} {s['kw']:>6} {s['sem']:>6} {s['overlap']:>6} "
          f"{s['sem_only']:>7} {s['kw_only']:>7} {noise:>6.1f}%")

# ── 8. Quality spot-check: 5 random semantic-only finds per category ──
print("\n" + "=" * 110)
print("QUALITY SPOT-CHECK: Random semantic-only finds per category")
print("=" * 110)

for cat_idx, cat in enumerate(cat_names):
    finds = []
    for ticker, res in all_company_results.items():
        for rd in res["rows"]:
            if cat in rd["sem_cats"] and cat not in rd["kw_cats"]:
                finds.append({
                    "ticker": ticker,
                    "text": rd["text"][:250],
                    "speaker": rd["speaker"],
                    "diff": rd["diffs"][cat_idx],
                    "risk_sim": rd["risk_sims"][cat_idx],
                })

    print(f"\n── {cat} ({len(finds)} semantic-only finds) ──")
    if not finds:
        print("  (none)")
        continue

    for item in random.sample(finds, min(5, len(finds))):
        print(f"  [{item['ticker']}] diff={item['diff']:.3f} sim={item['risk_sim']:.3f}")
        print(f"  {item['text'][:200]}...")
        # Quick quality label
        print()

# ── 9. Hybrid detection summary for select companies ──
print("\n" + "=" * 110)
print("HYBRID DETECTION EXAMPLES (keyword + semantic)")
print("=" * 110)

for ticker in ["AAPL", "TSLA", "AMZN", "NFLX", "META"]:
    if ticker not in all_company_results:
        continue
    res = all_company_results[ticker]
    print(f"\n{'─' * 40} {ticker} {'─' * 40}")
    print(f"  Keywords: {res['kw']} | Semantic: {res['sem']} | "
          f"Overlap: {res['overlap']} | New: {res['sem_only']}")

    # Show top 3 semantic-only finds for this company
    sem_finds = []
    for rd in res["rows"]:
        new_cats = rd["sem_cats"] - rd["kw_cats"]
        if new_cats:
            best_cat = max(new_cats, key=lambda c: rd["diffs"][cat_names.index(c)])
            cidx = cat_names.index(best_cat)
            sem_finds.append({
                "cat": best_cat,
                "diff": rd["diffs"][cidx],
                "text": rd["text"][:200],
                "speaker": rd["speaker"],
            })

    sem_finds.sort(key=lambda x: -x["diff"])
    if sem_finds:
        print(f"  Top semantic-only finds:")
        for item in sem_finds[:3]:
            print(f"    [{item['cat']}] diff={item['diff']:.3f} — {item['speaker']}")
            print(f"    {item['text']}...")
            print()
