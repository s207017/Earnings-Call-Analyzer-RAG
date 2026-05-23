"""Test semantic risk detection using sentence-transformers embeddings.

Compares keyword-based detection vs embedding similarity approach.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent

# ── 1. Load model ──
print("Loading sentence-transformer model...")
t0 = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Model loaded in {time.time() - t0:.1f}s\n")

# ── 2. Build rich risk category descriptions ──
# These go beyond keywords — they describe what each risk *means* semantically
RISK_DESCRIPTIONS = {
    "demand_risk": [
        "Customer demand is declining and orders are being cancelled.",
        "Revenue is falling because fewer customers are buying products.",
        "The market is shrinking and consumers are spending less.",
        "Sales volume has dropped significantly compared to last quarter.",
        "We are seeing weakness in our pipeline and fewer new deals.",
        "Subscriber growth has stalled and churn rates are increasing.",
    ],
    "margin_risk": [
        "Profit margins are being squeezed due to rising input costs.",
        "We cannot raise prices fast enough to offset cost inflation.",
        "Gross margins declined because of unfavorable product mix.",
        "Operating expenses grew faster than revenue this quarter.",
        "Our profitability is under pressure from competitive pricing.",
        "Higher raw material and labor costs are eating into margins.",
    ],
    "supply_chain_risk": [
        "Our supply chain has been disrupted by component shortages.",
        "We are experiencing delays in shipping and logistics.",
        "Inventory levels are too high and we may need write-downs.",
        "We depend on a single supplier for critical components.",
        "Manufacturing capacity constraints are limiting our output.",
        "Lead times have extended significantly for key materials.",
    ],
    "regulatory_risk": [
        "We face potential fines from government regulatory agencies.",
        "New regulations could increase our compliance costs significantly.",
        "There is an ongoing antitrust investigation into our practices.",
        "Data privacy laws like GDPR are creating operational challenges.",
        "We are involved in litigation that could result in material settlements.",
        "Tax authorities are auditing our international transfer pricing.",
    ],
    "competition_risk": [
        "Competitors are gaining market share with aggressive pricing.",
        "New entrants are disrupting our traditional business model.",
        "We are losing deals to competitors who offer lower prices.",
        "Our competitive differentiation is eroding in the market.",
        "The market is becoming commoditized and margins are shrinking.",
        "Our win rates have declined as competition intensifies.",
    ],
    "fx_macro_risk": [
        "The strong US dollar is creating significant revenue headwinds.",
        "Macroeconomic uncertainty is causing customers to delay purchases.",
        "Rising interest rates are increasing our borrowing costs.",
        "Inflation is affecting consumer spending in our key markets.",
        "Economic recession fears are dampening enterprise IT spending.",
        "Currency fluctuations reduced our international revenue by several points.",
    ],
    "liquidity_risk": [
        "Our cash burn rate is unsustainable without additional funding.",
        "We need to refinance debt that matures within the next year.",
        "Free cash flow turned negative due to heavy capital investments.",
        "Our credit rating is at risk of being downgraded.",
        "Working capital requirements have increased substantially.",
        "We suspended the share buyback program to preserve cash.",
    ],
    "tech_execution_risk": [
        "Our product launch has been delayed due to technical issues.",
        "We experienced a cybersecurity breach that compromised customer data.",
        "The cloud migration is taking longer and costing more than planned.",
        "Legacy system integration is creating significant engineering challenges.",
        "Our platform scalability is being tested by rapid user growth.",
        "Quality issues forced us to recall and patch our latest release.",
    ],
    "labor_risk": [
        "We are struggling to hire and retain skilled engineering talent.",
        "Employee attrition rates have increased to concerning levels.",
        "We announced layoffs affecting a significant portion of our workforce.",
        "Rising wage costs are putting pressure on our operating margins.",
        "The transition to hybrid work has created productivity challenges.",
        "Union negotiations could result in higher labor costs next year.",
    ],
    "geopolitical_risk": [
        "Trade tensions between the US and China affect our operations.",
        "New tariffs on imported goods will increase our costs.",
        "Political instability in key markets is disrupting our business.",
        "We are diversifying our supply chain away from geopolitical hotspots.",
        "Export restrictions are limiting our ability to sell in certain countries.",
        "The conflict in the region has disrupted our local operations.",
    ],
}

# ── 3. Embed all risk descriptions ──
print("Embedding risk category descriptions...")
category_embeddings = {}
for cat, descriptions in RISK_DESCRIPTIONS.items():
    embeddings = model.encode(descriptions, show_progress_bar=False)
    # Use mean of all description embeddings as the category centroid
    category_embeddings[cat] = np.mean(embeddings, axis=0)

print(f"Embedded {len(category_embeddings)} risk categories\n")

# ── 4. Load real transcript chunks ──
chunks_path = PROJECT_ROOT / "data" / "processed" / "chunks_with_risk.parquet"
df = pd.read_parquet(chunks_path)

# Parse risk_categories if stored as JSON strings
if "risk_categories" in df.columns and df["risk_categories"].dtype == object:
    df["risk_categories"] = df["risk_categories"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

print(f"Loaded {len(df)} chunks from dataset")
print(f"Chunks with keyword-detected risks: {(df['risk_count'] > 0).sum()}")
print()

# ── 5. Test on a sample ──
# Pick some interesting test cases:
# a) Chunks that keywords flagged as risky
# b) Chunks that keywords missed but might be semantically risky
# c) Clearly safe chunks

# Sample from MSFT for consistency
msft = df[df["ticker"] == "MSFT"].copy()
print(f"MSFT chunks: {len(msft)}, with keyword risks: {(msft['risk_count'] > 0).sum()}\n")

# Take a mix of chunks
sample_risky = msft[msft["risk_count"] > 0].head(5)
sample_clean = msft[msft["risk_count"] == 0].sample(10, random_state=42)
sample = pd.concat([sample_risky, sample_clean])

# ── 6. Semantic scoring ──
print("=" * 80)
print("SEMANTIC RISK DETECTION RESULTS")
print("=" * 80)

THRESHOLD = 0.35  # Similarity threshold for risk detection

# Stack all category centroids
cat_names = list(category_embeddings.keys())
cat_matrix = np.stack([category_embeddings[c] for c in cat_names])

# Embed all sample texts at once
t0 = time.time()
sample_texts = sample["text"].tolist()
text_embeddings = model.encode(sample_texts, show_progress_bar=False)
embed_time = time.time() - t0

# Compute cosine similarities
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(text_embeddings, cat_matrix)

print(f"\nEmbedding time for {len(sample)} chunks: {embed_time:.2f}s")
print(f"({embed_time/len(sample)*1000:.0f}ms per chunk)\n")

for i, (_, row) in enumerate(sample.iterrows()):
    text_preview = row["text"][:150].replace("\n", " ")
    keyword_cats = row["risk_categories"] if isinstance(row["risk_categories"], list) else []

    # Semantic detections above threshold
    chunk_sims = similarities[i]
    semantic_cats = [(cat_names[j], chunk_sims[j]) for j in range(len(cat_names))
                     if chunk_sims[j] > THRESHOLD]
    semantic_cats.sort(key=lambda x: -x[1])

    # Format
    kw_str = ", ".join(keyword_cats) if keyword_cats else "none"
    sem_str = ", ".join(f"{c} ({s:.3f})" for c, s in semantic_cats) if semantic_cats else "none"

    # Highlight differences
    kw_set = set(keyword_cats)
    sem_set = set(c for c, _ in semantic_cats)
    new_finds = sem_set - kw_set
    missed = kw_set - sem_set

    print(f"─── Chunk: {row['speaker']} ({row['role']}) ───")
    print(f"Text: {text_preview}...")
    print(f"  Keywords:  {kw_str}")
    print(f"  Semantic:  {sem_str}")
    if new_finds:
        print(f"  ✦ NEW semantic finds: {', '.join(new_finds)}")
    if missed:
        print(f"  ✧ Keywords only (semantic missed): {', '.join(missed)}")
    print()

# ── 7. Full evaluation on MSFT ──
print("=" * 80)
print("FULL EVALUATION — ALL MSFT CHUNKS")
print("=" * 80)

all_embeddings = model.encode(msft["text"].tolist(), show_progress_bar=True, batch_size=32)
all_sims = cosine_similarity(all_embeddings, cat_matrix)

# Compare keyword vs semantic at different thresholds
for threshold in [0.30, 0.35, 0.40, 0.45]:
    keyword_total = 0
    semantic_total = 0
    overlap = 0
    semantic_only = 0
    keyword_only = 0

    for i, (_, row) in enumerate(msft.iterrows()):
        kw_cats = set(row["risk_categories"]) if isinstance(row["risk_categories"], list) else set()
        sem_cats = set(cat_names[j] for j in range(len(cat_names))
                       if all_sims[i][j] > threshold)

        keyword_total += len(kw_cats)
        semantic_total += len(sem_cats)
        overlap += len(kw_cats & sem_cats)
        semantic_only += len(sem_cats - kw_cats)
        keyword_only += len(kw_cats - sem_cats)

    print(f"\nThreshold: {threshold}")
    print(f"  Keyword detections:  {keyword_total}")
    print(f"  Semantic detections: {semantic_total}")
    print(f"  Overlap (both):      {overlap}")
    print(f"  Semantic only (new): {semantic_only}")
    print(f"  Keyword only:        {keyword_only}")

# ── 8. Show top semantic-only finds (what keywords missed) ──
print("\n" + "=" * 80)
print("TOP SEMANTIC-ONLY DETECTIONS (keywords missed these)")
print("=" * 80)

EVAL_THRESHOLD = 0.40
semantic_only_finds = []

for i, (_, row) in enumerate(msft.iterrows()):
    kw_cats = set(row["risk_categories"]) if isinstance(row["risk_categories"], list) else set()
    for j, cat in enumerate(cat_names):
        if all_sims[i][j] > EVAL_THRESHOLD and cat not in kw_cats:
            semantic_only_finds.append({
                "text": row["text"][:200],
                "speaker": row["speaker"],
                "category": cat,
                "similarity": all_sims[i][j],
            })

semantic_only_finds.sort(key=lambda x: -x["similarity"])
for item in semantic_only_finds[:15]:
    print(f"\n  Category: {item['category']} (sim={item['similarity']:.3f})")
    print(f"  Speaker: {item['speaker']}")
    print(f"  Text: {item['text']}...")
