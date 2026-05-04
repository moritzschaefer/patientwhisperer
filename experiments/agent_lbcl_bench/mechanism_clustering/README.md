# Clustering and Novelty Analysis of Level 3 Agent-Generated Mechanisms

## Goal

From the 713 per-patient mechanism instances (712 unique free-text descriptions across 79 patients), produce:
1. A **deduplicated set of distinct mechanism types** (e.g., "T cell exhaustion" clusters ~50 patient-specific phrasings)
2. A **classification of each cluster** as: (a) matching LBCL-Bench, (b) known in literature but not in LBCL-Bench, or (c) potentially novel
3. **Counts**: how many distinct mechanism types were identified, how many are novel, how many patients contribute to each

These numbers feed directly into the Emmy Noether proposal (v11, preliminary results section).

## Input Data

- `results/step3_evaluation/all_patient_mechanisms.csv` — 713 rows: patient_id, response, mechanism (free text), confidence, direction
- `data/lbcl_bench_filtered.csv` — 21 benchmark mechanisms with verbal_summary and consolidated_description
- `results/step3_evaluation/bench_mechanism_patient_counts.csv` — which benchmark mechanisms were matched (13/21)

## Approach

### Step 1: Embed all mechanism descriptions

Use a text embedding model to embed all 713 mechanism descriptions + 21 LBCL-Bench verbal summaries into the same space.

```python
# Option A: sentence-transformers (local, fast)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # or 'all-mpnet-base-v2' for higher quality
embeddings = model.encode(mechanism_texts)

# Option B: OpenAI embeddings (higher quality, API cost)
# Use text-embedding-3-small via Stanford AI API
```

**Recommendation:** Start with sentence-transformers locally. If clusters are noisy, try OpenAI embeddings.

### Step 2: Cluster mechanism descriptions

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np

# Try a range of distance thresholds
for n_clusters in [20, 30, 40, 50, 60, 80]:
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
    labels = clustering.fit_predict(embeddings[:713])  # only agent mechanisms
    score = silhouette_score(embeddings[:713], labels, metric='cosine')
    print(f"n_clusters={n_clusters}, silhouette={score:.3f}")

# Alternative: HDBSCAN for automatic cluster count
from hdbscan import HDBSCAN
clusterer = HDBSCAN(min_cluster_size=3, metric='cosine', min_samples=1)
labels = clusterer.fit_predict(embeddings[:713])
```

**Expected output:** ~30–60 distinct mechanism types (rough estimate based on scanning the mechanism_frequency.csv — themes like exhaustion, metabolic fitness, Tregs, CD8 cytotoxicity, monocyte contamination, senescence, etc. recur across patients).

### Step 3: Label each cluster

For each cluster, generate a representative label:
```python
# Take the 3 most central mechanisms in each cluster
# Feed to an LLM: "These are 3 examples from a cluster of N mechanism descriptions 
# identified by AI agents in CAR T cell therapy analysis. 
# Give a concise (5-10 word) label for this mechanism type."
```

### Step 4: Match clusters to LBCL-Bench

For each cluster, compute cosine similarity to each of the 21 LBCL-Bench verbal summaries. Threshold at e.g. 0.6 for "matching."

```python
from sklearn.metrics.pairwise import cosine_similarity

# cluster_centroids: mean embedding per cluster
# bench_embeddings: 21 LBCL-Bench embeddings
sim_matrix = cosine_similarity(cluster_centroids, bench_embeddings)

for i, cluster_label in enumerate(cluster_labels):
    best_bench_idx = sim_matrix[i].argmax()
    best_sim = sim_matrix[i].max()
    status = "BENCH-MATCH" if best_sim > 0.6 else "NOVEL"
    print(f"{cluster_label}: {status} (sim={best_sim:.2f} to {bench_summaries[best_bench_idx]})")
```

**Alternative (more reliable):** Use an LLM judge (same as used for step3_evaluate.py) to classify each cluster:
```
Given this cluster of agent-identified mechanisms:
[top 5 examples]

And this LBCL-Bench entry:
[verbal_summary]

Is this cluster describing the same or a substantially overlapping biological mechanism? (yes/partial/no)
```

### Step 5: Classify novel clusters

For clusters that don't match LBCL-Bench, classify as:
- **(b) Known in broader CAR T/immunotherapy literature** — use LLM: "Is this mechanism well-established in the CAR T cell therapy or cancer immunology literature?"
- **(c) Potentially novel** — not established in literature

### Step 6: Output

1. **Table: mechanism clusters** — cluster_id, label, n_patients, n_OR, n_NR, direction, confidence_distribution, LBCL-Bench match (yes/no/which), novelty category
2. **Summary statistics:**
   - Total distinct mechanism types identified: X
   - Matching LBCL-Bench: Y (these are the 13 we already know)
   - Known but not in LBCL-Bench: Z
   - Potentially novel: W
3. **For the proposal:** "Per-patient analysis identified X distinct mechanism types across 79 patients. Of these, Y matched known LBCL-Bench entries (62% benchmark recall), Z were consistent with broader immunotherapy literature, and W represent potentially novel candidate mechanisms."

## Implementation Plan

```
cluster_novel_mechanisms.py
├── load_data()           # Read all_patient_mechanisms.csv + lbcl_bench_filtered.csv
├── embed_mechanisms()    # sentence-transformers or OpenAI
├── cluster_mechanisms()  # AgglomerativeClustering or HDBSCAN
├── label_clusters()      # LLM-based cluster labeling
├── match_to_bench()      # Cosine similarity or LLM judge
├── classify_novelty()    # LLM-based literature check
└── export_results()      # CSV + summary stats
```

Estimated runtime: ~5 min for embedding, ~10 min for LLM-based labeling/matching (if using ~50 clusters × ~25 LLM calls).

## Notes

- The existing `step3_evaluate.py` uses batched LLM matching (15 patients/batch). The same infrastructure can be reused for cluster-to-bench matching.
- The `step2_evaluate_recall.py` has the LLM-as-judge prompts that can be adapted.
- Run locally (embeddings are small) or on SNAP for LLM API calls.
